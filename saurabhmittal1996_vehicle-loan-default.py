#Importing Required Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#SMOTE to balance the Imbalance Data
from imblearn.over_sampling import SMOTE

#for Spliting Data and Hyperparameter Tuning 
from sklearn.model_selection import train_test_split, GridSearchCV

#Importing Machine Learning Model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from catboost import CatBoostClassifier
    
#Bagging Algo
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

#To tranform data
from sklearn import preprocessing

#statistical Tools
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc

#Setting Format
pd.options.display.float_format = '{:.5f}'.format
pd.options.display.max_columns = None
pd.options.display.max_rows = None
train = pd.read_csv("../input/lt-vehicle-loan-default-prediction/train.csv")
train.loan_default.value_counts().plot(kind='bar')
train.loan_default.value_counts()
test = pd.read_csv("../input/lt-vehicle-loan-default-prediction/test.csv")
test.shape,train.shape
data = pd.concat([train,test],ignore_index=True)

#Replacing all the Spaces with '_'
data.columns=data.columns.str.replace('.','_')
data.shape
#Lets looks at data description
info = pd.read_csv("../input/lt-vehicle-loan-default-prediction/data_dictionary.csv")
info
data.head()
#Lets look at all the unique values in the data

for i in data.columns:
    print(i," : distinct_value")
    print(data[i].nunique()," : No. of unique Items")
    print(data[i].unique())
    print("-"*30)
    print("")
train.isna().sum()

#So only Employment Type data is missing
round(100*(test.isna().sum())/len(test), 2)
round(100*(data.isna().sum())/len(data), 2)
#Data Correlation
plt.figure(figsize=(12,8))
sns.heatmap(data.corr())
data.corr()['loan_default'].sort_values(ascending = False)
data.columns
#Lets Look at few columns

columns_unique = ['UniqueID','MobileNo_Avl_Flag',
         'Current_pincode_ID','Employee_code_ID',
         'NO_OF_INQUIRIES','State_ID',
         'branch_id','manufacturer_id','supplier_id']


unique_col = data[columns_unique]
unique_col.head()
#Looking at all unique values
for i in unique_col.columns:
    print(i," : distinct_value")
    print(unique_col[i].nunique()," : No. of unique Items")
    print(unique_col[i].unique())
    print("-"*30)
    print("")
unique_col.hist(bins=5,figsize=(16,12))
data.drop(unique_col,axis=1,inplace=True)
data.describe(include='all').T
#Now we have 2 Columns named "AVERAGE_ACCT_AGE" & "CREDIT_HISTORY_LENGTH".
#They have AplhNumeric Values Lets change them to Months

def change_col_month(col):
    year = int(col.split()[0].replace('yrs',''))
    month = int(col.split()[1].replace('mon',''))
    return year*12+month

data['CREDIT_HISTORY_LENGTH'] = data['CREDIT_HISTORY_LENGTH'].apply(change_col_month)
data['AVERAGE_ACCT_AGE'] = data['AVERAGE_ACCT_AGE'].apply(change_col_month)
data.head()
plot = data.iloc[:test.shape[0]]
plot = plot[plot['AVERAGE_ACCT_AGE']<175]
sns.lineplot(x=plot['AVERAGE_ACCT_AGE'],y=plot['loan_default'])
plot = data.iloc[:test.shape[0]]
plot = plot[plot['CREDIT_HISTORY_LENGTH']<200]
sns.lineplot(x=plot['CREDIT_HISTORY_LENGTH'],y=plot['loan_default'])
data.PERFORM_CNS_SCORE_DESCRIPTION.value_counts()
def replace_not_scored(n):
    #here we are spliting letters before '-'.
    score=n.split("-")
    
    if len(score)!=1:
       
        return score[0]
    else:
        return 'N'
data['CNS_SCORE_DESCRIPTION']=data['PERFORM_CNS_SCORE_DESCRIPTION'].apply(replace_not_scored).astype(np.object)
data.head()
#Now Transform CNS Score Description data into Numbers

sub_risk = {'N':-1, 'K':0, 'J':1, 'I':2, 'H':3, 'G':4, 'E':5,'F':6, 'L':7, 'M':8, 'B':9, 'D':10, 'A':11, 'C':12}

data['CNS_SCORE_DESCRIPTION'] = data['CNS_SCORE_DESCRIPTION'].apply(lambda x: sub_risk[x])
plt.figure(figsize=(12,8))
sns.countplot(x=data['CNS_SCORE_DESCRIPTION'])
#Replacing all the values into Common Group

data['PERFORM_CNS_SCORE_DESCRIPTION'].replace({'C-Very Low Risk':'Very Low Risk',
                                             'A-Very Low Risk':'Very Low Risk',
                                             'D-Very Low Risk':'Very Low Risk',
                                             'B-Very Low Risk':'Very Low Risk',
                                             'M-Very High Risk':'Very High Risk',
                                             'L-Very High Risk':'Very High Risk',
                                             'F-Low Risk':'Low Risk',
                                             'E-Low Risk':'Low Risk',
                                             'G-Low Risk':'Low Risk',
                                             'H-Medium Risk':'Medium Risk',
                                             'I-Medium Risk':'Medium Risk',
                                             'J-High Risk':'High Risk',
                                             'K-High Risk':'High Risk'},
                                              inplace=True)
#Transformin them into Numeric Features

risk_map = {'No Bureau History Available':-1, 
              'Not Scored: No Activity seen on the customer (Inactive)':-1,
              'Not Scored: Sufficient History Not Available':-1,
              'Not Scored: No Updates available in last 36 months':-1,
              'Not Scored: Only a Guarantor':-1,
              'Not Scored: More than 50 active Accounts found':-1,
              'Not Scored: Not Enough Info available on the customer':-1,
              'Very Low Risk':4,
              'Low Risk':3,
              'Medium Risk':2, 
              'High Risk':1,
              'Very High Risk':0}

data['PERFORM_CNS_SCORE_DESCRIPTION'] = data['PERFORM_CNS_SCORE_DESCRIPTION'].map(risk_map)
data['PERFORM_CNS_SCORE_DESCRIPTION'].value_counts()
sns.countplot(x=data['PERFORM_CNS_SCORE_DESCRIPTION'])
data.Employment_Type.value_counts()
data['Employment_Type'] = data['Employment_Type'].fillna('Not_employed')
data.Employment_Type.value_counts()
employment_map = {'Self employed':0, 'Salaried':1, 'Not_employed':-1}

data['Employment_Type'] = data['Employment_Type'].apply(lambda x: employment_map[x])

data['Employment_Type'].value_counts()
sns.countplot(x=data['Employment_Type'])
pri_columns = ['PRI_NO_OF_ACCTS','SEC_NO_OF_ACCTS',
           'PRI_ACTIVE_ACCTS','SEC_ACTIVE_ACCTS',
           'PRI_OVERDUE_ACCTS','SEC_OVERDUE_ACCTS',
           'PRI_CURRENT_BALANCE','SEC_CURRENT_BALANCE',
           'PRI_SANCTIONED_AMOUNT','SEC_SANCTIONED_AMOUNT',
           'PRI_DISBURSED_AMOUNT','SEC_DISBURSED_AMOUNT',
           'PRIMARY_INSTAL_AMT', 'SEC_INSTAL_AMT']

pri_df = data[pri_columns]
pri_df.head()
#Creating and Sorting Columns

data['NO_OF_ACCTS'] = data['PRI_NO_OF_ACCTS'] + data['SEC_NO_OF_ACCTS']

data['ACTIVE_ACCTS'] = data['PRI_ACTIVE_ACCTS'] + data['SEC_ACTIVE_ACCTS']

data['OVERDUE_ACCTS'] = data['PRI_OVERDUE_ACCTS'] + data['SEC_OVERDUE_ACCTS']

data['CURRENT_BALANCE'] = data['PRI_CURRENT_BALANCE'] + data['SEC_CURRENT_BALANCE']

data['SANCTIONED_AMOUNT'] = data['PRI_SANCTIONED_AMOUNT'] + data['SEC_SANCTIONED_AMOUNT']

data['DISBURSED_AMOUNT'] = data['PRI_DISBURSED_AMOUNT'] + data['SEC_DISBURSED_AMOUNT']

data['INSTAL_AMT'] = data['PRIMARY_INSTAL_AMT'] + data['SEC_SANCTIONED_AMOUNT']
data.drop(pri_columns, axis=1, inplace=True)
data.shape
new_columns = ['NO_OF_ACCTS', 'ACTIVE_ACCTS', 'OVERDUE_ACCTS', 'CURRENT_BALANCE',
       'SANCTIONED_AMOUNT', 'DISBURSED_AMOUNT', 'INSTAL_AMT']
for i in new_columns:
    print(i," : distinct_value")
    print(data[i].nunique()," : No. of unique Items")
    print(data[i].unique())
    print("-"*30)
    print("")
sns.scatterplot(data=data['ACTIVE_ACCTS'])
li = list(data['ACTIVE_ACCTS'].sort_values()[-3:].index)
data['ACTIVE_ACCTS'][li] = int(data.drop(li)['ACTIVE_ACCTS'].mode())
sns.scatterplot(data=data['NO_OF_ACCTS'])
li = list(data['NO_OF_ACCTS'].sort_values()[-4:].index)
data['NO_OF_ACCTS'][li] = int(data.drop(li)['NO_OF_ACCTS'].mode())
sns.scatterplot(data=data['OVERDUE_ACCTS'])
li = list(data['OVERDUE_ACCTS'].sort_values()[-10:].index)
data['OVERDUE_ACCTS'][li] = int(data.drop(li)['OVERDUE_ACCTS'].mode())
sns.scatterplot(data = data['CURRENT_BALANCE'])
sns.scatterplot(x=data['PERFORM_CNS_SCORE'],y=data['ACTIVE_ACCTS'])
data.head()
data.Date_of_Birth.min(),data.Date_of_Birth.max()
df_age = data[['disbursed_amount', 'asset_cost', 'ltv', 'Date_of_Birth','DisbursalDate','loan_default']]
df_age.tail()
def age(dob):
    yr = int(dob[-2:])
    if yr >=0 and yr < 20:
        return yr + 2000
    else:
         return yr + 1900
        
df_age['Date_of_Birth'] = df_age['Date_of_Birth'].apply(age)
df_age['DisbursalDate'] = df_age['DisbursalDate'].apply(age)
df_age['Age']=df_age['DisbursalDate']-df_age['Date_of_Birth']
df_age=df_age.drop(['DisbursalDate','Date_of_Birth'],axis=1)

df_age.head()
ax = plt.subplots(figsize=(10,7))
sns.countplot(x=df_age['Age'],alpha=.8)
data['Date_of_Birth'] = data['Date_of_Birth'].apply(age)
data['DisbursalDate'] = data['DisbursalDate'].apply(age)
data['Age'] = data['DisbursalDate'] - data['Date_of_Birth']
data = data.drop( ['DisbursalDate', 'Date_of_Birth'], axis=1)
data.shape
data.head()
data_copy = data.copy()
train_data = data.iloc[0:train.shape[0]]
test_data = data.iloc[train.shape[0]:]
train_data.shape, test_data.shape
test_data.drop(['loan_default'], axis=1, inplace=True)
X = train_data.drop(['loan_default'], axis=1)
y = train_data['loan_default']
X.shape, y.shape
smote = SMOTE()
X_tf,y_tf = smote.fit_resample(X,y)
X_tf.shape, y_tf.shape
scaler = preprocessing.RobustScaler()
X_tf = scaler.fit_transform(X_tf)
# Split the data into training and testing sets 
x_train,x_test,y_train,y_test = train_test_split(X_tf,y_tf,test_size=.2, random_state = 3300)
lr = LogisticRegression(C=5.0)
knn = KNeighborsClassifier(weights='distance', algorithm='auto', n_neighbors=15)
rfc = RandomForestClassifier(n_estimators=200,criterion='gini')
dtc = DecisionTreeClassifier()
bnb = BernoulliNB()
xgb = XGBClassifier()
cat = CatBoostClassifier(verbose=0)
ada = AdaBoostClassifier()
gbc = GradientBoostingClassifier()
accuracy = {}
roc_r = {}

def train_model(model):
    # Checking accuracy
    model = model.fit(x_train, y_train)
    pred = model.predict(x_test)
    acc = accuracy_score(y_test, pred)*100
    accuracy[model] = acc
    print('accuracy_score',acc)
    print('precision_score',precision_score(y_test, pred)*100)
    print('recall_score',recall_score(y_test, pred)*100)
    print('f1_score',f1_score(y_test, pred)*100)
    roc_score = roc_auc_score(y_test, pred)*100
    roc_r[model] = roc_score
    print('roc_auc_score',roc_score)
    # confusion matrix
    print('confusion_matrix')
    print(pd.DataFrame(confusion_matrix(y_test, pred)))
    fpr, tpr, threshold = roc_curve(y_test, pred)
    roc_auc = auc(fpr, tpr)*100

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
train_model(knn)
train_model(lr)
train_model(dtc)
train_model(bnb)
train_model(rfc)
train_model(xgb)
train_model(ada)
train_model(gbc)
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier()
train_model(mlp)
# Checking accuracy
model = cat.fit(x_train, y_train)
pred = cat.predict(x_test)
acc = accuracy_score(y_test, pred)*100
accuracy['Cat Boost Classifier'] = acc
print('accuracy_score',acc)
print('precision_score',precision_score(y_test, pred)*100)
print('recall_score',recall_score(y_test, pred)*100)
print('f1_score',f1_score(y_test, pred)*100)
roc_score = roc_auc_score(y_test, pred)*100
roc_r["Cat Boost Classifier"] = roc_score
print('roc_auc_score',roc_score)
# confusion matrix
print('confusion_matrix')
print(pd.DataFrame(confusion_matrix(y_test, pred)))
fpr, tpr, threshold = roc_curve(y_test, pred)
roc_auc = auc(fpr, tpr)*100

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# Predicted values
y_head_lr = lr.predict(x_test)
y_head_knn = knn.predict(x_test)
y_head_xgb = xgb.predict(x_test)
y_head_nb = bnb.predict(x_test)
y_head_dtc = dtc.predict(x_test)
y_head_rfc = rfc.predict(x_test)
y_head_cat = cat.predict(x_test)
y_head_ada = ada.predict(x_test)
y_head_gbc = gbc.predict(x_test)
y_head_mlp = mlp.predict(x_test)
cm_lr = confusion_matrix(y_test,y_head_lr)
cm_knn = confusion_matrix(y_test,y_head_knn)
cm_xgb = confusion_matrix(y_test,y_head_xgb)
cm_nb = confusion_matrix(y_test,y_head_nb)
cm_dtc = confusion_matrix(y_test,y_head_dtc)
cm_rfc = confusion_matrix(y_test,y_head_rfc)

cm_cat = confusion_matrix(y_test,y_head_cat)
cm_ada = confusion_matrix(y_test,y_head_ada)
cm_gbc = confusion_matrix(y_test,y_head_gbc)
cm_mlp = confusion_matrix(y_test,y_head_mlp)
plt.figure(figsize=(24,12))

plt.suptitle("Confusion Matrixes",fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

plt.subplot(4,3,5)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(4,3,6)
plt.title("K Nearest Neighbors Confusion Matrix")
sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(4,3,2)
plt.title("XGB Confusion Matrix")
sns.heatmap(cm_xgb,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(4,3,4)
plt.title("Naive Bayes Confusion Matrix")
sns.heatmap(cm_nb,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(4,3,3)
plt.title("Decision Tree Classifier Confusion Matrix")
sns.heatmap(cm_dtc,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(4,3,1)
plt.title("Random Forest Gini Confusion Matrix")
sns.heatmap(cm_rfc,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(4,3,7)
plt.title("CatBooost Confusion Matrix")
sns.heatmap(cm_cat,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(4,3,8)
plt.title("Ada Boost Confusion Matrix")
sns.heatmap(cm_ada,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(4,3,9)
plt.title("Gradient boost Classifier Confusion Matrix")
sns.heatmap(cm_gbc,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(4,3,10)
plt.title("MLP CLassifier Confusion Matrix")
sns.heatmap(cm_mlp,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})


plt.show()
sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
plt.xticks(rotation = 90)
sns.barplot(x=list(accuracy.keys()), y=list(accuracy.values()), palette="cubehelix")
plt.show()
sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,100,10))
plt.ylabel("ROC_score %")
plt.xlabel("Algorithms")
plt.xticks(rotation = 90)
sns.barplot(x=list(roc_r.keys()), y=list(roc_r.values()))
plt.show()
'''
criterion = ['gini','entropy']
n_estimators = [100, 200, 300, 500, 1000, 1200]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10] 

hyperF = dict(criterion = criterion,
            n_estimators = n_estimators, max_depth = max_depth,  
            min_samples_split = min_samples_split, 
            min_samples_leaf = min_samples_leaf)

gridF = GridSearchCV(rfc, hyperF, cv = 3, verbose = 1, 
                      n_jobs = -1)
bestF = gridF.fit(x_train, y_train)
'''
rfc = RandomForestClassifier(criterion = 'gini',
            n_estimators = 200, max_depth = 30,  
            min_samples_split = 2, 
            min_samples_leaf = 2)
rfc.fit(X_tf,y_tf)
test_data.reset_index(inplace=True)
test_data.head()
test_data.drop(['index'], axis=1, inplace=True)
test_data['loan_default'] = rfc.predict(test_data)
test_data['loan_default'] = test_data['loan_default'].astype(int)
test_data
test_data.to_csv("correct.csv", index=None)
