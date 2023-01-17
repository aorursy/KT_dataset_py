#load packages



import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import random as rnd
# loading Common Model Algorithms



from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC,LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.datasets import load_digits

from sklearn import tree



#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns
train_df= pd.read_csv('../input/lt-vehicle-loan-default-prediction/train.csv')

test_df= pd.read_csv('../input/lt-vehicle-loan-default-prediction/test.csv')
# train_df

# preview the data



train_df.head(10)
# train_df

#data info



train_df.info(max_cols=1000)
# train_df

# data describe



train_df.describe()
# train_df

# data describe for object



categorical_varaibles=train_df.describe(include=['O'])

categorical_varaibles
print(test_df.shape,train_df.shape)
#checking missing values(in percentage) in test data

print(train_df.isnull().sum()*100/train_df.shape[0])
#checking missing values(in percentage) in test data

print(test_df.isnull().sum()*100/test_df.shape[0])
train_df = train_df.fillna(train_df.mode().iloc[0])

test_df = test_df.fillna(test_df.mode().iloc[0])
print(train_df['Employment.Type'].isnull().sum(),test_df['Employment.Type'].isnull().sum())
test_df.shape
train_df.shape
train_df.nunique()
train_df=train_df.drop(columns ='MobileNo_Avl_Flag')

test_df=test_df.drop(columns ='MobileNo_Avl_Flag')
#lets get all the categorical train data present 

str_cols = train_df.select_dtypes(include = 'object').columns

train_df[str_cols].head()
#lets get all the categorical test data 

str_cols = test_df.select_dtypes(include = 'object').columns

test_df[str_cols].head()
train_df.dtypes
#For our analysis we need to change date of birth to age so that its more relevant and acceptable to the model

now = pd.Timestamp('now')

train_df['Date.of.Birth'] = pd.to_datetime(train_df['Date.of.Birth'], format='%d-%m-%y')

train_df['Date.of.Birth'] = train_df['Date.of.Birth'].where(train_df['Date.of.Birth'] < now, train_df['Date.of.Birth'] -  np.timedelta64(100, 'Y'))

train_df['Age'] = (now - train_df['Date.of.Birth']).astype('<m8[Y]')

train_df=train_df.drop('Date.of.Birth',axis=1)

#doing the same for our test data

now = pd.Timestamp('now')

test_df['Date.of.Birth'] = pd.to_datetime(test_df['Date.of.Birth'], format='%d-%m-%y')

test_df['Date.of.Birth'] = test_df['Date.of.Birth'].where(test_df['Date.of.Birth'] < now, test_df['Date.of.Birth'] -  np.timedelta64(100, 'Y'))

test_df['Age'] = (now - test_df['Date.of.Birth']).astype('<m8[Y]')

test_df=test_df.drop('Date.of.Birth',axis=1)
train_df['Age'].head(5)
test_df['Age'].head(5)
#For our analysis we need to change date of birth to age so that its more relevant and acceptable to the model

now = pd.Timestamp('now')

train_df['DisbursalDate'] = pd.to_datetime(train_df['DisbursalDate'], format='%d-%m-%y')

train_df['DisbursalDate'] = train_df['DisbursalDate'].where(train_df['DisbursalDate'] < now, train_df['DisbursalDate'] -  np.timedelta64(100, 'Y'))

train_df['time_since_loan_dispursed_in_yrs'] = (now - train_df['DisbursalDate']).astype('<m8[Y]')

train_df=train_df.drop('DisbursalDate',axis=1)



#For our analysis we need to change date of birth to age so that its more relevant and acceptable to the model

now = pd.Timestamp('now')

test_df['DisbursalDate'] = pd.to_datetime(test_df['DisbursalDate'], format='%d-%m-%y')

test_df['DisbursalDate'] = test_df['DisbursalDate'].where(test_df['DisbursalDate'] < now, test_df['DisbursalDate'] -  np.timedelta64(100, 'Y'))

test_df['time_since_loan_dispursed_in_yrs'] = (now - test_df['DisbursalDate']).astype('<m8[Y]')

test_df=test_df.drop('DisbursalDate',axis=1)
print(train_df['time_since_loan_dispursed_in_yrs'].head(10),test_df['time_since_loan_dispursed_in_yrs'].head(10))
#so whats left now.....lets get all the categorical test data 

str_cols = test_df.select_dtypes(include = 'object').columns

test_df[str_cols].head()
#Creating a function for encoding features with only 2 classes (we have only 1 variable like that but still a useful function in case we need in the future)

def two_feat_encoding(df_to_transform):

    le = LabelEncoder()



    for cols in df_to_transform:

        if df_to_transform[cols].dtype == 'object':

            if len(list(df_to_transform[cols].unique())) == 2:

                le.fit(df_to_transform[cols])

                df_to_transform[cols] = le.transform(df_to_transform[cols])

    return df_to_transform

train_df=two_feat_encoding(train_df)

test_df=two_feat_encoding(test_df)
#yet again..... whats left now.....lets get all the categorical test data 

str_cols = test_df.select_dtypes(include = 'object').columns

test_df[str_cols].head()
# for PERFORM_CNS_SCORE_DESCRIPTION we can reduce the 20 classes by replacing wherever risk information

train_df['PERFORM_CNS.SCORE.DESCRIPTION'].replace(to_replace=['Not Scored: More than 50 active Accounts found','No Bureau History Available','Not Scored: No Activity seen on the customer (Inactive)','Not Scored: No Updates available in last 36 months','Not Enough Info available on the customer','Not Scored: Only a Guarantor','Not Scored: Sufficient History Not Available','Not Scored: Not Enough Info available on the customer'], value= 'No_score', inplace = True)

test_df['PERFORM_CNS.SCORE.DESCRIPTION'].replace(to_replace=['Not Scored: More than 50 active Accounts found','No Bureau History Available','Not Scored: No Activity seen on the customer (Inactive)','Not Scored: No Updates available in last 36 months','Not Enough Info available on the customer','Not Scored: Only a Guarantor','Not Scored: Sufficient History Not Available','Not Scored: Not Enough Info available on the customer'], value= 'No_score', inplace = True)

train_df['PERFORM_CNS.SCORE.DESCRIPTION'].nunique()


vlow_risk=['A-Very Low Risk','B-Very Low Risk','C-Very Low Risk','D-Very Low Risk']

low_risk= ['E-Low Risk','F-Low Risk','G-Low Risk']

mid_risk= ['H-Medium Risk','I-Medium Risk']

high_risk= ['J-High Risk','K-High Risk']

vhigh_risk=['L-Very High Risk','M-Very High Risk']



train_df['PERFORM_CNS.SCORE.DESCRIPTION'].replace(to_replace='No_score',value = 0,inplace = True)

test_df['PERFORM_CNS.SCORE.DESCRIPTION'].replace(to_replace='No_score',value = 0,inplace = True)

train_df['PERFORM_CNS.SCORE.DESCRIPTION'].replace(to_replace=vlow_risk, value= 1, inplace = True)

test_df['PERFORM_CNS.SCORE.DESCRIPTION'].replace(to_replace=vlow_risk, value= 1, inplace = True)

train_df['PERFORM_CNS.SCORE.DESCRIPTION'].replace(to_replace=low_risk, value= 2, inplace = True)

test_df['PERFORM_CNS.SCORE.DESCRIPTION'].replace(to_replace=low_risk, value= 2, inplace = True)

train_df['PERFORM_CNS.SCORE.DESCRIPTION'].replace(to_replace=mid_risk, value= 3, inplace = True)

test_df['PERFORM_CNS.SCORE.DESCRIPTION'].replace(to_replace=mid_risk, value= 3, inplace = True)

train_df['PERFORM_CNS.SCORE.DESCRIPTION'].replace(to_replace=high_risk, value= 4, inplace = True)

test_df['PERFORM_CNS.SCORE.DESCRIPTION'].replace(to_replace=high_risk, value= 4, inplace = True)

train_df['PERFORM_CNS.SCORE.DESCRIPTION'].replace(to_replace=vhigh_risk, value= 5, inplace = True)

test_df['PERFORM_CNS.SCORE.DESCRIPTION'].replace(to_replace=vhigh_risk, value= 5, inplace = True)

test_df['PERFORM_CNS.SCORE.DESCRIPTION'].head()
#now yet again..... whats left now.....lets get all the categorical train data 

str_cols = test_df.select_dtypes(include = 'object').columns

train_df[str_cols].head()
train_df[['AVERAGE.ACCT.AGE_1','AVERAGE.ACCT.AGE_2']]=train_df['AVERAGE.ACCT.AGE'].str.split(expand=True)

test_df[['AVERAGE.ACCT.AGE_1','AVERAGE.ACCT.AGE_2']]=test_df['AVERAGE.ACCT.AGE'].str.split(expand=True)

train_df=train_df.drop(columns ='AVERAGE.ACCT.AGE')

test_df=test_df.drop(columns ='AVERAGE.ACCT.AGE')
train_df[['CREDIT.HISTORY.LENGTH_1','CREDIT.HISTORY.LENGTH_2']]=train_df['CREDIT.HISTORY.LENGTH'].str.split(expand=True)

test_df[['CREDIT.HISTORY.LENGTH_1','CREDIT.HISTORY.LENGTH_2']]=test_df['CREDIT.HISTORY.LENGTH'].str.split(expand=True)

train_df=train_df.drop(columns ='CREDIT.HISTORY.LENGTH')

test_df=test_df.drop(columns ='CREDIT.HISTORY.LENGTH')
#so whats left now.....lets get all the categorical test data 

str_cols = train_df.select_dtypes(include = 'object').columns

train_df[str_cols].head()
#stripping months and years

train_df['CREDIT.HISTORY.LENGTH_1']=train_df['CREDIT.HISTORY.LENGTH_1'].str.strip('yrs')

test_df['CREDIT.HISTORY.LENGTH_1']=test_df['CREDIT.HISTORY.LENGTH_1'].str.strip('yrs')

train_df['CREDIT.HISTORY.LENGTH_2']=train_df['CREDIT.HISTORY.LENGTH_2'].str.strip('mon')

test_df['CREDIT.HISTORY.LENGTH_2']=test_df['CREDIT.HISTORY.LENGTH_2'].str.strip('mon')

train_df['AVERAGE.ACCT.AGE_1']=train_df['AVERAGE.ACCT.AGE_1'].str.strip('yrs')

test_df['AVERAGE.ACCT.AGE_1']=test_df['AVERAGE.ACCT.AGE_1'].str.strip('yrs')

train_df['AVERAGE.ACCT.AGE_2']=train_df['AVERAGE.ACCT.AGE_2'].str.strip('mon')

test_df['AVERAGE.ACCT.AGE_2']=test_df['AVERAGE.ACCT.AGE_2'].str.strip('mon')



#converting datatype

train_df['CREDIT.HISTORY.LENGTH_1'] = train_df['CREDIT.HISTORY.LENGTH_1'].astype(int)

test_df['CREDIT.HISTORY.LENGTH_1'] = test_df['CREDIT.HISTORY.LENGTH_1'].astype(int)

train_df['CREDIT.HISTORY.LENGTH_2'] = train_df['CREDIT.HISTORY.LENGTH_2'].astype(int)

test_df['CREDIT.HISTORY.LENGTH_2'] = test_df['CREDIT.HISTORY.LENGTH_2'].astype(int)

train_df['AVERAGE.ACCT.AGE_1'] = train_df['AVERAGE.ACCT.AGE_1'].astype(int)

test_df['AVERAGE.ACCT.AGE_1'] = test_df['AVERAGE.ACCT.AGE_1'].astype(int)

train_df['AVERAGE.ACCT.AGE_2'] = train_df['AVERAGE.ACCT.AGE_2'].astype(int)

test_df['AVERAGE.ACCT.AGE_2'] = test_df['AVERAGE.ACCT.AGE_2'].astype(int)



# since we need to conctanate month value lets divide by 12 and round them off

train_df['CREDIT.HISTORY.LENGTH_2']=round((train_df['CREDIT.HISTORY.LENGTH_2']/12),2)

test_df['CREDIT.HISTORY.LENGTH_2']=round((test_df['CREDIT.HISTORY.LENGTH_2']/12),2)

train_df['AVERAGE.ACCT.AGE_2']=round((train_df['AVERAGE.ACCT.AGE_2']/12),2)

test_df['AVERAGE.ACCT.AGE_2']=round((test_df['AVERAGE.ACCT.AGE_2']/12),2)



#concatenating and converting

columnss=['AVERAGE.ACCT.AGE_1','AVERAGE.ACCT.AGE_2','CREDIT.HISTORY.LENGTH_1','CREDIT.HISTORY.LENGTH_2']

train_df['AVERAGE.ACCT.AGE']= train_df['AVERAGE.ACCT.AGE_1'].astype(float) + train_df['AVERAGE.ACCT.AGE_2'].astype(float)

test_df['AVERAGE.ACCT.AGE']= test_df['AVERAGE.ACCT.AGE_1'].astype(float) + test_df['AVERAGE.ACCT.AGE_2'].astype(float)

train_df['CREDIT.HISTORY.LENGTH']= train_df['CREDIT.HISTORY.LENGTH_1'].astype(float) + train_df['CREDIT.HISTORY.LENGTH_2'].astype(float)

test_df['CREDIT.HISTORY.LENGTH']= test_df['CREDIT.HISTORY.LENGTH_1'].astype(float) + test_df['CREDIT.HISTORY.LENGTH_2'].astype(float)

train_df['AVERAGE.ACCT.AGE']=train_df['AVERAGE.ACCT.AGE'].astype(float)

test_df['AVERAGE.ACCT.AGE']=test_df['AVERAGE.ACCT.AGE'].astype(float)

train_df['CREDIT.HISTORY.LENGTH']=train_df['CREDIT.HISTORY.LENGTH'].astype(float)

test_df['CREDIT.HISTORY.LENGTH']=test_df['CREDIT.HISTORY.LENGTH'].astype(float)

train_df['AVERAGE.ACCT.AGE'].head(12)
train_df=train_df.drop(columns=columnss)

test_df=test_df.drop(columns=columnss)

train_df.columns
#so whats left now.....lets get all the categorical test data 

str_cols = test_df.select_dtypes(include = 'object').columns

test_df[str_cols].head()
train_df.nunique()
ids_to_drop = ['UniqueID','supplier_id','Current_pincode_ID','branch_id','Employee_code_ID']

train_df=train_df.drop(columns=ids_to_drop)

test_df=test_df.drop(columns=ids_to_drop)
train_df.nunique()
related_to_drop = ['PERFORM_CNS.SCORE','PRI.NO.OF.ACCTS','SEC.NO.OF.ACCTS']

train_df=train_df.drop(columns=related_to_drop)

test_df=test_df.drop(columns=related_to_drop)
train_df.head(4)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(train_df.drop(columns='loan_default'))

scaler.fit(test_df)
scaled_values_1=scaler.transform(train_df.drop(columns='loan_default'))

scaled_values_2=scaler.transform(test_df)

num_data = list(train_df._get_numeric_data().columns)

num_data.remove('loan_default')

scaler = StandardScaler()

scaler.fit(train_df[num_data])

normalized = scaler.transform(train_df[num_data])

normalized2 = scaler.transform(test_df[num_data])

normalized_train = pd.DataFrame(normalized , columns=num_data)

normalized2_test = pd.DataFrame(normalized2 , columns=num_data)

print("The shape of normalised numerical data : " , normalized.shape)

print("The shape of normalised numerical data : " , normalized2.shape)
#adding our predictor variable

normalized_train['loan_default']=train_df['loan_default']



X=normalized_train.drop(columns='loan_default')

y=normalized_train['loan_default']
normalized_train.head(2)
from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

model = ExtraTreesClassifier()

model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(15).plot(kind='barh')

plt.show()
feat_importances.nlargest(15)
# here we will select the top 10 variables

new_train_df=train_df[['ltv','disbursed_amount','asset_cost','Age','State_ID','manufacturer_id','CREDIT.HISTORY.LENGTH','AVERAGE.ACCT.AGE','PRIMARY.INSTAL.AMT','PRI.CURRENT.BALANCE','PRI.DISBURSED.AMOUNT','PRI.SANCTIONED.AMOUNT','NO.OF_INQUIRIES','PRI.ACTIVE.ACCTS','PERFORM_CNS.SCORE.DESCRIPTION','loan_default']]

new_test_df=test_df[['ltv','disbursed_amount','asset_cost','Age','State_ID','manufacturer_id','CREDIT.HISTORY.LENGTH','AVERAGE.ACCT.AGE','PRIMARY.INSTAL.AMT','PRI.CURRENT.BALANCE','PRI.DISBURSED.AMOUNT','PRI.SANCTIONED.AMOUNT','NO.OF_INQUIRIES','PRI.ACTIVE.ACCTS','PERFORM_CNS.SCORE.DESCRIPTION']]
new_train_df.columns
X=new_train_df.drop(columns='loan_default')

y=new_train_df['loan_default']
# this is a function for complete evaluation of models

def model_performance(model):

    model.fit(X_train,y_train)

    RF_training_labels = model.predict(X_train)

    RF_test_labels = model.predict(X_test)

    Training_accuracy = model.score(X_train, y_train, sample_weight=None)

    Test_accuracy = model.score(X_test, y_test, sample_weight=None)

    F1_score_train = f1_score(y_train, RF_training_labels, average = 'weighted')

    F1_score_test = f1_score(y_test, RF_test_labels, average = 'weighted')

    Recall_train = recall_score(y_train, RF_training_labels, average = 'weighted') 

    Recall_test  = recall_score(y_test, RF_test_labels, average = 'weighted') 

    Precision_train = precision_score(y_train, RF_training_labels, average = 'weighted')

    Precision_test = precision_score(y_test, RF_test_labels, average = 'weighted')

    accuracy_train = accuracy_score(y_train, RF_training_labels, )

    accuracy_test = accuracy_score(y_test, RF_test_labels)

    rf_cm_tr = confusion_matrix(y_train, RF_training_labels)

    rf_cm_te = confusion_matrix(y_test, RF_test_labels)

    print("Training_accuracy - ", Training_accuracy)

    print("Test_accuracy - ", Test_accuracy)

    print("F1_score_train - ", F1_score_train)

    print("F1_score_test - ", F1_score_test)

    print("Recall_train - ", Recall_train)

    print("Recall_test - ", Recall_test)

    print("Precision_train - ", Precision_train)

    print("Precision_test - ", Precision_test)

    #Confusion Matrix

    class_names=[0,1] # name  of classes

    fig, ax = plt.subplots()

    tick_marks = np.arange(len(class_names))

    plt.xticks(tick_marks, class_names)

    plt.yticks(tick_marks, class_names)

    # create heatmap

    sns.heatmap(pd.DataFrame(rf_cm_te), annot=True, cmap="YlGnBu" ,fmt='g')

    ax.xaxis.set_label_position("top")

    plt.tight_layout()

    plt.title('Test Confusion matrix', y=1.1)

    plt.ylabel('Actual label')

    plt.xlabel('Predicted label')



    


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)



len(X_train)

len(X_test)







m1 = tree.DecisionTreeClassifier()

m2 = BaggingClassifier()

m3 = RandomForestClassifier()

m4 = AdaBoostClassifier()

m5 = GradientBoostingClassifier()

m6= GaussianNB()



models=[m1,m2,m3,m4,m5,m6]



for i in range(0,len(models)):

   print(model_performance(models[i]))

    

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
m1 = LogisticRegression()

m2 = KNeighborsClassifier()



models=[m1,m2]

for i in range(0,len(models)):

   print(model_performance(models[i]))


clf = DecisionTreeClassifier(random_state=0)

clf.fit(X_train,y_train)
path = clf.cost_complexity_pruning_path(X_train, y_train)

ccp_alphas, impurities = path.ccp_alphas, path.impurities
ccp_alphas


#clfs = []

#for ccp_alpha in ccp_alphas:

#    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)

#    clf.fit(X_train, y_train)

#    clfs.append(clf)

#print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(

#      clfs[-1].tree_.node_count, ccp_alphas[-1]))
predictions = clf.predict(new_test_df)
y_pred=pd.DataFrame(predictions)
new_test_df['predicted default on loan']=y_pred
new_test_df.to_csv(r'C:\Users\Admin\Desktop\Data_science_projects\Submissions.csv', index = False)
