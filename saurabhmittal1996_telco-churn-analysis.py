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

from sklearn import svm

from sklearn.neural_network import MLPClassifier



#Bagging Algo

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier



#To tranform data

from sklearn import preprocessing



#statistical Tools

from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score,f1_score

from sklearn.metrics import confusion_matrix, roc_curve, auc, plot_confusion_matrix



#Setting Format

pd.options.display.float_format = '{:.5f}'.format

pd.options.display.max_columns = None

pd.options.display.max_rows = None
data = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data.shape
data.head()
data.describe(include='all').T
(data['Churn'].value_counts()/data.shape[0]).plot(kind='bar')
#Lets Convert Churn column into Numerical data

data['Churn'] = data['Churn'].map({'Yes':1, 'No':0})
data.head()
data.info()
data['Churn'] = data['Churn'].astype('int')
data.isna().sum()
data[data==0].sum()
for i in data.columns:

    print(i,'(', data[i].dtype, ')' ,": Distinct Values")

    print(data[i].nunique(), "Total Unique Values")

    print(data.shape[0], "Total Values")

    print(data[i].unique())

    print("-"*30)

    print("")
# Senior Citizen should be int dtype lets change it to object

#This Columns means whether or not customer is Senior so it should be Object Type 

data['SeniorCitizen'] = data['SeniorCitizen'].astype(object)
#Total Charges should be numeric not object



data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.info()
data.isna().sum()
data.head()
data[data.isna().any(axis=1)]
df = data[data.isna().any(axis=1)]

df.shape
df['TotalCharges'] = df['MonthlyCharges']
df.head()
data = pd.concat([data,df],ignore_index=True)
data.dropna(inplace=True)

data.shape
sns.countplot(x=data['SeniorCitizen'])
for i in data.select_dtypes(include='O'):

    sns.countplot(data[i])

    plt.xticks(rotation = 90)

    plt.show()
sns.pairplot(data)
for i in data.select_dtypes(exclude='O'):

    sns.distplot(data[i], bins=10)

    plt.show()
data['tenure'] = np.log1p(data['tenure'])

data['MonthlyCharges'] = np.log1p(data['MonthlyCharges'])

data['TotalCharges'] = np.log1p(data['TotalCharges'])
for i in data.select_dtypes(exclude='O'):

    sns.distplot(data[i], bins=10)

    plt.show()
data.head()
data.drop(['customerID'], axis=1, inplace=True)
cat_col = data.select_dtypes(include="O").columns

cat_col
df = data.copy()
df = pd.get_dummies(df)
df.head()
plt.figure(figsize=(24,12))

sns.heatmap(df.corr())
df.corr()['Churn'].sort_values(ascending = False)
X = df.drop(['Churn'], axis=1)

y = df['Churn']
sm = SMOTE(random_state=50)
X_tf,y_tf = sm.fit_resample(X,y)
scaler = preprocessing.RobustScaler()

df_x = scaler.fit_transform(X_tf)
# Using Skicit-learn to split data into training and testing sets 

# Split the data into training and testing sets 

x_train,x_test,y_train,y_test = train_test_split(df_x,y_tf,test_size=.2, random_state = 100)
lr = LogisticRegression(C=5.0)

knn = KNeighborsClassifier(weights='distance', algorithm='auto', n_neighbors=15)

rfc = RandomForestClassifier(n_estimators=200,criterion='gini', n_jobs=-1)

dtc = DecisionTreeClassifier()

bnb = BernoulliNB()

xgb = XGBClassifier(n_jobs=-1)

cat = CatBoostClassifier(verbose=0)

ada = AdaBoostClassifier()

gbc = GradientBoostingClassifier()

svc = svm.SVC(kernel = 'poly', C=4, gamma='scale', degree = 2)
def train_model(model):

    # Checking accuracy

    model = model.fit(x_train, y_train)

    pred = model.predict(x_test)

    print('accuracy_score',accuracy_score(y_test, pred)*100)

    print('precision_score',precision_score(y_test, pred)*100)

    print('recall_score',recall_score(y_test, pred)*100)

    print('f1_score',f1_score(y_test, pred)*100)

    print('roc_auc_score',roc_auc_score(y_test, pred)*100)

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
train_model(lr)
train_model(dtc)
train_model(rfc)
train_model(xgb)
train_model(ada)
train_model(gbc)
train_model(cat)
train_model(bnb)
train_model(knn)
mlp = MLPClassifier()

train_model(mlp)
train_model(svc)
# Predicted values

y_head_lr = lr.predict(x_test)

y_head_rfc = rfc.predict(x_test)

y_head_xgb = xgb.predict(x_test)

y_head_ada = ada.predict(x_test)

y_head_dtc = dtc.predict(x_test)

y_head_gbc = gbc.predict(x_test)

y_head_cat = cat.predict(x_test)

y_head_knn = knn.predict(x_test)

y_head_nb = bnb.predict(x_test)

y_head_mlp = mlp.predict(x_test)

y_head_svm = svc.predict(x_test)


cm_lr = confusion_matrix(y_test,y_head_lr)

cm_rfc = confusion_matrix(y_test,y_head_rfc)

cm_xgb = confusion_matrix(y_test,y_head_xgb)

cm_ada = confusion_matrix(y_test,y_head_ada)

cm_dtc = confusion_matrix(y_test,y_head_dtc)

cm_gbc = confusion_matrix(y_test,y_head_gbc)

cm_cat = confusion_matrix(y_test,y_head_cat)



cm_knn = confusion_matrix(y_test,y_head_knn)

cm_nb = confusion_matrix(y_test,y_head_nb)

cm_mlp = confusion_matrix(y_test,y_head_mlp)

cm_svm = confusion_matrix(y_test,y_head_svm)
plt.figure(figsize=(30,20))



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



plt.subplot(4,3,11)

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



plt.subplot(4,3,3)

plt.title("Support Vector CLassifier Confusion Matrix")

sns.heatmap(cm_svm,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.show()
'''

max_feature = ['auto', 'sqrt']



n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [1, 2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}



grid = GridSearchCV(rfc, random_grid, cv=3, verbose=True, n_jobs=-1)

grid.fit(x_train, y_train)'''
rfc_1 = RandomForestClassifier(bootstrap=False, max_depth=90, max_features='auto',

                              min_samples_leaf=2, min_samples_split=5, n_estimators=600, )



train_model(rfc_1)