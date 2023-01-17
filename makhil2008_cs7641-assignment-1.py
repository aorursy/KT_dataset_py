# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd 

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 



import gc

from datetime import datetime 

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn import svm

from sklearn.model_selection import GridSearchCV





from sklearn import tree

from sklearn.ensemble import AdaBoostClassifier

from catboost import CatBoostClassifier

import lightgbm as lgb

import xgboost as xgb

from sklearn.ensemble import GradientBoostingClassifier

import sklearn.metrics as metrics







pd.set_option('display.max_columns', 100)
raw_data=pd.read_csv('../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')
raw_data.head()
raw_data.describe(include='all')
raw_data[['default.payment.next.month']].value_counts()/30000
temp = raw_data["default.payment.next.month"].value_counts()

df = pd.DataFrame({'default.payment.next.month': temp.index,'values': temp.values})

plt.figure(figsize = (6,6))

plt.title('Default Credit Card Clients')

sns.set_color_codes("pastel")

sns.barplot(x = 'default.payment.next.month', y="values", data=df)

locs, labels = plt.xticks()

plt.xlabel('Default Payment')

plt.ylabel('# People', fontsize=16)

plt.xticks([0,1],['Not Default','Default'])

plt.show()
plt.figure(figsize = (14,6))

plt.title('Amount of credit limit - Density Plot')

sns.set_color_codes("pastel")

sns.distplot(raw_data['LIMIT_BAL'],kde=True,bins=200, color="blue")

plt.show()
# fig, (ax1, ax2) = plt.subplots(ncols=1, figsize=(12,6))

# # s = sns.boxplot(ax = ax1, x="SEX", y="LIMIT_BAL", hue="SEX",data=raw_data, palette="PRGn",showfliers=True)

s = sns.boxplot( x="SEX", y="LIMIT_BAL", hue="SEX",data=raw_data, palette="PRGn",showfliers=False,)

s.set_xticklabels(['Male','Female'])

s.set_xticklabels

s.set(xlabel='SEX', ylabel='Credit Limit')

s.legend([],[], frameon=False)

plt.show();
var = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']



plt.figure(figsize = (8,8))

plt.title('Amount of bill statement (Apr-Sept) \ncorrelation plot (Pearson)')

corr = raw_data[var].corr()

sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,vmin=-1, vmax=1)

plt.show()
var = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5']



plt.figure(figsize = (8,8))

plt.title('Amount of previous payment (Apr-Sept) \ncorrelation plot (Pearson)')

corr = raw_data[var].corr()

sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,vmin=-1, vmax=1)

plt.show()
target = 'default.payment.next.month'

predictors = [  'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 

                'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 

                'BILL_AMT1','BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',

                'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
train_df, val_df = train_test_split(raw_data, test_size=0.3, shuffle=True ,random_state=1)
train_df_bkp = train_df.copy()

val_df_bkp = val_df.copy()
rocs=pd.DataFrame(columns=['max_depth','val','train'])

for i in range(10,110,10):

    train_df = train_df_bkp.copy()

    train_df2=train_df.rename(columns={'default.payment.next.month':'default'})

    train_df21=train_df2.query('default==1')

    train_df20=train_df2.query('default==0').sample(round(len(train_df2.query('default==0').index)*i/100),random_state=42)

    train_df2=train_df21.append(train_df20).rename(columns={'default':'default.payment.next.month'})

    train_df=train_df2

    clf = tree.DecisionTreeClassifier(random_state=42,max_depth=4)

    clf.fit(train_df[predictors], train_df[target].values)

    preds = clf.predict(val_df[predictors])

    rocs=rocs.append(pd.DataFrame([[i,metrics.f1_score(val_df[target].values, preds),metrics.f1_score(train_df[target].values, clf.predict(train_df[predictors]))]],columns=['max_depth','val','train']),ignore_index=True)

plt.figure(figsize=(10,5))

plt.plot(rocs["max_depth"], rocs["train"], color='blue', marker='o',label="Training")        # specify color by name

plt.plot(rocs["max_depth"], rocs["val"], color='red', marker='o', linestyle='--',label="Validation")        # specify color by name

plt.xlabel('% Non-Defaults in Traning Data')

plt.ylabel('F1 Score')

plt.legend(loc="upper right")
rocs
train_df = train_df_bkp.copy()

train_df2=train_df.rename(columns={'default.payment.next.month':'default'})

train_df21=train_df2.query('default==1')

train_df20=train_df2.query('default==0').sample(round(len(train_df2.query('default==0').index)*50/100),random_state=42)

train_df2=train_df21.append(train_df20).rename(columns={'default':'default.payment.next.month'})

train_df=train_df2
clf = tree.DecisionTreeClassifier(random_state=42)

clf.fit(train_df[predictors], train_df[target].values)

preds = clf.predict(val_df[predictors])

print("Validation Accuracy:"+str(metrics.f1_score(val_df[target].values, preds)))

print("Training accuracy:"+str(metrics.f1_score(train_df[target].values, clf.predict(train_df[predictors]))))
text_representation = tree.export_text(clf)

print(text_representation)
rocs=pd.DataFrame(columns=['max_depth','val','train'])

for i in range(2,10):

    clf = tree.DecisionTreeClassifier(max_depth=i,random_state=42)

    clf.fit(train_df[predictors], train_df[target].values)

    preds = clf.predict(val_df[predictors])

    rocs=rocs.append(pd.DataFrame([[i,metrics.f1_score(val_df[target].values, preds),metrics.f1_score(train_df[target].values, clf.predict(train_df[predictors]))]],columns=['max_depth','val','train']),ignore_index=True)

plt.figure(figsize=(10,5))

plt.plot(rocs["max_depth"], rocs["train"], color='blue', marker='o',label="Training")        # specify color by name

plt.plot(rocs["max_depth"], rocs["val"], color='red', marker='o', linestyle='--',label="Validation")        # specify color by name

plt.xlabel('Maximum Depth')

plt.ylabel('F1 Score')

plt.legend(loc="upper right")

rocs
rocs=pd.DataFrame(columns=['max_depth','val','train'])

for i in range(2,10):

    clf = GradientBoostingClassifier( max_depth=i, random_state=42)

    clf.fit(train_df[predictors], train_df[target].values)

    preds = clf.predict(val_df[predictors])

    rocs=rocs.append(pd.DataFrame([[i,metrics.f1_score(val_df[target].values, preds),metrics.f1_score(train_df[target].values, clf.predict(train_df[predictors]))]],columns=['max_depth','val','train']),ignore_index=True)

plt.figure(figsize=(10,5))

plt.plot(rocs["max_depth"], rocs["train"], color='blue', marker='o',label="Training")        # specify color by name

plt.plot(rocs["max_depth"], rocs["val"], color='red', marker='o', linestyle='--',label="Validation")        # specify color by name

plt.xlabel('Maximum Depth')

plt.ylabel('F1 Score')

plt.legend(loc="upper right")

rocs
lr_list = [0.01,0.02,0.05, 0.1,0.15,0.2, 0.25, 0.5, 0.75, 1]

rocs=pd.DataFrame(columns=['max_depth','train','val'])



for learning_rate in lr_list:

    clf = GradientBoostingClassifier( learning_rate=learning_rate, max_depth=4, random_state=42)

    clf.fit(train_df[predictors], train_df[target].values)

    preds = clf.predict(val_df[predictors])

    rocs=rocs.append(pd.DataFrame([[learning_rate,metrics.f1_score(val_df[target].values, preds),metrics.f1_score(train_df[target].values, clf.predict(train_df[predictors]))]],columns=['max_depth','val','train']),ignore_index=True)

plt.figure(figsize=(10,5))

plt.plot(rocs["max_depth"], rocs["train"], color='blue', marker='o',label="Training")        # specify color by name

plt.plot(rocs["max_depth"], rocs["val"], color='red', marker='o', linestyle='--',label="Validation")        # specify color by name

plt.xlabel('Learning Rate')

plt.ylabel('F1 Score')    

plt.legend(loc="upper right")

rocs
lr_list = [25,50,100,200,300,400]

rocs=pd.DataFrame(columns=['max_depth','train','val'])



for learning_rate in lr_list:

    clf = GradientBoostingClassifier( n_estimators=learning_rate, max_depth=4,learning_rate=0.2, random_state=42)

    clf.fit(train_df[predictors], train_df[target].values)

    preds = clf.predict(val_df[predictors])

    rocs=rocs.append(pd.DataFrame([[learning_rate,metrics.f1_score(val_df[target].values, preds),metrics.f1_score(train_df[target].values, clf.predict(train_df[predictors]))]],columns=['max_depth','val','train']),ignore_index=True)

plt.figure(figsize=(10,5))

plt.plot(rocs["max_depth"], rocs["train"], color='blue', marker='o',label="Training")        # specify color by name

plt.plot(rocs["max_depth"], rocs["val"], color='red', marker='o', linestyle='--',label="Validation")        # specify color by name

plt.xlabel('Number of estimators')

plt.ylabel('F1 Score')    

plt.legend(loc="upper right")

rocs
from sklearn import preprocessing

temp1=train_df.copy()

temp1['train_val']='train'

temp2=val_df.copy()

temp2['train_val']='Val'

temp=temp1.append(temp2)

temp[predictors]=preprocessing.scale(temp[predictors])

tempdf=pd.DataFrame(temp,columns=temp1.columns)

train_df_sc=tempdf.query('train_val=="train"').drop(['train_val'],axis=1)

val_df_sc=tempdf.query('train_val=="Val"').drop(['train_val'],axis=1)

rocs=pd.DataFrame(columns=['max_depth','train','val'])



#Import knearest neighbors Classifier model

from sklearn.neighbors import KNeighborsClassifier

for i in range(1,15):

    #Create KNN Classifier

    knn = KNeighborsClassifier(n_neighbors=i)

    #Train the model using the training sets

    knn.fit(train_df_sc[predictors], train_df_sc[target].values)

    preds = knn.predict(val_df_sc[predictors])

    rocs=rocs.append(pd.DataFrame([[i,metrics.f1_score(val_df_sc[target].values, preds),metrics.f1_score(train_df_sc[target].values, knn.predict(train_df_sc[predictors]))]],columns=['max_depth','val','train']),ignore_index=True)

plt.figure(figsize=(10,5))

plt.plot(rocs["max_depth"], rocs["train"], color='blue', marker='o',label="Training")        # specify color by name

plt.plot(rocs["max_depth"], rocs["val"], color='red', marker='o', linestyle='--',label="Validation")        # specify color by name

plt.xlabel('Number of Nearest Neighbors')

plt.ylabel('F1 Score')    

plt.legend(loc="upper right")    

#Predict the response for test dataset
rocs
rocs=pd.DataFrame(columns=['max_depth','uniform','distance'])



#Import knearest neighbors Classifier model

from sklearn.neighbors import KNeighborsClassifier

for i in range(1,15):

    #Create KNN Classifier

    knn = KNeighborsClassifier(n_neighbors=i)

    #Train the model using the training sets

    knn.fit(train_df_sc[predictors], train_df_sc[target].values)

    preds = knn.predict(val_df_sc[predictors])

    

    knn2 = KNeighborsClassifier(n_neighbors=i,weights='distance')

    #Train the model using the training sets

    knn2.fit(train_df_sc[predictors], train_df_sc[target].values)

    preds2 = knn2.predict(val_df_sc[predictors])

    

    rocs=rocs.append(pd.DataFrame([[i,metrics.f1_score(val_df_sc[target].values, preds),metrics.f1_score(val_df_sc[target].values, preds2)]],columns=['max_depth','uniform','distance']),ignore_index=True)

plt.figure(figsize=(10,5))

plt.plot(rocs["max_depth"], rocs["uniform"], color='blue', marker='o',label="Uniform Weight")        # specify color by name

plt.plot(rocs["max_depth"], rocs["distance"], color='red', marker='o', linestyle='--',label="Distance based Weight")        # specify color by name

plt.xlabel('Number of Nearest Neighbors')

plt.ylabel('F1 Score')    

plt.legend(loc="lower right")    

#Predict the response for test dataset
rocs
params = [{ 'momentum' :0 ,'learning_rate_init': 0.1},

          {  'momentum' :0 ,'learning_rate_init': 0.3},

          { 'momentum' :0 , 'learning_rate_init': 0.5},

          { 'momentum' :0 ,'learning_rate_init': 0.7},

           {'momentum' :0 , 'learning_rate_init': 1},

          { 'momentum' :0.5 ,'learning_rate_init': 0.1},

          {  'momentum' :0.5 ,'learning_rate_init': 0.3},

          { 'momentum' :0.5 , 'learning_rate_init': 0.5},

          { 'momentum' :0.5 ,'learning_rate_init': 0.7},

           {'momentum' :0.5 , 'learning_rate_init': 1},

          { 'momentum' :0.9 ,'learning_rate_init': 0.1},

          {  'momentum' :0.9 ,'learning_rate_init': 0.3},

          { 'momentum' :0.9 , 'learning_rate_init': 0.5},

          { 'momentum' :0.9 ,'learning_rate_init': 0.7},

           {'momentum' :0.9 , 'learning_rate_init': 1},



         ]
rocs=pd.DataFrame(columns=['Params','train','val'])

plt.figure(figsize=(15,10))

for param in params:

    i=i+1

    clf = MLPClassifier( verbose=0, random_state=42,

                            max_iter=1000, **param,tol=0.00000001,solver= 'sgd',learning_rate= 'constant')

    clf.fit(train_df_sc[predictors], train_df_sc[target].values)

    preds = clf.predict(val_df_sc[predictors])

    rocs=rocs.append(pd.DataFrame([[param,metrics.f1_score(val_df_sc[target].values, preds),metrics.f1_score(train_df_sc[target].values, clf.predict(train_df_sc[predictors]))]],columns=['Params','val','train']),ignore_index=True)

    plt.plot(clf.loss_curve_,label=param)

plt.legend(loc="upper right")

plt.show()

# rocs=rocs.append(pd.DataFrame([[i,roc_auc_score(val_df[target].values, preds),roc_auc_score(train_df[target].values, clf.predict(train_df[predictors]))]],columns=['max_depth','train','val']),ignore_index=True)

# plt.plot(rocs["max_depth"], rocs["train_roc"], color='blue')        # specify color by name

# plt.plot(rocs["max_depth"], rocs["val_roc"], color='red')        # specify color by name



# plt.figure(figsize=(10,5))

# plt.plot(rocs["max_depth"], rocs["train"], color='blue', marker='o')        # specify color by name

# plt.plot(rocs["max_depth"], rocs["val"], color='red', marker='o', linestyle='--')        # specify color by name

# plt.xlabel('Maximum Depth')

# plt.ylabel('F1 Score')
rocs
params = [

          {  'C' :0.1 ,'gamma': 0.001},

          { 'C' :1 , 'gamma': 0.001},

          { 'C' :10 ,'gamma': 0.001},

           {'C' :100 , 'gamma': 0.001},

          {  'C' :0.1 ,'gamma': 0.01},

          { 'C' :1 , 'gamma': 0.01},

          { 'C' :10 ,'gamma': 0.01},

           {'C' :100 , 'gamma': 0.01},

          {  'C' :0.1 ,'gamma': 0.1},

          { 'C' :1 , 'gamma': 0.1},

          { 'C' :10 ,'gamma': 0.1},

           {'C' :100 , 'gamma': 0.1},

          {  'C' :0.1 ,'gamma': 1},

          { 'C' :1 , 'gamma': 1},

          { 'C' :10 ,'gamma': 1},

           {'C' :100 , 'gamma': 1}



         ]
lr_list = [0.01, 0.1, 1, 10,100,1000]

rocs=pd.DataFrame(columns=['max_depth','train','val'])



for param in params:

    clf = svm.SVC(kernel='rbf',**param)

    clf.fit(train_df_sc[predictors], train_df_sc[target].values)

    preds = clf.predict(val_df_sc[predictors])

    rocs=rocs.append(pd.DataFrame([[param,metrics.f1_score(val_df_sc[target].values, preds),metrics.f1_score(train_df_sc[target].values, clf.predict(train_df_sc[predictors]))]],columns=['max_depth','val','train']),ignore_index=True)

# plt.figure(figsize=(10,5))

# plt.plot(rocs["max_depth"], rocs["train"], color='blue', marker='o',label="Training")        # specify color by name

# plt.plot(rocs["max_depth"], rocs["val"], color='red', marker='o', linestyle='--',label="Validation")        # specify color by name

# plt.xlabel('C')

# plt.ylabel('F1 Score')    

# plt.legend(loc="upper right")
params = [

     

          {  'C' :0.1 ,'gamma': 0.001},

          { 'C' :1 , 'gamma': 0.001},

          {  'C' :0.1 ,'gamma': 0.01},

          { 'C' :1 , 'gamma': 0.01},

          { 'C' :10 ,'gamma': 0.01},

          {  'C' :0.1 ,'gamma': 0.1}

         

         ]
rocs=pd.DataFrame(columns=['params','train','val'])



for param in params:

    print(param)

    clf = svm.SVC(kernel='poly',**param,random_state=42,cache_size=500)

    clf.fit(train_df_sc[predictors], train_df_sc[target].values)

    preds = clf.predict(val_df_sc[predictors])

    rocs=rocs.append(pd.DataFrame([[param,metrics.f1_score(val_df_sc[target].values, preds),metrics.f1_score(train_df_sc[target].values, clf.predict(train_df_sc[predictors]))]],columns=['params','val','train']),ignore_index=True)
rocs
data = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data.TotalCharges = pd.to_numeric(data.TotalCharges, errors='coerce')

data.isnull().sum()
data.dropna(inplace = True)

#Remove customer IDs from the data set

df2 = data.iloc[:,1:]

#Convertin the predictor variable in a binary numeric variable

df2['Churn'].replace(to_replace='Yes', value=1, inplace=True)

df2['Churn'].replace(to_replace='No',  value=0, inplace=True)



#Let's convert all the categorical variables into dummy variables

df_dummies = pd.get_dummies(df2)

df_dummies.head()
df_dummies.Churn.value_counts()
temp = df_dummies["Churn"].value_counts()

df = pd.DataFrame({'Churn': temp.index,'values': temp.values})

plt.figure(figsize = (6,6))

plt.title('Customer Churn')

sns.set_color_codes("pastel")

sns.barplot(x = 'Churn', y="values", data=df)

locs, labels = plt.xticks()

plt.xlabel('Default Payment')

plt.ylabel('# People', fontsize=16)

plt.xticks([0,1],['Not Churn','Churn'])

plt.show()
target = 'Churn'

predictors = [  'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',

       'gender_Female', 'gender_Male', 'Partner_No', 'Partner_Yes',

       'Dependents_No', 'Dependents_Yes', 'PhoneService_No',

       'PhoneService_Yes', 'MultipleLines_No',

       'MultipleLines_No phone service', 'MultipleLines_Yes',

       'InternetService_DSL', 'InternetService_Fiber optic',

       'InternetService_No', 'OnlineSecurity_No',

       'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',

       'OnlineBackup_No', 'OnlineBackup_No internet service',

       'OnlineBackup_Yes', 'DeviceProtection_No',

       'DeviceProtection_No internet service', 'DeviceProtection_Yes',

       'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',

       'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',

       'StreamingMovies_No', 'StreamingMovies_No internet service',

       'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year',

       'Contract_Two year', 'PaperlessBilling_No', 'PaperlessBilling_Yes',

       'PaymentMethod_Bank transfer (automatic)',

       'PaymentMethod_Credit card (automatic)',

       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']
train_df, val_df = train_test_split(df_dummies, test_size=0.3, shuffle=True ,random_state=42)
train_df_bkp = train_df.copy()

val_df_bkp = val_df.copy()
rocs=pd.DataFrame(columns=['max_depth','val','train'])

for i in range(10,110,10):

    train_df2 = train_df_bkp.copy()

    train_df21=train_df2.query('Churn==1')

    train_df20=train_df2.query('Churn==0').sample(round(len(train_df2.query('Churn==0').index)*i/100),random_state=42)

    train_df2=train_df21.append(train_df20)

    train_df=train_df2

    clf = tree.DecisionTreeClassifier(random_state=42,max_depth=5)

    clf.fit(train_df[predictors], train_df[target].values)

    preds = clf.predict(val_df[predictors])

    rocs=rocs.append(pd.DataFrame([[i,metrics.f1_score(val_df[target].values, preds),metrics.f1_score(train_df[target].values, clf.predict(train_df[predictors]))]],columns=['max_depth','val','train']),ignore_index=True)

plt.figure(figsize=(10,5))

plt.plot(rocs["max_depth"], rocs["train"], color='blue', marker='o',label="Training")        # specify color by name

plt.plot(rocs["max_depth"], rocs["val"], color='red', marker='o', linestyle='--',label="Validation")        # specify color by name

plt.xlabel('% Non-Defaults in Traning Data')

plt.ylabel('F1 Score')

plt.legend(loc="upper right")
rocs
train_df, val_df = train_test_split(df_dummies, test_size=0.3, shuffle=True ,random_state=42)
clf = tree.DecisionTreeClassifier(random_state=42)

clf.fit(train_df[predictors], train_df[target].values)

preds = clf.predict(val_df[predictors])

print("Validation Accuracy:"+str(metrics.f1_score(val_df[target].values, preds)))

print("Training accuracy:"+str(metrics.f1_score(train_df[target].values, clf.predict(train_df[predictors]))))
text_representation = tree.export_text(clf)

print(text_representation)
rocs=pd.DataFrame(columns=['max_depth','val','train'])

for i in range(2,10):

    clf = tree.DecisionTreeClassifier(max_depth=i,random_state=42)

    clf.fit(train_df[predictors], train_df[target].values)

    preds = clf.predict(val_df[predictors])

    rocs=rocs.append(pd.DataFrame([[i,metrics.f1_score(val_df[target].values, preds),metrics.f1_score(train_df[target].values, clf.predict(train_df[predictors]))]],columns=['max_depth','val','train']),ignore_index=True)

plt.figure(figsize=(10,5))

plt.plot(rocs["max_depth"], rocs["train"], color='blue', marker='o',label="Training")        # specify color by name

plt.plot(rocs["max_depth"], rocs["val"], color='red', marker='o', linestyle='--',label="Validation")        # specify color by name

plt.xlabel('Maximum Depth')

plt.ylabel('F1 Score')

plt.legend(loc="upper right")

rocs
rocs=pd.DataFrame(columns=['max_depth','val','train'])

for i in range(2,10):

    clf = GradientBoostingClassifier( max_depth=i, random_state=42)

    clf.fit(train_df[predictors], train_df[target].values)

    preds = clf.predict(val_df[predictors])

    rocs=rocs.append(pd.DataFrame([[i,metrics.f1_score(val_df[target].values, preds),metrics.f1_score(train_df[target].values, clf.predict(train_df[predictors]))]],columns=['max_depth','val','train']),ignore_index=True)

plt.figure(figsize=(10,5))

plt.plot(rocs["max_depth"], rocs["train"], color='blue', marker='o',label="Training")        # specify color by name

plt.plot(rocs["max_depth"], rocs["val"], color='red', marker='o', linestyle='--',label="Validation")        # specify color by name

plt.xlabel('Maximum Depth')

plt.ylabel('F1 Score')

plt.legend(loc="upper right")
rocs
lr_list = [0.01,0.02,0.05, 0.1,0.15,0.2, 0.25, 0.5, 0.75, 1]

rocs=pd.DataFrame(columns=['max_depth','train','val'])



for learning_rate in lr_list:

    clf = GradientBoostingClassifier( learning_rate=learning_rate, max_depth=4, random_state=42)

    clf.fit(train_df[predictors], train_df[target].values)

    preds = clf.predict(val_df[predictors])

    rocs=rocs.append(pd.DataFrame([[learning_rate,metrics.f1_score(val_df[target].values, preds),metrics.f1_score(train_df[target].values, clf.predict(train_df[predictors]))]],columns=['max_depth','val','train']),ignore_index=True)

plt.figure(figsize=(10,5))

plt.plot(rocs["max_depth"], rocs["train"], color='blue', marker='o',label="Training")        # specify color by name

plt.plot(rocs["max_depth"], rocs["val"], color='red', marker='o', linestyle='--',label="Validation")        # specify color by name

plt.xlabel('Learning Rate')

plt.ylabel('F1 Score')    

plt.legend(loc="upper right")

rocs
lr_list = [25,50,100,200,300,400]

rocs=pd.DataFrame(columns=['max_depth','train','val'])



for learning_rate in lr_list:

    clf = GradientBoostingClassifier( n_estimators=learning_rate, max_depth=4,learning_rate=0.05, random_state=42)

    clf.fit(train_df[predictors], train_df[target].values)

    preds = clf.predict(val_df[predictors])

    rocs=rocs.append(pd.DataFrame([[learning_rate,metrics.f1_score(val_df[target].values, preds),metrics.f1_score(train_df[target].values, clf.predict(train_df[predictors]))]],columns=['max_depth','val','train']),ignore_index=True)

plt.figure(figsize=(10,5))

plt.plot(rocs["max_depth"], rocs["train"], color='blue', marker='o',label="Training")        # specify color by name

plt.plot(rocs["max_depth"], rocs["val"], color='red', marker='o', linestyle='--',label="Validation")        # specify color by name

plt.xlabel('Number of estimators')

plt.ylabel('F1 Score')    

plt.legend(loc="upper right")

rocs
from sklearn import preprocessing

temp1=train_df.copy()

temp1['train_val']='train'

temp2=val_df.copy()

temp2['train_val']='Val'

temp=temp1.append(temp2)

temp[predictors]=preprocessing.scale(temp[predictors])

tempdf=pd.DataFrame(temp,columns=temp1.columns)

train_df_sc=tempdf.query('train_val=="train"').drop(['train_val'],axis=1)

val_df_sc=tempdf.query('train_val=="Val"').drop(['train_val'],axis=1)

rocs=pd.DataFrame(columns=['max_depth','train','val'])



#Import knearest neighbors Classifier model

from sklearn.neighbors import KNeighborsClassifier

for i in range(1,15):

    #Create KNN Classifier

    knn = KNeighborsClassifier(n_neighbors=i)

    #Train the model using the training sets

    knn.fit(train_df_sc[predictors], train_df_sc[target].values)

    preds = knn.predict(val_df_sc[predictors])

    rocs=rocs.append(pd.DataFrame([[i,metrics.f1_score(val_df_sc[target].values, preds),metrics.f1_score(train_df_sc[target].values, knn.predict(train_df_sc[predictors]))]],columns=['max_depth','val','train']),ignore_index=True)

plt.figure(figsize=(10,5))

plt.plot(rocs["max_depth"], rocs["train"], color='blue', marker='o',label="Training")        # specify color by name

plt.plot(rocs["max_depth"], rocs["val"], color='red', marker='o', linestyle='--',label="Validation")        # specify color by name

plt.xlabel('Number of Nearest Neighbors')

plt.ylabel('F1 Score')    

plt.legend(loc="upper right")    

#Predict the response for test dataset
rocs
rocs=pd.DataFrame(columns=['max_depth','uniform','distance'])



#Import knearest neighbors Classifier model

from sklearn.neighbors import KNeighborsClassifier

for i in range(1,15):

    #Create KNN Classifier

    knn = KNeighborsClassifier(n_neighbors=i)

    #Train the model using the training sets

    knn.fit(train_df_sc[predictors], train_df_sc[target].values)

    preds = knn.predict(val_df_sc[predictors])

    

    knn2 = KNeighborsClassifier(n_neighbors=i,weights='distance')

    #Train the model using the training sets

    knn2.fit(train_df_sc[predictors], train_df_sc[target].values)

    preds2 = knn2.predict(val_df_sc[predictors])

    

    rocs=rocs.append(pd.DataFrame([[i,metrics.f1_score(val_df_sc[target].values, preds),metrics.f1_score(val_df_sc[target].values, preds2)]],columns=['max_depth','uniform','distance']),ignore_index=True)

plt.figure(figsize=(10,5))

plt.plot(rocs["max_depth"], rocs["uniform"], color='blue', marker='o',label="Uniform Weight")        # specify color by name

plt.plot(rocs["max_depth"], rocs["distance"], color='red', marker='o', linestyle='--',label="Distance based Weight")        # specify color by name

plt.xlabel('Number of Nearest Neighbors')

plt.ylabel('F1 Score')    

plt.legend(loc="lower right")    

#Predict the response for test dataset
rocs
params = [{ 'momentum' :0 ,'learning_rate_init': 0.1},

          {  'momentum' :0 ,'learning_rate_init': 0.3},

          { 'momentum' :0 , 'learning_rate_init': 0.5},

          { 'momentum' :0 ,'learning_rate_init': 0.7},

           {'momentum' :0 , 'learning_rate_init': 1},

          { 'momentum' :0.5 ,'learning_rate_init': 0.1},

          {  'momentum' :0.5 ,'learning_rate_init': 0.3},

          { 'momentum' :0.5 , 'learning_rate_init': 0.5},

          { 'momentum' :0.5 ,'learning_rate_init': 0.7},

           {'momentum' :0.5 , 'learning_rate_init': 1},

          { 'momentum' :0.9 ,'learning_rate_init': 0.1},

          {  'momentum' :0.9 ,'learning_rate_init': 0.3},

          { 'momentum' :0.9 , 'learning_rate_init': 0.5},

          { 'momentum' :0.9 ,'learning_rate_init': 0.7},

           {'momentum' :0.9 , 'learning_rate_init': 1},



         ]
rocs=pd.DataFrame(columns=['Params','train','val'])

plt.figure(figsize=(15,10))

for param in params:

    i=i+1

    clf = MLPClassifier( verbose=0, random_state=42,

                            max_iter=1000, **param,tol=0.00000001,solver= 'sgd',learning_rate= 'constant')

    clf.fit(train_df_sc[predictors], train_df_sc[target].values)

    preds = clf.predict(val_df_sc[predictors])

    rocs=rocs.append(pd.DataFrame([[param,metrics.f1_score(val_df_sc[target].values, preds),metrics.f1_score(train_df_sc[target].values, clf.predict(train_df_sc[predictors]))]],columns=['Params','val','train']),ignore_index=True)

    plt.plot(clf.loss_curve_,label=param)

plt.legend(loc="upper right")

plt.show()

# rocs=rocs.append(pd.DataFrame([[i,roc_auc_score(val_df[target].values, preds),roc_auc_score(train_df[target].values, clf.predict(train_df[predictors]))]],columns=['max_depth','train','val']),ignore_index=True)

# plt.plot(rocs["max_depth"], rocs["train_roc"], color='blue')        # specify color by name

# plt.plot(rocs["max_depth"], rocs["val_roc"], color='red')        # specify color by name



# plt.figure(figsize=(10,5))

# plt.plot(rocs["max_depth"], rocs["train"], color='blue', marker='o')        # specify color by name

# plt.plot(rocs["max_depth"], rocs["val"], color='red', marker='o', linestyle='--')        # specify color by name

# plt.xlabel('Maximum Depth')

# plt.ylabel('F1 Score')
rocs
params = [

          {  'C' :0.1 ,'gamma': 0.001},

          { 'C' :1 , 'gamma': 0.001},

          { 'C' :10 ,'gamma': 0.001},

           {'C' :100 , 'gamma': 0.001},

          {  'C' :0.1 ,'gamma': 0.01},

          { 'C' :1 , 'gamma': 0.01},

          { 'C' :10 ,'gamma': 0.01},

           {'C' :100 , 'gamma': 0.01},

          {  'C' :0.1 ,'gamma': 0.1},

          { 'C' :1 , 'gamma': 0.1},

          { 'C' :10 ,'gamma': 0.1},

           {'C' :100 , 'gamma': 0.1},

          {  'C' :0.1 ,'gamma': 1},

          { 'C' :1 , 'gamma': 1},

          { 'C' :10 ,'gamma': 1},

           {'C' :100 , 'gamma': 1}



         ]
lr_list = [0.01, 0.1, 1, 10,100,1000]

rocs=pd.DataFrame(columns=['params','train','val'])



for param in params:

    clf = svm.SVC(kernel='rbf',**param)

    clf.fit(train_df_sc[predictors], train_df_sc[target].values)

    preds = clf.predict(val_df_sc[predictors])

    rocs=rocs.append(pd.DataFrame([[param,metrics.f1_score(val_df_sc[target].values, preds),metrics.f1_score(train_df_sc[target].values, clf.predict(train_df_sc[predictors]))]],columns=['params','val','train']),ignore_index=True)

# plt.figure(figsize=(10,5))

# plt.plot(rocs["max_depth"], rocs["train"], color='blue', marker='o',label="Training")        # specify color by name

# plt.plot(rocs["max_depth"], rocs["val"], color='red', marker='o', linestyle='--',label="Validation")        # specify color by name

# plt.xlabel('C')

# plt.ylabel('F1 Score')    

# plt.legend(loc="upper right")
rocs
params = [

          {  'C' :0.1 ,'gamma': 0.001},

          { 'C' :1 , 'gamma': 0.001},

          { 'C' :10 ,'gamma': 0.001},

           {'C' :100 , 'gamma': 0.001},

          {  'C' :0.1 ,'gamma': 0.01},

          { 'C' :1 , 'gamma': 0.01},

          { 'C' :10 ,'gamma': 0.01},

           {'C' :100 , 'gamma': 0.01},

          {  'C' :0.1 ,'gamma': 0.1},

          { 'C' :1 , 'gamma': 0.1},

          { 'C' :10 ,'gamma': 0.1}



         ]
rocs=pd.DataFrame(columns=['params','train','val'])



for param in params:

    print(param)

    clf = svm.SVC(kernel='poly',**param,random_state=42,cache_size=500)

    clf.fit(train_df_sc[predictors], train_df_sc[target].values)

    preds = clf.predict(val_df_sc[predictors])

    rocs=rocs.append(pd.DataFrame([[param,metrics.f1_score(val_df_sc[target].values, preds),metrics.f1_score(train_df_sc[target].values, clf.predict(train_df_sc[predictors]))]],columns=['params','val','train']),ignore_index=True)
rocs