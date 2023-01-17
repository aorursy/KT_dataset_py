%matplotlib inline

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
all_data = pd.read_csv("../input/Churn_Modelling.csv")
all_data.info()
all_data.describe()
all_data['Geography'].value_counts()
sns.barplot(x='Geography',y='Exited', hue='Gender',data=all_data)
all_data.columns
dummy_gender = pd.get_dummies(all_data['Gender'], prefix='Gender')
dummy_geo = pd.get_dummies(all_data['Geography'],prefix = 'Geo')
dummy_NoOfProducts=pd.get_dummies(all_data['NumOfProducts'],prefix='NOP')
a=sns.FacetGrid(all_data,hue='Exited',aspect=6)

a.map(sns.kdeplot,'Age',shade=True)

a.set(xlim=[0,all_data['Age'].max()])

a.add_legend()
bins = [18,22,34,40,60,80,100]

labels = ['18-22','23-34','35-40','41-60','61-80','81-100']

dummy_age_labels=pd.cut(all_data['Age'],bins,labels=labels,right=False)
all_data['Age_labeled']= dummy_age_labels
sns.barplot(x='Age_labeled',y='NumOfProducts',hue='Gender', data = all_data)
sns.barplot(x = 'Age_labeled',y='Exited',hue='Gender',data=all_data)
dummy_age=pd.get_dummies(all_data['Age_labeled'],prefix='Age')
bins =[300,579,669,739,799,850]

labels =['Very Poor','Fair','Good','Very Good','Exceptional']

dummy_crdscore_labels=pd.cut(all_data['CreditScore'],bins,labels=labels)

all_data['CreditScore_labled']= dummy_crdscore_labels

dummy_creditscore = pd.get_dummies(all_data['CreditScore_labled'], prefix = 'CreditLevel')
sns.barplot(x = 'CreditScore_labled',y='Exited',hue='Gender',data=all_data)
bins = [0,1,5,8,11]

labels = ['0-1','1-5','5-8','8-11']

dummy_tenure_labels=pd.cut(all_data['Tenure'],bins,labels=labels,right=False)

all_data['Tenure_labeled']= dummy_tenure_labels

dummy_tenure = pd.get_dummies(all_data['Tenure_labeled'],prefix = 'Tenure')
sns.barplot(x = 'Tenure_labeled',y='Exited',hue='Gender',data=all_data)
from sklearn.preprocessing import StandardScaler

all_data['Balance'] = StandardScaler().fit_transform(all_data.filter(['Balance']))

all_data['EstimatedSalary'] = StandardScaler().fit_transform(all_data.filter(['EstimatedSalary']))
data_combined = pd.concat([

    all_data, dummy_age, dummy_tenure, dummy_creditscore, dummy_geo,

    dummy_gender, dummy_NoOfProducts

],

                          axis=1)
data_combined.drop(

    ['Gender', 'Age', 'CreditScore', 'Geography', 'NumOfProducts', 'Tenure'],

    axis=1,

    inplace=True)
data_combined.drop([

    'Surname', 'CustomerId', 'Age_labeled', 'CreditScore_labled',

    'Tenure_labeled'

],

                   axis=1,

                   inplace=True)
data_combined.head()
#获取标签值 

#Label value

y_label=data_combined['Exited']

y_label.shape
#去掉标签值，保留训练特征

#Remove lable value, keep training featurers only.

data_combined.drop(['Exited'],axis=1, inplace = True)
X_data = data_combined
#把数据划分出训练集和测试集

#Split data into trainging set and testing set

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X_data,y_label,test_size=0.2,random_state=2)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

param_grid = [

    {

        'n_estimators': [50, 100, 150],

        'max_leaf_nodes':[50,100,150]

    },

    {

        'bootstrap': [False],

        'n_estimators': [50, 100,150],

    },

]

churn_mdl_rf_sm = RandomForestClassifier()

grid_search = GridSearchCV(

    churn_mdl_rf_sm, param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(x_train, y_train)
#查看最佳参数

#Check the best parameters

grid_search.best_params_
#使用最佳参数建立分类器

#Use the best parameters for the classifier

churn_mdl_rf = RandomForestClassifier(

    random_state=2,

    n_estimators=150,

    n_jobs=4,

    max_leaf_nodes=150

)

churn_mdl_rf.fit(x_train, y_train)
from sklearn.model_selection import cross_val_score

cross_val_score(churn_mdl_rf,x_train, y_train,cv=5)
cross_val_score(churn_mdl_rf,x_test, y_test,cv=5)
from sklearn.model_selection import cross_val_score

cross_val_score(churn_mdl_rf,X_data, y_label,cv=5)