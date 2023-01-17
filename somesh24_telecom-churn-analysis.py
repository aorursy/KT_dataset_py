#import libraries

import os

import numpy as np

import pandas as pd

from scipy import stats



#import visualization libs

import matplotlib.pyplot as plt

import seaborn as sns



#import preprocessing libraries

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler

from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV



#import dimensions related libraries

from sklearn.decomposition import PCA



from sklearn.pipeline import Pipeline



#import algos

from sklearn.linear_model import LogisticRegression,SGDClassifier

import xgboost as xgb

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier



#import error metrics

from sklearn.metrics import plot_confusion_matrix, plot_roc_curve

#get the path of our dataset

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#read and analyze the data

data = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')



print("Shape",data.shape)



print("Info", data.info())



data.head()
#check the null values

((data.isnull().sum()*100)/len(data)).sort_values(ascending=False)
#check the unique values

data.nunique()
#Lets do some EDA

churn = data[data.Churn == 'Yes']

non_churn = data[data.Churn == 'No']
#Churn Vs Gender



sns.set_palette(sns.color_palette('muted', 7))

plt.figure(figsize=(8,4))

sns.countplot('gender', data=data, hue='Churn')
#Churn Vs SeniorCitizen



sns.set_palette(sns.color_palette('hls', 7))

plt.figure(figsize=(8,4))

sns.countplot('SeniorCitizen', data=data, hue='Churn')
#tenure relation with churn



sns.set_palette(sns.color_palette('Blues_d', 2))

sns.boxplot(x='Churn' ,y='tenure', data=data)



#seems most of the people who left service had less tenure
sns.set_palette(sns.color_palette('RdBu', 2))



sns.boxplot(x='Churn',y='MonthlyCharges',data=data)
#First convert the object column into type float

data['TotalCharges'] = data['TotalCharges'].replace('[^\d.]', '', regex = True).replace('',np.nan).astype(float)

sns.boxplot(x='Churn',y='TotalCharges',data=data)
sns.kdeplot(data[data['Churn'] == 'No']['TotalCharges'], label= 'Churn: No')

sns.kdeplot(data[data['Churn'] == 'Yes']['TotalCharges'], label= 'Churn: Yes')
sns.kdeplot(data[data['Churn'] == 'No']['tenure'], label= 'Churn: No')

sns.kdeplot(data[data['Churn'] == 'Yes']['tenure'], label= 'Churn: Yes')
#now check the numeric columns and check the distribution of data

t= (data.dtypes != 'object')

#now convert the data into list

num_cols = list(t[t].index)

num_cols
#Balance the data for numeric columns after EDA



sns.set_palette(sns.color_palette('muted', 1))

sns.distplot(data['tenure'], bins=20, hist=True,label='tenure')
sns.distplot(data['MonthlyCharges'], bins=20, hist=True,label='MonthlyCharges')
sns.distplot(data['TotalCharges'], bins=10, hist=True,label='TotalCharges')
sns.set_palette(sns.color_palette('hls', 4))

sns.catplot(y="Churn", x="MonthlyCharges", row="PaymentMethod", kind="box", data=data, height=2, aspect=4, orient='h')
sns.boxplot(x=data['MonthlyCharges'])

sns.boxplot(x=data['TotalCharges'])
#Tenure V/s Monthly Charges



sns.set(style="ticks")

x = data['tenure']

y = data['MonthlyCharges']

sns.jointplot(x, y, kind="hex", color="#4CB391")
# Show the joint distribution using kernel density estimation



g = sns.jointplot(x, y, kind="kde",

                  height=4, space=0)
sns.set(style="darkgrid")

g = sns.jointplot("TotalCharges", "tenure", 

                   data=data, kind="reg",

                   color="m", height=6)
#lets calculate the IQR

Q1 = data['TotalCharges'].quantile(0.25)

Q3 = data['TotalCharges'].quantile(0.75)

IQR = Q3 - Q1

print(IQR)
print((data['TotalCharges'] < (Q1 - 1.5 * IQR)) |(data['TotalCharges'] > (Q3 + 1.5 * IQR)))
totalcharge_out = ((data['TotalCharges'] < (Q1 - 1.5 * IQR)) |(data['TotalCharges'] > (Q3 + 1.5 * IQR)))

TotalCharges = data['TotalCharges'][~totalcharge_out]

TotalCharges.shape
#Lets check with zscore

z = np.abs(stats.zscore(data['TotalCharges']))

print(z)
threshold = 3

print(np.where(z > 3))
#Create a heatmap

plt.figure(figsize=(15, 10))

data.drop(['customerID'], axis=1, inplace=True)

# We will get the numeric representation of the object columns to numeric ones by applying pd.factorize.

corr = data.apply(lambda x: pd.factorize(x)[0]).corr()

ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, 

                 linewidths=.2, cmap="YlGnBu", annot=True)







#lets first use the pearson coefficient to check the correlation

print(data.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1))  



#check the correlation between columns which is greater than 0.95

corr = data.corr()

columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if corr.iloc[i,j] >= 0.95:

            if columns[j]:

                columns[j] = False

                

selected_columns = corr.columns[columns]

corr = corr[selected_columns]

corr.head()
#now check the object columns

ob= (data.dtypes == 'object')

#now convert the data into list

obj_cols = list(ob[ob].index)

print("List of Object columns",obj_cols)

print()

print("Number of Unique Values\n",data.nunique())
#lets split the data into train and validation set

data_x, data_y = data.drop(columns = 'Churn'), data['Churn']

skf = StratifiedKFold(n_splits=20, shuffle=True, random_state= 142)

for train_index, val_index in skf.split(data_x,data_y):

    train_x, val_x = data_x.iloc[train_index], data_x.iloc[val_index]

    train_y, val_y = data_y.iloc[train_index], data_y.iloc[val_index]

    

train_x.shape, val_x.shape    
train_x = pd.get_dummies(train_x)

print(train_x.head())

val_x = pd.get_dummies(val_x)

val_x.head()
#Lets first start with Logistic Regression

logReg = LogisticRegression()

sc = StandardScaler()

pca = PCA()



# Create a pipeline of three steps. First, standardize the data.

# Second, tranform the data with PCA.

# Third, train a logistic regression on the data.

pipe = Pipeline(steps=[('sc', sc),

                       ('pca', pca),

                       ('logistic', logReg)])



# Create Parameter Space

# Create a list of a sequence of integers from 1 to 30 (the number of features in X + 1)

n_components = list(range(1,train_x.shape[1]+1,1))

# Create a list of values of the regularization parameter

C = np.logspace(-4, 4, 50)

# Create a list of options for the regularization penalty

penalty = ['l2']

# Create a dictionary of all the parameter options 

# Note has you can access the parameters of steps of a pipeline by using '__’

parameters = dict(pca__n_components=n_components,

                  logistic__C=C,

                  logistic__penalty=penalty)



# Conduct Parameter Optmization With Pipeline

# Create a grid search object

clf = GridSearchCV(pipe, parameters)
train_x = train_x.reset_index()

val_x = val_x.reset_index()



#np.nan_to_num(train_x)

train_x = train_x.fillna(train_x.mean())

val_x = val_x.fillna(val_x.mean())

print(np.all(np.isfinite(train_x)))

print(np.any(np.isnan(train_x)))

print(train_x.isnull().sum())

clf.fit(train_x,train_y)



# View The Best Parameters

print('Best Penalty:', clf.best_estimator_.get_params()['logistic__penalty'])

print('Best C:', clf.best_estimator_.get_params()['logistic__C'])

print('Best Number Of Components:', clf.best_estimator_.get_params()['pca__n_components'])

print();print(clf.best_estimator_.get_params()['logistic'])



clf_pred = clf.best_estimator_.predict(val_x)



# Use Cross Validation To Evaluate Model

CV_Result = cross_val_score(clf, val_x, val_y, cv=4, n_jobs=-1)

print(CV_Result)

print(CV_Result.mean())

print(CV_Result.std())
# Check the accuracy of the model

from sklearn.metrics import confusion_matrix

cm = confusion_matrix( val_y, clf_pred)

print(cm)



# Lets check classification report also 

from sklearn.metrics import classification_report

cr = classification_report( val_y, clf_pred)

print(cr)
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



le = LabelEncoder()

le_val = le.fit(val_y)

le_val = le.transform(val_y)

clf_pred = le.transform(clf_pred)



# Now we will check ROC curve

def plot_roc_curve(fpr, tpr):

    plt.plot(fpr, tpr, color='orange', label='ROC')

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()

    plt.show()

    

print('AUC: %.2f'% roc_auc_score(le_val, clf_pred))



# plot the curve

fpr, tpr, thresholds = roc_curve(le_val, clf_pred)

plot_roc_curve(fpr, tpr)
from sklearn.ensemble import RandomForestClassifier



rfc_model=RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=42)



rfc_model.fit(train_x,train_y)



rfc_pred = rfc_model.predict(val_x)
rfc_pred = le.transform(rfc_pred)



print('AUC: %.2f'% roc_auc_score(le_val, rfc_pred))



# plot the curve

fpr, tpr, thresholds = roc_curve(le_val, rfc_pred)

plot_roc_curve(fpr, tpr)
#Import Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB



gnb_model = GaussianNB()



gnb_model.fit(train_x,train_y)



gnb_pred = gnb_model.predict(val_x)



gnb_pred = le.transform(gnb_pred)



# Now check the accuracy of the model

print('AUC: %.2f'% roc_auc_score(le_val, gnb_pred))



# plot the curve

fpr, tpr, thresholds = roc_curve(le_val, gnb_pred)

plot_roc_curve(fpr, tpr)