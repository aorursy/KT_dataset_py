# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#create dataframe to read the csv file
TelCo_Cust_Data=pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
#Check Data
TelCo_Cust_Data.head()
#Check for basic data type checks. Python reads Dtype = object as text. 
TelCo_Cust_Data.info()
#Before converting object to int/float, decide the categorical variables. We can do this using nunique function
TelCo_Cust_Data.nunique()
#Have look at unique values in column 
TelCo_Cust_Data.gender.unique()
TelCo_Cust_Data.PaymentMethod.unique()
TelCo_Cust_Data.Contract.unique()
#Get distribution of "churn" which is our target variable
TelCo_Cust_Data.Churn.value_counts()
#Plot churn distribution
TelCo_Cust_Data.Churn.value_counts().plot(kind='pie');
TelCo_Cust_Data.Churn.value_counts().plot(kind='bar');
# Gender wise churn count
TelCo_Cust_Data.groupby("Churn").gender.value_counts()
#plot Gender wise churn count
sns.countplot('gender',hue='Churn',data=TelCo_Cust_Data);
#Chanses of customer leaving the telecom comapny based on the :payment methods
TelCo_Cust_Data.groupby("Churn").PaymentMethod.value_counts()
sns.countplot(hue='Churn',data=TelCo_Cust_Data,x='PaymentMethod');
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)
#Average monthly charges
TelCo_Cust_Data.groupby("Churn").MonthlyCharges.mean()
#Tenure

TelCo_Cust_Data.groupby("Churn").tenure.mean()
#churn based on age
TelCo_Cust_Data.groupby("Churn").SeniorCitizen.value_counts()
#Plot churn based on age
sns.countplot(x='SeniorCitizen',hue="Churn",data=TelCo_Cust_Data)

#So based on graph we can see, non senior citizen are less likely to churn
#internet service
TelCo_Cust_Data.groupby("Churn").InternetService.value_counts()
sns.countplot(x='InternetService',hue="Churn",data=TelCo_Cust_Data)
#we can use pearson,spearman or kendall.
TelCo_Cust_Data.corr(method="pearson")
#Note : corr() from pandas will calculate the correlation of non-object columns only

#Let us assume we want to analyse this churn across internet service options 1)DSL 2)Fiber Optics 3) No Internet Service
#This method is called one hot coding method which assigns boolean value per internet type option
pd.get_dummies(data=TelCo_Cust_Data,columns=['InternetService'],prefix='IntServ')
#Select column as featured columns:
#feature column are defined based on how the target variable is dependent on these columns. Selecting right column give 
#better model accuracy
#Using Pearson Correlation
plt.figure(figsize=(7,7))
cor = TelCo_Cust_Data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
#above code gives only 3 columns as part of correlation because rest of the columns are object /text.
#Map Interger values to object columns which you think can be used for prediction

TelCo_Cust_Data.gender=TelCo_Cust_Data.gender.map({'Female':1,'Male':0})
TelCo_Cust_Data.Churn=TelCo_Cust_Data.Churn.map({'No':1,'Yes':0})
TelCo_Cust_Data.PaymentMethod=TelCo_Cust_Data.PaymentMethod.map({'Electronic check':1,'Mailed check':2,'Bank transfer (automatic)':3,'Credit card (automatic)':4})
TelCo_Cust_Data.Contract=TelCo_Cust_Data.Contract.map({'Month-to-month':1,'One year':2,'Two year':3})
TelCo_Cust_Data.MultipleLines=TelCo_Cust_Data.MultipleLines.map({'No':1,'Yes':0,'No phone service':2})
TelCo_Cust_Data.OnlineBackup=TelCo_Cust_Data.OnlineBackup.map({'No':1,'Yes':0,'No phone service':2})
TelCo_Cust_Data.DeviceProtection=TelCo_Cust_Data.DeviceProtection.map({'No':1,'Yes':0,'No phone service':2})
TelCo_Cust_Data.TechSupport=TelCo_Cust_Data.TechSupport.map({'No':1,'Yes':0,'No phone service':2})
TelCo_Cust_Data.Partner=TelCo_Cust_Data.Partner.map({'No':1,'Yes':0})
TelCo_Cust_Data.head()
TelCo_Cust_Data.head()
plt.figure(figsize=(12,12))
cor = TelCo_Cust_Data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
#Correlation with output variable : We should ideally consider cor_targer >0.5, but as we are not getting anyone near to it, I choose 0.25, to continue the code
cor_target = abs(cor["Churn"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.25]
relevant_features
TelCo_Cust_Data.columns

featured_columns=['tenure','TechSupport','Contract','PaymentMethod']
#create another dataframe for same csv to compare with heatmap
TelCo_Cust_Data_tree=pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
pd.get_dummies(data=TelCo_Cust_Data_tree,columns=['InternetService'],prefix='IntServ')

#TelCo_Cust_Data_tree.gender=TelCo_Cust_Data_tree.gender.map({'Male':1,'Female':0})
TelCo_Cust_Data_tree.gender=TelCo_Cust_Data_tree.gender.map({'Female':1,'Male':0})
TelCo_Cust_Data_tree.Churn=TelCo_Cust_Data_tree.Churn.map({'No':1,'Yes':0})
TelCo_Cust_Data_tree.PaymentMethod=TelCo_Cust_Data_tree.PaymentMethod.map({'Electronic check':1,'Mailed check':2,'Bank transfer (automatic)':3,'Credit card (automatic)':4})
TelCo_Cust_Data_tree.Contract=TelCo_Cust_Data_tree.Contract.map({'Month-to-month':1,'One year':2,'Two year':3})
TelCo_Cust_Data_tree.MultipleLines=TelCo_Cust_Data_tree.MultipleLines.map({'No':1,'Yes':0,'No phone service':2})
TelCo_Cust_Data_tree.OnlineBackup=TelCo_Cust_Data_tree.OnlineBackup.map({'No':1,'Yes':0,'No phone service':2})
TelCo_Cust_Data_tree.DeviceProtection=TelCo_Cust_Data_tree.DeviceProtection.map({'No':1,'Yes':0,'No phone service':2})
TelCo_Cust_Data_tree.TechSupport=TelCo_Cust_Data_tree.TechSupport.map({'No':1,'Yes':0,'No phone service':2})
TelCo_Cust_Data_tree.Partner=TelCo_Cust_Data_tree.Partner.map({'No':1,'Yes':0})
TelCo_Cust_Data_tree.head()
TelCo_Cust_Data_tree.columns
TelCo_Cust_Data_tree.info()
#Check NaN values per column
TelCo_Cust_Data_tree.isnull().sum()
#TelCo_Cust_Data_tree.isnull().values.any()
#TelCo_Cust_Data_tree.count()
#TelCo_Cust_Data_tree.gender.unique()
#Drop the rows where at least one element is missing /NaN ( as fit function for train and test data will not accept missing values)
TelCo_Cust_Data_tree.dropna(inplace=True)
TelCo_Cust_Data_tree.isnull().sum()
#No NaN/Missing values present now
#Select non-object featured columns and give it as input to decision tree algorithm
featured_columns=['gender','tenure','PaymentMethod','Contract','MultipleLines','OnlineBackup','DeviceProtection','TechSupport','Partner',
                 'MonthlyCharges']
X=TelCo_Cust_Data_tree[featured_columns]
y=TelCo_Cust_Data_tree.Churn
from sklearn.model_selection import train_test_split
#create trainm test data using train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=13)
from sklearn.tree import DecisionTreeClassifier
TelCo_Data=DecisionTreeClassifier(max_depth=3,random_state=13)

TelCo_Data.fit(X_train,y_train)
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(TelCo_Data,max_depth=3,feature_names=featured_columns);
#gives probable featured column in bar graph. inbuild function in matplotlib
plt.barh(featured_columns,TelCo_Data.feature_importances_)
