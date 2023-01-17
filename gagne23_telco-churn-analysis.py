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
# import dataset and name it telco
telco=pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# view the datasets head
telco.head()
# Looking a the summary statistics of the numerical variables
telco.describe()
# Looking at the types of each columns
telco.info()
# Explore NAs
telco.isna().sum()
#Explore unique variables from all columns
telco.nunique()
# Null model
print(telco['Churn'].value_counts()/ telco.shape[0])

# If we had predict non churn, this model would be right 73.46% of the time. ML model must beat that score
# Change the TotalCharges to type :Float and SeniorCitizen to object and look at the data
telco['TotalCharges']= pd.to_numeric(telco['TotalCharges'],errors='coerce')
telco['SeniorCitizen']=telco['SeniorCitizen'].astype('object')
telco.info()
#Look at the distribution of the TotalCharges columns
import seaborn as sns
sns.distplot(telco['TotalCharges'])
#Impute median on TotalCharges Columns
telco.loc[telco['TotalCharges'].isna(),'TotalCharges']=telco['TotalCharges'].median()

#Look at the final missing values in each columns
telco.isna().sum()
# Separate features, categorical and numerical data 
telco.drop('customerID',axis=1,inplace=True)

#choose only the column name without the churn columns
features= [column_name for column_name in telco.columns if column_name!='Churn']

#Categorical features
categorical = [column_name for column_name in features if telco[column_name].dtype=='object']

# numerical features
numeric = [column_name for column_name in features if column_name not in categorical]
#Visualisation of churned customer with categorical features
import matplotlib.pyplot as plt
plt.rcParams["axes.labelsize"] = 5
sns.set(font_scale=5) 

# lets visualize these code we just did with Churn target
fig,axes= plt.subplots(5,3,figsize= (100,100))

for ax, column in zip(axes.flatten(),categorical):
    sns.countplot(x=column, hue='Churn',ax=ax, data=telco)
    ax.set_title(column)
    
plt.show()
#Visualisation of numerical data
sns.set(font_scale=1) 
 
# Create figure and axes
fig, axes = plt.subplots(1, 3, figsize = (20, 8))

# Iterate over each axes, and plot a boxplot with numeric columns
for ax, column in zip(axes.flatten(), numeric):
    
    # Create a boxplot
    sns.boxplot(x = "Churn", y = column, data = telco, ax = ax)
    
    # Set title
    ax.set_title(column)
# split the dataset into train and test set
from sklearn.model_selection import train_test_split

X= telco[features]
y= telco['Churn'].replace({'Yes':1,'No':0})

test_size=0.2

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=42)
#scale the numerical data with StandardScaler
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train[numeric])
X_test_scaled=scaler.fit_transform(X_test[numeric])

#bring back the scaled data to the original X_train and X_test set
X_train[numeric]=X_train_scaled
X_test[numeric]=X_test_scaled
X_train.head()


#Now the numerical values are scaled around 0
# Transform the categorical variable into binary variable with pd.get_dummies(when using a dataframe) OneHotEncoder otherwise
X_train=pd.get_dummies(X_train,columns=categorical)
X_test=pd.get_dummies(X_test,columns=categorical)

#look at the values
X_train.head()

# import the datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score,f1_score
# Create lots of model
rfc= RandomForestClassifier(max_depth=50,n_estimators=200,max_features=5)
knn=KNeighborsClassifier(n_neighbors=5)
ada=AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),n_estimators=100,algorithm='SAMME.R',learning_rate=0.5)
log_reg=LogisticRegression(C=5,random_state=42)
svc=SVC(kernel='poly',gamma='scale',degree=2,probability=True)
#Create the definition to train them all and look for training score
def training_model(model):
    
    #build models
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    #print metrics
    print('accuracy: ',accuracy_score(y_test,y_pred))
    print('precision: ',precision_score(y_test,y_pred))
    print('recall: ',recall_score(y_test,y_pred))
    print('f1 score: ',f1_score(y_test,y_pred))
    
    #print confusion matrix
    print('confusion_matrix')
    print(pd.DataFrame(confusion_matrix(y_test, y_pred)))

# RANDOM FOREST CLASSIFIER

training_model(rfc)
# KNN
training_model(knn)
# ADABOOST CLASSIFIER
training_model(ada)
# LOGISTIC REGRESSION
training_model(log_reg)
#SUPPORT VECTOR MACHINE
training_model(svc)
#Set the parameters to tune
log_reg.get_params()
params=[{'C':[1,3,5],'max_iter':[10,30,50,100]}]
# set and find the best params
log_reg_2=LogisticRegression(C=5,max_iter = 100)
model=log_reg_2.fit(X_train,y_train)
model.coef_

# Printin accuracy_score with tune modeling
y_pred=model.predict(X_test)
print('accuracy score= ',accuracy_score(y_pred,y_test))
# To get the weights of all the variables
weights = pd.Series(model.coef_[0],index=X_train.columns)
weights.sort_values(ascending = False)