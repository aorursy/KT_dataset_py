#Import the neccesary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

sns.set()



from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm
path = '/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv'



df = pd.read_csv(path)
df.head()
df.describe(include = 'all')
df.info()
#Value count of the column

df['TotalCharges'].value_counts()
#Replacing the empty value with zero 

df['TotalCharges'].replace(' ', 0, inplace = True)
df[df['TotalCharges'].apply (lambda x: x== ' ')]
#Changing the column datatype to float

df['TotalCharges'] = df['TotalCharges'].astype('float')
sns.distplot(df['tenure']) #This distribution plot appears to be normal with no outlier
sns.distplot(df['MonthlyCharges']) #This distribution plot appears to be normal with no outlier
sns.distplot(df['TotalCharges'])#This distribution plot appears to be having a few outliers. Let's explore it further
df.describe()
#Selecting Totalcharges above 8500 to see if they are outliers

df[df['TotalCharges'].apply (lambda x: x > 8500)]

#Upon further exploration they are not outliers
#Getting Variables in our dataframe

df.columns.values
from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.tools.tools import add_constant



# the target column (in this case 'churn') should not be included in variables

#Categorical variables already turned into dummy indicator may or maynot be added if any

variable = df[['tenure', 'MonthlyCharges','TotalCharges',]]

X = add_constant(variable)

vif = pd.DataFrame()

vif['VIF']  = [variance_inflation_factor(X.values, i) for i in range (X.shape[1])]

vif['features'] = X.columns



vif

#Using 10 as the minimum vif values i.e any independent variable 10 and above will have to be dropped

#From the results all independent variable are below 10
#Selecting the variable

scale_int = df[['MonthlyCharges']]



scaler = StandardScaler()#Selecting the standardscaler

scaler.fit(scale_int)#fitting our independent variables
df['scaled_monthly']= scaler.transform(scale_int)#scaling
scale_int = df[['tenure']] #Selecting the variable



scaler = StandardScaler()#Selecting the standardscaler

scaler.fit(scale_int)#fitting our independent variables
df['scaled_tenure']= scaler.transform(scale_int)#scaling
scale_int = df[['tenure']] #Selecting the variable



scaler = StandardScaler()#Selecting the standardscaler

scaler.fit(scale_int)#fitting our independent variables
df['scaled_charges']= scaler.transform(scale_int)#scaling
df.describe()# Checking our scaled results
df.describe(include = 'all')
#Dropping columns not needed

df.drop(['tenure','MonthlyCharges','customerID', 'TotalCharges'], axis = 1, inplace = True)
#Turning Churn to a dummy indicator with 1 standing yes and 0 standing for no

df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})
#Variables in our dataframe

df.columns.values
#new dataframe with dummies

df_dummies = pd.get_dummies(df, drop_first = True)



df_dummies
#Declaring independent variable i.e x

#Declaring Target variable i.e y

x = df_dummies.drop('Churn', axis = 1)

y = df_dummies['Churn']
#Splitting our data into train and test dataframe

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 24)
reg = LogisticRegression() #Selecting the model

reg.fit(x_train, y_train) #training the model with x_train and y_train
#Predicting with our already trained model using x_test

y_hat = reg.predict(x_test)
#Getting the accuracy of our model

acc = metrics.accuracy_score(y_hat, y_test)

acc
#The intercept for our regression

reg.intercept_
#Coefficient for all our variables

reg.coef_
cm = confusion_matrix(y_hat,y_test)

cm
# Format for easier understanding

cm_df = pd.DataFrame(cm)

cm_df.columns = ['Predicted 0','Predicted 1']

cm_df = cm_df.rename(index={0: 'Actual 0',1:'Actual 1'})

cm_df
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier # for K nearest neighbours

from sklearn import svm #for Support Vector Machine (SVM) 
dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)

y1 = dt.predict(x_test)

acc1 = metrics.accuracy_score(y1, y_test)

acc1
kk = KNeighborsClassifier()

kk.fit(x_train,y_train)

y2 = kk.predict(x_test)

acc2 = metrics.accuracy_score(y2, y_test)

acc2
sv = svm.SVC()

sv.fit(x_train,y_train)

y3 = sv.predict(x_test)

acc3 = metrics.accuracy_score(y3, y_test)

acc3
result = pd.DataFrame(data = x.columns.values, columns = ['features'] )

result['weight'] = np.transpose(reg.coef_)

result['odds'] = np.exp(np.transpose(reg.coef_))



result