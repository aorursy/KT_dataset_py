# Importing the necessary libraries

import numpy as np

import pandas as pd

import pandas_profiling as pp

import matplotlib.pyplot as plt

import seaborn as sns







# Model development libraries

from sklearn.linear_model import BayesianRidge

from fancyimpute import IterativeImputer as MICE

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from catboost import CatBoostClassifier



from sklearn.metrics import accuracy_score,confusion_matrix,classification_report



%matplotlib inline
# Getting the data



pima = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
# See a sample of the data. In this case the first five rows



pima.head()
pima.info()



# There are 768 rows and 9 columns. It is a pretty small dataset
pima.describe(include = 'all')
pima.columns
# To check whether it is an imbalanced dataset



pima['Outcome'].value_counts(normalize = True)*100
# Checking for missing values



pima.isnull().sum()
# Carry out a whole profile report 



pp.ProfileReport(pima)
sns.boxplot(data= pima)



sns.set(rc = {'figure.figsize':(10,10)})
#Replace zeros with np.nans



pima['Glucose'].replace(0,np.nan,inplace = True)
pima['BloodPressure'].replace(0,np.nan,inplace = True)

pima['SkinThickness'].replace(0,np.nan,inplace = True)

pima['BMI'].replace(0,np.nan,inplace = True)
pima.isnull().sum()
imputer = MICE(BayesianRidge())

impute_data = pd.DataFrame(imputer.fit_transform(pima))
cols = list(pima)

impute_data.columns = cols
impute_data.isnull().sum()
pima_diabetes = impute_data
pima_diabetes.info()
pima_diabetes = pima_diabetes.astype({"Pregnancies" : int,

                                     "Glucose": int,

                                     "BloodPressure": int,

                                     "SkinThickness" : int,

                                     "Insulin": int,

                                     "Age": int,

                                     "Outcome": int})
pima_diabetes.info()
# Lets do a correlation matrix

x = pima_diabetes.iloc[:,:]

y = pima_diabetes.iloc[:, 0]

corrmat = pima_diabetes.corr()

top_features = corrmat.index

plt.figure(figsize=(10,10))

matrix =sns.heatmap(pima_diabetes[top_features].corr(),annot=True,cmap="RdYlGn")
# Lets do a boxplot



sns.boxplot(data= pima_diabetes)



sns.set(rc = {'figure.figsize':(16,16)})
sns.countplot('Outcome' , data = pima_diabetes)
pp.ProfileReport(pima_diabetes)
# Split the features and the targets

x = pima_diabetes.drop(['Outcome'],axis = 1)

y = pima_diabetes['Outcome']
x_train,x_test,y_train,y_test = train_test_split( x , y , test_size = 0.2 , random_state = 7 )
#Logistic Regressor



lr = LogisticRegression()



lr.fit(x_train, y_train)



lr_pred = lr.predict(x_test)



accuracy = accuracy_score(y_test, lr_pred)



accuracy * 100
print(classification_report(y_test, lr_pred))
#Decision tree



dt = DecisionTreeClassifier()



dt.fit(x_train, y_train)



dt_pred = dt.predict(x_test)



accuracy = accuracy_score(y_test, dt_pred)



accuracy * 100
print(classification_report(y_test, dt_pred))
#Random Forest



rf = RandomForestClassifier(n_estimators = 500)

rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)

print(classification_report(y_test,y_pred_rf))