#importing relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
sns.set_style("darkgrid")
#importing raw data from CSV
raw_data = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
raw_data.info()
raw_data.describe()
#Probability of 1s to all outcomes
raw_data['Outcome'].sum()/raw_data['Outcome'].count()
#Setting the Quantile limit to 80%
quantile_limit = 0.8
no_diabetes = raw_data[raw_data['Outcome'] == 0].copy()
no_diabetes.describe()
#From the previous analyses we can see there are a lot of outliers in Insulin
sns.distplot(no_diabetes['Insulin'])
quantile_insuline = no_diabetes['Insulin'].quantile(quantile_limit)
quantile_insuline
#Also from the previous analyses we can see there are a lot of outliers in Age
sns.distplot(no_diabetes['Age'])
quantile_age = no_diabetes['Age'].quantile(quantile_limit)
quantile_age
data_norm = raw_data.copy()
#Discarding 0 outcomes above quantile_age
data_norm = data_norm[(data_norm['Outcome'] == 1) | (data_norm['Age'] <= quantile_age)]
data_norm.describe()
#Probability of 1s to all outcomes
data_norm['Outcome'].sum()/data_norm['Outcome'].count()
#Checking 0 data points
no_diabetes = data_norm[raw_data['Outcome'] == 0].copy()
no_diabetes.describe()
quantile_insulin = no_diabetes['Insulin'].quantile(quantile_limit)
quantile_insulin
#Discarding 0 outcomes above Insulin 130
data_norm = data_norm[(data_norm['Outcome'] == 1) | (data_norm['Insulin'] <= quantile_insulin)]
#Probability of 1s to all outcomes
data_norm['Outcome'].sum()/data_norm['Outcome'].count()
data_pp = data_norm.copy()
data_pp.info()
data_pp.describe()
#raw inputs
unscaled_X = data_pp.drop('Outcome',axis=1)
#outputs/targets
y = data_pp['Outcome']
unscaled_X.head()
scaler = StandardScaler()
X = scaler.fit_transform(unscaled_X)
X
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=108)
#Instantiating our model
pima_diabetes_model = LogisticRegression()
#Fitting and training our model
pima_diabetes_model.fit(X_train,y_train)
#Evaluating our model on training data
pima_diabetes_model.score(X_train, y_train)
#Evaluating our model on test data
pima_diabetes_model.score(X_test, y_test)
#Evaluating accuracy - Part1
y_pred = pima_diabetes_model.predict(X_test)
#Evaluating accuracy - Part2
print(classification_report(y_test,y_pred))