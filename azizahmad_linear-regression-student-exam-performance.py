import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
data.head()
data.info()
sns.pairplot(data,hue='gender')
sns.pairplot(data,hue='lunch',markers="o", palette='rainbow')
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
data['male'] = data['gender'].apply(lambda x: 1 if x =='male' else 0)
race = pd.get_dummies(data['race/ethnicity'],drop_first=True)
parents = pd.get_dummies(data['parental level of education'],drop_first=True)
lunch = pd.get_dummies(data['lunch'],drop_first=True)
testprep = pd.get_dummies(data['test preparation course'],drop_first=True)
df = pd.concat([data,race,parents,lunch,testprep],axis=1)
df.drop(['gender','race/ethnicity','parental level of education','lunch','test preparation course'],axis=1,inplace=True)
df.head()
X_train,X_test,y_train,y_test = train_test_split(df.drop('math score',axis=1),df['math score'],test_size=0.3)
model = LinearRegression().fit(X_train,y_train)
predictions = model.predict(X_test)
plt.hist(y_test-predictions,bins=20)
from sklearn.metrics import mean_absolute_error,mean_squared_error
print('MAE', mean_absolute_error(y_test,predictions))

print('MSE', mean_squared_error(y_test,predictions))

print('RMSE', np.sqrt(mean_squared_error(y_test,predictions)))