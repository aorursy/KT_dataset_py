# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/fish-market/Fish.csv')
df.head()
df.info()
df.isnull().sum()
df.describe()
df.Species.value_counts()
px.pie(df,names='Species')
sns.pairplot(df,hue='Species')
plt.figure(figsize=(12,5))
plt.plot(df['Length1'],label='Length Vertical')
plt.plot(df['Length2'],label='Length Diagonal')
plt.plot(df['Length3'],label='Length Cross')
plt.legend()
plt.figure(figsize=(12,5))
#plt.plot(df['Weight'],label='Weight')
plt.plot(df['Width'],label='Width')
plt.plot(df['Height'],label='Height')
plt.legend()
px.scatter(df,x='Height',y='Length1',size='Weight',color='Species')
px.scatter_3d(df,x='Length1',y='Height',z='Width',size='Weight',color='Species')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X=df.loc[:,'Weight':'Width']
y=df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('Dimenstions of Train Set of features:',np.shape(X_train))
print('Dimensions of Train Set of the target:',np.shape(y_train))
print('Dimenstions of Test Set of features:',np.shape(X_test))
print('Dimenstions of Test Set of the target:',np.shape(y_test))
#first I will use default values in Logistic Regression
logreg=LogisticRegression(max_iter=100000).fit(X_train,y_train)
print('Model intercept: ', logreg.intercept_)
print('Model coefficients: ',logreg.coef_)
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))
from sklearn.metrics import classification_report,confusion_matrix
pred1=logreg.predict(X_test)
print(confusion_matrix(y_test,pred1))
print('\n')
print(classification_report(y_test,pred1))
logreg.predict([[400,26.0,30.0,37.0,12.3,5.0]])
