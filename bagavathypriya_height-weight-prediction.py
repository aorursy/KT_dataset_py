# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/weight-height/weight-height.csv')
df.head()
df.info()
df.describe()
sns.scatterplot('Height','Weight',data=df,hue='Gender')
sns.countplot('Gender',data=df)
sns.jointplot(x='Height',y='Weight',data=df,kind='reg')
sns.heatmap(df.corr(),annot=True)
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
df['Gender']=lab.fit_transform(df['Gender'])
df.head()
df.tail()
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(xtrain,ytrain)
ypred=reg.predict(xtest)
print('Accuracy is {}'.format(reg.score(xtest,ytest)*100))

trainypred=reg.predict(xtrain)
testypred=reg.predict(xtest)
from sklearn.metrics import r2_score
print('Training set score {}'.format(r2_score(ytrain,trainypred)))
print('Testing set score {}'.format(r2_score(ytest,testypred)))
from mpl_toolkits.mplot3d import Axes3D 
ytest=np.array(ytest)
xtest.shape
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xtrain['Gender'],xtrain['Height'],ytrain,c='red', marker='o', alpha=0.5)
ax.plot_surface(xtest['Gender'],xtest['Height'],ytest, color='blue', alpha=0.3)
ax.set_xlabel('Price')
ax.set_ylabel('AdSpends')
ax.set_zlabel('Sales')
plt.show()
