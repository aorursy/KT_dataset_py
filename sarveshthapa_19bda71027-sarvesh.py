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
import pandas as pd#for reading csv
import numpy as np#for numerical operations
import matplotlib.pyplot as plt#for plotting
import seaborn as sns#for interactive plotting
%matplotlib inline
#for inline executions
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
train=pd.read_csv("../input/bda-2019-ml-test/Train_Mask.csv")#reading train data
test=pd.read_csv("../input/bda-2019-ml-test/Test_Mask_Dataset.csv")#reading test data
train.head()#printing first 5 values of train dataset
test.head()#printing first 5 values of test dataset 
train.shape#printing the number of rows and columns for train
test.shape#printing the number of rows and columns for test
train.isnull().sum()
test.isnull().sum()
train.describe()
train.info()
#Looking for important features
sns.pairplot(train,palette='bwr')
train.corr()
sns.heatmap(train.corr())
#counts of customer churn cases vs not churn in dataset
target= train['flag'].value_counts()
levels = ['0','1']
trace = go.Pie(labels=target.index,values=target.values,
               marker=dict(colors=('orange','green')))
layout = dict(title="Period of belt", margin=dict(l=150), width=500, height=500)
figdata = [trace]
fig = go.Figure(data=figdata, layout=layout)
iplot(fig)
#print target class counts
print(target)
plt.figure(figsize=(10,6))
ax = sns.boxplot(x='flag', y = 'timeindex', data=train)
ax.set_title('Effect of belt', fontsize=18)
ax.set_ylabel('Time index', fontsize = 15)
ax.set_xlabel('Period of belt', fontsize = 15)
plt.figure(figsize=(10,6))
ax = sns.boxplot(x='flag', y = 'currentBack', data=train)
ax.set_title('Effect of belt', fontsize=18)
ax.set_ylabel('Current back chain', fontsize = 15)
ax.set_xlabel('Period of belt', fontsize = 15)
plt.figure(figsize=(10,6))
ax = sns.boxplot(x='flag', y = 'currentFront', data=train)
ax.set_title('Effect of belt', fontsize=18)
ax.set_ylabel('current front chain', fontsize = 15)
ax.set_xlabel('Period of belt', fontsize = 15)
plt.figure(figsize=(10,6))
ax = sns.boxplot(x='flag', y = 'motorTempBack', data=train)
ax.set_title('Effect of belt', fontsize=18)
ax.set_ylabel('Motor temp back', fontsize = 15)
ax.set_xlabel('Period of belt', fontsize = 15)
plt.figure(figsize=(10,6))
ax = sns.boxplot(x='flag', y = 'motorTempFront', data=train)
ax.set_title('Effect of belt', fontsize=18)
ax.set_ylabel('motor Temp Front', fontsize = 15)
ax.set_xlabel('Period of belt', fontsize = 15)
plt.figure(figsize=(10,6))
ax = sns.boxplot(x='flag', y = 'refPositionBack', data=train)
ax.set_title('Effect of belt', fontsize=18)
ax.set_ylabel('velocityFront', fontsize = 15)
ax.set_xlabel('Period of belt', fontsize = 15)
plt.figure(figsize=(10,6))
ax = sns.boxplot(x='flag', y = 'velocityBack', data=train)
ax.set_title('Effect of belt', fontsize=18)
ax.set_ylabel('velocityFront', fontsize = 15)
ax.set_xlabel('Period of belt', fontsize = 15)
plt.figure(figsize=(10,6))
ax = sns.boxplot(x='flag', y = 'trackingDeviationBack', data=train)
ax.set_title('Effect of belt', fontsize=18)
ax.set_ylabel('velocityFront', fontsize = 15)
ax.set_xlabel('Period of belt', fontsize = 15)
plt.figure(figsize=(10,6))
ax = sns.boxplot(x='flag', y = 'trackingDeviationFront', data=train)
ax.set_title('Effect of belt', fontsize=18)
ax.set_ylabel('velocityFront', fontsize = 15)
ax.set_xlabel('Period of belt', fontsize = 15)
plt.figure(figsize=(10,6))
ax = sns.boxplot(x='flag', y = 'refVelocityFront', data=train)
ax.set_title('Effect of belt', fontsize=18)
ax.set_ylabel('velocityFront', fontsize = 15)
ax.set_xlabel('Period of belt', fontsize = 15)
plt.figure(figsize=(10,6))
ax = sns.boxplot(x='flag', y = 'refVelocityFront', data=train)
ax.set_title('Effect of belt', fontsize=18)
ax.set_ylabel('velocityFront', fontsize = 15)
ax.set_xlabel('Period of belt', fontsize = 15)
train.columns
#from the above boxplots, I have to see that dataset is not structured at all and has many outliers
#Working on the Outliers
from scipy import stats
z = np.abs(stats.zscore(train))
print(z)
sum(z)/len(z) #finding average of the each array in z to find the threshold value
#other method for managing outliers is by using quartiles
Q1 = train.quantile(0.25)
Q3 = train.quantile(0.75)
IQR = Q3 - Q1
print(IQR)   
print(train < (Q1 - 1.5 * IQR)) |(train > (Q3 + 1.5 * IQR))
train_f=train[~((train < (Q1 - 1.5 * IQR)) |(train > (Q3 + 1.5 * IQR))).any(axis=1)]
#removing outliers, or the points which are present ourtside the whiskers of the quartiles
train_f.shape
#as we can see that more than 30% was dropped because of outliers, but this is not right approach as dropping more than 30% of the data is waste. So we will look for some other way
#logistic Regression
from sklearn.model_selection import train_test_split
X = train.drop('flag',axis=1)
y = train['flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
#random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=6000)
rfc.fit(X_train,y_train)
predictions2 = rfc.predict(X_test)
print(classification_report(y_test,predictions2))
df = pd.DataFrame(predictions2,columns=['flag'])
df.head()
df['flag'].value_counts()
df.to_csv(r'C:\Users\sarve\Downloads\ML lab external\19BDA71027.csv',header=['flag'])
