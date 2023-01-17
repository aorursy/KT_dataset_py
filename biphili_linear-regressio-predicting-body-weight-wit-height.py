# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
df=pd.read_csv('../input/weight-height/weight-height.csv')

df.head()
df.shape
df.info()
df.describe()
df['Gender'].value_counts()
df.plot(kind='scatter',x='Height',y='Weight');
males=df[df['Gender']=='Male']

females=df[df['Gender']=='Female']
fig,ax = plt.subplots()

males.plot(kind='scatter',x='Height',y='Weight',

          ax=ax,color='blue',alpha=0.3,

          title='Male and Female Populations')

females.plot(kind='scatter',x='Height',y='Weight',

          ax=ax,color='red',alpha=0.3,

          title='Male and Female Populations');
df['Genddercolor'] = df['Gender'].map({'Male':'blue','Female':'red'})
df.plot(kind='scatter',x='Height',y='Weight',c=df['Genddercolor'],alpha=0.3,title='Male & Female Population');
fig,ax = plt.subplots()

ax.plot(males['Height'],males['Weight'],'ob',females['Height'],females['Weight'],'or',alpha=0.3)

plt.xlabel('Height')

plt.ylabel('Weight')

plt.title('Male & Female Populations');
males['Height'].plot(kind='hist',bins=50,range=(50,80),alpha=0.3,color='blue')

females['Height'].plot(kind='hist',bins=50,range=(50,80),alpha=0.3,color='red')

plt.title('Height distribution')

plt.legend(['Males','Females'])

plt.xlabel('Height in')

plt.axvline(males['Height'].mean(),color='blue',linewidth=2)

plt.axvline(females['Height'].mean(),color='red',linewidth=2);
males['Height'].plot(kind='hist',bins=200,range=(50,80),alpha=0.3,color='blue',cumulative=True,normed=True)

females['Height'].plot(kind='hist',bins=200,range=(50,80),alpha=0.3,color='red',cumulative=True,normed=True)



plt.title('Height Distribution')

plt.legend(['Males','Females'])

plt.xlabel('Height (in)')



plt.axhline(0.8)

plt.axhline(0.5)

plt.axhline(0.2);
dfpvt=df.pivot(columns='Gender',values='Weight')

dfpvt.head(2)
dfpvt.plot(kind='box');

plt.title('Weight Box Plot')

plt.ylabel('Weight (lb)')
X=df['Height'].values[:,None]

X.shape
y=df.iloc[:,2].values

y.shape
from sklearn.model_selection import train_test_split 

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LinearRegression

lm=LinearRegression()

lm.fit(X_train,y_train)
y_test=lm.predict(X_test)

print(y_test)
plt.scatter(X,y,color='b')

plt.plot(X_test,y_test,color='black',linewidth=3)

plt.xlabel('Height in inches')

plt.ylabel('Weigth in Pounds')

plt.show()
y_train_pred=lm.predict(X_train).ravel()

y_test_pred=lm.predict(X_test).ravel()
from sklearn.metrics import mean_squared_error as mse,r2_score
print("The Mean Squared Error on Train set is:\t{:0.1f}".format(mse(y_train,y_train_pred)))

print("The Mean Squared Error on Test set is:\t{:0.1f}".format(mse(y_test,y_test_pred)))
print("The R2 score on the Train set is:\t{:0.1f}".format(r2_score(y_train,y_train_pred)))

print("The R2 score on the Test set is:\t{:0.1f}".format(r2_score(y_test,y_test_pred)))