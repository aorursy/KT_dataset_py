import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('../input/salary-data-simple-linear-regression/Salary_Data.csv')
# Displaying the columns of the dataset

df.columns
# displaying first five entries of the dataset

df.head()
# shape of the Dataset

df.shape
# Info about the dataset

df.info()
# Checking for the missing values in the dataset

df.isnull().sum()

# There are no missing values in the Dataset
df.dtypes
# Describtion of the Dataset

df.describe()
# Checking for the outliers

plt.boxplot(df['YearsExperience'])
sns.distplot(df['YearsExperience'])
# Checking for the presence of outliers in the dependent variable

plt.boxplot(df['Salary'])
sns.distplot(df['Salary'])
sns.scatterplot(x='YearsExperience',y='Salary',data=df)
plt.title('Correlation Matrix')

sns.heatmap(df.corr(),annot=True)
X=df.iloc[:,0]

Y=df.iloc[:,1]
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1)
print("Shape of X_train is: ",X_train.shape)

print("Shape of X_test is: ",X_test.shape)

print("Shape of Y_train is: ",Y_train.shape)

print("Shape of Y_test is: ",Y_test.shape)
#Adding new column

X_train=X_train[:,np.newaxis]

X_test=X_test[:,np.newaxis]
from sklearn.linear_model import LinearRegression

lrg=LinearRegression()
lrg.fit(X_train,Y_train)
lrg.score(X_train,Y_train)
# predicting the output of test dataset

Y_pred=lrg.predict(X_test)
data={'Y_test':Y_test,'Y_pred':Y_pred}

pd.DataFrame(data=data)
print(lrg.intercept_)

print(lrg.coef_)
plt.title('Residual Plot',size=16)

sns.residplot(Y_test,Y_pred,color='r')

plt.xlabel('Y_pred',size=12)

plt.ylabel('Residues',size=12)
from sklearn.metrics import r2_score

r2=r2_score(Y_test,Y_pred)

print('r2_score is: ',r2)