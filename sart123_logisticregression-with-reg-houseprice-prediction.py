# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df=pd.read_csv('../input/train.csv')

test_df=pd.read_csv('../input/test.csv')

test_df.describe()

test_df.head()
#train_df.head()

train_df.columns.values

Y_train=train_df['SalePrice']

train_df=train_df.drop("SalePrice",axis=1)

Y_train=Y_train.fillna(Y_train.median())

print(Y_train)
#merge test and train data

#replace all the null categorical data with mode

#repalce all the null numerical data with the median of the column

df=pd.concat([train_df,test_df])

for col in df.columns.values:

    if df[col].dtype=='object':

        if np.sum(df[col].isnull()) > 100:

            df = df.drop(col, axis = 1)

        elif np.sum(df[col].isnull()) > 0:

            mode = df[col].mode()[0]

            idx = np.where(df[col].isnull())[0]

            df[col].iloc[idx] = mode

    else:

        if np.sum(df[col].isnull()) > 100:

            df = df.drop(col, axis = 1)

        elif np.sum(df[col].isnull()) > 0:

            median = df[col].median()

            idx = np.where(df[col].isnull())[0]

            df[col].iloc[idx] = median

col_val=df.columns.values

df.head()       
df.isnull().sum()
df.count()
df=pd.get_dummies(df)

df.head()
#split data

train_df=df.iloc[:1460]

test_df=df.iloc[1460:]

#train_df.count()

test_df.count()
#create a Random Forest model

from sklearn.ensemble import RandomForestRegressor

from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt

X_train ,X_test,Y_training,Y_test=train_test_split(train_df,Y_train,test_size=0.1)

clf=RandomForestRegressor(n_estimators=800,n_jobs=-1)

clf.fit(X_train,Y_training)



coef = pd.Series(clf.feature_importances_, index = X_train.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 5))

coef.head(25).plot(kind='bar')

plt.title('Feature Significance')

plt.tight_layout()
#predict using the model

import matplotlib.pyplot as plt

Y_pred=clf.predict(X_test)

plt.figure(figsize=(10, 5))

plt.scatter(Y_test, Y_pred, s=20)

plt.title('Predicted vs. Actual')

plt.xlabel('Actual Sale Price')

plt.ylabel('Predicted Sale Price')



plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)])

plt.tight_layout()
#preprocess the test data

#take only the features that are used for classification 

#make the null values 

test_df.head()

results= clf.predict(test_df)

indices=np.arange(1,len(results)+1)

result_df = pd.DataFrame({'Id':test_df['Id'],'SalePrice':results})

result_df.head()

result_df.to_csv('out.csv',index=False)

#print (indices)

print(len(results))

#print(len(indices))