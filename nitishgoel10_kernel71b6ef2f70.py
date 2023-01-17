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



# code to read the csv file

print(os.listdir("../input"))

hd_df = pd.read_csv("../input/heart_disease_train - Copy.csv") 

hd_df.head()



# Get total number of rows in the dataset/dataframe

print(len(hd_df))

# Any results you write to the current directory are saved as output.



hd_df.info()
# Get the column name which has null values

hd_df.isnull().any()

hd_df.isnull().values.any()



# Get number of rows having null value in each column

cols = hd_df.columns[hd_df.isnull().any()]

for col in cols:

    print(col, len(hd_df[hd_df[col].isnull()]))

#     drop_row_col = hd_df[hd_df[col].isnull()]

#     df.dropna(axis=0, how='any', inplace=True)

# remove all the rows where value is null and storing it in a DF

    hd_df = hd_df[pd.notnull(hd_df[col])]

    print(col, len(hd_df[hd_df[col].isnull()]))



# drop the columns

# df = df.drop(df[col],axis=1)



# other ways of null value imputation: Mean/Mode/median

# for feature in cols:

#     mode_value = df[df[feature]!='#NULL!'][feature].mode().iloc[0]

#     df.loc[df[feature]=='#NULL!', feature]=mode_value

# for features in cols:

#     mean_value = df[df[features]!='#NULL!'][features].astype(float).mean()

#     df.loc[df[features]=='#NULL!', features]=mean_value
hd_df.describe()

hd_df.info()
#box plot

#convert age into years

hd_df['age_yrs'] = hd_df['age']/365

hd_df['age_yrs']= hd_df['age_yrs'].round(3)

hd_df.head()
# Get dummies - converting the categorical value 

hd_df= pd.get_dummies(hd_df, columns = ['cholesterol', 'gluc'])

hd_df.drop(['id', 'age'], inplace=True, axis=1)

hd_df.head()

print(hd_df['cardio'].unique())

from sklearn import preprocessing

from sklearn.model_selection import train_test_split





x = hd_df.loc[:,hd_df.columns!='cardio']

print(x.shape)

y = hd_df['cardio'].values

print(y.shape)

# x= preprocessing.normalize(x)

x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.3, random_state = 7)

from sklearn.tree import DecisionTreeClassifier 

from sklearn import metrics

clf = DecisionTreeClassifier()

clf = clf.fit(x_train,y_train)

print("Accuracy:",metrics.accuracy_score(x_train, y_train))

print("Accuracy:",metrics.accuracy_score(x_test, y_test))