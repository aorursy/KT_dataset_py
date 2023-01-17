# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# For preprocessing the data

from sklearn.preprocessing import Imputer

from sklearn import preprocessing

# To split the dataset into train and test datasets

from sklearn.cross_validation import train_test_split

# To model the Gaussian Navie Bayes classifier

from sklearn.naive_bayes import GaussianNB

# To calculate the accuracy score of the model

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv('../input/HR_comma_sep.csv')

#df.head()

# df.satisfaction_level.describe()

# df.describe(include=[np.number])



# Encoding non-numeric data: sales, salary

le = preprocessing.LabelEncoder()

sales_cat = le.fit_transform(df.sales)

salary_cat = le.fit_transform(df.salary)



df['sales_cat'] = sales_cat

df['salary_cat'] = salary_cat



# Copy to a new data frame

df_rev = df



#drop the old categorical columns from dataframe

dummy_fields = ['sales', 'salary']

df_rev = df_rev.drop(dummy_fields, axis = 1)



# df.isnull().sum()



num_features = ['satisfaction_level', 'last_evaluation', 'number_project', 

                'average_montly_hours', 'time_spend_company',

                'Work_accident', 'promotion_last_5years', 'sales_cat',

                'salary_cat']



df_rev = df_rev.reindex_axis(['satisfaction_level', 'last_evaluation', 'number_project', 

                                'average_montly_hours', 'time_spend_company',

                                'Work_accident', 'promotion_last_5years', 'sales_cat',

                                'salary_cat', 'left'], axis= 1)

 

# df_rev.head()



# Handling missing data





# standardize the data columns

scaled_features = {}

for each in num_features:

    mean = df_rev[each].mean()

    std = df_rev[each].std()

    scaled_features[each] = [mean, std]

    df_rev.loc[:, each] = (df_rev[each] - mean)/std

    

# Slice data

features = df_rev.values[:,:9]

# print(features)

target = df_rev.values[:, 9]

features_train, features_test, target_train, target_test = train_test_split(features,target, test_size = 0.33, random_state = 42)



# print(np.unique(target_train))

# print(target_train)

# Gaussian Naive Bayes

clf = GaussianNB()

clf.fit(features_train, target_train)

target_pred = clf.predict(features_test)



# Calculate the accuracy of the model

accuracy_score(target_test, target_pred, normalize = True)