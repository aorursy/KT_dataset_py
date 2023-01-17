# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train_features = pd.read_csv('../input/train_features.csv')

df_train_labels = pd.read_csv("../input/train_labels.csv")

df_test_features = pd.read_csv("../input/test_features.csv")

sample_submission = pd.read_csv("../input/sample_submission.csv")
df_train_labels.columns
df_train_features.shape
df_test_features.shape
import numpy as np

majority_class = df_train_labels['status_group'].mode()[0]

#print(majority_class)



y_pred = np.full(shape=df_train_labels['status_group'].shape, fill_value=majority_class)
df_train_labels.status_group.shape, y_pred.shape
all(y_pred==majority_class)
from sklearn.metrics import accuracy_score 

accuracy_score(df_train_labels['status_group'], y_pred)
df_train_labels['status_group'].value_counts()
df_train_labels['status_group'].value_counts(normalize=True)
%matplotlib inline

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression

from mlxtend.plotting import plot_decision_regions





#let's import the warning before running any sophisticated methods

import warnings

from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
print(classification_report(df_train_labels['status_group'], y_pred) )
#Let's merge train and test

full_df = pd.concat([df_train_features, df_test_features])
full_df.shape
#these were the number of rows in the orignal sets

59400 + 14358
full_df.head()
full_df.isna().sum()
#data['Native Country'] = data['Native Country'].fillna(data['Native Country'].mode()[0])

full_df['funder'] = full_df['funder'].fillna(full_df['funder'].mode()[0])
full_df['installer'] = full_df['installer'].fillna(full_df['installer'].mode()[0])
full_df['subvillage'] = full_df['subvillage'].fillna(full_df['subvillage'].mode()[0])
full_df['public_meeting'] = full_df['public_meeting'].fillna(full_df['public_meeting'].mode()[0])
full_df['scheme_management'] = full_df['scheme_management'].fillna(full_df['scheme_management'].mode()[0])
full_df['permit'] = full_df['permit'].fillna(full_df['permit'].mode()[0])
full_df.isna().sum()
full_df = full_df.drop(columns = 'scheme_name')
#split the data back

X_cleaned = full_df[:-14358]

X_test_cleaned = full_df[-14358:]

y = df_train_labels['status_group']
X_cleaned.shape, X_test_cleaned.shape, y.shape

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y, test_size=0.25, random_state=42, shuffle=True)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
X_train.dtypes
X_train_numeric = X_train.select_dtypes(np.number)
X_test_numeric = X_test.select_dtypes(np.number)
#let's see how our model does here



from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train_numeric, y_train)

y_pred = model.predict(X_test_numeric)

accuracy_score(y_test, y_pred)
from sklearn.preprocessing import LabelEncoder

def dummyEncode(df):

        columnsToEncode = list(df.select_dtypes(include=['category','object']))

        le = LabelEncoder()

        for feature in columnsToEncode:

            try:

                df[feature] = le.fit_transform(df[feature])

            except:

                print('Error encoding '+feature)

        return df
#encode our train df that we split from the full_df

cat_coded_df = dummyEncode(X_cleaned)
cat_coded_df.head()
#let's also encode out test set we split from full_df

X_cleaned_test = dummyEncode(X_test_cleaned)
X_cleaned_test.head()
#split our train set that we just encoded (and assigned into cat_coded_df)

#into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(cat_coded_df, y, test_size=0.25, random_state=42, shuffle=True)
import warnings

from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
#run multinomial logistic regression

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='newton-cg', multi_class='multinomial')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test, y_pred)
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline



pipeline = make_pipeline(StandardScaler(), 

                        LogisticRegression(solver='newton-cg', multi_class='multinomial'))



pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
accuracy_score(y_test, y_pred)