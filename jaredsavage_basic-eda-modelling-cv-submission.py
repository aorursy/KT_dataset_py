import os

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

%matplotlib inline



#Load data

df_train = pd.read_csv('../input/learn-together/train.csv')

df_test = pd.read_csv('../input/learn-together/test.csv')

#Concatenate train and test sets into one DataFrame

df_full = pd.concat([df_train,df_test], sort=True, ignore_index = True)
# count the number of missing data and show the top ten

total = df_full.isnull().sum().sort_values(ascending=False)

percent = (df_full.isnull().sum()/df_full.isnull().count()).sort_values(ascending=False)*100

missing_data = pd.concat([total, round(percent,2)], axis=1, keys=['Total', 'Percent'])

missing_data.head(10)
df_train.info()
df_train.describe()
sns.distplot(df_train['Elevation'])
sns.distplot(df_train['Hillshade_9am'], hist_kws={'color':'b'}, kde_kws={"label":"9am",'color':'b'}) 

sns.distplot(df_train['Hillshade_Noon'], hist_kws={'color':'r'}, kde_kws={"label":"Noon", 'color':'r'})

sns.distplot(df_train['Hillshade_3pm'], hist_kws={'color':'g'}, kde_kws={"label":"3pm", 'color':'g'})

df_train['Cover_Type'].value_counts(ascending = True)
sns.scatterplot(df_train['Horizontal_Distance_To_Hydrology'],df_train['Vertical_Distance_To_Hydrology'])
dims = (20, 12)

fig, ax = plt.subplots(figsize=dims)

sns.scatterplot(df_full['Horizontal_Distance_To_Hydrology'],df_full['Vertical_Distance_To_Hydrology'], hue = df_full['Horizontal_Distance_To_Fire_Points'], ax=ax)
X = df_train.iloc[:, 1:55].values #Excludes the Id attribute

y = df_train.iloc[:, -1].values #Separates the dependant variable



# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) #Uses 80% for training and 20% for testing



# Fitting XGBoost to the Training set

from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

#from sklearn.metrics import confusion_matrix

#cm = confusion_matrix(y_test, y_pred)



# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

accuracies.mean()
# Predicting the Test set results

test_id = df_test['Id']

test_set = df_test.iloc[:,1:55]

X_train = pd.DataFrame(X_train)

test_set = pd.DataFrame(data=test_set.values)

test_set = np.array(test_set)
y_pred_test = classifier.predict(test_set)
sub = pd.DataFrame({'Id':  test_id, 'Cover_Type': y_pred_test})

sub.to_csv('submission_xgb.csv', index = False)
y_pred_test