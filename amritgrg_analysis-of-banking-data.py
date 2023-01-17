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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# set some display options:

sns.set(style="whitegrid")

pd.set_option("display.max_columns", 36)
# load datas

train = pd.read_csv('/kaggle/input/banking-dataset-marketing-targets/train.csv')

test = pd.read_csv('/kaggle/input/banking-dataset-marketing-targets/test.csv')
train.head()
test.head()
# print train and test data shape

print('train set size is : ', train.shape)

print('test set size is : ', test.shape)
# lets see the features of train set

train.columns
# ID coluns have no value in classification so lets drop it.



train.drop('ID', axis = 1, inplace = True)

test.drop('ID', axis = 1, inplace = True)
# lets check infos of train set

train.info()
train.isna().sum()
# subscribed feature is our target variabe wwe will convert it to numeric 



train['subscribed'].replace('no', 0 , inplace=True)

train['subscribed'].replace('yes', 1, inplace=True)
# numerical features

numerical_features = [cols for cols in train.columns if train[cols].dtype != 'O']

numerical_features = train[numerical_features]

numerical_features.head()
numerical_features.describe()
# correlation between independent features and dependent features

corr = train.corr()

corr
# correlation in heatmap

sns.heatmap(corr)
numerical_features.hist(figsize = (15,15))

plt.show()
# categorical features

categorical_features = [cols for cols in train.columns if train[cols].dtype == 'O']

categorical_features = train[categorical_features]

categorical_features.head()
print('Lets see the unique values in Categorical_features:')

print('#########################################################\n')



for feature in categorical_features:

    print('The unique values in '+ feature + " " + 'feature are:' )

    print(train[feature].unique())

    print('\n')
# Checking the target features

train.subscribed.value_counts()
sns.countplot(train.subscribed)

plt.show()
# Distribution of job feature

sns.countplot(train.job)

plt.xticks(rotation=90)

plt.show()
# Distribution of marital features

sns.countplot(train.marital)

plt.xticks(rotation=0)

plt.show()
# Distribution of marital feature

sns.countplot(train.education)

plt.xticks(rotation=0)

plt.show()
# Distribution of default feature

sns.countplot(train.default)

plt.xticks(rotation=0)

plt.show()
# Distribution of housing features

sns.countplot(train.housing)

plt.xticks(rotation=0)

plt.show()
# Distribution of loan features

sns.countplot(train.loan)

plt.xticks(rotation=0)

plt.show()
# Distribution of contact feature

sns.countplot(train.contact)

plt.xticks(rotation=0)

plt.show()
# Distribution of month feature

sns.countplot(train.month)

plt.xticks(rotation=90)

plt.show()
sns.countplot(x="marital", hue="subscribed", data=train)
sns.countplot(x="education", hue="subscribed", data=train)

plt.xticks(rotation=0)

plt.show()
sns.countplot(x="job", hue="subscribed", data=train)

plt.xticks(rotation=90)

plt.show()
sns.countplot(x="housing", hue="subscribed", data=train)

plt.xticks(rotation=0)

plt.show()
sns.countplot(x="loan", hue="subscribed", data=train)

plt.xticks(rotation=0)

plt.show()
sns.countplot(x="default", hue="subscribed", data=train)

plt.xticks(rotation=0)

plt.show()
sns.countplot(x="contact", hue="subscribed", data=train)

plt.xticks(rotation=0)

plt.show()
sns.countplot(x="month", hue="subscribed", data=train)

plt.xticks(rotation=90)

plt.show()
sns.countplot(x="poutcome", hue="subscribed", data=train)

plt.xticks(rotation=0)

plt.show()
# combine train adn test data together

data = pd.concat([train,test], ignore_index=True)
train.shape, test.shape, data.shape
data.head()
# checking null values

data.isna().sum()
# lets separate newly combined data into numerical and categorical features



numerical_features = [cols for cols in data.columns if data[cols].dtype != 'O']

categorical_features = [cols for cols in data.columns if data[cols].dtype == 'O']
data[numerical_features].head()
data[categorical_features].head()
# get dummies of categorical variable. Its same as one hot encoding.

# we do this one hot encoding of categorical variables because our ML algorithms only works with numeric



cat_dummies = pd.get_dummies(data[categorical_features])

cat_dummies.head()
# concat numerical features and one hot encoded features fromm above



newdata = pd.concat([data[numerical_features], cat_dummies], axis = 1)

newdata.head()
newdata.shape
# create feature and target vectors



features = newdata.drop('subscribed', axis = 1)

target = newdata.subscribed

features.describe()
features.shape, target.shape
cols = features.columns # columns of features 
# scaling is done here

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

features = scaler.fit_transform(features)
# After scaling features above it is changed into ndarray. so we will change it back to Dataframe

features = pd.DataFrame(features, columns = [cols])

print(features.shape)

features.head()
features.describe()
# lets add subscribed to our features 

features['subscribed'] = target
features.head()
# separate traina dnd test data



train2 = features.iloc[:31647]

test2 = features.iloc[31647:]

train2.shape, test2.shape
# drop target feature from test data

test2.drop('subscribed', axis = 1, inplace = True)

print(test2.shape)
train2.head()
# import necessary modules

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Define feature and target vectors

x = train2.iloc[:,:-1]

y = train2.iloc[:,-1]

x.shape, y.shape
# split data into train and test set



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=0)
# model creation

logreg = LogisticRegression()

logreg.fit(x_train,y_train)
# predict result for x_test



y_pred = logreg.predict(x_test)

y_pred
# check model accuracy



print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
# check training set accuracy



y_pred_train = logreg.predict(x_train)

y_pred_train



print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
# print the scores on training and test set



print('Training set score: {:.4f}'.format(logreg.score(x_train, y_train)))



print('Test set score: {:.4f}'.format(logreg.score(x_test, y_test)))
# predicting target class for unseen test data.

# this data is test.csv given by kaggle



y_pred_test2 = logreg.predict(test2)

y_pred_test2
df_y_pred_test2 = pd.DataFrame(y_pred_test2, columns = ['test2pred'])

df_y_pred_test2.head()
df_y_pred_test2['test2pred'].unique()
df_y_pred_test2['test2pred'].value_counts()
# confusion matrix and classification report



cm = confusion_matrix(y_test,y_pred)

print('confusion matrix: \n ', cm)

print('\n')

print('classification report: \n ', classification_report(y_test,y_pred))
y.value_counts()
from sklearn.utils import resample



# separate the maority and minority class observation

data_major = train2[y == 0.0]

data_minor = train2[y == 1.0]



# over-sample the minority class observations

data_minor_oversample = resample(data_minor, replace = True, n_samples=27932, random_state = 0)



# finally combine the majority class observation and oversampled minoiry class observation

data_oversampled = pd.concat([data_major, data_minor_oversample])
# class label count after oversampled.we will see that minoity class now is proportionate to majority class

data_oversampled.iloc[:,-1].value_counts()
# again lets splt our over samoled data into feature and traget variables

X = data_oversampled.iloc[:,:-1]

Y = data_oversampled.iloc[:,-1]



# lets split data into train and test set

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)



# model building

logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)



# Lets evaluate our model

y_pred_train = logreg.predict(x_train)

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))



print("Accuracy score: ", accuracy_score(y_test,y_pred))

print('\n')

print(confusion_matrix(y_test, y_pred))

print('\n')

print(classification_report(y_test, y_pred))