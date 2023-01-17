

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

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# load data

dataset = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")

dataset.head()
dataset.shape
# check for infos

dataset.info()
# check for missing values 

dataset.isna().sum()
# we drop time feature from our dataset

data = dataset.copy()

data = data.drop('Time', axis = 1)

data.head()
data.hist(figsize = (20,20))

plt.show()
amount = pd.DataFrame(data['Amount'])

amount.head()
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

data['Amount'] = sc.fit_transform(amount)

#data['Amount'] = data['scaleAmount']



data.head()
data.Class.value_counts()
sns.countplot(data.Class)
# Creating feature and target vectors

X = data.drop('Class', axis = 1)

Y = data.Class



X.shape, Y.shape
# spliting data into train set and test set



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report, confusion_matrix



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)





y_pred_train = logreg.predict(x_train)

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))



print("Accuracy score: ", accuracy_score(y_test,y_pred))

print('\n')

print(confusion_matrix(y_test, y_pred))

print('\n')

print(classification_report(y_test, y_pred))
from sklearn.utils import resample



# separate the maority and minority class observation

data_major = data[data['Class'] == 0]

data_minor = data[data['Class'] == 1]



# over-sample the minority class observations

data_minor_oversample = resample(data_minor, replace = True, n_samples=284315, random_state = 0)



# finally combine the majority class observation and oversampled minoiry class observation

data_oversampled = pd.concat([data_major, data_minor_oversample])
# class label count after oversampled.we will see that minoity class now is proportionate to majority class

data_oversampled['Class'].value_counts()
# again lets splt our over samoled data into feature and traget variables

X = data_oversampled.drop('Class', axis = 1)

Y = data_oversampled.Class



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
from sklearn.utils import resample



# separate majority and minority class observation

data_major = data[data['Class'] == 0]

data_minor = data[data['Class'] == 1]



# perform undersampling in majority class data

data_major_undersample = resample(data_major, replace = False, n_samples=492, random_state = 0)



# finally concat the minority class data and undersampled majority class data

data_undersampled = pd.concat([data_minor, data_major_undersample])
# class lbel count after undersampling

data_undersampled.Class.value_counts()
# lets create feature and target variables from above undersampled data

X = data_undersampled.drop('Class', axis = 1)

Y = data_undersampled.Class



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
print('Initially the class distribution counts as below')

data.Class.value_counts()
# creating feature and target vectors

x = data.drop('Class', axis = 1)

y = data.Class



x.shape, y.shape
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

x_train.shape, x_test.shape
print('Label counts in splited y_train before applying SMOTE algorithm:\n')

print("counts of label '1': {}".format(sum(y_train == 1))) 

print("counts of label '0': {} \n".format(sum(y_train == 0))) 
from imblearn.over_sampling import SMOTE 

smote = SMOTE(random_state = 0) 

x_trainN, y_trainN = smote.fit_sample(x_train, y_train.ravel()) 



print('Lets see the sample size again')

print('\nx_train size: ', x_trainN.shape)

print(' y_train size: ', y_trainN.shape)
print('Label counts in splited y_train AFTER applying SMOTE algorithm:\n')

print("counts of label '1': {}".format(sum(y_trainN == 1))) 

print("counts of label '0': {} \n".format(sum(y_trainN == 0))) 
# model building

logreg = LogisticRegression() 

logreg.fit(x_trainN, y_trainN.ravel()) 

y_pred = logreg.predict(x_test) 





# Lets evaluate our model

y_pred_train = logreg.predict(x_trainN)

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_trainN, y_pred_train)))



print("Accuracy score: ", accuracy_score(y_test,y_pred))

print('\n')

print(confusion_matrix(y_test, y_pred))

print('\n')

print(classification_report(y_test, y_pred))  

