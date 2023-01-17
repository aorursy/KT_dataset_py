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

from sklearn.preprocessing import LabelEncoder,MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split, ParameterGrid

from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, log_loss

import operator

import json

from IPython import display

import os

import warnings



np.random.seed(0)

warnings.filterwarnings("ignore")

THRESHOLD = 4

train = pd.read_csv("../input/padhai-module1-assignment/train.csv")

train.head()

test = pd.read_csv("../input/padhai-module1-assignment/test.csv")

test.head()

train_filtered= train[['PhoneId','Capacity','Expandable Memory','FM Radio','Fingerprint Sensor','Height','Internal Memory','Pixel Density','Processor','RAM','Resolution','SIM Size','Screen Protection','Screen Size','Screen to Body Ratio (calculated)','Thickness','Weight','Width']]

train_filtered.head()

test_filtered= train[['PhoneId','Capacity','Expandable Memory','FM Radio','Fingerprint Sensor','Height','Internal Memory','Pixel Density','Processor','RAM','Resolution','SIM Size','Screen Protection','Screen Size','Screen to Body Ratio (calculated)','Thickness','Weight','Width']]

test_filtered.head()





print("Number of data points in train: %d" % train_filtered.shape[0])

print("Number of features in train: %d" % train_filtered.shape[1])

print("Number of data points in test: %d" % test_filtered.shape[0])

print("Number of features in test: %d" % test_filtered.shape[1])

train_filtered = train_filtered[(train_filtered.isnull().sum(axis=1) <= 15)]

print("Number of data points in train: %d" % train_filtered.shape[0])

print("Number of features in train: %d" % train_filtered.shape[1])

test_filtered = test_filtered[(test_filtered.isnull().sum(axis=1) <= 15)]

print("Number of data points in test: %d" % test_filtered.shape[0])

print("Number of features in test: %d" % test_filtered.shape[1])
def for_integer(test):

    try:

        test = test.strip()

        return int(test.split(' ')[0])

    except IOError:

           pass

    except ValueError:

        pass

    except:

        pass



def for_string(test):

    try:

        test = test.strip()

        return (test.split(' ')[0])

    except IOError:

        pass

    except ValueError:

        pass

    except:

        pass



def for_float(test):

    try:

        test = test.strip()

        return float(test.split(' ')[0])

    except IOError:

        pass

    except ValueError:

        pass

    except:

        pass

def find_freq(test):

    try:

        test = test.strip()

        test = test.split(' ')

        if test[2][0] == '(':

            return float(test[2][1:])

        return float(test[2])

    except IOError:

        pass

    except ValueError:

        pass

    except:

        pass



    

def for_Internal_Memory(test):

    try:

        test = test.strip()

        test = test.split(' ')

        if test[1] == 'GB':

            return int(test[0])

        if test[1] == 'MB':

#             print("here")

            return (int(test[0]) * 0.001)

    except IOError:

           pass

    except ValueError:

        pass

    except:

        pass

    

def find_freq(test):

    try:

        test = test.strip()

        test = test.split(' ')

        if test[2][0] == '(':

            return float(test[2][1:])

        return float(test[2])

    except IOError:

        pass

    except ValueError:

        pass

    except:

        pass

    

def data_clean_2(x):

    data = x.copy()

    

    data['Capacity'] = data['Capacity'].apply(for_integer)



    data['Height'] = data['Height'].apply(for_float)

    data['Height'] = data['Height'].fillna(data['Height'].mean())



    data['Internal Memory'] = data['Internal Memory'].apply(for_Internal_Memory)



    data['Pixel Density'] = data['Pixel Density'].apply(for_integer)



    data['Internal Memory'] = data['Internal Memory'].fillna(data['Internal Memory'].median())

    data['Internal Memory'] = data['Internal Memory'].astype(int)



    data['RAM'] = data['RAM'].apply(for_integer)

    data['RAM'] = data['RAM'].fillna(data['RAM'].median())

    data['RAM'] = data['RAM'].astype(int)



    data['Resolution'] = data['Resolution'].apply(for_integer)

    data['Resolution'] = data['Resolution'].fillna(data['Resolution'].median())

    data['Resolution'] = data['Resolution'].astype(int)



    data['Screen Size'] = data['Screen Size'].apply(for_float)



    data['Thickness'] = data['Thickness'].apply(for_float)

    data['Thickness'] = data['Thickness'].fillna(data['Thickness'].mean())

    data['Thickness'] = data['Thickness'].round(2)





    data['Screen to Body Ratio (calculated)'] = data['Screen to Body Ratio (calculated)'].apply(for_float)

    data['Screen to Body Ratio (calculated)'] = data['Screen to Body Ratio (calculated)'].fillna(data['Screen to Body Ratio (calculated)'].mean())

    data['Screen to Body Ratio (calculated)'] = data['Screen to Body Ratio (calculated)'].round(2)



    data['Width'] = data['Width'].apply(for_float)

    data['Width'] = data['Width'].fillna(data['Width'].mean())

    data['Width'] = data['Width'].round(2)





    data['SIM Size'][data['SIM Size'].isna() == True] = "Other"





    data['Fingerprint Sensor'][data['Fingerprint Sensor'].isna() == True] = "Other"



    data['Expandable Memory'][data['Expandable Memory'].isna() == True] = "No"



    data['Weight'] = data['Weight'].apply(for_integer)

    data['Weight'] = data['Weight'].fillna(data['Weight'].mean())

    data['Weight'] = data['Weight'].astype(int)



    

    

    return data

	

	
train_filtered = data_clean_2(train_filtered)

train_filtered.head()

test_filtered = data_clean_2(test_filtered)

test_filtered.head()



train_ids = train_filtered['PhoneId']

train_ids.head()



test_ids = test_filtered['PhoneId']

test_ids.head()





train_without_phoneid=train_filtered.drop(columns=['PhoneId'])

train_without_phoneid.head()

test_without_phoneid=test_filtered.drop(columns=['PhoneId'])

test_without_phoneid.head()
train_with_phoneid_rating_merge = train_filtered.merge(train[['PhoneId', 'Rating']], on='PhoneId')

train_with_phoneid_rating_merge.head()

train_with_phoneid_rating_merge.shape



test_with_phoneid_rating_merge = test_filtered.merge(train[['PhoneId', 'Rating']], on='PhoneId')

test_with_phoneid_rating_merge.head()

test_with_phoneid_rating_merge.shape
from random import randint

from sklearn.metrics import accuracy_score

class MPNeuron:

    def __init__(self):

        self.b = 0

    def model(self,x):

        print(x)

        return(sum(x) >= self.b)

    def predict(self,X):

        Y = []

        for x in X:

            result = self.model(x)

            Y.append(result)

        return(np.array(Y))

    def fit(self, X, Y):

        accuracy = {}

        for b in range(X.shape[1] + 1):

            self.b = b

            Y_pred = self.predict(X)

            accuracy[b] = accuracy_score(Y_pred,Y)

    

        best_b = max(accuracy,key = accuracy.get)

        self.b = best_b

        print('Optimal value of b: ',best_b)

        print('Highest accuracy is: ',accuracy[best_b])

        return(accuracy[best_b])
Y_binarised_3_train = train['Rating'].map(lambda x: 0 if x < 4 else 1)

Y_binarised_3_train







X_train_binarised = np.array([

#X_train_new['Weight'].map(lambda x: 1 if x > 153 else 0), # Removed due to correlation and accuracy increased

#X_train_new['Height'].map(lambda x: 1 if x > 151 else 0), # Removed due to correlation and accuracy increased

#X_train_new['Screen to Body Ratio (calculated)'].map(lambda x: 1 if x >= 55  else 0), # Removed due to correlation and accuracy increased

train_filtered['Pixel Density'].map(lambda x: 1 if x > 270 else 0),

train_filtered['Screen Size'].map(lambda x: 1 if x > 4.8 else 0),

train_filtered['RAM'].map(lambda x: 1 if x > 4 else 0),

#X_train_new['Resolution'].map(lambda x: 1 if x > 20 else 0), # Removed due to correlation and accuracy increased

train_filtered['Internal Memory'].map(lambda x: 1 if x > 64 else 0),

train_filtered['Capacity'].map(lambda x: 1 if x > 2100 else 0)

    

]) 



X_train_binarised = X_train_binarised.T

X_train_binarised.shape

Y_binarised_3_train.head()

Y_binarised_3_train=Y_binarised_3_train.drop(Y_binarised_3_train.index[2])

Y_binarised_3_train.shape

x=X_train_binarised

y=Y_binarised_3_train

mp_neuron = MPNeuron()

train_accuracy = mp_neuron.fit(x,y)

print(train_accuracy)



X_test_binarised = np.array([

#X_train_new['Weight'].map(lambda x: 1 if x > 153 else 0), # Removed due to correlation and accuracy increased

#X_train_new['Height'].map(lambda x: 1 if x > 151 else 0), # Removed due to correlation and accuracy increased

#X_train_new['Screen to Body Ratio (calculated)'].map(lambda x: 1 if x >= 55  else 0), # Removed due to correlation and accuracy increased

test_filtered['Pixel Density'].map(lambda x: 1 if x > 270 else 0),

test_filtered['Screen Size'].map(lambda x: 1 if x > 4.8 else 0),

test_filtered['RAM'].map(lambda x: 1 if x > 4 else 0),

#X_train_new['Resolution'].map(lambda x: 1 if x > 20 else 0), # Removed due to correlation and accuracy increased

test_filtered['Internal Memory'].map(lambda x: 1 if x > 64 else 0),

test_filtered['Capacity'].map(lambda x: 1 if x > 2100 else 0)

    

]) 





X_test_binarised.shape

x=X_test_binarised.T

x



mp_neuron = MPNeuron()

Y_test_pred = mp_neuron.predict(x)

    

# Convert True, False to 1,0

Y_test_pred = Y_test_pred.astype(int)

Y_test_pred