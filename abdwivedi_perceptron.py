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



import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder,MinMaxScaler, StandardScaler

from sklearn.metrics import accuracy_score

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
#Reading the files

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



#Checking the no of features and data points in train

print("No of data points in train is %d" % train.shape[0])

print("No of features in train is %d" % train.shape[1])



#Checking the no of features and data points in test

print("No of data points in test is %d" % test.shape[0])

print("No of features in test is %d" % test.shape[1])
def data_clean(data):

    

    # Let's first remove all missing value features

    columns_to_remove = ['Also Known As','Applications','Audio Features','Bezel-less display'

                         'Browser','Build Material','Co-Processor','Browser'

                         'Display Colour','Mobile High-Definition Link(MHL)',

                         'Music', 'Email','Fingerprint Sensor Position',

                         'Games','HDMI','Heart Rate Monitor','IRIS Scanner', 

                         'Optical Image Stabilisation','Other Facilities',

                         'Phone Book','Physical Aperture','Quick Charging',

                         'Ring Tone','Ruggedness','SAR Value','SIM 3','SMS',

                         'Screen Protection','Screen to Body Ratio (claimed by the brand)',

                         'Sensor','Software Based Aperture', 'Special Features',

                         'Standby time','Stylus','TalkTime', 'USB Type-C',

                         'Video Player', 'Video Recording Features','Waterproof',

                         'Wireless Charging','USB OTG Support', 'Video Recording','Java']



    columns_to_retain = list(set(data.columns)-set(columns_to_remove))

    data = data[columns_to_retain]



    #Features having very low variance 

    columns_to_remove = ['Architecture','Audio Jack','GPS','Loudspeaker','Network','Network Support','VoLTE']

    columns_to_retain = list(set(data.columns)-set(columns_to_remove))

    data = data[columns_to_retain]



    # Multivalued:

    columns_to_remove = ['Architecture','Launch Date','Audio Jack','GPS','Loudspeaker','Network','Network Support','VoLTE', 'Custom UI']

    columns_to_retain = list(set(data.columns)-set(columns_to_remove))

    data = data[columns_to_retain]



    # Not much important

    columns_to_remove = ['Bluetooth', 'Settings','Wi-Fi','Wi-Fi Features']

    columns_to_retain = list(set(data.columns)-set(columns_to_remove))

    data = data[columns_to_retain]

    

    return data
train = data_clean(train)

test = data_clean(test)
train = train[(train.isnull().sum(axis=1) <= 15)]

# You shouldn't remove data points from test set

#test = test[(test.isnull().sum(axis=1) <= 15)]
# check the number of features and data points in train

print("Number of data points in train: %d" % train.shape[0])

print("Number of features in train: %d" % train.shape[1])



# check the number of features and data points in test

print("Number of data points in test: %d" % test.shape[0])

print("Number of features in test: %d" % test.shape[1])
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



    data['Type'] = data['Type'].fillna('Li-Polymer')



    data['Screen to Body Ratio (calculated)'] = data['Screen to Body Ratio (calculated)'].apply(for_float)

    data['Screen to Body Ratio (calculated)'] = data['Screen to Body Ratio (calculated)'].fillna(data['Screen to Body Ratio (calculated)'].mean())

    data['Screen to Body Ratio (calculated)'] = data['Screen to Body Ratio (calculated)'].round(2)



    data['Width'] = data['Width'].apply(for_float)

    data['Width'] = data['Width'].fillna(data['Width'].mean())

    data['Width'] = data['Width'].round(2)



    data['Flash'][data['Flash'].isna() == True] = "Other"



    data['User Replaceable'][data['User Replaceable'].isna() == True] = "Other"



    data['Num_cores'] = data['Processor'].apply(for_string)

    data['Num_cores'][data['Num_cores'].isna() == True] = "Other"





    data['Processor_frequency'] = data['Processor'].apply(find_freq)

    #because there is one entry with 208MHz values, to convert it to GHz

    data['Processor_frequency'][data['Processor_frequency'] > 200] = 0.208

    data['Processor_frequency'] = data['Processor_frequency'].fillna(data['Processor_frequency'].mean())

    data['Processor_frequency'] = data['Processor_frequency'].round(2)



    data['Camera Features'][data['Camera Features'].isna() == True] = "Other"



    #simplifyig Operating System to os_name for simplicity

    data['os_name'] = data['Operating System'].apply(for_string)

    data['os_name'][data['os_name'].isna() == True] = "Other"



    data['Sim1'] = data['SIM 1'].apply(for_string)



    data['SIM Size'][data['SIM Size'].isna() == True] = "Other"



    data['Image Resolution'][data['Image Resolution'].isna() == True] = "Other"



    data['Fingerprint Sensor'][data['Fingerprint Sensor'].isna() == True] = "Other"



    data['Expandable Memory'][data['Expandable Memory'].isna() == True] = "No"



    data['Weight'] = data['Weight'].apply(for_integer)

    data['Weight'] = data['Weight'].fillna(data['Weight'].mean())

    data['Weight'] = data['Weight'].astype(int)



    data['SIM 2'] = data['SIM 2'].apply(for_string)

    data['SIM 2'][data['SIM 2'].isna() == True] = "Other"

    

    return data
train = data_clean_2(train)

test = data_clean_2(test)



# check the number of features and data points in train

print("Number of data points in train: %d" % train.shape[0])

print("Number of features in train: %d" % train.shape[1])



# check the number of features and data points in test

print("Number of data points in test: %d" % test.shape[0])

print("Number of features in test: %d" % test.shape[1])
def data_clean_3(x):

    

    data = x.copy()



    columns_to_remove = ['User Available Storage','SIM Size','Chipset','Processor','Autofocus','Aspect Ratio','Touch Screen',

                        'Bezel-less display','Operating System','SIM 1','USB Connectivity','Other Sensors','Graphics','FM Radio',

                        'NFC','Shooting Modes','Browser','Display Colour' ]



    columns_to_retain = list(set(data.columns)-set(columns_to_remove))

    data = data[columns_to_retain]





    columns_to_remove = [ 'Screen Resolution','User Replaceable','Camera Features',

                        'Thickness', 'Display Type']



    columns_to_retain = list(set(data.columns)-set(columns_to_remove))

    data = data[columns_to_retain]





    columns_to_remove = ['Fingerprint Sensor', 'Flash', 'Rating Count', 'Review Count','Image Resolution','Type','Expandable Memory',\

                        'Colours','Width','Model']

    columns_to_retain = list(set(data.columns)-set(columns_to_remove))

    data = data[columns_to_retain]



    return data
train = data_clean_3(train)

test = data_clean_3(test)



# check the number of features and data points in train

print("Number of data points in train: %d" % train.shape[0])

print("Number of features in train: %d" % train.shape[1])



# check the number of features and data points in test

print("Number of data points in test: %d" % test.shape[0])

print("Number of features in test: %d" % test.shape[1])
# one hot encoding



train_ids = train['PhoneId']

test_ids = test['PhoneId']



cols = list(test.columns)

cols.remove('PhoneId')

cols.insert(0, 'PhoneId')



combined = pd.concat([train.drop('Rating', axis=1)[cols], test[cols]])

print(combined.shape)

print(combined.columns)



combined = pd.get_dummies(combined)

print(combined.shape)

print(combined.columns)



train_new = combined[combined['PhoneId'].isin(train_ids)]

test_new = combined[combined['PhoneId'].isin(test_ids)]
train_new = train_new.merge(train[['PhoneId', 'Rating']], on='PhoneId')
# check the number of features and data points in train

print("Number of data points in train: %d" % train_new.shape[0])

print("Number of features in train: %d" % train_new.shape[1])



# check the number of features and data points in test

print("Number of data points in test: %d" % test_new.shape[0])

print("Number of features in test: %d" % test_new.shape[1])
train_new.head()
print(train_new.mean())
test_new.head()
print(test_new.mean())
y_train = train_new['Rating'].map(lambda x: 0 if x < 3.85 else 1)

y_train.head()
x_train1 = train_new.drop('Rating', axis=1)
x_train1.head()
x_train = x_train1.drop('PhoneId', axis=1)
x_train.head()
x_test = test_new.drop('PhoneId', axis=1)
x_test.head()
y_train = y_train.values

x_train = x_train.values

x_test = x_test.values
type(y_train)

type(x_train)

type(x_test)
class Perceptron:

  

  def __init__ (self):

    self.w = None

    self.b = None

    

  def model(self, x):

    return 1 if (np.dot(self.w, x) >= self.b) else 0

    

  def predict(self, X):

    Y = []

    for x in X:

      result = self.model(x)

      Y.append(result)

    return np.array(Y)

    

  def fit(self, X, Y, epochs = 1, lr = 1):

    

    self.w = np.ones(X.shape[1])

    self.b = 0

    

    accuracy = {}

    max_accuracy = 0

    

    wt_matrix = []

    

    for i in range(epochs):

      for x, y in zip(X, Y):

        y_pred = self.model(x)

        if y == 1 and y_pred == 0:

          self.w = self.w + lr * x

          self.b = self.b - lr * 1

        elif y == 0 and y_pred == 1:

          self.w = self.w - lr * x

          self.b = self.b + lr * 1

          

      wt_matrix.append(self.w)    

          

      accuracy[i] = accuracy_score(self.predict(X), Y)

      if (accuracy[i] > max_accuracy):

        max_accuracy = accuracy[i]

        chkptw = self.w

        chkptb = self.b

        

    self.w = chkptw

    self.b = chkptb

        

    print(max_accuracy)

    

    plt.plot(accuracy.values())

    plt.ylim([0, 1])

    plt.show()

    

    return np.array(wt_matrix)
perceptron = Perceptron()

wt_matrix = perceptron.fit(x_train, y_train, 10000, 1)
Y_test = []

Y_test = perceptron.predict(x_test)

Y_test= 1*Y_test

Y_test
submission = pd.DataFrame({'PhoneId':test_new['PhoneId'], 'Class':Y_test})

submission = submission[['PhoneId', 'Class']]

submission.head()
submission.to_csv("submission.csv", index=False)