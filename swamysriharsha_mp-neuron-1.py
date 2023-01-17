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
# read data from file

train = pd.read_csv("../input/train.csv") 

test = pd.read_csv("../input/test.csv")



# check the number of features and data points in train

print("Number of data points in train: %d" % train.shape[0])

print("Number of features in train: %d" % train.shape[1])



# check the number of features and data points in test

print("Number of data points in test: %d" % test.shape[0])

print("Number of features in test: %d" % test.shape[1])
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
test_new.head()
"""

WRITE YOUR MODELLING CODE HERE

"""
len(train_new[train_new['Rating'] >=4])
len(train_new[train_new['Rating'] < 4])
train_new['Rating'] = train_new['Rating'].apply(lambda x: 1 if x>=4 else 0)
train_new['Rating'].value_counts()
import matplotlib.pyplot as plt

import seaborn as sns
train_new['RAM'][train_new['Rating']==0].value_counts()
train_new['RAM'][train_new['Rating']==1].value_counts()
ax = sns.scatterplot(x="Capacity", y="Pixel Density", hue="Rating", data=train_new)
train_new = train_new[train_new['Capacity']<6000]
train_new = train_new[train_new['Screen to Body Ratio (calculated)']>50]
train_new = train_new[train_new['Weight']>=120]
train_new = train_new[train_new['Internal Memory']<=256]
train_new = train_new[train_new['Height']<=170]
train_new = train_new[train_new['RAM']<100]
train_new.shape
train_new.head()
ax = sns.scatterplot(x="Internal Memory", y="Pixel Density", hue="Rating", data=train_new)
train_new['RAM'] = train_new['RAM'].apply(lambda x: 1 if x>=4 else 0)

test_new['RAM'] = test_new['RAM'].apply(lambda x: 1 if x>=4 else 0)
train_new['Internal Memory'][train_new['Rating']==0].value_counts()
train_new['Internal Memory'][train_new['Rating']==1].value_counts()
train_new['Internal Memory'] = train_new['Internal Memory'].apply(lambda x: 1 if x>=64 else 0)

test_new['Internal Memory'] = test_new['Internal Memory'].apply(lambda x: 1 if x>=64 else 0)
ax = sns.scatterplot(x="Resolution", y="Pixel Density", hue="Rating", data=train_new)
train_new['Resolution'][train_new['Rating']==0].value_counts()
train_new['Resolution'][train_new['Rating']==1].value_counts()
train_new['Resolution'] = train_new['Resolution'].apply(lambda x: 1 if x>20 else 0)

test_new['Resolution'] = test_new['Resolution'].apply(lambda x: 1 if x>20 else 0)
ax = sns.scatterplot(x="Capacity", y="Pixel Density", hue="Rating", data=train_new)
ax = sns.scatterplot(x="Processor_frequency", y="Pixel Density", hue="Rating", data=train_new)
train_new['Rating'][train_new['Processor_frequency']>2].value_counts()
train_new['Processor_frequency'] = train_new['Processor_frequency'].apply(lambda x: 1 if x>2 else 0)

test_new['Processor_frequency'] = test_new['Processor_frequency'].apply(lambda x: 1 if x>2 else 0)
ax = sns.scatterplot(x="Screen Size", y="Pixel Density", hue="Rating", data=train_new)
train_new['Rating'][train_new['Screen Size']>5.5].value_counts()
train_new['Screen Size'] = train_new['Screen Size'].apply(lambda x: 1 if x>5.5 else 0)

test_new['Screen Size'] = test_new['Screen Size'].apply(lambda x: 1 if x>5.5 else 0)
ax = sns.scatterplot(x="Screen to Body Ratio (calculated)", y="Pixel Density", hue="Rating", data=train_new)
train_new['Rating'][train_new['Screen to Body Ratio (calculated)']<75].value_counts()
ax = sns.scatterplot(x="Height", y="Pixel Density", hue="Rating", data=train_new)
train_new['Rating'][train_new['Height']<136].value_counts()
train_new['Height'] = train_new['Height'].apply(lambda x: 1 if x>=136 else 0)

test_new['Height'] = test_new['Height'].apply(lambda x: 1 if x>=136 else 0)
ax = sns.scatterplot(x="Capacity", y="Pixel Density", hue="Rating", data=train_new)
train_new['Capacity'] = train_new['Capacity'].apply(lambda x: 1 if x>train_new['Capacity'].mean() else 0)

test_new['Capacity'] = test_new['Capacity'].apply(lambda x: 1 if x>train_new['Capacity'].mean() else 0)
train_new['Weight'] = train_new['Weight'].apply(lambda x: 1 if x<train_new['Weight'].mean() else 0)

test_new['Weight'] = test_new['Weight'].apply(lambda x: 1 if x<train_new['Weight'].mean() else 0)
ax = sns.scatterplot(x="Screen to Body Ratio (calculated)", y="Pixel Density", hue="Rating", data=train_new)
#train_new['Rating'][train_new['Screen to Body Ratio (calculated)']>78].value_counts()
X = train_new.drop('Rating',axis=1)

y = train_new['Rating']
print(X.shape, y.shape)
test_new.shape
names = ['Pixel Density', 'Screen to Body Ratio (calculated)']
X[names].head()
test_new[names].head()
plt.plot(X[names].T, '.')

plt.xticks(rotation='vertical')

plt.show()
X[names] = X[names].apply(pd.cut, bins=2, labels=[1,0])
test_new[names] = test_new[names].apply(pd.cut, bins=2, labels=[1,0])
plt.plot(X[names].T, '.')

plt.xticks(rotation='vertical')

plt.show()
X.head()
test_new.head()
plt.plot(X.T, '.')

plt.xticks(rotation='vertical')

plt.show()
plt.plot(test_new.T, '.')

plt.xticks(rotation='vertical')

plt.show()
'''from imblearn.combine import SMOTETomek



smt = SMOTETomek(ratio='auto')

X_smt, y_smt = smt.fit_sample(X, y)'''
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.1, stratify=y, random_state=42)
#model predict fit



class MPNeuron:

  

  def __init__(self):

    self.b = None

    

  def model(self, x):

    return (1 if sum(x) >= self.b else 0)

    

  def predict(self, X): #total training data as an input

    y_pred = []

    for x in X: #for each data point

      res = self.model(x)

      y_pred.append(res)

    return np.array(y_pred)

  

  def fit(self, X, Y): #training i.e; finidng best_b in case of MP Neuron

    acc = {}

    for b in range(X.shape[1]+1):

      self.b = b

      y_predb = self.predict(X)

      acc[b] = accuracy_score(y_predb, Y)

      print(b, acc[b])

    best_b = max(acc, key=acc.get)

    self.b = best_b

    print("final result", best_b, acc[best_b])

      

    

    
mp_neuron = MPNeuron()
X_train = X_train.values

Y_train = Y_train.values
X_test = X_test.values

Y_test = Y_test.values
print(type(X), type(y))

print(type(X_train), type(Y_train))

print(type(X_test), type(Y_test))
print(X.shape, y.shape)

print(X_train.shape, Y_train.shape)

print(X_test.shape, Y_test.shape)
X_train_new = X_train[:,1:]

X_test_new = X_test[:,1:]
print(X_train.shape, X_train_new.shape)

print(X_test.shape, X_test_new.shape)
mp_neuron.fit(X_train_new, Y_train)
y_ = mp_neuron.predict(X_test_new)

print(accuracy_score(y_, Y_test))
X_val = test_new.values
X_val = X_val[:,1:]
type(X_val)
X_val.shape
y_pred = mp_neuron.predict(X_val)
y_pred.shape
submission = pd.DataFrame({'PhoneId':test_new['PhoneId'], 'Class':y_pred})

submission = submission[['PhoneId', 'Class']]

submission.head()
submission.to_csv("submission.csv", index=False)