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
#train_new

#test_new

class MPNeuron:

  

  def __init__(self):

    self.b = None

    

  def model(self, x):

    return(sum(x) >= self.b)

  

  def predict(self, X):

    Y = []

    for x in X:

      result = self.model(x)

      Y.append(result)

    return np.array(Y)

  

  def fit(self, X, Y):

    accuracy = {}

    

    for b in range(X.shape[1] + 1):

      self.b = b

      Y_pred = self.predict(X)

      accuracy[b] = accuracy_score(Y_pred, Y)

      

    best_b = max(accuracy, key = accuracy.get)

    self.b = best_b

    

    print('Optimal value of b is', best_b)

    print('Highest accuracy is', accuracy[best_b])
def rating(x):

    return 1 if (x >= 4.0 ) else 0

train_new['Rating'] = train_new['Rating'].apply(rating,)

def RAM(x):

    if x >= 2 : #x >= 6

        return 1

    else:

        return 0



train_new['RAM'] = train_new['RAM'].apply(RAM)

test_new['RAM'] = test_new['RAM'].apply(RAM)

def Height(x):

    if x >= 136.5: #x >= 150

        return 1

    else:

        return 0

    

train_new['Height'] = train_new['Height'].apply(Height)

test_new['Height'] = test_new['Height'].apply(Height)

def Processor_frequency(x):

    if x >= 1.4: #x > 1.5

        return 1

    else:

        return 0

    

train_new['Processor_frequency'] = train_new['Processor_frequency'].apply(Processor_frequency)

test_new['Processor_frequency'] = test_new['Processor_frequency'].apply(Processor_frequency)

def Capacity(x):

    if x >= 2350: #x >= 4100

        return 1

    else:

        return 0 

    

train_new['Capacity'] = train_new['Capacity'].apply(Capacity)

test_new['Capacity'] = test_new['Capacity'].apply(Capacity)

def Pixel_Density(x):

    if x >= 234: #x >= 271

        return 1

    else:

        return 0



train_new['Pixel Density'] = train_new['Pixel Density'].apply(Pixel_Density)

test_new['Pixel Density'] = test_new['Pixel Density'].apply(Pixel_Density)

def Resolution(x):

    if x >= 5: # x > 8 

        return 1

    else:

        return 0

    

train_new['Resolution'] = train_new['Resolution'].apply(Resolution)

test_new['Resolution'] = test_new['Resolution'].apply(Resolution)

def Weight(x):

    if x >= 129: # x > 164

        return 1

    else:

        return 0

    

train_new['Weight'] = train_new['Weight'].apply(Weight)

test_new['Weight'] = test_new['Weight'].apply(Weight)

def Screen_to_Body_Ratio(x):

    if x >= 65.23 : #x > 65.47

        return 1

    else:

        return 0

    

train_new['Screen to Body Ratio (calculated)'] = train_new['Screen to Body Ratio (calculated)'].apply(Screen_to_Body_Ratio)

test_new['Screen to Body Ratio (calculated)'] = test_new['Screen to Body Ratio (calculated)'].apply(Screen_to_Body_Ratio)

def Screen_Size(x):

    if x >= 4.7: #x > 5.5

        return 1

    else:

        return 0

    

train_new['Screen Size'] = train_new['Screen Size'].apply(Screen_Size)

test_new['Screen Size'] = test_new['Screen Size'].apply(Screen_Size)

def Internal_memory(x):

    if x >= 16: #x >= 32

        return 1

    else:

        return 0

    

train_new['Internal Memory'] = train_new['Internal Memory'].apply(Internal_memory)

test_new['Internal Memory'] = test_new['Internal Memory'].apply(Internal_memory)
train_new.head()
#X = train_new[['Capacity','Height','Pixel Density', 'Processor_frequency','Resolution','RAM', 'Screen Size','Screen to Body Ratio (calculated)','Internal Memory','Weight' ]]

X = train_new.drop(['PhoneId','Rating'], axis=1)

#X_bin = X.apply(pd.cut, bins=2, labels=[1,0])

y = train_new['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, )
X_train = X_train.values

X_test = X_test.values
mpneuron =MPNeuron()

mpneuron.fit(X_train, y_train)

predict =mpneuron.predict(X_test)

accuracy = accuracy_score(predict, y_test)

print(accuracy)
test = test_new.drop('PhoneId', axis=1)
#test = test_new.apply(pd.cut, bins=2, labels=[1,0])
test = test.values
predict =mpneuron.predict(test)
predict
predict = pd.DataFrame(predict, columns=['Rating'], dtype='int')

predict
submission = pd.DataFrame({'PhoneId':test_new['PhoneId'], 'Class':predict['Rating']})

submission = submission[['PhoneId', 'Class']]

submission.head()
submission
submission.to_csv("submission.csv", index=False)
"""

WRITE YOUR MODELLING CODE HERE

"""