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

import seaborn as sns

sns.set()



np.random.seed(0)

warnings.filterwarnings("ignore")

THRESHOLD = 4
# read data from file

#train = pd.read_csv("../input/train.csv") 

#test = pd.read_csv("../input/test.csv")

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
# MP Neuron Template class



class MPNeuron:

    

    def __init__(self):

        self.b = None

        

    def model(self, x):

        return int((sum(x) >= self.b))

    

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

            y_pred = self.predict(X)

            accuracy[b] = accuracy_score(y_pred, Y)

        

        best_b = max(accuracy, key = accuracy.get)

        self.b = best_b

        

        print("Optimal Value of b is", best_b)

        print("Highest accuracy is", accuracy[best_b])
#columns_to_drop = ['PhoneId','Sim1_2G','SIM 2_3G','Sim1_3G','os_name_Blackberry','os_name_Tizen','os_name_KAI','Num_cores_Deca','SIM Slot(s)_Dual SIM, GSM+CDMA','Brand_Lyf','Brand_Karbonn','Brand_Jivi','Brand_Lephone','Brand_LG','Brand_Micromax','Brand_Reliance', 'Brand_Mobiistar', 'Brand_Nubia', 'Brand_Razer', 'Brand_Spice', 'Brand_Ulefone', 'Brand_Yu', 'Brand_VOTO', 'Brand_Do', 'Brand_InFocus', 'Brand_Intex', 'Brand_HTC', 'Brand_Coolpad', 'Brand_Billion', 'Brand_iVooMi', 'Brand_10.or','Brand_Blackberry']

#columns_to_drop += ['Weight', 'Height', 'Screen to Body Ratio (calculated)']

# columns_to_drop = ['PhoneId']



# X_train = train_new.drop(columns_to_drop, axis = 1)

# X_train['Rating'] = train_new['Rating'].map(lambda x: 1 if x >= THRESHOLD else 0)



#ls = X_train.groupby('Rating').mean()

#ls

#with pd.option_context('display.max_rows', None, 'display.max_columns', None):

    #print(ls)
# for i in range(X_train.shape[1] - 1):

#     zero = ls[X_train.columns[i]][0]

#     one = ls[X_train.columns[i]][1]

#     if zero == 0 and one == 0:

#         print(X_train.columns[i], zero, one)

#         columns_to_drop.append(X_train.columns[i])

#     elif one == 0:

#         print(X_train.columns[i], zero, one)

#         columns_to_drop.append(X_train.columns[i])

#     elif zero > one:

#         print(X_train.columns[i], zero, one)

#         columns_to_drop.append(X_train.columns[i])
columns_to_drop = ['PhoneId', 'Sim1_3G', 'os_name_Blackberry', 'os_name_KAI', 'os_name_Tizen', 'Brand_10.or',

'Brand_Billion', 'Brand_Blackberry', 'Brand_Coolpad', 'Brand_Do', 'Brand_Gionee', 'Brand_HTC',

'Brand_InFocus', 'Brand_Infinix', 'Brand_Intex', 'Brand_Jivi', 'Brand_Karbonn', 'Brand_LG', 'Brand_Lava',

'Brand_Lenovo', 'Brand_Lephone', 'Brand_Lyf', 'Brand_Micromax', 'Brand_Mobiistar', 'Brand_Moto', 'Brand_Nokia',

'Brand_Nubia', 'Brand_Panasonic', 'Brand_Razer', 'Brand_Reliance', 'Brand_Sony', 'Brand_Spice', 'Brand_VOTO',

'Brand_Yu', 'Brand_iVooMi', 'SIM Slot(s)_Dual SIM, GSM+GSM', 'SIM 2_2G', 'SIM 2_3G']

X_train = train_new.drop(columns_to_drop + ['Rating'], axis = 1)

Y_train = train_new['Rating']

Y_binarised_train = train_new['Rating'].map(lambda x: 1 if x >= THRESHOLD else 0)



X_test = test_new.drop(columns_to_drop, axis = 1)
#X_train_visual_rating_binary = pd.concat([X_train, Y_binarised_train], axis = 1)



#plt.figure(figsize = (14,8))

#sns.scatterplot(x = 'Screen Size', y = 'Height', hue = 'Rating', data = X_train_visual_rating_binary)
#columns_to_drop += ['Weight', 'Height', 'Screen to Body Ratio (calculated)']



X_binarised_train = train_new.drop(columns_to_drop + ['Rating'], axis = 1)



X_binarised_train['Capacity'] = X_binarised_train['Capacity'].map(lambda x: 1 if x > 2200 else 0)

X_binarised_train['Height'] = X_binarised_train['Height'].map(lambda x: 1 if x > 135 else 0)

X_binarised_train['Pixel Density'] = X_binarised_train['Pixel Density'].map(lambda x: 1 if x > 250 else 0)

X_binarised_train['Screen Size'] = X_binarised_train['Screen Size'].map(lambda x: 1 if x >= 5 else 0)

X_binarised_train['Processor_frequency'] = X_binarised_train['Processor_frequency'].map(lambda x: 1 if x > 1.5 else 0)

X_binarised_train['Resolution'] = X_binarised_train['Resolution'].map(lambda x: 1 if x >= 5 else 0)

X_binarised_train['Internal Memory'] = X_binarised_train['Internal Memory'].map(lambda x: 1 if x >= 30 else 0)

X_binarised_train['Screen to Body Ratio (calculated)'] = X_binarised_train['Screen to Body Ratio (calculated)'].map(lambda x: 1 if x > 65 else 0)

X_binarised_train['RAM'] = X_binarised_train['RAM'].map(lambda x: 1 if (x > 2 or x < 511) else 0)

X_binarised_train['Weight'] = X_binarised_train['Weight'].map(lambda x: 1 if x > 132 else 0)



#X_binarised_train.head()
X = X_binarised_train.values

Y = Y_binarised_train.values



mp_neuron = MPNeuron()

mp_neuron.fit(X, Y)
X_binarised_test = test_new.drop(columns_to_drop, axis = 1)



X_binarised_test['Capacity'] = X_binarised_test['Capacity'].map(lambda x: 1 if x > 2200 else 0)

X_binarised_test['Height'] = X_binarised_test['Height'].map(lambda x: 1 if x > 135 else 0)

X_binarised_test['Pixel Density'] = X_binarised_test['Pixel Density'].map(lambda x: 1 if x > 250 else 0)

X_binarised_test['Screen Size'] = X_binarised_test['Screen Size'].map(lambda x: 1 if x >= 5 else 0)

X_binarised_test['Processor_frequency'] = X_binarised_test['Processor_frequency'].map(lambda x: 1 if x > 1.5 else 0)

X_binarised_test['Resolution'] = X_binarised_test['Resolution'].map(lambda x: 1 if x >= 5 else 0)

X_binarised_test['Internal Memory'] = X_binarised_test['Internal Memory'].map(lambda x: 1 if x >= 30 else 0)

X_binarised_test['Screen to Body Ratio (calculated)'] = X_binarised_test['Screen to Body Ratio (calculated)'].map(lambda x: 1 if x > 65 else 0)

X_binarised_test['RAM'] = X_binarised_test['RAM'].map(lambda x: 1 if (x > 2 or x < 511) else 0)

X_binarised_test['Weight'] = X_binarised_test['Weight'].map(lambda x: 1 if x > 132 else 0)



#X_binarised_test.head()
X = X_binarised_test.values



Y_test_predicted = mp_neuron.predict(X)
submission = pd.DataFrame({'PhoneId':test_new['PhoneId'], 'Class':Y_test_predicted})

submission = submission[['PhoneId', 'Class']]

submission.head()
submission.to_csv("submission.csv", index=False)