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
df=train_new['Rating'].apply(lambda x:1 if x>=4.0 else 0)

train_new['Rating']=df

columns_remove=['os_name_Blackberry','os_name_KAI','os_name_Nokia','Brand_Motorola','Brand_Mobiistar',

 'os_name_Tizen','SIM Slot(s)_Dual SIM, GSM+CDMA','Sim1_3G','SIM 2_3G','SIM 2_Other','Brand_Intex',

 'SIM Slot(s)_Dual SIM, GSM+GSM','Num_cores_312','Num_cores_Deca','Num_cores_Tru-Octa','Brand_Panasonic',

 'Brand_10.or','Brand_Billion','Brand_Comio','Brand_Coolpad','Brand_Do','Brand_Jivi',

 'Brand_Karbonn','Brand_LeEco','Brand_Lephone','Brand_Lyf','os_name_Other','Brand_Yu',

 'Brand_Nubia','Brand_Razer','Brand_Reliance','Brand_Spice','Brand_Ulefone','Brand_VOTO','Brand_iVooMi']

avail_columns=['PhoneId','Height', 'Resolution', 'Screen Size' ,'Processor_frequency', 'RAM',

 'Capacity', 'Weight' ,'Pixel Density' ,'Internal Memory',

 'Screen to Body Ratio (calculated)' ,'os_name_Android','os_name_Blackberry','os_name_KAI',

 'os_name_Nokia','os_name_Other','os_name_Tizen','os_name_iOS',

 'SIM Slot(s)_Dual SIM, GSM+CDMA','SIM Slot(s)_Dual SIM, GSM+GSM',

 'SIM Slot(s)_Dual SIM, GSM+GSM, Dual VoLTE','SIM Slot(s)_Single SIM, GSM',

 'Num_cores_312','Num_cores_Deca','Num_cores_Dual','Num_cores_Hexa',

 'Num_cores_Octa','Num_cores_Other','Num_cores_Quad','Num_cores_Tru-Octa',

 'Sim1_2G','Sim1_3G','Sim1_4G','SIM 2_2G','SIM 2_3G','SIM 2_4G',

 'SIM 2_Other','Brand_10.or','Brand_Apple','Brand_Asus','Brand_Billion',

 'Brand_Blackberry','Brand_Comio','Brand_Coolpad','Brand_Do',

 'Brand_Gionee','Brand_Google','Brand_HTC','Brand_Honor','Brand_Huawei',

 'Brand_InFocus','Brand_Infinix','Brand_Intex','Brand_Itel','Brand_Jivi',

 'Brand_Karbonn','Brand_LG','Brand_Lava','Brand_LeEco','Brand_Lenovo',

 'Brand_Lephone','Brand_Lyf','Brand_Meizu','Brand_Micromax',

 'Brand_Mobiistar','Brand_Moto','Brand_Motorola','Brand_Nokia',

 'Brand_Nubia','Brand_OPPO','Brand_OnePlus','Brand_Oppo','Brand_Panasonic',

 'Brand_Razer','Brand_Realme','Brand_Reliance','Brand_Samsung',

 'Brand_Sony','Brand_Spice','Brand_Tecno','Brand_Ulefone','Brand_VOTO',

 'Brand_Vivo','Brand_Xiaomi','Brand_Xiaomi Poco','Brand_Yu','Brand_iVooMi',

 'Rating']
for i in columns_remove:

    avail_columns.remove(i)

avail_columns.remove('PhoneId')
train_new=train_new[avail_columns]

avail_columns.remove('Rating')

X_test=test_new[avail_columns]
train_new['Height']=train_new['Height'].apply(lambda x: 0 if(x>156.5 and x<=157.9) else 1)

X_test['Height']=X_test['Height'].apply(lambda x: 0 if(x>156.5 and x<=157.9) else 1)
train_new['Processor_frequency']=train_new['Processor_frequency'].apply(lambda x: 1 if x>=1.66 else 0)

X_test['Processor_frequency']=X_test['Processor_frequency'].apply(lambda x: 1 if x>=1.66 else 0)
for i in avail_columns:

    print(i,train_new[train_new['Rating']==1][i].mean()*238)
train_new.shape
train_new['Resolution']=train_new['Resolution'].apply(lambda x: 1 if(x>=9) else 0)

X_test['Resolution']=X_test['Resolution'].apply(lambda x: 1 if(x>=9) else 0)
train_new['RAM']=train_new['RAM'].apply(lambda x: 1 if(x>=2 and x<=128) else 0)

X_test['RAM']=X_test['RAM'].apply(lambda x: 1 if(x>=2 and x<=128) else 0)
train_new['Capacity']=train_new['Capacity'].apply(lambda x: 1)

X_test['Capacity']=X_test['Capacity'].apply(lambda x: 1) 
train_new['Weight']=train_new['Weight'].apply(lambda x: 1 if x<=160 else 0)

X_test['Weight']=X_test['Weight'].apply(lambda x: 1 if x<=160 else 0)
train_new['Screen to Body Ratio (calculated)']=train_new['Screen to Body Ratio (calculated)'].apply(lambda x:1 if(x>=79) else 0)

X_test['Screen to Body Ratio (calculated)']=X_test['Screen to Body Ratio (calculated)'].apply(lambda x:1 if(x>=79) else 0)
train_new['Pixel Density']=train_new['Pixel Density'].apply(lambda x:1 if(x>=234) else 0)

X_test['Pixel Density']=X_test['Pixel Density'].apply(lambda x:1 if(x>=234) else 0)
train_new['Screen Size']=train_new['Screen Size'].apply(lambda x:1 if(x>=5.5) else 0)

X_test['Screen Size']=X_test['Screen Size'].apply(lambda x:1 if(x>=5.5) else 0)
train_new['Internal Memory']=train_new['Internal Memory'].apply(lambda x:1 if(x>=32) else 0)

X_test['Internal Memory']=X_test['Internal Memory'].apply(lambda x:1 if(x>=32) else 0)
X_binarised_train=train_new.drop('Rating',axis=1)

Y_train=train_new['Rating']
X_binarised_train=X_binarised_train.values

Y_train=Y_train.values

X_test=X_test.values
class MPNeuron:

  

  def __init__(self):

    self.b = None

    

  def model(self, x):

    if(sum(x) >= self.b):

        return 1

    else:

        return 0

  

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
mp_neuron = MPNeuron()

mp_neuron.fit(X_binarised_train, Y_train)
Y_test_pred = mp_neuron.predict(X_test)
submission = pd.DataFrame({'PhoneId':test_new['PhoneId'], 'Class':Y_test_pred})

submission = submission[['PhoneId', 'Class']]

submission.head()
submission.to_csv("submission.csv", index=False)