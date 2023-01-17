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
# submission = pd.DataFrame({'PhoneId':test_new['PhoneId'], 'Class':[0]*test_new.shape[0]})

# submission = submission[['PhoneId', 'Class']]

# submission.head()
# submission.to_csv("submission.csv", index=False)
# Scatter Plots of each input variable against Rating to determine relationships and arrive at the best direction & split position for binarization



for i in range(1,len(train_new.columns)):

  print(train_new.columns[i])

  plt.scatter(train_new[train_new.columns[i]], train_new[train_new.columns[87]])

  plt.show()
# MP Neuron Class (Reused from Padhai course, but edited to return 0 or 1 instead of boolean value; Changed to handle single input feature)



class MPNeuron_trial:



  def __init__(self):

    self.b = None

    

  def model(self, x):

    if x >= self.b:

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

    

    for b in range(2):

      self.b = b

      Y_pred = self.predict(X)

      accuracy[b] = accuracy_score(Y_pred, Y)

      

    best_b = max(accuracy, key = accuracy.get)

    self.b = best_b

    

    print('Optimal value of b is', best_b)

    print('Highest accuracy is', accuracy[best_b])
# MP Neuron trial code - Instantiate



mp_neuron_trial = MPNeuron_trial()
# Binarization - Train Rating

Train_Rating_Binarized = train_new['Rating'].map(lambda x: 0 if x < THRESHOLD else 1)



binarised_train = train_new.drop('Rating', axis=1)



# Convert Binarized Rating Pandas Series to Numpy Array



Train_Rating_Binarized_nparray = np.asarray(Train_Rating_Binarized)
# For value_counts by dummy variables



for_value_counts = train_new



for_value_counts['Rating'] = for_value_counts['Rating'].map(lambda x: 0 if x < THRESHOLD else 1)
for_value_counts.groupby('Rating')['Weight'].describe()
for_value_counts.groupby('Rating')['RAM'].describe()
for_value_counts.groupby('Rating')['Screen to Body Ratio (calculated)'].describe()
for_value_counts.groupby('Rating')['Pixel Density'].describe()
for_value_counts.groupby('Rating')['Processor_frequency'].describe()
for_value_counts.groupby('Rating')['Screen Size'].describe()
for_value_counts.groupby('Rating')['Capacity'].describe()
for_value_counts.groupby('Rating')['Height'].describe()
for_value_counts.groupby('Rating')['Internal Memory'].describe()
for_value_counts.groupby('Rating')['Resolution'].describe()
for_value_counts.groupby('Num_cores_312')['Rating'].value_counts()
for_value_counts.groupby('SIM Slot(s)_Dual SIM, GSM+CDMA')['Rating'].value_counts()
for_value_counts.groupby('Sim1_2G')['Rating'].value_counts()
for_value_counts.groupby('Sim1_3G')['Rating'].value_counts()
for_value_counts.groupby('Sim1_4G')['Rating'].value_counts()
for_value_counts.groupby('os_name_Blackberry')['Rating'].value_counts()
for_value_counts.groupby('os_name_KAI')['Rating'].value_counts()
for_value_counts.groupby('os_name_Nokia')['Rating'].value_counts()
for_value_counts.groupby('os_name_Other')['Rating'].value_counts()
for_value_counts.groupby('os_name_Tizen')['Rating'].value_counts()
for_value_counts.groupby('Num_cores_Deca')['Rating'].value_counts()
for_value_counts.groupby('Num_cores_Other')['Rating'].value_counts()
for_value_counts.groupby('Num_cores_Tru-Octa')['Rating'].value_counts()
for_value_counts.groupby('SIM 2_3G')['Rating'].value_counts()
for_value_counts.groupby('Brand_10.or')['Rating'].value_counts()
for_value_counts.groupby('Brand_Asus')['Rating'].value_counts()
for_value_counts.groupby('Brand_Blackberry')['Rating'].value_counts()
for_value_counts.groupby('Brand_Comio')['Rating'].value_counts()
for_value_counts.groupby('Brand_Coolpad')['Rating'].value_counts()
for_value_counts.groupby('Brand_Gionee')['Rating'].value_counts()
for_value_counts.groupby('Brand_HTC')['Rating'].value_counts()
for_value_counts.groupby('Brand_Honor')['Rating'].value_counts()
for_value_counts.groupby('Brand_InFocus')['Rating'].value_counts()
for_value_counts.groupby('Brand_Infinix')['Rating'].value_counts()
for_value_counts.groupby('Brand_Intex')['Rating'].value_counts()
for_value_counts.groupby('Brand_Jivi')['Rating'].value_counts()
for_value_counts.groupby('Brand_Karbonn')['Rating'].value_counts()
for_value_counts.groupby('Brand_Lava')['Rating'].value_counts()
for_value_counts.groupby('Brand_LeEco')['Rating'].value_counts()
for_value_counts.groupby('Brand_Lenovo')['Rating'].value_counts()
for_value_counts.groupby('Brand_Lephone')['Rating'].value_counts()
for_value_counts.groupby('Brand_Lyf')['Rating'].value_counts()
for_value_counts.groupby('Brand_Meizu')['Rating'].value_counts()
for_value_counts.groupby('Brand_Micromax')['Rating'].value_counts()
for_value_counts.groupby('Brand_Mobiistar')['Rating'].value_counts()
for_value_counts.groupby('Brand_Moto')['Rating'].value_counts()
for_value_counts.groupby('Brand_Motorola')['Rating'].value_counts()
for_value_counts.groupby('Brand_Nubia')['Rating'].value_counts()
for_value_counts.groupby('Brand_Oppo')['Rating'].value_counts()
for_value_counts.groupby('Brand_Panasonic')['Rating'].value_counts()
for_value_counts.groupby('Brand_Razer')['Rating'].value_counts()
for_value_counts.groupby('Brand_Reliance')['Rating'].value_counts()
for_value_counts.groupby('Brand_VOTO')['Rating'].value_counts()
for_value_counts.groupby('Brand_Yu')['Rating'].value_counts()
for_value_counts.groupby('Brand_iVooMi')['Rating'].value_counts()
for_value_counts.groupby('SIM Slot(s)_Dual SIM, GSM+GSM')['Rating'].value_counts()
for_value_counts.groupby('SIM Slot(s)_Dual SIM, GSM+GSM, Dual VoLTE')['Rating'].value_counts()
for_value_counts.groupby('SIM Slot(s)_Single SIM, GSM')['Rating'].value_counts()
for_value_counts.groupby('os_name_Android')['Rating'].value_counts()
for_value_counts.groupby('os_name_iOS')['Rating'].value_counts()
for_value_counts.groupby('Num_cores_Dual')['Rating'].value_counts()
for_value_counts.groupby('Num_cores_Hexa')['Rating'].value_counts()
for_value_counts.groupby('Num_cores_Octa')['Rating'].value_counts()
for_value_counts.groupby('Num_cores_Quad')['Rating'].value_counts()
for_value_counts.groupby('SIM 2_2G')['Rating'].value_counts()
for_value_counts.groupby('SIM 2_4G')['Rating'].value_counts()
for_value_counts.groupby('SIM 2_Other')['Rating'].value_counts()
for_value_counts.groupby('Brand_Apple')['Rating'].value_counts()
for_value_counts.groupby('Brand_Billion')['Rating'].value_counts()
for_value_counts.groupby('Brand_Do')['Rating'].value_counts()
for_value_counts.groupby('Brand_Google')['Rating'].value_counts()
for_value_counts.groupby('Brand_Huawei')['Rating'].value_counts()
for_value_counts.groupby('Brand_Itel')['Rating'].value_counts()
for_value_counts.groupby('Brand_LG')['Rating'].value_counts()
for_value_counts.groupby('Brand_Nokia')['Rating'].value_counts()
for_value_counts.groupby('Brand_OPPO')['Rating'].value_counts()
for_value_counts.groupby('Brand_OnePlus')['Rating'].value_counts()
for_value_counts.groupby('Brand_Realme')['Rating'].value_counts()
for_value_counts.groupby('Brand_Samsung')['Rating'].value_counts()
for_value_counts.groupby('Brand_Sony')['Rating'].value_counts()
for_value_counts.groupby('Brand_Spice')['Rating'].value_counts()
for_value_counts.groupby('Brand_Tecno')['Rating'].value_counts()
for_value_counts.groupby('Brand_Ulefone')['Rating'].value_counts()
for_value_counts.groupby('Brand_Vivo')['Rating'].value_counts()
for_value_counts.groupby('Brand_Xiaomi')['Rating'].value_counts()
for_value_counts.groupby('Brand_Xiaomi Poco')['Rating'].value_counts()
# Explore Cut values for Weight



for wei in range(60,80,1):

    Weight_In = binarised_train['Weight'].map(lambda x: 0 if x < wei else 1)

    print('Weight Value ', wei)

    mp_neuron_trial.fit(Weight_In, Train_Rating_Binarized_nparray)
Y_train_pred_trial = mp_neuron_trial.predict(Weight_In)



confusion_matrix(Train_Rating_Binarized_nparray, Y_train_pred_trial, sample_weight=None)
# Explore Cut values for Capacity



for cap in range(0, 14000, 500):

   Capacity_In = binarised_train['Capacity'].map(lambda x: 0 if x > cap else 1)

   print('Capacity Value ', cap)

   mp_neuron_trial.fit(Capacity_In, Train_Rating_Binarized_nparray)
# Explore Cut values for RAM



for ram in range(250,350,5):

    RAM_In = binarised_train['RAM'].map(lambda x: 0 if x > ram else 1)

    print('RAM Value ', ram)

    mp_neuron_trial.fit(RAM_In, Train_Rating_Binarized_nparray)
Y_train_pred_trial = mp_neuron_trial.predict(RAM_In)



confusion_matrix(Train_Rating_Binarized_nparray, Y_train_pred_trial, sample_weight=None)
# Explore Cut values for Height



for hei in range(110,180,5):

    Height_In = binarised_train['Height'].map(lambda x: 0 if x > hei else 1)

    print('Height Value ', hei)

    mp_neuron_trial.fit(Height_In, Train_Rating_Binarized_nparray)
# Explore Cut values for Screen to Body Ratio (calculated)



for sbr in range(20,90,1):

    SBR_In = binarised_train['Screen to Body Ratio (calculated)'].map(lambda x: 0 if x < sbr else 1)

    print('SBR Value ', sbr)

    mp_neuron_trial.fit(SBR_In, Train_Rating_Binarized_nparray)
Y_train_pred_trial = mp_neuron_trial.predict(SBR_In)



confusion_matrix(Train_Rating_Binarized_nparray, Y_train_pred_trial, sample_weight=None)
# Explore Cut values for Pixel Density



for pix in range(240,270,1):

    Pixel_In = binarised_train['Pixel Density'].map(lambda x: 0 if x < pix else 1)

    print('Pixel Density Value ', pix)

    mp_neuron_trial.fit(Pixel_In, Train_Rating_Binarized_nparray)
Y_train_pred_trial = mp_neuron_trial.predict(Pixel_In)



confusion_matrix(Train_Rating_Binarized_nparray, Y_train_pred_trial, sample_weight=None)
# Explore Cut values for Internal Memory



for im in range(0,550,10):

    IM_In = binarised_train['Internal Memory'].map(lambda x: 0 if x > im else 1)

    print('Internal Memory Value ', im)

    mp_neuron_trial.fit(IM_In, Train_Rating_Binarized_nparray)
# Explore Cut values for Processor Frequency



for pf in range(100,150,5):

    PF_In = binarised_train['Processor_frequency'].map(lambda x: 0 if x < (pf/100) else 1)

    print('Processor Frequency Value ', pf)

    mp_neuron_trial.fit(PF_In, Train_Rating_Binarized_nparray)
Y_train_pred_trial = mp_neuron_trial.predict(PF_In)



confusion_matrix(Train_Rating_Binarized_nparray, Y_train_pred_trial, sample_weight=None)
# Explore Cut values for Screen Size



for ss in range(20,70,1):

    Screen_In = binarised_train['Screen Size'].map(lambda x: 0 if x < (ss/10) else 1)

    print('Screen Size Value ', ss)

    mp_neuron_trial.fit(Screen_In, Train_Rating_Binarized_nparray)
Y_train_pred_trial = mp_neuron_trial.predict(Screen_In)



confusion_matrix(Train_Rating_Binarized_nparray, Y_train_pred_trial, sample_weight=None)
# Explore Cut values for Resolution



for res in range(0,25,1):

    Resolution_In = binarised_train['Resolution'].map(lambda x: 0 if x > res else 1)

    print('Resolution Value ', res)

    mp_neuron_trial.fit(Resolution_In, Train_Rating_Binarized_nparray)
for val in range(0,11,5):

    Test_In = binarised_train['SIM Slot(s)_Dual SIM, GSM+GSM'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)
for val in range(0,11,5):

    Test_In = binarised_train['SIM Slot(s)_Dual SIM, GSM+GSM'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)
for val in range(0,11,5):

    Test_In = binarised_train['SIM Slot(s)_Dual SIM, GSM+GSM, Dual VoLTE'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)
for val in range(0,11,5):

    Test_In = binarised_train['SIM Slot(s)_Single SIM, GSM'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)
for val in range(0,11,5):

    Test_In = binarised_train['os_name_Android'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)
for val in range(0,11,5):

    Test_In = binarised_train['os_name_iOS'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)
for val in range(0,11,5):

    Test_In = binarised_train['Num_cores_Dual'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)
for val in range(0,11,5):

    Test_In = binarised_train['Num_cores_Hexa'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)
for val in range(0,11,5):

    Test_In = binarised_train['Num_cores_Octa'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)
for val in range(0,11,5):

    Test_In = binarised_train['Num_cores_Quad'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)
for val in range(0,11,5):

    Test_In = binarised_train['SIM 2_2G'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)  
for val in range(0,11,5):

    Test_In = binarised_train['SIM 2_4G'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)   
for val in range(0,11,5):

    Test_In = binarised_train['SIM 2_Other'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)  
for val in range(0,11,5):

    Test_In = binarised_train['Brand_Apple'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray) 
for val in range(0,11,5):

    Test_In = binarised_train['Brand_Billion'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray) 
for val in range(0,11,5):

    Test_In = binarised_train['Brand_Do'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)
for val in range(0,11,5):

    Test_In = binarised_train['Brand_Google'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)
for val in range(0,11,5):

    Test_In = binarised_train['Brand_Huawei'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)
for val in range(0,11,5):

    Test_In = binarised_train['Brand_Itel'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)
for val in range(0,11,5):

    Test_In = binarised_train['Brand_LG'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)
for val in range(0,11,5):

    Test_In = binarised_train['Brand_Nokia'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)
for val in range(0,11,5):

    Test_In = binarised_train['Brand_OPPO'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)
for val in range(0,11,5):

    Test_In = binarised_train['Brand_OnePlus'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)
for val in range(0,11,5):

    Test_In = binarised_train['Brand_Realme'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)
for val in range(0,11,5):

    Test_In = binarised_train['Brand_Samsung'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)
for val in range(0,11,5):

    Test_In = binarised_train['Brand_Sony'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)
for val in range(0,11,5):

    Test_In = binarised_train['Brand_Spice'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)
for val in range(0,11,5):

    Test_In = binarised_train['Brand_Tecno'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)
for val in range(0,11,5):

    Test_In = binarised_train['Brand_Ulefone'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)
for val in range(0,11,5):

    Test_In = binarised_train['Brand_Vivo'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)
for val in range(0,11,5):

    Test_In = binarised_train['Brand_Xiaomi'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)
for val in range(0,11,5):

    Test_In = binarised_train['Brand_Xiaomi Poco'].map(lambda x: 0 if x < (val/10) else 1)

    print('Value ', val/10)

    mp_neuron_trial.fit(Test_In, Train_Rating_Binarized_nparray)
plt.plot(train_new.T, '*')

plt.xticks(rotation = 'vertical')

plt.show()
plt.plot(test_new.T, '*')

plt.xticks(rotation = 'vertical')

plt.show()
train_new.groupby('Num_cores_312')['Rating'].value_counts()
binarised_test = test_new
#  def data_clean_4(x):

#    

#    data = x.copy()

#      

#    columns_to_remove = ['Capacity', 'Height', 'Internal Memory', 'Resolution']

#    columns_to_retain = list(set(data.columns)-set(columns_to_remove))

#    data = data[columns_to_retain]

#

#    return data  
#binarised_train = data_clean_4(binarised_train)

#binarised_test = data_clean_4(binarised_test)
def data_clean_5(x):

    

    data = x.copy()

      

    columns_to_remove = ['os_name_Android', 'os_name_iOS', 'Num_cores_Dual', 'Num_cores_Hexa', 'Num_cores_Octa',\

                         'Num_cores_Quad', 'SIM 2_2G', 'SIM 2_4G', 'SIM 2_Other', 'Brand_Apple', 'Brand_Billion',\

                         'Brand_Do', 'Brand_Google', 'Brand_Huawei', 'Brand_Itel', 'Brand_Nokia', 'Brand_Comio',\

                         'Brand_OPPO', 'Brand_OnePlus', 'Brand_Realme', 'Brand_Samsung', 'Brand_Spice',\

                         'Brand_Tecno', 'Brand_Ulefone', 'Brand_Vivo', 'Brand_Xiaomi', 'Brand_Xiaomi Poco',\

                        'SIM Slot(s)_Dual SIM, GSM+GSM', 'SIM Slot(s)_Dual SIM, GSM+GSM, Dual VoLTE',\

                        'SIM Slot(s)_Single SIM, GSM']



    columns_to_retain = list(set(data.columns)-set(columns_to_remove))

    data = data[columns_to_retain]

    

    columns_to_remove = ['Weight', 'Capacity', 'RAM', 'Height','Screen to Body Ratio (calculated)',\

                         'Pixel Density', 'Processor_frequency', 'Internal Memory', 'Screen Size', 'Resolution']



    columns_to_retain = list(set(data.columns)-set(columns_to_remove))

    data = data[columns_to_retain]



    return data
binarised_train = data_clean_5(binarised_train)

binarised_test = data_clean_5(binarised_test)
def data_clean_6(x):

    

    data = x.copy()

      

    columns_to_remove = ['SIM Slot(s)_Dual SIM, GSM+CDMA', 'Sim1_2G', 'Sim1_4G',\

                         'os_name_Nokia', 'os_name_Other', 'Num_cores_312', 'Num_cores_Other',\

                         'Num_cores_Tru-Octa', 'SIM 2_3G', 'Brand_Asus',\

                         'Brand_Gionee', 'Brand_Honor', 'Brand_Infinix',\

                         'Brand_Lava', 'Brand_LeEco', 'Brand_Lenovo',\

                         'Brand_Meizu', 'Brand_Moto', 'Brand_Motorola',\

                         'Brand_Oppo', 'Brand_Panasonic']

    columns_to_retain = list(set(data.columns)-set(columns_to_remove))

    data = data[columns_to_retain]



    columns_to_remove = ['Brand_Reliance', 'os_name_Blackberry']

    columns_to_retain = list(set(data.columns)-set(columns_to_remove))

    data = data[columns_to_retain]

    

    return data
binarised_train = data_clean_6(binarised_train)

binarised_test = data_clean_6(binarised_test)
# Binarization - Train 



# Cut limits for Continuous Variables determined by looking at Scatter Plots in a previous Code section; Rating column Binarized in a previous section separately



# Cut values are adjusted again after looking at Confusion matrix to adjust False Positive vs False Negative rate; 

#     By default the model comes out as biased towards the majority representation of Positive cases



# binarised_train['Weight'] = binarised_train['Weight'].map(lambda x: 0 if x < 280 else 1)

# Try 50 to 350 range

# Previously chosen optimal value = 70; Adjusted to improve prediction of negative cases as per Confusion Matrix



# binarised_train['RAM'] = binarised_train['RAM'].map(lambda x: 0 if x > 265 else 1)

# Try 0 to 550 range

# Previously chosen optimal value = 300; Adjusted to improve prediction of negative cases as per Confusion Matrix



# binarised_train['Screen to Body Ratio (calculated)'] = binarised_train['Screen to Body Ratio (calculated)'].map(lambda x: 0 if x < 70 else 1)

# Try 20 to 90 range



# binarised_train['Pixel Density'] = binarised_train['Pixel Density'].map(lambda x: 0 if x < 250 else 1)

# Try 100 to 600 range

# Previously chosen optimal value = 250; Adjusted to improve prediction of negative cases as per Confusion Matrix



# binarised_train['Processor_frequency'] = binarised_train['Processor_frequency'].map(lambda x: 0 if x < 1.75 else 1)

# Try 0.75 to 3 range

# Previously chosen optimal value = 1.35; Adjusted to improve prediction of negative cases as per Confusion Matrix



# binarised_train['Screen Size'] = binarised_train['Screen Size'].map(lambda x: 0 if x < 4.7 else 1)

# Try 2 to 7 range



# binarised_train['Capacity'] = binarised_train['Capacity'].map(lambda x: 0 if x < 1400 else 1)

# Try 0 to 14000 range (Drop Capacity)



# binarised_train['Height'] = binarised_train['Height'].map(lambda x: 0 if x < 135 else 1)

# Try 0 to 180 range (Drop Height)



# binarised_train['Internal Memory'] = binarised_train['Internal Memory'].map(lambda x: 0 if x < 10 else 1)

# Try 0 to 550 range (Drop Internal Memory)



# binarised_train['Resolution'] = binarised_train['Resolution'].map(lambda x: 0 if x > 15 else 1)

# Try 0 to 25 range (Drop Resolution)





# Other Dummy Variables with clear cut values



# binarised_train['SIM Slot(s)_Dual SIM, GSM+CDMA'] = binarised_train['SIM Slot(s)_Dual SIM, GSM+CDMA'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_train['Sim1_2G'] = binarised_train['Sim1_2G'].map(lambda x: 0 if x > 0.5 else 1)

binarised_train['Sim1_3G'] = binarised_train['Sim1_3G'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_train['Sim1_4G'] = binarised_train['Sim1_4G'].map(lambda x: 0 if x < 0.5 else 1)

#binarised_train['os_name_Blackberry'] = binarised_train['os_name_Blackberry'].map(lambda x: 0 if x > 0.5 else 1)

binarised_train['os_name_KAI'] = binarised_train['os_name_KAI'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_train['os_name_Nokia'] = binarised_train['os_name_Nokia'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_train['os_name_Other'] = binarised_train['os_name_Other'].map(lambda x: 0 if x > 0.5 else 1)

binarised_train['os_name_Tizen'] = binarised_train['os_name_Tizen'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_train['Num_cores_312'] = binarised_train['Num_cores_312'].map(lambda x: 0 if x > 0.5 else 1)

binarised_train['Num_cores_Deca'] = binarised_train['Num_cores_Deca'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_train['Num_cores_Other'] = binarised_train['Num_cores_Other'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_train['Num_cores_Tru-Octa'] = binarised_train['Num_cores_Tru-Octa'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_train['SIM 2_3G'] = binarised_train['SIM 2_3G'].map(lambda x: 0 if x > 0.5 else 1)

binarised_train['Brand_10.or'] = binarised_train['Brand_10.or'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_train['Brand_Asus'] = binarised_train['Brand_Asus'].map(lambda x: 0 if x > 0.5 else 1)

binarised_train['Brand_Blackberry'] = binarised_train['Brand_Blackberry'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_train['Brand_Comio'] = binarised_train['Brand_Comio'].map(lambda x: 0 if x > 0.5 else 1)

binarised_train['Brand_Coolpad'] = binarised_train['Brand_Coolpad'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_train['Brand_Gionee'] = binarised_train['Brand_Gionee'].map(lambda x: 0 if x > 0.5 else 1)

binarised_train['Brand_HTC'] = binarised_train['Brand_HTC'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_train['Brand_Honor'] = binarised_train['Brand_Honor'].map(lambda x: 0 if x > 0.5 else 1)

binarised_train['Brand_InFocus'] = binarised_train['Brand_InFocus'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_train['Brand_Infinix'] = binarised_train['Brand_Infinix'].map(lambda x: 0 if x > 0.5 else 1)

binarised_train['Brand_Intex'] = binarised_train['Brand_Intex'].map(lambda x: 0 if x > 0.5 else 1)

binarised_train['Brand_Jivi'] = binarised_train['Brand_Jivi'].map(lambda x: 0 if x > 0.5 else 1)

binarised_train['Brand_Karbonn'] = binarised_train['Brand_Karbonn'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_train['Brand_Lava'] = binarised_train['Brand_Lava'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_train['Brand_LeEco'] = binarised_train['Brand_LeEco'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_train['Brand_Lenovo'] = binarised_train['Brand_Lenovo'].map(lambda x: 0 if x > 0.5 else 1)

binarised_train['Brand_Lephone'] = binarised_train['Brand_Lephone'].map(lambda x: 0 if x > 0.5 else 1)

binarised_train['Brand_Lyf'] = binarised_train['Brand_Lyf'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_train['Brand_Meizu'] = binarised_train['Brand_Meizu'].map(lambda x: 0 if x > 0.5 else 1)

binarised_train['Brand_Micromax'] = binarised_train['Brand_Micromax'].map(lambda x: 0 if x > 0.5 else 1)

binarised_train['Brand_Mobiistar'] = binarised_train['Brand_Mobiistar'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_train['Brand_Moto'] = binarised_train['Brand_Moto'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_train['Brand_Motorola'] = binarised_train['Brand_Motorola'].map(lambda x: 0 if x > 0.5 else 1)

binarised_train['Brand_Nubia'] = binarised_train['Brand_Nubia'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_train['Brand_Oppo'] = binarised_train['Brand_Oppo'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_train['Brand_Panasonic'] = binarised_train['Brand_Panasonic'].map(lambda x: 0 if x > 0.5 else 1)

binarised_train['Brand_Razer'] = binarised_train['Brand_Razer'].map(lambda x: 0 if x > 0.5 else 1)

#binarised_train['Brand_Reliance'] = binarised_train['Brand_Reliance'].map(lambda x: 0 if x > 0.5 else 1)

binarised_train['Brand_VOTO'] = binarised_train['Brand_VOTO'].map(lambda x: 0 if x > 0.5 else 1)

binarised_train['Brand_Yu'] = binarised_train['Brand_Yu'].map(lambda x: 0 if x > 0.5 else 1)

binarised_train['Brand_iVooMi'] = binarised_train['Brand_iVooMi'].map(lambda x: 0 if x > 0.5 else 1)



# Re-introduced after checking value_counts though Scatter Plots suggested elimination



binarised_train['Brand_LG'] = binarised_train['Brand_LG'].map(lambda x: 0 if x > 0.5 else 1)

binarised_train['Brand_Sony'] = binarised_train['Brand_Sony'].map(lambda x: 0 if x > 0.5 else 1)



binarised_train.head()
# Binarization - Test (Cut limits reused from analysis on Train dataset)



### Previously selected continuous variables

# binarised_test['Screen Size'] = binarised_test['Screen Size'].map(lambda x: 0 if x < 4.7 else 1)

# binarised_test['RAM'] = binarised_test['RAM'].map(lambda x: 0 if x > 265 else 1)

# binarised_test['Pixel Density'] = binarised_test['Pixel Density'].map(lambda x: 0 if x < 250 else 1)

# binarised_test['Screen to Body Ratio (calculated)'] = binarised_test['Screen to Body Ratio (calculated)'].map(lambda x: 0 if x < 70 else 1)

# binarised_test['Processor_frequency'] = binarised_test['Processor_frequency'].map(lambda x: 0 if x < 1.75 else 1)

# binarised_test['Height'] = binarised_test['Height'].map(lambda x: 0 if x < 135 else 1)

# binarised_test['Weight'] = binarised_test['Weight'].map(lambda x: 0 if x < 280 else 1)



### Previously removed continuous variables from Scatter Plots & individual loop tests with trial MP Neuron Model

# binarised_test['Capacity'] = binarised_test['Capacity'].map(lambda x: 0 if x < 1400 else 1)

# binarised_test['Height'] = binarised_test['Height'].map(lambda x: 0 if x < 135 else 1)

# binarised_test['Internal Memory'] = binarised_test['Internal Memory'].map(lambda x: 0 if x < 10 else 1)

# binarised_test['Resolution'] = binarised_test['Resolution'].map(lambda x: 0 if x > 15 else 1)







# binarised_test['Sim1_2G'] = binarised_test['Sim1_2G'].map(lambda x: 0 if x > 0.5 else 1)

binarised_test['Sim1_3G'] = binarised_test['Sim1_3G'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_test['Sim1_4G'] = binarised_test['Sim1_4G'].map(lambda x: 0 if x < 0.5 else 1)

#binarised_test['os_name_Blackberry'] = binarised_test['os_name_Blackberry'].map(lambda x: 0 if x > 0.5 else 1)

binarised_test['os_name_KAI'] = binarised_test['os_name_KAI'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_test['os_name_Nokia'] = binarised_test['os_name_Nokia'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_test['os_name_Other'] = binarised_test['os_name_Other'].map(lambda x: 0 if x > 0.5 else 1)

binarised_test['os_name_Tizen'] = binarised_test['os_name_Tizen'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_test['Num_cores_312'] = binarised_test['Num_cores_312'].map(lambda x: 0 if x > 0.5 else 1)

binarised_test['Num_cores_Deca'] = binarised_test['Num_cores_Deca'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_test['Num_cores_Other'] = binarised_test['Num_cores_Other'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_test['Num_cores_Tru-Octa'] = binarised_test['Num_cores_Tru-Octa'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_test['SIM 2_3G'] = binarised_test['SIM 2_3G'].map(lambda x: 0 if x > 0.5 else 1)

binarised_test['Brand_10.or'] = binarised_test['Brand_10.or'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_test['Brand_Asus'] = binarised_test['Brand_Asus'].map(lambda x: 0 if x > 0.5 else 1)

binarised_test['Brand_Blackberry'] = binarised_test['Brand_Blackberry'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_test['Brand_Comio'] = binarised_test['Brand_Comio'].map(lambda x: 0 if x > 0.5 else 1)

binarised_test['Brand_Coolpad'] = binarised_test['Brand_Coolpad'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_test['Brand_Gionee'] = binarised_test['Brand_Gionee'].map(lambda x: 0 if x > 0.5 else 1)

binarised_test['Brand_HTC'] = binarised_test['Brand_HTC'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_test['Brand_Honor'] = binarised_test['Brand_Honor'].map(lambda x: 0 if x > 0.5 else 1)

binarised_test['Brand_InFocus'] = binarised_test['Brand_InFocus'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_test['Brand_Infinix'] = binarised_test['Brand_Infinix'].map(lambda x: 0 if x > 0.5 else 1)

binarised_test['Brand_Intex'] = binarised_test['Brand_Intex'].map(lambda x: 0 if x > 0.5 else 1)

binarised_test['Brand_Jivi'] = binarised_test['Brand_Jivi'].map(lambda x: 0 if x > 0.5 else 1)

binarised_test['Brand_Karbonn'] = binarised_test['Brand_Karbonn'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_test['Brand_Lava'] = binarised_test['Brand_Lava'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_test['Brand_LeEco'] = binarised_test['Brand_LeEco'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_test['Brand_Lenovo'] = binarised_test['Brand_Lenovo'].map(lambda x: 0 if x > 0.5 else 1)

binarised_test['Brand_Lephone'] = binarised_test['Brand_Lephone'].map(lambda x: 0 if x > 0.5 else 1)

binarised_test['Brand_Lyf'] = binarised_test['Brand_Lyf'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_test['Brand_Meizu'] = binarised_test['Brand_Meizu'].map(lambda x: 0 if x > 0.5 else 1)

binarised_test['Brand_Micromax'] = binarised_test['Brand_Micromax'].map(lambda x: 0 if x > 0.5 else 1)

binarised_test['Brand_Mobiistar'] = binarised_test['Brand_Mobiistar'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_test['Brand_Moto'] = binarised_test['Brand_Moto'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_test['Brand_Motorola'] = binarised_test['Brand_Motorola'].map(lambda x: 0 if x > 0.5 else 1)

binarised_test['Brand_Nubia'] = binarised_test['Brand_Nubia'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_test['Brand_Oppo'] = binarised_test['Brand_Oppo'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_test['Brand_Panasonic'] = binarised_test['Brand_Panasonic'].map(lambda x: 0 if x > 0.5 else 1)

binarised_test['Brand_Razer'] = binarised_test['Brand_Razer'].map(lambda x: 0 if x > 0.5 else 1)

#binarised_test['Brand_Reliance'] = binarised_test['Brand_Reliance'].map(lambda x: 0 if x > 0.5 else 1)

binarised_test['Brand_VOTO'] = binarised_test['Brand_VOTO'].map(lambda x: 0 if x > 0.5 else 1)

binarised_test['Brand_Yu'] = binarised_test['Brand_Yu'].map(lambda x: 0 if x > 0.5 else 1)

binarised_test['Brand_iVooMi'] = binarised_test['Brand_iVooMi'].map(lambda x: 0 if x > 0.5 else 1)

# binarised_test['SIM Slot(s)_Dual SIM, GSM+CDMA'] = binarised_test['SIM Slot(s)_Dual SIM, GSM+CDMA'].map(lambda x: 0 if x > 0.5 else 1)



# Re-introduced after checking value_counts though Scatter Plots suggested elimination



binarised_test['Brand_LG'] = binarised_test['Brand_LG'].map(lambda x: 0 if x > 0.5 else 1)

binarised_test['Brand_Sony'] = binarised_test['Brand_Sony'].map(lambda x: 0 if x > 0.5 else 1)





binarised_test.head()
binarised_train = binarised_train.drop('PhoneId',axis=1)

binarised_test = binarised_test.drop('PhoneId',axis=1)
#binarised_train_continuous = binarised_train[['Weight', 'RAM', 'Screen to Body Ratio (calculated)', 'Pixel Density', 'Processor_frequency', 'Screen Size']]

#binarised_test_continuous = binarised_test[['Weight', 'RAM', 'Screen to Body Ratio (calculated)', 'Pixel Density', 'Processor_frequency', 'Screen Size']]
#type(binarised_train)
#binarised_test_continuous = binarised_test_continuous.values

#binarised_train_continuous = binarised_train_continuous.values
# Convert Pandas Datagrames to Numpy Arrays



binarised_test = binarised_test.values

binarised_train = binarised_train.values
# MP Neuron Class (Reused from Padhai course, but edited to return 0 or 1 instead of boolean value)



class MPNeuron:



  def __init__(self):

    self.b = None

    

  def model(self, x):

    if sum(x) >= self.b:

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

# MP Neuron - Instantiate and Fit on Train



mp_neuron = MPNeuron()

mp_neuron.fit(binarised_train, Train_Rating_Binarized_nparray)
# Display Confusion Matrix for Train



Y_train_pred = mp_neuron.predict(binarised_train)



confusion_matrix(Train_Rating_Binarized_nparray, Y_train_pred, sample_weight=None)
# MP Neuron - Predict for Test



Y_test_pred = mp_neuron.predict(binarised_test)
# Create Submission file



submission = pd.DataFrame({'PhoneId':test_new['PhoneId'], 'Class':Y_test_pred})

submission = submission[['PhoneId', 'Class']]



submission.head()
# print(submission)
# Write to CSV



submission.to_csv("submission-13.csv", index=False)