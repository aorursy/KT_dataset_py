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

train = pd.read_csv("../input/padhai-module1-assignment/train.csv")

test = pd.read_csv("../input/padhai-module1-assignment/test.csv")



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
# save the cleaned csv 

train_new.to_csv("train_cleaned.csv", index=False)

test_new.to_csv("test_cleaned.csv", index=False)
test_new.head()
train_new.head()
train_new.describe()
train_new.shape
plt.plot(train_new['Rating'].T,'*')

plt.xticks(rotation = 'vertical')

plt.show()
train_new['Rating'] = train_new['Rating'].map(lambda x: 0 if x<=THRESHOLD else 1)
plt.plot(train_new['Rating'].T,'*')

plt.xticks(rotation = 'vertical')

plt.show()
train_new.head(20)
ratings = train_new['Rating']
all_details = train_new.drop('Rating', axis =1)
plt.plot(train_new['PhoneId'].T,'*')

plt.xticks(rotation = 'vertical')

plt.show()
all_details = all_details.drop('PhoneId', axis =1)
all_details.head()
print(train_new['Screen to Body Ratio (calculated)'].mean())

plt.plot(train_new['Screen to Body Ratio (calculated)'].T,'*')

plt.xticks(rotation = 'vertical')

plt.show()
all_details['Screen to Body Ratio (calculated)'] = all_details['Screen to Body Ratio (calculated)'].map(lambda x: 0 if x <= 72.35 else 1)
test_new['Screen to Body Ratio (calculated)'] = test_new['Screen to Body Ratio (calculated)'].map(lambda x: 0 if x<= 72.35 else 1)
print(train_new['Screen Size'].mean())

plt.plot(train_new['Screen Size'].T,'*')

plt.xticks(rotation = 'vertical')

plt.show()
all_details['Screen Size'] = all_details['Screen Size'].map(lambda x: 0 if x <= 5.46 else 1)
test_new['Screen Size'] = test_new['Screen Size'].map(lambda x: 0 if x<= 5.46 else 1)
print(train_new['Processor_frequency'].mean())

plt.plot(train_new['Processor_frequency'].T,'*')

plt.xticks(rotation = 'vertical')

plt.show()
all_details['Processor_frequency'] = all_details['Processor_frequency'].map(lambda x: 0 if x <= 1.79 else 1)
test_new['Processor_frequency'] = test_new['Processor_frequency'].map(lambda x: 0 if x <= 1.76 else 1)
print(train_new['RAM'].mean())

plt.plot(train_new['RAM'].T,'*')

plt.show()
all_details['RAM'] = all_details['RAM'].map(lambda x: 0 if x <= 1 else 1)
test_new['RAM'] = test_new['RAM'].map(lambda x: 0 if x <= 1 else 1)
print(train_new['Internal Memory'].value_counts())
print(train_new['Internal Memory'].mean())

plt.plot(train_new['Internal Memory'],'*')

plt.show()
all_details['Internal Memory'] = all_details['Internal Memory'].map(lambda x: 0 if x<=32 else 1)
test_new['Internal Memory'] = test_new['Internal Memory'].map(lambda x: 0 if x <= 32 else 1)
print(train_new['Weight'].value_counts())
print(train_new['Weight'].mean())

plt.plot(train_new['Weight'],'*')

plt.show()
all_details['Weight'] = all_details['Weight'].map(lambda x: 0 if x<= 160 else 1)
test_new['Weight'] = test_new['Weight'].map(lambda x:0 if x<= 160 else 1)
print(train_new['Resolution'].value_counts())

print(train_new['Resolution'].median())

plt.plot(train_new['Resolution'],'*')

plt.show()
all_details['Resolution'] = all_details['Resolution'].map(lambda x: 0 if x<=7 else 1)
test_new['Resolution'] = test_new['Resolution'].map(lambda x:0 if x<=7 else 1)
print(train_new['Pixel Density'].value_counts())

print(train_new['Pixel Density'].mean())

print(train_new['Pixel Density'].median())

plt.plot(train_new['Pixel Density'],'*')

plt.show()
all_details['Pixel Density'] = all_details['Pixel Density'].map(lambda x:0 if x<=326 else 1)
test_new['Pixel Density'] = test_new['Pixel Density'].map(lambda x:0 if x<= 326 else 1)
print(train_new['Height'].value_counts())

print(train_new['Height'].mean())

print(train_new['Height'].median())

plt.plot(train_new['Height'],'*')

plt.show()
all_details['Height'] = all_details['Height'].map(lambda x: 0 if x<= 150 else 1)
test_new['Height'] = test_new['Height'].map(lambda x: 0 if x<= 150 else 1)
print(train_new['Capacity'].value_counts())

print(train_new['Capacity'].mean())

print(train_new['Capacity'].median())

plt.plot(train_new['Capacity'],'*')

plt.show()
all_details['Capacity'] = all_details['Capacity'].map(lambda x: 0 if x<=3000 else 1)
test_new['Capacity'] = test_new['Capacity'].map(lambda x: 0 if x<=3000 else 1)
all_details.head(20)
test_new.head(20)
testphoneid = test_new['PhoneId']
test_new = test_new.drop('PhoneId',axis =1)
test_new.head(20)
all_details = all_details.values

test_new = test_new.values

ratings = ratings.values
all_details.shape
test_new.shape
ratings.shape
b = 3

i = 30

print("for row ", i)

if (np.sum(all_details[i,:]) >= b):

    print("MP Neuron inference is liked")

else:

    print("MP Neuron inference is not like")



if (ratings[i] == 1):

    print("ground result is liked")

else:

    print("ground result is not liked")
b = 8



Y_Pred_Train = []

accurate_rows = 0



# iterate 2 vectors simultaneously so for that I'll use zip function

for x, y in zip(all_details, ratings):

    Y_Pred = (np.sum(x) >= b)    # inference of the model

    Y_Pred_Train.append(Y_Pred)

    accurate_rows += (y == Y_Pred)



print(accurate_rows, accurate_rows/all_details.shape[0])
for b in range(all_details.shape[1] + 1):

    Y_Pred_Train = []

    accurate_rows = 0



  # iterate 2 vectors simultaneously so for that I'll use zip function

    for x, y in zip(all_details, ratings):

        Y_Pred = (np.sum(x) >= b)    # inference of the model

        Y_Pred_Train.append(Y_Pred)

        accurate_rows += (y == Y_Pred)



    print(b, accurate_rows/all_details.shape[0])
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
mp_neuron = MPNeuron()

mp_neuron.fit(all_details, ratings)
Y_testPred = mp_neuron.predict(test_new)
Y_testPred = Y_testPred.astype(int)
submission = pd.DataFrame({'PhoneId':testphoneid, 'Class':Y_testPred})

submission = submission[['PhoneId', 'Class']]

submission.head()
submission.to_csv("submission.csv", index=False)