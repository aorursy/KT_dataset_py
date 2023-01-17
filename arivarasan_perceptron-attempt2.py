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
submission = pd.DataFrame({'PhoneId':test_new['PhoneId'], 'Class':[0]*test_new.shape[0]})

submission = submission[['PhoneId', 'Class']]

submission.head()
submission.to_csv("submission.csv", index=False)
"""

WRITE YOUR MODELLING CODE HERE

"""
train_data = train_new.copy()

train_data['is_liked'] = train_data['Rating'].map(lambda x: 0 if x<4 else 1)
X = train_data.drop(['PhoneId','Rating'], axis=1)

Y = train_data['is_liked']
X.groupby('is_liked').mean()
def normalize(data):

    normalized_data=[]

    for x in data:

        norm = (x-min(data))/(max(data)-min(data))

        normalized_data.append(norm)

    return(np.array(normalized_data))
X['RAM'] = X['RAM'].apply(lambda x: 1 if(x>8) else x)

X['Internal Memory'] = X['Internal Memory'].apply(lambda x: 1 if(x>128) else x)



X['RAM'] = normalize(X['RAM'])

X['Internal Memory'] = normalize(X['Internal Memory'])

X['Pixel Density'] = normalize(X['Pixel Density'])

X['Capacity'] = normalize(X['Capacity'])

X['Resolution'] = normalize(X['Resolution'])

X['Screen Size'] = normalize(X['Screen Size'])

X['Screen to Body Ratio (calculated)'] = normalize(X['Screen to Body Ratio (calculated)'])

X['Height'] = normalize(X['Height'])

X['Processor_frequency'] = normalize(X['Processor_frequency'])

X['Weight'] = normalize(X['Weight'])

X.head()
#Likeability scores of binary features

def compute_likeability(feature):

    no_of_likes = 0

    for x,y in zip(X[feature],X['is_liked']):

        if(x==1 and y==1):

            no_of_likes+=1

    total_x = sum(X[feature]==1)

    if(total_x==0):

        total_x=1

    likeability_score = no_of_likes/total_x

    return likeability_score



def return_likeability(features):

    likeability={}

    for feature in features:

        #print(feature)

        likeability[feature] = compute_likeability(feature)

    return likeability
os_features = ['os_name_Android','os_name_Blackberry','os_name_KAI','os_name_Other','os_name_Tizen','os_name_Nokia','os_name_iOS']

brand_features = ['Brand_Apple', 'Brand_Asus', 'Brand_Billion', 'Brand_Blackberry',

      'Brand_Comio', 'Brand_Coolpad', 'Brand_Do', 'Brand_Gionee',

    'Brand_Google', 'Brand_HTC', 'Brand_Honor', 'Brand_Huawei',

    'Brand_InFocus', 'Brand_Infinix', 'Brand_Intex', 'Brand_Itel',

      'Brand_Jivi', 'Brand_Karbonn', 'Brand_LG', 'Brand_Lava', 'Brand_LeEco',

      'Brand_Lenovo', 'Brand_Lephone', 'Brand_Lyf', 'Brand_Meizu',

      'Brand_Micromax', 'Brand_Mobiistar', 'Brand_Moto', 'Brand_Motorola',

      'Brand_Nokia', 'Brand_Nubia', 'Brand_OPPO', 'Brand_OnePlus',

     'Brand_Oppo', 'Brand_Panasonic', 'Brand_Razer', 'Brand_Realme',

     'Brand_Reliance', 'Brand_Samsung', 'Brand_Sony', 'Brand_Spice',

      'Brand_Tecno', 'Brand_Ulefone', 'Brand_VOTO', 'Brand_Vivo',

      'Brand_Xiaomi', 'Brand_Xiaomi Poco', 'Brand_Yu', 'Brand_iVooMi','Brand_10.or']



numcores_features = ['Num_cores_312','Num_cores_Deca','Num_cores_Dual','Num_cores_Dual','Num_cores_Hexa','Num_cores_Octa','Num_cores_Other','Num_cores_Quad','Num_cores_Tru-Octa']

sim1_features = ['Sim1_2G','Sim1_3G','Sim1_3G','Sim1_4G']

sim2_features = ['SIM 2_2G','SIM 2_3G','SIM 2_4G','SIM 2_Other']

simslot_features = ['SIM Slot(s)_Dual SIM, GSM+CDMA','SIM Slot(s)_Dual SIM, GSM+GSM','SIM Slot(s)_Dual SIM, GSM+GSM, Dual VoLTE','SIM Slot(s)_Single SIM, GSM']
os_likeability = return_likeability(os_features)

X['os_likeability'] = np.zeros(X.shape[0])

X['os_likeability'].loc[np.where(X['os_name_Android']==1)[0]] = 0.6893

X['os_likeability'].loc[np.where(X['os_name_Other']==1)[0]] = 0.714

X['os_likeability'].loc[np.where(X['os_name_Nokia']==1)[0]] = 1

X['os_likeability'].loc[np.where(X['os_name_iOS']==1)[0]] = 1

X = X.drop(os_features, axis=1)



brand_likeability = return_likeability(brand_features)

X['brand_likeability'] = np.zeros(X.shape[0])

X['brand_likeability'].loc[np.where(X['Brand_Apple']==1)[0]] = 1

X['brand_likeability'].loc[np.where(X['Brand_Asus']==1)[0]] = 0.875

X['brand_likeability'].loc[np.where(X['Brand_Blackberry']==1)[0]] = 0.2

X['brand_likeability'].loc[np.where(X['Brand_Comio']==1)[0]] = 1

X['brand_likeability'].loc[np.where(X['Brand_Gionee']==1)[0]] = 0.6

X['brand_likeability'].loc[np.where(X['Brand_Google']==1)[0]] = 1

X['brand_likeability'].loc[np.where(X['Brand_HTC']==1)[0]] = 0.1667

X['brand_likeability'].loc[np.where(X['Brand_Honor']==1)[0]] = 0.882

X['brand_likeability'].loc[np.where(X['Brand_Huawei']==1)[0]] = 1

X['brand_likeability'].loc[np.where(X['Brand_Infinix']==1)[0]] = 0.6667

X['brand_likeability'].loc[np.where(X['Brand_Itel']==1)[0]] = 0.75

X['brand_likeability'].loc[np.where(X['Brand_Lava']==1)[0]] = 1

X['brand_likeability'].loc[np.where(X['Brand_LG']==1)[0]] = 0.333

X['brand_likeability'].loc[np.where(X['Brand_LeEco']==1)[0]] = 1

X['brand_likeability'].loc[np.where(X['Brand_Lenovo']==1)[0]] = 0.6

X['brand_likeability'].loc[np.where(X['Brand_Meizu']==1)[0]] = 1

X['brand_likeability'].loc[np.where(X['Brand_Micromax']==1)[0]] = 0.2

X['brand_likeability'].loc[np.where(X['Brand_Mobiistar']==1)[0]] = 0.25

X['brand_likeability'].loc[np.where(X['Brand_Moto']==1)[0]] = 0.6667

X['brand_likeability'].loc[np.where(X['Brand_Motorola']==1)[0]] = 1

X['brand_likeability'].loc[np.where(X['Brand_Nokia']==1)[0]] = 0.619

X['brand_likeability'].loc[np.where(X['Brand_OPPO']==1)[0]] = 1

X['brand_likeability'].loc[np.where(X['Brand_OnePlus']==1)[0]] = 1

X['brand_likeability'].loc[np.where(X['Brand_Oppo']==1)[0]] = 0.833

X['brand_likeability'].loc[np.where(X['Brand_Panasonic']==1)[0]] = 0.5

X['brand_likeability'].loc[np.where(X['Brand_Realme']==1)[0]] = 1

X['brand_likeability'].loc[np.where(X['Brand_Samsung']==1)[0]] = 0.8636

X['brand_likeability'].loc[np.where(X['Brand_Sony']==1)[0]] = 0.444

X['brand_likeability'].loc[np.where(X['Brand_Tecno']==1)[0]] = 0.875

X['brand_likeability'].loc[np.where(X['Brand_Ulefone']==1)[0]] = 1

X['brand_likeability'].loc[np.where(X['Brand_Vivo']==1)[0]] = 0.923

X['brand_likeability'].loc[np.where(X['Brand_Xiaomi']==1)[0]] = 0.9047

X['brand_likeability'].loc[np.where(X['Brand_Xiaomi Poco']==1)[0]] = 1

X['brand_likeability'].loc[np.where(X['Brand_Yu']==1)[0]] = 0.25

X = X.drop(brand_features,axis=1)



num_cores_likeability = return_likeability(numcores_features)

print(num_cores_likeability)

X['numcores_likeability'] = np.zeros(X.shape[0])

X['numcores_likeability'].loc[np.where(X['Num_cores_312']==1)[0]] = 1

X['numcores_likeability'].loc[np.where(X['Num_cores_Deca']==1)[0]] = 0

X['numcores_likeability'].loc[np.where(X['Num_cores_Dual']==1)[0]] = 0.6

X['numcores_likeability'].loc[np.where(X['Num_cores_Hexa']==1)[0]] = 0.9167

X['numcores_likeability'].loc[np.where(X['Num_cores_Octa']==1)[0]] = 0.773

X['numcores_likeability'].loc[np.where(X['Num_cores_Other']==1)[0]] = 0.833

X['numcores_likeability'].loc[np.where(X['Num_cores_Quad']==1)[0]] = 0.5495

X['numcores_likeability'].loc[np.where(X['Num_cores_Tru-Octa']==1)[0]] = 1

X = X.drop(numcores_features, axis=1)



sim1_likeability = return_likeability(sim1_features)

print(sim1_likeability)

X['sim1_likeability'] = np.zeros(X.shape[0])

X['sim1_likeability'].loc[np.where(X['Sim1_2G']==1)[0]] = 0.857

X['sim1_likeability'].loc[np.where(X['Sim1_3G']==1)[0]] = 0.2857

X['sim1_likeability'].loc[np.where(X['Sim1_4G']==1)[0]] = 0.703

X = X.drop(sim1_features, axis = 1)



sim2_likeability = return_likeability(sim2_features)

print(sim2_likeability)

X['sim2_likeability'] = np.zeros(X.shape[0])

X['sim2_likeability'].loc[np.where(X['SIM 2_2G']==1)[0]] = 0.5277

X['sim2_likeability'].loc[np.where(X['SIM 2_3G']==1)[0]] = 0.5714

X['sim2_likeability'].loc[np.where(X['SIM 2_4G']==1)[0]] = 0.7424

X['sim2_likeability'].loc[np.where(X['SIM 2_Other']==1)[0]] = 0.79

X = X.drop(sim2_features, axis=1)



simslot_likeability = return_likeability(simslot_features)

print(simslot_likeability)

X['simslot_likeability'] = np.zeros(X.shape[0])

X['simslot_likeability'].loc[np.where(X['SIM Slot(s)_Dual SIM, GSM+CDMA']==1)[0]] = 1

X['simslot_likeability'].loc[np.where(X['SIM Slot(s)_Dual SIM, GSM+GSM']==1)[0]] = 0.6409

X['simslot_likeability'].loc[np.where(X['SIM Slot(s)_Dual SIM, GSM+GSM, Dual VoLTE']==1)[0]] = 0.923

X['simslot_likeability'].loc[np.where(X['SIM Slot(s)_Single SIM, GSM']==1)[0]] = 0.793

X = X.drop(simslot_features, axis=1)



X.head()
x_train = X.drop('is_liked', axis=1)

x_train = x_train.values

y_train = Y.values

x_train.shape
class Perceptron():

    def __init__(self):

        self.w = None

        self.b = None

    

    def model(self, x):

        return 1 if (np.dot(self.w, x) >= self.b) else 0

    

    def predict(self, X):

        y_pred = []

        for x in X:

            pred = self.model(x)

            y_pred.append(pred)

        return(np.array(y_pred))

    

    def fit(self, X, Y, epochs = 1, lr = 1):

        self.w = np.ones(X.shape[1])

        #self.w = np.random.random_sample(X.shape[1])

        self.b = 0

        accuracy = {}

        max_accuracy = 0

        wt_matrix = []

        for epoch in range(epochs):         

            for x,y in zip(X,Y):

                pred = self.model(x)

                if(pred == 0 and y == 1):

                    self.w = self.w + lr * x

                    self.b = self.b - lr * 1

                elif(pred == 1 and y == 0):

                    self.w = self.w - lr * x

                    self.b = self.b + lr * 1        

            wt_matrix.append(self.w)

            accuracy[epoch] = accuracy_score(self.predict(X), Y)

            if(accuracy[epoch]>max_accuracy):

                chkptw = self.w

                chkptb = self.b

                max_accuracy = accuracy[epoch]

        self.w = chkptw

        self.b = chkptb

        plt.plot(accuracy.values())

        plt.ylim([0,1])

        plt.show()

        print('Max Accuracy: ', max_accuracy)

        return([wt_matrix, accuracy])
perceptron = Perceptron()
[wt_matrix, accuracy_dict] = perceptron.fit(x_train, y_train, epochs=5000, lr=0.2)
test_new.head()
test_data = test_new.drop('PhoneId', axis=1)
test_data['RAM'] = test_data['RAM'].apply(lambda x: 1 if(x>8) else x)

test_data['Internal Memory'] = test_data['Internal Memory'].apply(lambda x: 1 if(x>128) else x)



test_data['RAM'] = normalize(test_data['RAM'])

test_data['Internal Memory'] = normalize(test_data['Internal Memory'])

test_data['Pixel Density'] = normalize(test_data['Pixel Density'])

test_data['Capacity'] = normalize(test_data['Capacity'])

test_data['Resolution'] = normalize(test_data['Resolution'])

test_data['Screen Size'] = normalize(test_data['Screen Size'])

test_data['Screen to Body Ratio (calculated)'] = normalize(test_data['Screen to Body Ratio (calculated)'])

test_data['Height'] = normalize(test_data['Height'])

test_data['Processor_frequency'] = normalize(test_data['Processor_frequency'])

test_data['Weight'] = normalize(test_data['Weight'])

test_data.head()
test_data['os_likeability'] = np.zeros(test_data.shape[0])

test_data['os_likeability'].loc[np.where(test_data['os_name_Android']==1)[0]] = 0.6893

test_data['os_likeability'].loc[np.where(test_data['os_name_Other']==1)[0]] = 0.714

test_data['os_likeability'].loc[np.where(test_data['os_name_Nokia']==1)[0]] = 1

test_data['os_likeability'].loc[np.where(test_data['os_name_iOS']==1)[0]] = 1

test_data = test_data.drop(os_features, axis=1)





test_data['brand_likeability'] = np.zeros(test_data.shape[0])

test_data['brand_likeability'].loc[np.where(test_data['Brand_Apple']==1)[0]] = 1

test_data['brand_likeability'].loc[np.where(test_data['Brand_Asus']==1)[0]] = 0.875

test_data['brand_likeability'].loc[np.where(test_data['Brand_Blackberry']==1)[0]] = 0.2

test_data['brand_likeability'].loc[np.where(test_data['Brand_Comio']==1)[0]] = 1

test_data['brand_likeability'].loc[np.where(test_data['Brand_Gionee']==1)[0]] = 0.6

test_data['brand_likeability'].loc[np.where(test_data['Brand_Google']==1)[0]] = 1

test_data['brand_likeability'].loc[np.where(test_data['Brand_HTC']==1)[0]] = 0.1667

test_data['brand_likeability'].loc[np.where(test_data['Brand_Honor']==1)[0]] = 0.882

test_data['brand_likeability'].loc[np.where(test_data['Brand_Huawei']==1)[0]] = 1

test_data['brand_likeability'].loc[np.where(test_data['Brand_Infinix']==1)[0]] = 0.6667

test_data['brand_likeability'].loc[np.where(test_data['Brand_Itel']==1)[0]] = 0.75

test_data['brand_likeability'].loc[np.where(test_data['Brand_Lava']==1)[0]] = 1

test_data['brand_likeability'].loc[np.where(test_data['Brand_LG']==1)[0]] = 0.333

test_data['brand_likeability'].loc[np.where(test_data['Brand_LeEco']==1)[0]] = 1

test_data['brand_likeability'].loc[np.where(test_data['Brand_Lenovo']==1)[0]] = 0.6

test_data['brand_likeability'].loc[np.where(test_data['Brand_Meizu']==1)[0]] = 1

test_data['brand_likeability'].loc[np.where(test_data['Brand_Micromax']==1)[0]] = 0.2

test_data['brand_likeability'].loc[np.where(test_data['Brand_Mobiistar']==1)[0]] = 0.25

test_data['brand_likeability'].loc[np.where(test_data['Brand_Moto']==1)[0]] = 0.6667

test_data['brand_likeability'].loc[np.where(test_data['Brand_Motorola']==1)[0]] = 1

test_data['brand_likeability'].loc[np.where(test_data['Brand_Nokia']==1)[0]] = 0.619

test_data['brand_likeability'].loc[np.where(test_data['Brand_OPPO']==1)[0]] = 1

test_data['brand_likeability'].loc[np.where(test_data['Brand_OnePlus']==1)[0]] = 1

test_data['brand_likeability'].loc[np.where(test_data['Brand_Oppo']==1)[0]] = 0.833

test_data['brand_likeability'].loc[np.where(test_data['Brand_Panasonic']==1)[0]] = 0.5

test_data['brand_likeability'].loc[np.where(test_data['Brand_Realme']==1)[0]] = 1

test_data['brand_likeability'].loc[np.where(test_data['Brand_Samsung']==1)[0]] = 0.8636

test_data['brand_likeability'].loc[np.where(test_data['Brand_Sony']==1)[0]] = 0.444

test_data['brand_likeability'].loc[np.where(test_data['Brand_Tecno']==1)[0]] = 0.875

test_data['brand_likeability'].loc[np.where(test_data['Brand_Ulefone']==1)[0]] = 1

test_data['brand_likeability'].loc[np.where(test_data['Brand_Vivo']==1)[0]] = 0.923

test_data['brand_likeability'].loc[np.where(test_data['Brand_Xiaomi']==1)[0]] = 0.9047

test_data['brand_likeability'].loc[np.where(test_data['Brand_Xiaomi Poco']==1)[0]] = 1

test_data['brand_likeability'].loc[np.where(test_data['Brand_Yu']==1)[0]] = 0.25

test_data = test_data.drop(brand_features,axis=1)





test_data['numcores_likeability'] = np.zeros(test_data.shape[0])

test_data['numcores_likeability'].loc[np.where(test_data['Num_cores_312']==1)[0]] = 1

test_data['numcores_likeability'].loc[np.where(test_data['Num_cores_Deca']==1)[0]] = 0

test_data['numcores_likeability'].loc[np.where(test_data['Num_cores_Dual']==1)[0]] = 0.6

test_data['numcores_likeability'].loc[np.where(test_data['Num_cores_Hexa']==1)[0]] = 0.9167

test_data['numcores_likeability'].loc[np.where(test_data['Num_cores_Octa']==1)[0]] = 0.773

test_data['numcores_likeability'].loc[np.where(test_data['Num_cores_Other']==1)[0]] = 0.833

test_data['numcores_likeability'].loc[np.where(test_data['Num_cores_Quad']==1)[0]] = 0.5495

test_data['numcores_likeability'].loc[np.where(test_data['Num_cores_Tru-Octa']==1)[0]] = 1

test_data = test_data.drop(numcores_features, axis=1)



test_data['sim1_likeability'] = np.zeros(test_data.shape[0])

test_data['sim1_likeability'].loc[np.where(test_data['Sim1_2G']==1)[0]] = 0.857

test_data['sim1_likeability'].loc[np.where(test_data['Sim1_3G']==1)[0]] = 0.2857

test_data['sim1_likeability'].loc[np.where(test_data['Sim1_4G']==1)[0]] = 0.703

test_data = test_data.drop(sim1_features, axis = 1)





test_data['sim2_likeability'] = np.zeros(test_data.shape[0])

test_data['sim2_likeability'].loc[np.where(test_data['SIM 2_3G']==1)[0]] = 0.5714

test_data['sim2_likeability'].loc[np.where(test_data['SIM 2_2G']==1)[0]] = 0.5277

test_data['sim2_likeability'].loc[np.where(test_data['SIM 2_4G']==1)[0]] = 0.7424

test_data['sim2_likeability'].loc[np.where(test_data['SIM 2_Other']==1)[0]] = 0.79

test_data = test_data.drop(sim2_features, axis=1)





test_data['simslot_likeability'] = np.zeros(test_data.shape[0])

test_data['simslot_likeability'].loc[np.where(test_data['SIM Slot(s)_Dual SIM, GSM+GSM']==1)[0]] = 0.6409

test_data['simslot_likeability'].loc[np.where(test_data['SIM Slot(s)_Dual SIM, GSM+CDMA']==1)[0]] = 1

test_data['simslot_likeability'].loc[np.where(test_data['SIM Slot(s)_Dual SIM, GSM+GSM, Dual VoLTE']==1)[0]] = 0.923

test_data['simslot_likeability'].loc[np.where(test_data['SIM Slot(s)_Single SIM, GSM']==1)[0]] = 0.793

test_data = test_data.drop(simslot_features, axis=1)



test_data.head()
x_test = test_data.values

x_test.shape

x_train.shape
test_pred = perceptron.predict(x_test)

print(test_pred)
submission = pd.DataFrame({'PhoneId':test_new['PhoneId'], 'Class':test_pred})

submission = submission[['PhoneId', 'Class']]

submission.to_csv("submission.csv", index=False)