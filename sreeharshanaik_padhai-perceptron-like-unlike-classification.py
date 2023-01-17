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



import random



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
#Check the shape of the training data and testing data

train_new.shape, test_new.shape
train_new.head()

# There is a PhoneID column which is an indicator column and does not play any role decision making.

# We see categorical variables coded as dummy variables which is an important observation and will be 

# used later on in the model
test_new.head()

# There is a PhoneID column which is an indicator column and does not play any role decision making

# There is no Rating column in test_new since this you will have to predict by building a working model

# on the train_data
# Create X_train which will hold all columns except PhoneID and Rating using train_new

# Create Y_train which will only hold the Rating column present in train_new, note that the dataframe maintains integrity of the PhoneID 

# which is very essential



# Create X_test which will hold all columns except PhoneID

# There is no Y_test for obvious reasons as this is what you will be predicting 



X_train = train_new.drop(['PhoneId','Rating'],axis=1)

Y_train = train_new['Rating'].map(lambda x: 1 if x >= 4 else 0) # Notice that in the Perceptron model the output should be binary



X_test = test_new.drop(['PhoneId'],axis=1)
# Checking for correlation only for the first 10 discrete variables

X_train_corr = X_train.iloc[:,[0,1,2,3,4,5,6,7,8,9]]

X_train_corr.head()
correlations = X_train_corr.corr().abs().unstack().sort_values(kind="quicksort").reset_index()

correlations = correlations[correlations['level_0'] != correlations['level_1']]



correlations.tail(10)



# The tail of the dataframe has the information about highest correlation between features,

# where as the head has details about features that have least correlation



# Observe that Screen Size and Screen to Body Ratio are highly correlated, We don't know if they are

# positively or negatively correlated yet. We will find that out visually

# Similarly Height and Screen Size are correlated.



# Intution tells me that they will be positvely correlated, meaning any increase in Height of the 

# phone will result in an increase of the Screen Size and vice versa. 



#A bigger phone screen (Screen Size) means the phone is lenghtier (Height). 
corr = X_train.corr()

fig = plt.figure()

fig.set_size_inches(20,20)

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(X_train.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(X_train.columns)

ax.set_yticklabels(X_train.columns)

plt.show()



# We can clearly see that Screen Size is positively correlated with Height and Screen to body Ratio

# There are other positively correlated variables too. Another good easy to understand example is 

# Brand_Apple and os_name_iOS.



# Look against the line of 'Brand_Apple' and compare it with all the top columns 

# It's obvious isn't it that an Apple iPhone / product user will have an iOS operating 

# system on his/her device. He/she cannot have an Android on his Apple iPhone. 

# This now introduces to you negative correlation. Notice how Brand_Apple and os_name_Andriod are

# negatively correlated (dark blue), which means Apple folks cannot have Android OS
X_train = np.array([

X_train['Weight'], 

#X_train['Height'],# Column removed due to correlation

X_train['Screen to Body Ratio (calculated)'],

X_train['Pixel Density'], 

X_train['Processor_frequency'], 

#X_train['Screen Size'],# Column removed due to correlation

X_train['RAM'], 

X_train['Resolution'], 

X_train['Internal Memory'], 

#X_train['Capacity'],# Column removed due to correlation

X_train['Brand_10.or'],

X_train['Brand_Apple'],

X_train['Brand_Asus'], 

#X_train['Brand_Billion'], # No Rows found, Removed to increase train accuracy due to scope of assignment

X_train['Brand_Blackberry'], 

X_train['Brand_Comio'], 

X_train['Brand_Coolpad'], 

#X_train['Brand_Do'], # No Rows found, Removed to increase train accuracy due to scope of assignment

X_train['Brand_Gionee'], 

X_train['Brand_Google'], 

X_train['Brand_HTC'], 

X_train['Brand_Honor'], 

X_train['Brand_Huawei'], 

X_train['Brand_InFocus'], 

X_train['Brand_Infinix'], 

X_train['Brand_Intex'], 

X_train['Brand_Itel'], 

X_train['Brand_Jivi'], 

X_train['Brand_Karbonn'], 

X_train['Brand_LG'], 

X_train['Brand_Lava'], 

X_train['Brand_LeEco'], 

X_train['Brand_Lenovo'], 

X_train['Brand_Lephone'], 

X_train['Brand_Lyf'], 

X_train['Brand_Meizu'], 

X_train['Brand_Micromax'], 

X_train['Brand_Mobiistar'], 

X_train['Brand_Moto'], 

X_train['Brand_Motorola'], 

X_train['Brand_Nokia'], 

X_train['Brand_Nubia'], 

X_train['Brand_OPPO'], 

X_train['Brand_OnePlus'],

X_train['Brand_Oppo'], 

X_train['Brand_Panasonic'], 

X_train['Brand_Razer'], 

X_train['Brand_Realme'], 

#X_train['Brand_Reliance'],# Removed due to correlation 

X_train['Brand_Samsung'], 

X_train['Brand_Sony'], 

#X_train['Brand_Spice'],# No Rows found, Removed to increase train accuracy due to scope of assignment 

X_train['Brand_Tecno'], 

X_train['Brand_Ulefone'], 

X_train['Brand_VOTO'], 

X_train['Brand_Vivo'], 

X_train['Brand_Xiaomi'], 

X_train['Brand_Xiaomi Poco'], 

X_train['Brand_Yu'], 

X_train['Brand_iVooMi'], 

X_train['SIM Slot(s)_Dual SIM, GSM+CDMA'], 

X_train['SIM Slot(s)_Dual SIM, GSM+GSM'], 

X_train['SIM Slot(s)_Dual SIM, GSM+GSM, Dual VoLTE'], 

# X_train['SIM Slot(s)_Single SIM, GSM'],# Removed due to correlation

X_train['Num_cores_312'], 

X_train['Num_cores_Deca'], 

X_train['Num_cores_Dual'], 

X_train['Num_cores_Hexa'], 

X_train['Num_cores_Octa'], 

X_train['Num_cores_Other'],# Food for thought column - Retained to prove correlation theory, If I remove this column my 71% data gives 

X_train['Num_cores_Quad'],# 85% accuracy compared to my current 83.33 % accuracy. I sacrificed my leaderboard rank for this !!

X_train['Num_cores_Tru-Octa'], 

#X_train['Sim1_2G'],# Column removed due to correlation

X_train['Sim1_3G'], 

#X_train['Sim1_4G'],# Column removed due to correlation

X_train['SIM 2_2G'], 

X_train['SIM 2_3G'], # Food for thought columns - What happens if I remove this and retain SIM 2_4G ?  

X_train['SIM 2_4G'], # Food for thought columns - What happens if I remove this and retain SIM 2_3G ?  

X_train['SIM 2_Other'], 

X_train['os_name_Android'], 

#X_train['os_name_Blackberry'], # Removed due to correlation  

#X_train['os_name_KAI'], # Removed due to correlation  

X_train['os_name_Nokia'], 

#X_train['os_name_Other'],# Removed due to correlation

X_train['os_name_Tizen'],  

#X_train['os_name_iOS'],# Removed due to correlation

    

]) 



X_train = X_train.T
X_test = np.array([

X_test['Weight'], 

#X_test['Height'],# Column removed due to correlation

X_test['Screen to Body Ratio (calculated)'], 

X_test['Pixel Density'], 

X_test['Processor_frequency'], 

#X_test['Screen Size'],# Column removed due to correlation 

X_test['RAM'], 

X_test['Resolution'], 

X_test['Internal Memory'], 

#X_test['Capacity'], # Column removed due to correlation

X_test['Brand_10.or'],

X_test['Brand_Apple'],

X_test['Brand_Asus'], 

#X_test['Brand_Billion'], # No Rows found, Removed to increase train accuracy due to scope of assignment

X_test['Brand_Blackberry'], 

X_test['Brand_Comio'], 

X_test['Brand_Coolpad'], 

#X_test['Brand_Do'], # No Rows found, Removed to increase train accuracy due to scope of assignment

X_test['Brand_Gionee'], 

X_test['Brand_Google'], 

X_test['Brand_HTC'], 

X_test['Brand_Honor'], 

X_test['Brand_Huawei'], 

X_test['Brand_InFocus'], 

X_test['Brand_Infinix'], 

X_test['Brand_Intex'], 

X_test['Brand_Itel'], 

X_test['Brand_Jivi'], 

X_test['Brand_Karbonn'], 

X_test['Brand_LG'], 

X_test['Brand_Lava'], 

X_test['Brand_LeEco'], 

X_test['Brand_Lenovo'], 

X_test['Brand_Lephone'], 

X_test['Brand_Lyf'], 

X_test['Brand_Meizu'], 

X_test['Brand_Micromax'], 

X_test['Brand_Mobiistar'], 

X_test['Brand_Moto'], 

X_test['Brand_Motorola'], 

X_test['Brand_Nokia'], 

X_test['Brand_Nubia'], 

X_test['Brand_OPPO'], 

X_test['Brand_OnePlus'],

X_test['Brand_Oppo'], 

X_test['Brand_Panasonic'], 

X_test['Brand_Razer'], 

X_test['Brand_Realme'], 

#X_test['Brand_Reliance'],# Removed due to correlation 

X_test['Brand_Samsung'], 

X_test['Brand_Sony'], 

#X_test['Brand_Spice'],# No Rows found, Removed to increase train accuracy due to scope of assignment 

X_test['Brand_Tecno'], 

X_test['Brand_Ulefone'], 

X_test['Brand_VOTO'], 

X_test['Brand_Vivo'], 

X_test['Brand_Xiaomi'], 

X_test['Brand_Xiaomi Poco'], 

X_test['Brand_Yu'], 

X_test['Brand_iVooMi'], 

X_test['SIM Slot(s)_Dual SIM, GSM+CDMA'], 

X_test['SIM Slot(s)_Dual SIM, GSM+GSM'], 

X_test['SIM Slot(s)_Dual SIM, GSM+GSM, Dual VoLTE'], 

# X_test['SIM Slot(s)_Single SIM, GSM'],# Removed due to correlation

X_test['Num_cores_312'], 

X_test['Num_cores_Deca'], 

X_test['Num_cores_Dual'], 

X_test['Num_cores_Hexa'], 

X_test['Num_cores_Octa'], 

X_test['Num_cores_Other'],# Food for thought column - Retained to prove correlation theory, If I remove this column my 71% data gives 

X_test['Num_cores_Quad'],# 85% accuracy compared to my current 83.33 % accuracy. I sacrificed my leaderboard rank for this !!

X_test['Num_cores_Tru-Octa'], 

#X_test['Sim1_2G'],# Column removed due to correlation

X_test['Sim1_3G'], 

#X_test['Sim1_4G'],# Column removed due to correlation

X_test['SIM 2_2G'], 

X_test['SIM 2_3G'], # Food for thought columns - What happens if I remove this and retain SIM 2_4G ?  

X_test['SIM 2_4G'], # Food for thought columns - What happens if I remove this and retain SIM 2_3G ?  

X_test['SIM 2_Other'], 

X_test['os_name_Android'], 

#X_test['os_name_Blackberry'], # Removed due to correlation  

# X_test['os_name_KAI'], # Removed due to correlation  

X_test['os_name_Nokia'], 

#X_test['os_name_Other'],# Removed due to correlation

X_test['os_name_Tizen'],  

#X_test['os_name_iOS'],# Removed due to correlation

    

]) 



X_test = X_test.T
from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

print(scaler.fit(X_train))

MinMaxScaler(copy=True, feature_range=(0, 1))



X_train = scaler.transform(X_train)



scaler = MinMaxScaler()

print(scaler.fit(X_test))

MinMaxScaler(copy=True, feature_range=(0, 1))



X_test = scaler.transform(X_test)
X_train[0:1]
class Perceptron:

    def __init__ (self):

        self.w = None

        self.b = None

 

    def model(self, x):

        return 1 if (np.dot(self.w, x) >= self.b) else 0

    

    def predict(self,X):

        Y = []

        for x in X:

            result=self.model(x)

            Y.append(result)

        return np.array(Y)

    

    def fit(self,X, Y, epochs = 1, lr = 1):

        #Weights = 0

        self.w = np.ones(X.shape[1]) #np.random.rand(73)

        self.b = 0 #random.randint(0,1)

        

        accuracy = {}

        max_accuracy = 0

        

        for i in range(epochs):

            for x,y in zip(X, Y):

                y_pred = self.model(x)

                if y==1 and y_pred ==0 :

                    self.w = self.w + lr * x

                    self.b = self.b - lr * 1

                #elif y==1 and y_pred == 1:

                    #self.w = self.w

                    #self.b = self.b

                #elif y==0 and y_pred == 0:

                    #self.w = self.w

                    #self.b = self.b

                elif y==0 and y_pred ==1:

                    self.w = self.w - lr * x

                    self.b = self.b + lr * 1

            accuracy[i] = accuracy_score(self.predict(X),Y)

            if (accuracy[i] > max_accuracy):

                max_accuracy = accuracy[i]

                chkptw = self.w

                chkptb = self.b

            #    self.var = self.var - 1

            #elif (accuracy[i] < max_accuracy):

            #    self.var = self.var + 1

                

        self.w = chkptw

        self.b = chkptb

        

        print(max_accuracy)

        #print(self.var)

        print(self.w)

        plt.plot(accuracy.values())

        plt.ylim([0,1])

        plt.show()
perceptron = Perceptron()
perceptron.fit(X_train,Y_train,10000,.01)
plt.plot(perceptron.w)

plt.show()
Y_pred_train = perceptron.predict(X_train)
Y_pred_test = perceptron.predict(X_test)

Y_pred_test = list(Y_pred_test)

print(Y_pred_test)
submission = pd.DataFrame({'PhoneId':test_new['PhoneId'], 'Class':Y_pred_test})

submission = submission[['PhoneId', 'Class']]

submission.head()
submission.to_csv("submission.csv", index=False)