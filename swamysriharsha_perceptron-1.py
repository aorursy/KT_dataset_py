import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder,MinMaxScaler, StandardScaler, MaxAbsScaler

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
train_new['Rating'] = train_new['Rating'].apply(lambda x: 1 if x>=THRESHOLD else 0)
train_new.head()
test_new.head()
import seaborn as sns
ax = sns.scatterplot(x="Capacity", y="Pixel Density", hue="Rating", data=train_new)
train_new = train_new[train_new['Capacity']<6000]
ax = sns.scatterplot(x="Screen to Body Ratio (calculated)", y="Pixel Density", hue="Rating", data=train_new)
train_new = train_new[train_new['Screen to Body Ratio (calculated)']>50]
train_new.shape
ax = sns.scatterplot(x="Weight", y="Pixel Density", hue="Rating", data=train_new)
train_new['Weight'][train_new['Weight']<120].value_counts()
train_new = train_new[train_new['Weight']>=120]
train_new.shape
ax = sns.scatterplot(x="Internal Memory", y="Pixel Density", hue="Rating", data=train_new)
train_new['Internal Memory'][train_new['Internal Memory']>256].value_counts()
train_new = train_new[train_new['Internal Memory']<=256]
train_new.shape
ax = sns.scatterplot(x="Height", y="Pixel Density", hue="Rating", data=train_new)
train_new = train_new[train_new['Height']<=170]
train_new.shape
ax = sns.scatterplot(x="RAM", y="Pixel Density", hue="Rating", data=train_new)
train_new = train_new[train_new['RAM']<100]
train_new.shape
train_new.head()
from sklearn import preprocessing
scaler = MinMaxScaler()

X = train_new.drop('Rating',axis=1)

y = train_new['Rating']
print(X.shape, y.shape, test_new.shape)
X.shape
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek



smt = SMOTETomek(ratio='auto')

X_smt, y_smt = smt.fit_sample(X, y)
X_train, X_test, Y_train, Y_test = train_test_split(X_smt, y_smt, test_size=0.05, stratify = y_smt, random_state=1)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
X_train = np.delete(X_train, 63, 1)

X_test = np.delete(X_test, 63, 1)
X_train[:,63]
#X_train = X_train.values

#X_test = X_test.values
X_train = X_train[:,1:]

X_test = X_test[:,1:]
#X_train = preprocessing.normalize(X_train, norm='l2')

#X_test = preprocessing.normalize(X_test, norm='l2')
scaler.fit(X_train)

scaler.fit(X_test)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
import matplotlib.pyplot as plt
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

    

    #self.w = np.zeros(X.shape[1])

    mu, sigma = 0, 0.3 # mean and standard deviation

    self.w = np.random.normal(mu, sigma, X.shape[1])

    #self.w = np.random.uniform(-1,0,X.shape[1])

    self.b = 0.2

    

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

    

      if i%10000==0:

        print("epoch: {0} and Accuracy: {1}".format(i,accuracy[i]))

            

      if (accuracy[i] > max_accuracy):

        max_accuracy = accuracy[i]

        chkptw = self.w

        chkptb = self.b

        chkpti = i

        

    self.w = chkptw

    self.b = chkptb

        

    print(max_accuracy, "epoch ",chkpti)

    

    plt.plot(accuracy.values())

    plt.ylim([0, 1])

    plt.show()

    

    return np.array(wt_matrix)
perceptron = Perceptron()
wt_matrix = perceptron.fit(X_train, Y_train, 200000, 0.00001)
Y_pred_test = perceptron.predict(X_test)

print(accuracy_score(Y_pred_test, Y_test))
plt.plot(wt_matrix[144283,:])

plt.show()
newarr = wt_matrix[144283,:].reshape([-1,1])

minw = newarr[0]

for i in range(1,len(newarr)):

    if newarr[i]<minw:

        mini = i

        minw = newarr[i]

print(mini, newarr[i])        
X_val = test_new.drop('PhoneId',axis=1)
#X_val = preprocessing.normalize(X_val, norm='l2')
scaler.fit(X_val)
X_val = scaler.transform(X_val)
X_val.shape
type(X_val)
X_val = np.delete(X_val, 63, 1)
Y_val = perceptron.predict(X_val)
Y_val
submission = pd.DataFrame({'PhoneId':test_new['PhoneId'], 'Class':Y_val})

submission = submission[['PhoneId', 'Class']]

submission.head()
submission.to_csv("submission.csv", index=False)
"""

WRITE YOUR MODELLING CODE HERE

"""