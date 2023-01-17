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
import pandas as pd

import matplotlib.pyplot as plt



df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
# remove column

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
import numpy as np

df_train = data_clean(df_train)

df_test = data_clean(df_test)

df_train.head(1)

df_test.head(1)



# checking which column is not in l2

l3 = [a for a in list(set(df_train.columns)) if a not in list(set(df_test.columns))]

l3
# if no of feature is not avaiale more 15, then delete that row

df_train = df_train.dropna(thresh=15)
# check the number of features and data points in train

print("Number of data points in train: %d" % df_train.shape[0])

print("Number of features in train: %d" % df_train.shape[1])



# check the number of features and data points in test

print("Number of data points in test: %d" % df_test.shape[0])

print("Number of features in test: %d" % df_test.shape[1])
# missing value treatment

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
# apply missing value treatment

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
df_train = data_clean_2(df_train)

df_test = data_clean_2(df_test)



# check the number of features and data points in train

print("Number of data points in train: %d" % df_train.shape[0])

print("Number of features in train: %d" % df_train.shape[1])



# check the number of features and data points in test

print("Number of data points in test: %d" % df_test.shape[0])

print("Number of features in test: %d" % df_test.shape[1])
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
df_train = data_clean_3(df_train)

df_test = data_clean_3(df_test)



# check the number of features and data points in train

print("Number of data points in train: %d" % df_train.shape[0])

print("Number of features in train: %d" % df_train.shape[1])



# check the number of features and data points in test

print("Number of data points in test: %d" % df_test.shape[0])

print("Number of features in test: %d" % df_test.shape[1])
# one hot encoding



train_ids = df_train['PhoneId']

test_ids = df_test['PhoneId']



cols = list(df_test.columns)

cols.remove('PhoneId')

cols.insert(0, 'PhoneId')



combined = pd.concat([df_train.drop('Rating', axis=1)[cols], df_test[cols]])

print(combined.shape)

print(combined.columns)



combined = pd.get_dummies(combined)

print(combined.shape)

print(combined.columns)



train_new = combined[combined['PhoneId'].isin(train_ids)]

test_new = combined[combined['PhoneId'].isin(test_ids)]
train_new = train_new.merge(df_train[['PhoneId', 'Rating']], on='PhoneId')
# check the number of features and data points in train

print("Number of data points in train: %d" % train_new.shape[0])

print("Number of features in train: %d" % train_new.shape[1])



# check the number of features and data points in test

print("Number of data points in test: %d" % test_new.shape[0])

print("Number of features in test: %d" % test_new.shape[1])
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(train_new, train_new['Rating'], test_size=0.30, random_state=1)
X_binarised_train = X_train.apply(pd.cut, bins=2, labels=[1,0])

X_binarised_test = X_test.apply(pd.cut, bins=2, labels=[1,0])



# binarised test_new

X_binarised_test_New = test_new.apply(pd.cut, bins=2, labels=[1,0])
Y_binarised_train = Y_train.map(lambda x: 0 if x < 4 else 1)

Y_binarised_test = Y_test.map(lambda x: 0 if x < 4 else 1)
X_binarised_test_New['Rating'] = np.zeros(X_binarised_test_New.shape[0], dtype=np.int)
X_binarised_test = X_binarised_test.values

X_binarised_train = X_binarised_train.values

Y_binarised_train = Y_binarised_train.values

Y_binarised_test = Y_binarised_test.values



X_binarised_test_New = X_binarised_test_New.values
from sklearn.metrics import accuracy_score
# Perceptron class

class Perceptron:

  

  def __init__ (self):

    self.w = None

    self.b = None

    

  def model(self, x):

    #print(self.w.shape, ' ', x.shape)

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

wt_matrix = perceptron.fit(X_binarised_train, Y_binarised_train, 10000, 0.01)
Y_pred_test = perceptron.predict(X_binarised_test)

print(accuracy_score(Y_pred_test, Y_binarised_test))
Y_test_pred = perceptron.predict(X_binarised_test_New)

Y_test_pred
unique, counts = np.unique(Y_test_pred, return_counts=True)

print(unique, counts)
submission = pd.DataFrame({'PhoneId':test_new['PhoneId'], 'Class':np.array(Y_test_pred, dtype=int)})

submission = submission[['PhoneId', 'Class']]

submission.head(10)
submission.to_csv("submission.csv", index=False)