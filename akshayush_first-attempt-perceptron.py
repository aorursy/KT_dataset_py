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
#train.head()

combined.head()

#test[cols]

#cols
train_new = train_new.merge(train[['PhoneId', 'Rating']], on='PhoneId')
# check the number of features and data points in train

print("Number of data points in train: %d" % train_new.shape[0])

print("Number of features in train: %d" % train_new.shape[1])



# check the number of features and data points in test

print("Number of data points in test: %d" % test_new.shape[0])

print("Number of features in test: %d" % test_new.shape[1])
train_new.head()
test_new.head()

type(test_new)
#submission = pd.DataFrame({'PhoneId':test_new['PhoneId'], 'Class':[0]*test_new.shape[0]})

#submission = submission[['PhoneId', 'Class']]

#submission.head()
#submission.to_csv("submission.csv", index=False)
import math
class Perceptron:

  def __init__(self):

    self.w=None

    self.b=None

    

  

  def model(self,x):

    #return self.sigmoid(np.dot(self.w,x))

    return 1 if np.dot(self.w,x) >= self.b else 0

  

   

  

  def predict(self,x):

    y=[]

    for x_ in x:

      result=self.model(x_)

      y.append(result)

    return np.array(y)

   

  def fit(self,x,y,epochs=2,lr=1,seed=3):

    self.w=np.ones(x.shape[1])

    #np.random.seed(seed)

    #self.w=np.random.rand(1,x.shape[1])

    #self.b=np.random.rand(1)

    self.b=0

    accuracy={}

    max_accuracy=0

    wt_matrix=[]

    for i in range(epochs):

      for x_,y_ in zip(x,y):

        y_pred=self.model(x_)

        #error=y_-y_pred

        #adjustment=np.dot(x_.T, error**2 * self.sigmoid_derivative(y_pred))

        

        if y_==1 and y_pred==0:

          self.w=self.w+ lr * x_

          self.b=self.b+ lr * 1

        if y_==0 and y_pred==1:

          self.w=self.w- lr * x_

          self.b=self.b- lr * 1

      wt_matrix.append(self.w)

      accuracy[i]=accuracy_score(self.predict(x),y)

      #print(accuracy[i])

      if accuracy[i]>=max_accuracy:

        max_accuracy=accuracy[i]

        #print(max_accuracy)

        chktpw=self.w

        chktpb=self.b

    self.w=chktpw

    self.b=chktpb

    #plt.plot(accuracy.values())

    #plt.ylim([0,1])

    #plt.show

    print(max_accuracy)

    #return np.array(wt_matrix)
np.random.rand(1,test_new.shape[1])

np.random.rand(1)

np.random.seed(2)
#train_new.columns

print(test_new.shape)

print(train_new.shape)



#train_new.head()



cols_to_remove=['PhoneId','Rating']



train_new['label']=np.where(train_new['Rating']>=THRESHOLD,1,0)

train_columns=train_new.columns

feature_col=list(set(train_columns)-set(cols_to_remove))

label_col='label'

print(feature_col)

print(label_col)

#train_new.groupby('label').count()

x_train=train_new.copy()

#x_train,x_test=train_test_split(train_new,test_size=0.01,stratify=train_new['label'])

feature_col_test=list(set(test_new.columns)-set(['PhoneId']))

print(test_new.shape)
from sklearn.preprocessing import LabelEncoder,MinMaxScaler, StandardScaler,Normalizer,MaxAbsScaler

#scalar_test=MinMaxScaler()

scalar_test=MaxAbsScaler()

test_new_scaled=scalar_test.fit_transform(test_new[feature_col_test])

#test_new_scaled=test_new[feature_col_test].values

test_new.shape

test_new_scaled.shape





#scalar=MinMaxScaler() #0.8137254901960784

scalar=StandardScaler() #0.5261437908496732

scalar=MaxAbsScaler() 

#scalar=Normalizer()

train_new_scaled=scalar.fit_transform(x_train[feature_col_test])

#train_new_scaled=x_train[feature_col_test].values

#test_new_scaled=scalar.fit_transform(x_test[feature_col])

#print(feature_col)

#train_new_scaled.shape

train_new_scaled


y_train_new_scaled=x_train[label_col].values

perceptron=Perceptron()

#perceptron.fit(train_new_scaled,y_train_new_scaled,epochs=10000,lr=0.0001) #0.8137254901960784

#perceptron.fit(train_new_scaled,y_train_new_scaled,epochs=100,lr=0.001)

#e=[100,1000,3000,8000]

e=[8000]

#l=[0.1,0.3,0.5,0.8,0.01,0.05,0.08,0.001,0.002,0.003,0.005,0.008]

l=[0.008]

for epcoh in e:

    for lr in l:

        print('epcoh is {0} and learning rate is {1}'.format(epcoh,lr))

        perceptron.fit(train_new_scaled,y_train_new_scaled,epochs=epcoh,lr=lr)
test_pred=perceptron.predict(test_new_scaled)



print(test_pred.astype('int'))

submission_1=pd.DataFrame(test_pred.astype('int').tolist(),columns=["Class"])

submission_1['PhoneId']=test_new[['PhoneId']]

submission=submission_1[['PhoneId','Class']]



submission.to_csv("submission.csv", index=False)

'''

#y_test_new=x_test[label_col].values

print(len(test_new_scaled))

y_pred_test=perceptron.predict(test_new_scaled)

#print('Accuracy score is',accuracy_score(y_pred_test,y_test_new))

x_test['Class']=y_pred_test

x_test.head()

'''
