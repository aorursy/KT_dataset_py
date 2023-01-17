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
train_new = train_new.drop(columns=['PhoneId','Screen Size','Height'])

test_new = test_new.drop(columns=['Screen Size','Height'])

train_new.head()
# return the numerical columns only (!= catergorical columns)

def return_num_columns(df):

    num_columns = []

    for c in df.columns:

        if df[c].dtypes != np.dtype('uint8'):

            num_columns.append(c)

    return num_columns



num_columns = return_num_columns(train_new.iloc[:,0:len(train_new.T)-1])
# 1. find highly correlated variables

category_null = train_new.corr()[train_new.corr()['Rating'].isnull()].index



print("Variables with no correlation with rating")

print(category_null)
# remove variables with no correlation

train_new = train_new.drop(columns=category_null)

test_new = test_new.drop(columns=category_null)
# since all RAM values are in GB hence fixing the RAM >= 16 to MB by dividing it by 1000

def fix_outliers_RAM(df,indices):

    df['RAM'][indices] =  df['RAM'][indices]/1000

    return df

indices = train_new[train_new['RAM'] >= 16].index

train_new = fix_outliers_RAM(train_new,indices)
# removing variables with very less significant weight

#columns_less_wt = ['Brand_LG','Num_cores_Hexa','Num_cores_Octa']

#train_new = train_new.drop(columns=columns_less_wt)

#test_new = test_new.drop(columns=columns_less_wt)
# 1. Mapping the rating column into binary form

X_train_process = train_new.copy()

X_train_process['Rating'] = train_new['Rating'].map(lambda x: int(0) if x<4 else int(1))



# 2. extract the variables

m,n = X_train_process.shape

# not considering the first column i.e. the phoneID column

X_train = X_train_process.iloc[:,0:n-1].values

Y_train = X_train_process.iloc[:,n-1].values



#from sklearn.model_selection import train_test_split

#X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)
# From test_new, we have to remove the PhoneId while using it for predicting

np.shape(X_train),np.shape(Y_train),np.shape(test_new)
train_new.head()
# Standardizing only the numerical data

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()



def standardize_data(series):

    min_series = np.min(series)

    max_series = np.max(series)

    converted = []

    for x in series:

        value = (x-min_series)/(max_series-min_series)

        converted.append(value)

    return converted



def transform_data(X,split = len(num_columns)):

    # Categorical Data

    X_cat = X[:,split:]

    # Numerical Data

    X_num = X[:,0:split]

    # Standardizing the numerical data

    for i,series in enumerate(X_num.T):

        X_num[:,i] = standardize_data(series)

        

    # Concatenating the data

    X = np.concatenate((X_num,X_cat),axis=1)

    return X



# Standardizing the numerical data 

X_train = transform_data(X_train)

#X_test = transform_data(X_test)



np.shape(X_train)
class Perceptron:

    def __init__(self):

        self.w = None

        self.b = None

        self.accuracies = None

        self.max_accuracy = None



    def model(self,x):    

        value = np.dot(x,self.w)

        if (value)>=self.b:

            return 1

        else:

            return 0



    def predict(self,X):

        Y_pred = []

        for x in X:

            pred = self.model(x)

            Y_pred.append(pred)

        return np.array(Y_pred)

    

    def fit(self,X,Y,epochs=1,lr=1):



        # initial assumption

        self.w = np.ones(X.shape[1])

        self.b = 0

        accuracies = {}

        max_accuracy = 0    

        wt_matrix = []

        

        for i in range(epochs):

            for x,y in zip(X,Y):

                y_pred = self.model(x)

                if y==1 and y_pred==0:

                    self.w += lr*x

                    self.b += lr*1

                if y==0 and y_pred==1:

                    self.w -= lr*x

                    self.b -= lr*1

     

            wt_matrix.append(np.array(self.w))

            Y_pred = self.predict(X)

            accuracies[i] = accuracy_score(Y_pred,Y)

            

            if (accuracies[i] > max_accuracy):

                max_accuracy = accuracies[i]

                # checkpointing - whenever there is highest accuracy the iteration 

                # will store the parameters otherwise it will ignore

                chkptw = self.w

                chkptb = self.b



            self.w = chkptw

            self.b = chkptb



        self.accuracies = accuracies

        self.max_accuracy = max_accuracy

    

        return np.array(wt_matrix)
perceptron = Perceptron()
wts = perceptron.fit(X_train,Y_train,epochs=400,lr=0.2)

perceptron.max_accuracy
columns = train_new.columns[0:]

plot = {}

plt.figure(figsize=(25,10))

for i,value in enumerate(perceptron.w):

    if np.abs(value)<100:

        plot[columns[i]] = value

plt.bar(plot.keys(),plot.values())

plt.xticks(rotation='vertical')

plt.show()
plt.figure(figsize=(25,10))

plt.plot(perceptron.accuracies.keys(),perceptron.accuracies.values())

plt.ylim(0,1)

plt.show()
plt.figure(figsize=(25,10))

plt.plot(perceptron.w)

plt.show()
X_test = test_new.drop(columns=['PhoneId']).values

X_test = transform_data(X_test)

np.shape(X_test)
Y_pred = perceptron.predict(X_test)

Y_pred
submission = pd.DataFrame({'PhoneId':test_new['PhoneId'], 'Class':Y_pred})

submission = submission[['PhoneId', 'Class']]

submission.head()
submission.to_csv("submission.csv", index=False)