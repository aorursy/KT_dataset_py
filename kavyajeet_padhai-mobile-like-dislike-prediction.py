import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

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
# check the number of features and data points in train

print("Number of data points in train: %d" % train.shape[0])

print("Number of features in train: %d" % train.shape[1])



# check the number of features and data points in test

print("Number of data points in test: %d" % test.shape[0])

print("Number of features in test: %d" % test.shape[1])
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
train.head()

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

train_new.head()
# Binarising the output (ratings)

X_train_process = train_new.copy()

X_train_process['Rating'] = train_new['Rating'].map(lambda x: 0 if x<4 else 1)



# check the number of features and data points in train

print("Number of data points in train: %d" % train_new.shape[0])

print("Number of features in train: %d" % train_new.shape[1])



# check the number of features and data points in test

print("Number of data points in test: %d" % test_new.shape[0])

print("Number of features in test: %d" % test_new.shape[1])
X_train_0 = X_train_process[X_train_process['Rating']==0]

X_train_0 = fix_outliers_RAM(X_train_0,[46, 57, 74, 184, 297])

X_train_1 = X_train_process[X_train_process['Rating']==1]
column = 'Weight'

plt.figure(figsize=(15,10))

plt.boxplot(x=[X_train_0[column],X_train_1[column]],labels=['Disliked','Liked'])

plt.ylabel(column)

plt.show()

np.median(X_train_1[column])
## taking the data 

X_train = X_train_process.iloc[:,0:len(train_new.columns)-1]

Y_train = X_train_process.iloc[:,len(train_new.columns)-1]

X_train.head()
columns_to_remove = []

keys = ['os_name','Brand','Sim1','SIM 2','Num_cores','SIM Slot(s)']

for key in keys:

    for column in X_train_process.columns:

        if key in column:

            series = X_train_process[column]

            if np.sum(series) < 1:

                columns_to_remove.append(column)

columns_to_remove
# number of positive and negative examples

Y_train.value_counts()
# removing the variables with No correlation with rating

X_train_process.corr()[X_train_process.corr()['Rating'].isnull()].index
# removing the features PhoneID and Screen to Body Ratio



X_train_2 = X_train.drop(columns=['PhoneId','Screen Size'])

X_train_2 = X_train_2.drop(columns=columns_to_remove)
from scipy.stats import pearsonr

def return_high_corr(df,corr_threshold=0.70):

# check for correlation

    for column1 in df.columns[1:]:

        for column2 in df.columns[1:]:

            if column1!=column2:

                r = pearsonr(df[column1],df[column2])[0]

                if r > corr_threshold:

                    print(r,'|',column1,' vs ',column2)

                    

return_high_corr(X_train_2)
X_train_2 = X_train_2.drop(columns=['SIM Slot(s)_Single SIM, GSM'])
Y_train.value_counts()
# Removing the outliers/anomalies from each of the columns

# Anomalies can affect the binarisation process



from scipy.stats import zscore

def show_outliers(df):

    outliers = {}

    for c in df.columns:

        if df[c].dtype!=np.dtype('uint8'): 

            original= len(df)

            indices = df[zscore(df[c])>3].index

            if len(indices) > 0:

                outliers[c] = indices

    return outliers   



outliers =  show_outliers(X_train)

outliers
def fix_outliers_RAM(df,indices):

    df['RAM'][indices] =  df['RAM'][indices]/1000

    return df

X_train_3 = fix_outliers_RAM(X_train_2,[46, 57, 74, 184, 297])
# separating the data which are already binarised

plt.plot(X_train_3.T,'*')

plt.xticks(rotation='vertical')

plt.show()
def return_num(df):

    columns = [] 

    for column in df.columns:

        if df[column].dtype!=np.dtype('uint8'):

             columns.append(column)      

    return columns



num_columns = return_num(X_train_3)

num_columns
# Binarisation

def split_by_accuracy(df):

    

    splits = {}

    

    for column in df.columns:

        series = df[column]

        min_series = np.min(series)

        max_series = np.max(series)



        max_score = 0

        accuracies = {}

        best_split = 0



        for value in np.arange(min_series,max_series):

            split = series.map(lambda x: 0 if x<value else 1)

            score = accuracy_score(split,Y_train)

            if score > max_score:

                accuracies[score] = value

        max_score = max(accuracies.keys()) 

        best_split = accuracies[max_score]

        splits[column] = best_split

    return splits



def binarise_by_median(series,value=None):

    if value !=None:

        binary = series.map(lambda x: 0 if x < value else 1)

    else:

        cut = np.median(series)

        binary = series.map(lambda x: 0 if x < cut else 1)

    return binary



def binarise_by_mean(series,value=None):

    if value !=None:

        binary = series.map(lambda x: 0 if x < value else 1)

    else:

        cut = np.mean(series)

        binary = series.map(lambda x: 0 if x < cut else 1)

    return binary



X_train_4 = X_train_3.copy()

test_new=test_new[X_train_4.columns]

splits = split_by_accuracy(X_train_3[num_columns])



for column in X_train_4[num_columns].columns:

    X_train_4[column] = binarise_by_mean(X_train_4[column])

    test_new[column] = binarise_by_mean(test_new[column])

for column in X_train_4.columns:

    if X_train_4[column].dtype!=np.dtype('uint8'):

        print(column)

        print(X_train_4[column].value_counts())

        print()
X_train_3['Weight'].mean()


X_train_4['Capacity'] = X_train_3['Capacity'].map(lambda x: 0 if x<2500 else 1)

test_new['Capacity'] = test_new['Capacity'].map(lambda x: 0 if x<2500 else 1)



X_train_4['RAM'] = X_train_3['RAM'].map(lambda x: 0 if x<3 else 1)

test_new['RAM'] = test_new['RAM'].map(lambda x: 0 if x<3 else 1)



X_train_4['Weight'] = X_train_3['Weight'].map(lambda x: 0 if x>160 else 1)

test_new['Weight'] = test_new['Weight'].map(lambda x: 0 if x>160 else 1)



print(test_new['Capacity'].value_counts(),X_train_4['Capacity'].value_counts())

print(test_new['RAM'].value_counts(),X_train_4['RAM'].value_counts())

Y_train.value_counts()
X_train_4 = X_train_4.values
from sklearn.metrics import accuracy_score

class MPNeuron:

    def __init__(self):

        self.b = None

        

    def model(self,x):

        if (sum(x) >=self.b):

            return 1

        else:

            return 0



    def predict(self,X):

        Y_pred = []

        for x in X:

            Y_pred.append(self.model(x))

        return np.array(Y_pred)



    def fit(self,X,Y):

        accuracy = {}

        for b in range(0,len(X.T)+1):

            self.b = b

            y_pred = self.predict(X)

            accuracy[b] = accuracy_score(y_pred,Y)



        best_b = max(accuracy, key=accuracy.get)

        self.b = best_b

    

        print('optimal value of b is: '+str(best_b))

        print('Accuracy is: '+str(accuracy[best_b]))

        return accuracy

    

mp_neuron = MPNeuron()

accuracies = mp_neuron.fit(X_train_4,Y_train)

X_train_3.columns[0:mp_neuron.b+1]
test_new.head()
# removal of values from the test_set

Y_pred = mp_neuron.predict(test_new.values)

Y_pred
np.sum(Y_pred)/len(Y_pred)
submission = pd.DataFrame({'PhoneId':test['PhoneId'], 'Class':Y_pred})

submission = submission[['PhoneId', 'Class']]
submission.to_csv("submission.csv", index=False)