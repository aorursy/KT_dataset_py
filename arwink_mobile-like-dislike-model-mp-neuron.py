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

import matplotlib

%matplotlib inline
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

#combined = combined.drop(['Brand'],axis=1)

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
# Create X_train_new which will hold all columns except PhoneID and Rating using train_new

# Create Y_train_new which will only hold the Rating column present in train_new, note that the dataframe maintains integrity of the PhoneID 

# which is very essential



# Create X_test_new which will hold all columns except PhoneID

# There is no Y_test_new for obvious reasons as this is what you will be predicting 



X_train_new = train_new.drop(['PhoneId','Rating'],axis=1)

Y_train_new_rating_discrete = train_new['Rating'] 

Y_train_new_rating_binary   = train_new['Rating'].map(lambda x: 1 if x >= 4 else 0)



X_test_new = test_new.drop(['PhoneId'],axis=1)
X_train_new.describe()
X_test_new.describe()
# Checking for correlation only for the first 10 discrete variables

X_train_corr = X_train_new.iloc[:,[0,1,2,3,4,5,6,7,8,9]]

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
corr = X_train_new.corr()

fig = plt.figure()

fig.set_size_inches(20,20)

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(X_train_new.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(X_train_new.columns)

ax.set_yticklabels(X_train_new.columns)

plt.show()



# We can clearly see that Screen Size is positively correlated with Height and Screen to body Ratio

# There are other positively correlated variables too. Another good easy to understand example is 

# Brand_Apple and os_name_iOS.



# Look against the line of 'Brand_Apple' and compare it with all the top columns 

# It's obvious isn't it that an Apple iPhone / product user will have an iOS operating 

# system on his/her device. He/she cannot have an Android on his Apple iPhone. 

# This now introduces to you negative correlation. Notice how Brand_Apple and os_name_Andriod are

# negatively correlated (dark blue), which means Apple folks cannot have Android OS
#Assigning a dataframe to capture visuals on the discrete data, note that how we have two dataframes where the Y_train_new has been desgined to 

#hold discrete value as well as binary value for the ease of interpretation

X_train_visual_rating_binary = pd.concat([X_train_new,Y_train_new_rating_binary],axis=1)

X_train_visual_rating_discrete = pd.concat([X_train_new,Y_train_new_rating_discrete],axis=1)
import seaborn as sns
plt.figure(figsize=(14,8))

sns.barplot(x='RAM',y='Capacity',data=X_train_visual_rating_binary)
plt.figure(figsize=(14,8))

sns.scatterplot(x='RAM',y='Capacity',hue='Rating',data=X_train_visual_rating_binary)
plt.figure(figsize=(14,8))

plt.xticks(rotation='vertical')

sns.barplot(x='Screen Size',y='Pixel Density',data=X_train_visual_rating_binary)
plt.figure(figsize=(14,8))

sns.scatterplot(x='Screen Size',y='Pixel Density',hue='Rating',data=X_train_visual_rating_binary)
plt.figure(figsize=(14,8))

plt.xticks(rotation='vertical')

sns.barplot(x='Resolution',y='Pixel Density',data=X_train_visual_rating_binary)
plt.figure(figsize=(14,8))

sns.scatterplot(x='Resolution',y='Pixel Density',hue='Rating',data=X_train_visual_rating_binary)
plt.figure(figsize=(14,8))

sns.scatterplot(x='Internal Memory',y='Weight',hue='Rating',data=X_train_visual_rating_binary)
plt.figure(figsize=(14,8))

plt.xticks(rotation='vertical')

sns.barplot(x='Capacity',y='Weight',data=X_train_visual_rating_binary)

plt.figure(figsize=(14,8))

sns.scatterplot(x='Rating',y='Brand_Apple',hue='Rating',data=X_train_visual_rating_discrete)
plt.figure(figsize=(14,8))

sns.scatterplot(x='Rating',y='Brand_Blackberry',hue='Rating',data=X_train_visual_rating_discrete)


X_train_binarised = np.array([

#X_train_new['Weight'].map(lambda x: 1 if x > 153 else 0), # Removed due to correlation and accuracy increased

#X_train_new['Height'].map(lambda x: 1 if x > 151 else 0), # Removed due to correlation and accuracy increased

#X_train_new['Screen to Body Ratio (calculated)'].map(lambda x: 1 if x >= 55  else 0), # Removed due to correlation and accuracy increased

X_train_new['Pixel Density'].map(lambda x: 1 if x > 270 else 0),

X_train_new['Processor_frequency'].map(lambda x: 1 if x > 2.1 else 0),

X_train_new['Screen Size'].map(lambda x: 1 if x > 4.8 else 0),

X_train_new['RAM'].map(lambda x: 1 if x > 4 else 0),

#X_train_new['Resolution'].map(lambda x: 1 if x > 20 else 0), # Removed due to correlation and accuracy increased

X_train_new['Internal Memory'].map(lambda x: 1 if x > 64 else 0),

X_train_new['Capacity'].map(lambda x: 1 if x > 2100 else 0),

X_train_new['Brand_10.or'].map(lambda x: 0 if x >= 1 else 1), # 1 user has this phone and has not liked it. so its flipped

X_train_new['Brand_Apple'],#.map(lambda x: 0 if x=1), # 4 users have this phone and have liked it, hence no changes

X_train_new['Brand_Asus'],#.map(lambda x: 0 if x=1), # 4 users like this phone and 1 dislike, keep the field 

#X_train_new['Brand_Billion'].map(lambda x: 0 if x=1), # No Rows found, Removed to increase train accuracy due to scope of assignment

X_train_new['Brand_Blackberry'].map(lambda x: 0 if x>=1 else 1), # 3 Userd have disliked the phone hence its flipped

X_train_new['Brand_Comio'],#.map(lambda x: 0 if x=1), No changes 1 has liked it keep it

X_train_new['Brand_Coolpad'].map(lambda x: 0 if x >= 1 else 1), #flipped as 2 of them have disliked

#X_train_new['Brand_Do'].map(lambda x: 0 if x=1), # No Rows found, Removed to increase train accuracy due to scope of assignment

X_train_new['Brand_Gionee'],#.map(lambda x: 0 if x=1), # 50-50 column, as of now removed as there is no intellegence 

X_train_new['Brand_Google'],#.map(lambda x: 0 if x=1), Everyone has liked it

X_train_new['Brand_HTC'].map(lambda x: 0 if x>=1 else 1), #Max users dislike it

X_train_new['Brand_Honor'],#.map(lambda x: 0 if x=1), Max like it - 1 dislike, retained

X_train_new['Brand_Huawei'],#.map(lambda x: 0 if x=1), 

X_train_new['Brand_InFocus'].map(lambda x: 0 if x>=1 else 1), 

X_train_new['Brand_Infinix'],#.map(lambda x: 0 if x=1), 

X_train_new['Brand_Intex'].map(lambda x: 0 if x>=1 else 1),

X_train_new['Brand_Itel'],#.map(lambda x: 0 if x=1), 

X_train_new['Brand_Jivi'].map(lambda x: 0 if x>=1 else 1), 

X_train_new['Brand_Karbonn'].map(lambda x: 0 if x>=1 else 1),

X_train_new['Brand_LG'].map(lambda x: 0 if x>=1 else 1), # Many LG users don't like it

X_train_new['Brand_Lava'],#.map(lambda x: 0 if x=1), 50-50 user base, discared

X_train_new['Brand_LeEco'],#.map(lambda x: 0 if x=1), 

X_train_new['Brand_Lenovo'],#.map(lambda x: 0 if x=1), 50-50 user base, discarded

X_train_new['Brand_Lephone'].map(lambda x: 0 if x>=1 else 1), 

X_train_new['Brand_Lyf'].map(lambda x: 0 if x>=1 else 1),

X_train_new['Brand_Meizu'],#.map(lambda x: 0 if x=1), 

X_train_new['Brand_Micromax'].map(lambda x: 0 if x>=1 else 1), # Many micromax users don't like the phone

X_train_new['Brand_Mobiistar'].map(lambda x: 0 if x>=1 else 1), # On the line, 2 of them don't like 1 likes

X_train_new['Brand_Moto'],#.map(lambda x: 0 if x=1), 4 of them like, 3 of them don't

X_train_new['Brand_Motorola'],#.map(lambda x: 0 if x=1), 

X_train_new['Brand_Nokia'],#.map(lambda x: 0 if x=1),  50 - 50 User Base

X_train_new['Brand_Nubia'].map(lambda x: 0 if x>=1 else 1),

X_train_new['Brand_OPPO'],#.map(lambda x: 0 if x=1),

X_train_new['Brand_OnePlus'],#.map(lambda x: 0 if x=1), 

X_train_new['Brand_Oppo'],#.map(lambda x: 0 if x=1), 

X_train_new['Brand_Panasonic'].map(lambda x: 0 if x>=1 else 1), 

X_train_new['Brand_Razer'].map(lambda x: 0 if x>=1 else 1),

X_train_new['Brand_Realme'],#.map(lambda x: 0 if x=1), 

X_train_new['Brand_Reliance'].map(lambda x: 0 if x>=1 else 1), 

X_train_new['Brand_Samsung'],#.map(lambda x: 0 if x=1),

X_train_new['Brand_Sony'].map(lambda x: 0 if x>=1 else 1),

#X_train_new['Brand_Spice'].map(lambda x: 0 if x=1), # No Rows found, Removed to increase train accuracy due to scope of assignment

X_train_new['Brand_Tecno'],#.map(lambda x: 0 if x=1), 

X_train_new['Brand_Ulefone'],#.map(lambda x: 0 if x=1), 

X_train_new['Brand_VOTO'].map(lambda x: 0 if x>=1 else 1), 

X_train_new['Brand_Vivo'],#.map(lambda x: 0 if x=1), , 

X_train_new['Brand_Xiaomi'],#.map(lambda x: 0 if x=1), 

X_train_new['Brand_Xiaomi Poco'],#.map(lambda x: 0 if x=1), 

X_train_new['Brand_Yu'].map(lambda x: 0 if x>=1 else 1), 

X_train_new['Brand_iVooMi'].map(lambda x: 0 if x>=1 else 1), 

X_train_new['SIM Slot(s)_Dual SIM, GSM+CDMA'],#.map(lambda x: 0 if x>=1 else 1),

X_train_new['SIM Slot(s)_Dual SIM, GSM+GSM'].map(lambda x: 0 if x>=1 else 1),

X_train_new['SIM Slot(s)_Dual SIM, GSM+GSM, Dual VoLTE'],#.map(lambda x: 0 if x>=1 else 1),

X_train_new['SIM Slot(s)_Single SIM, GSM'],#.map(lambda x: 0 if x>=1 else 1)

X_train_new['Num_cores_312'],#.map(lambda x: 0 if x>=1 else 1),

X_train_new['Num_cores_Deca'].map(lambda x: 0 if x>=1 else 1),

X_train_new['Num_cores_Dual'].map(lambda x: 0 if x>=1 else 1),

X_train_new['Num_cores_Hexa'],#.map(lambda x: 0 if x>=1 else 1),

X_train_new['Num_cores_Octa'].map(lambda x: 0 if x>=1 else 1), # Slightly more than 50% dislike it

X_train_new['Num_cores_Other'],#.map(lambda x: 0 if x>=1 else 1),

X_train_new['Num_cores_Quad'].map(lambda x: 0 if x>=1 else 1),  # Slightly more than 50% dislike it

X_train_new['Num_cores_Tru-Octa'],#.map(lambda x: 0 if x>=1 else 1), 

X_train_new['Sim1_2G'],#.map(lambda x: 0 if x>=1 else 1), 

X_train_new['Sim1_3G'].map(lambda x: 0 if x>=1 else 1), 

X_train_new['Sim1_4G'],#.map(lambda x: 1 if x>=0 else 0), # Equal Spread

X_train_new['SIM 2_2G'].map(lambda x: 0 if x>=1 else 1), 

X_train_new['SIM 2_3G'],#.map(lambda x: 0 if x>=1 else 1),  # Equal Spread

X_train_new['SIM 2_4G'].map(lambda x: 0 if x>=1 else 1), 

X_train_new['SIM 2_Other'],#.map(lambda x: 0 if x>=1 else 1), 

X_train_new['os_name_Android'].map(lambda x: 0 if x>=1 else 1),

X_train_new['os_name_Blackberry'].map(lambda x: 0 if x>=1 else 1), 

X_train_new['os_name_KAI'].map(lambda x: 0 if x>=1 else 1), 

X_train_new['os_name_Nokia'],#.map(lambda x: 0 if x>=1 else 1), 

X_train_new['os_name_Other'],#.map(lambda x: 0 if x>=1 else 1), # Equal Spread

X_train_new['os_name_Tizen'].map(lambda x: 0 if x>=1 else 1), 

X_train_new['os_name_iOS'],#.map(lambda x: 0 if x>=1 else 1)

    

]) 

X_train_binarised = X_train_binarised.T
X_test_binarised = np.array([

#X_test_new['Weight'].map(lambda x: 1 if x > 153 else 0), # Removed due to correlation and accuracy increased

#X_test_new['Height'].map(lambda x: 1 if x > 151 else 0), # Removed due to correlation and accuracy increased

#X_test_new['Screen to Body Ratio (calculated)'].map(lambda x: 1 if x >= 55  else 0), # Removed due to correlation and accuracy increased

X_test_new['Pixel Density'].map(lambda x: 1 if x > 270 else 0),

X_test_new['Processor_frequency'].map(lambda x: 1 if x > 2.1 else 0),

X_test_new['Screen Size'].map(lambda x: 1 if x > 4.8 else 0),

X_test_new['RAM'].map(lambda x: 1 if x > 4 else 0),

#X_test_new['Resolution'].map(lambda x: 1 if x > 20 else 0), # Removed due to correlation and accuracy increased

X_test_new['Internal Memory'].map(lambda x: 1 if x > 64 else 0),

X_test_new['Capacity'].map(lambda x: 1 if x > 2100 else 0),

X_test_new['Brand_10.or'].map(lambda x: 0 if x >= 1 else 1), # 1 user has this phone and has not liked it. so its flipped

X_test_new['Brand_Apple'],#.map(lambda x: 0 if x=1), # 4 users have this phone and have liked it, hence no changes

X_test_new['Brand_Asus'],#.map(lambda x: 0 if x=1), # 4 users like this phone and 1 dislike, keep the field 

#X_test_new['Brand_Billion'].map(lambda x: 0 if x=1), # No Rows found, Removed to increase train accuracy due to scope of assignment

X_test_new['Brand_Blackberry'].map(lambda x: 0 if x>=1 else 1), # 3 Userd have disliked the phone hence its flipped

X_test_new['Brand_Comio'],#.map(lambda x: 0 if x=1), No changes 1 has liked it keep it

X_test_new['Brand_Coolpad'].map(lambda x: 0 if x >= 1 else 1), #flipped as 2 of them have disliked

#X_test_new['Brand_Do'].map(lambda x: 0 if x=1), # No Rows found, Removed to increase train accuracy due to scope of assignment

X_test_new['Brand_Gionee'],#.map(lambda x: 0 if x=1), # 50-50 column, as of now removed as there is no intellegence 

X_test_new['Brand_Google'],#.map(lambda x: 0 if x=1), Everyone has liked it

X_test_new['Brand_HTC'].map(lambda x: 0 if x>=1 else 1), #Max users dislike it

X_test_new['Brand_Honor'],#.map(lambda x: 0 if x=1), Max like it - 1 dislike, retained

X_test_new['Brand_Huawei'],#.map(lambda x: 0 if x=1), 

X_test_new['Brand_InFocus'].map(lambda x: 0 if x>=1 else 1), 

X_test_new['Brand_Infinix'],#.map(lambda x: 0 if x=1), 

X_test_new['Brand_Intex'].map(lambda x: 0 if x>=1 else 1),

X_test_new['Brand_Itel'],#.map(lambda x: 0 if x=1), 

X_test_new['Brand_Jivi'].map(lambda x: 0 if x>=1 else 1), 

X_test_new['Brand_Karbonn'].map(lambda x: 0 if x>=1 else 1),

X_test_new['Brand_LG'].map(lambda x: 0 if x>=1 else 1), # Many LG users don't like it

X_test_new['Brand_Lava'],#.map(lambda x: 0 if x=1), 50-50 user base, discared

X_test_new['Brand_LeEco'],#.map(lambda x: 0 if x=1), 

X_test_new['Brand_Lenovo'],#.map(lambda x: 0 if x=1), 50-50 user base, discarded

X_test_new['Brand_Lephone'].map(lambda x: 0 if x>=1 else 1), 

X_test_new['Brand_Lyf'].map(lambda x: 0 if x>=1 else 1),

X_test_new['Brand_Meizu'],#.map(lambda x: 0 if x=1), 

X_test_new['Brand_Micromax'].map(lambda x: 0 if x>=1 else 1), # Many micromax users don't like the phone

X_test_new['Brand_Mobiistar'].map(lambda x: 0 if x>=1 else 1), # On the line, 2 of them don't like 1 likes

X_test_new['Brand_Moto'],#.map(lambda x: 0 if x=1), 4 of them like, 3 of them don't

X_test_new['Brand_Motorola'],#.map(lambda x: 0 if x=1), 

X_test_new['Brand_Nokia'],#.map(lambda x: 0 if x=1),  50 - 50 User Base

X_test_new['Brand_Nubia'].map(lambda x: 0 if x>=1 else 1),

X_test_new['Brand_OPPO'],#.map(lambda x: 0 if x=1),

X_test_new['Brand_OnePlus'],#.map(lambda x: 0 if x=1), 

X_test_new['Brand_Oppo'],#.map(lambda x: 0 if x=1), 

X_test_new['Brand_Panasonic'].map(lambda x: 0 if x>=1 else 1), 

X_test_new['Brand_Razer'].map(lambda x: 0 if x>=1 else 1),

X_test_new['Brand_Realme'],#.map(lambda x: 0 if x=1), 

X_test_new['Brand_Reliance'].map(lambda x: 0 if x>=1 else 1), 

X_test_new['Brand_Samsung'],#.map(lambda x: 0 if x=1),

X_test_new['Brand_Sony'].map(lambda x: 0 if x>=1 else 1),

#X_test_new['Brand_Spice'].map(lambda x: 0 if x=1), # No Rows found, Removed to increase train accuracy due to scope of assignment

X_test_new['Brand_Tecno'],#.map(lambda x: 0 if x=1), 

X_test_new['Brand_Ulefone'],#.map(lambda x: 0 if x=1), 

X_test_new['Brand_VOTO'].map(lambda x: 0 if x>=1 else 1), 

X_test_new['Brand_Vivo'],#.map(lambda x: 0 if x=1), , 

X_test_new['Brand_Xiaomi'],#.map(lambda x: 0 if x=1), 

X_test_new['Brand_Xiaomi Poco'],#.map(lambda x: 0 if x=1), 

X_test_new['Brand_Yu'].map(lambda x: 0 if x>=1 else 1), 

X_test_new['Brand_iVooMi'].map(lambda x: 0 if x>=1 else 1), 

X_test_new['SIM Slot(s)_Dual SIM, GSM+CDMA'],#.map(lambda x: 0 if x>=1 else 1),

X_test_new['SIM Slot(s)_Dual SIM, GSM+GSM'].map(lambda x: 0 if x>=1 else 1),

X_test_new['SIM Slot(s)_Dual SIM, GSM+GSM, Dual VoLTE'],#.map(lambda x: 0 if x>=1 else 1),

X_test_new['SIM Slot(s)_Single SIM, GSM'],#.map(lambda x: 0 if x>=1 else 1)

X_test_new['Num_cores_312'],#.map(lambda x: 0 if x>=1 else 1),

X_test_new['Num_cores_Deca'].map(lambda x: 0 if x>=1 else 1),

X_test_new['Num_cores_Dual'].map(lambda x: 0 if x>=1 else 1),

X_test_new['Num_cores_Hexa'],#.map(lambda x: 0 if x>=1 else 1),

X_test_new['Num_cores_Octa'].map(lambda x: 0 if x>=1 else 1), # Slightly more than 50% dislike it

X_test_new['Num_cores_Other'],#.map(lambda x: 0 if x>=1 else 1),

X_test_new['Num_cores_Quad'].map(lambda x: 0 if x>=1 else 1),  # Slightly more than 50% dislike it

X_test_new['Num_cores_Tru-Octa'],#.map(lambda x: 0 if x>=1 else 1), 

X_test_new['Sim1_2G'],#.map(lambda x: 0 if x>=1 else 1), 

X_test_new['Sim1_3G'].map(lambda x: 0 if x>=1 else 1), 

X_test_new['Sim1_4G'],#.map(lambda x: 1 if x>=0 else 0), # Equal Spread

X_test_new['SIM 2_2G'].map(lambda x: 0 if x>=1 else 1), 

X_test_new['SIM 2_3G'],#.map(lambda x: 0 if x>=1 else 1),  # Equal Spread

X_test_new['SIM 2_4G'].map(lambda x: 0 if x>=1 else 1), 

X_test_new['SIM 2_Other'],#.map(lambda x: 0 if x>=1 else 1), 

X_test_new['os_name_Android'].map(lambda x: 0 if x>=1 else 1),

X_test_new['os_name_Blackberry'].map(lambda x: 0 if x>=1 else 1), 

X_test_new['os_name_KAI'].map(lambda x: 0 if x>=1 else 1), 

X_test_new['os_name_Nokia'],#.map(lambda x: 0 if x>=1 else 1), 

X_test_new['os_name_Other'],#.map(lambda x: 0 if x>=1 else 1), # Equal Spread

X_test_new['os_name_Tizen'].map(lambda x: 0 if x>=1 else 1), 

X_test_new['os_name_iOS'],#.map(lambda x: 0 if x>=1 else 1)

    

]) 



X_test_binarised = X_test_binarised.T
class MPNeuron:

    def __init__self(self):

        self.b = None

    def model(self,x):

        return(sum(x) >= self.b)

    def predict(self,X):

        Y=[]

        for x in X:

            result = self.model(x)

            Y.append(result)

        return np.array(Y)

    def fit(self,X,Y):

        accuracy = {}

        

        for b in range(X.shape[1]+1):

            self.b = b

            Y_pred = self.predict(X)

            accuracy[b] = accuracy_score(Y_pred, Y)

            

        best_b = max(accuracy, key = accuracy.get)

        self.b = best_b

        

        print('Optimal value of b is', best_b)

        print('Highest accuracy is', accuracy[best_b])
mp_neuron = MPNeuron()

mp_neuron.fit(X_train_binarised,Y_train_new_rating_binary)
Y_test_pred = mp_neuron.predict(X_test_binarised)

    

# Convert True, False to 1,0

Y_test_pred = Y_test_pred.astype(int)

Y_test_pred
submission = pd.DataFrame({'PhoneId':test_new['PhoneId'], 'Class':Y_test_pred})

submission = submission[['PhoneId', 'Class']]

submission.head()
submission.to_csv("submission_final.csv", index=False)