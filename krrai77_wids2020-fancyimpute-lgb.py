# In this notebook I have used fancyimpute technique to impute missing values in the entire WiDS2020 dataset. 

# I have used LGBM to train the model. I got accuracy of 0.88. Parameter tuning might help in bettring this score.

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from fancyimpute import KNN

from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import train_test_split

import lightgbm as lgb
#read the data and drop noisy columns

train=pd.read_csv("/kaggle/input/widsdatathon2020/training_v2.csv")

test=pd.read_csv("/kaggle/input/widsdatathon2020/unlabeled.csv")

solution = pd.read_csv("/kaggle/input/solutiontemplate/solution_template.csv")

trainv1=train.drop(['encounter_id','patient_id','icu_id', 'hospital_id', 'readmission_status','ethnicity'],axis=1)

testv1=test.drop(['encounter_id','patient_id','icu_id', 'hospital_id', 'readmission_status','hospital_death','ethnicity'],axis=1)

print("number of rows and columns in training set is \n",trainv1.shape)

print("number of rows and columns in test set is \n",testv1.shape)
#Exploring the data

trainv1.info()

trainv1.describe()

trainv1.isna().sum()

testv1.isna().sum()

trainv1['hospital_death'].value_counts()*100/len(trainv1['hospital_death'])

sns.countplot(trainv1['hospital_death'])
#Seperate categorical and numerical variables

cattrain=trainv1.select_dtypes('object')

numtrain=trainv1.select_dtypes('number')

cattest=testv1.select_dtypes('object')

numtest=testv1.select_dtypes('number')
#encoding categorical test variables

#instantiate both packages to use

encoder = OrdinalEncoder()

imputer = KNN()

# create a list of categorical columns to iterate over

cat_cols = cattest.columns



def encode(data):

    '''function to encode non-null data and replace it in the original data'''

    #retains only non-null values

    nonulls = np.array(data.dropna())

    #reshapes the data for encoding

    impute_reshape = nonulls.reshape(-1,1)

    #encode data

    impute_ordinal = encoder.fit_transform(impute_reshape)

    #Assign back encoded values to non-null values

    data.loc[data.notnull()] = np.squeeze(impute_ordinal)

    return data



#create a for loop to iterate through each column in the data

for columns in cat_cols:

    encode(cattest[columns])
#encoding categorical train variables

#instantiate both packages to use

encoder = OrdinalEncoder()

imputer = KNN()

# create a list of categorical columns to iterate over

cat_cols = cattrain.columns



def encode(data):

    '''function to encode non-null data and replace it in the original data'''

    #retains only non-null values

    nonulls = np.array(data.dropna())

    #reshapes the data for encoding

    impute_reshape = nonulls.reshape(-1,1)

    #encode data

    impute_ordinal = encoder.fit_transform(impute_reshape)

    #Assign back encoded values to non-null values

    data.loc[data.notnull()] = np.squeeze(impute_ordinal)

    return data



#create a for loop to iterate through each column in the data

for columns in cat_cols:

    encode(cattrain[columns])
#splitting train values into sections for faster imputing

numtrain1=numtrain[0:20000]

numtrain2=numtrain[20000:40000]

numtrain3=numtrain[40000:60000]

numtrain4=numtrain[60000:80000]

numtrain5=numtrain[80000:]



cattrain1=cattrain[0:20000]

cattrain2=cattrain[20000:40000]

cattrain3=cattrain[40000:60000]

cattrain4=cattrain[60000:80000]

cattrain5=cattrain[80000:]
#splitting test values into sections for faster imputing

cattest1=cattest[0:20000]

cattest2=cattest[20000:]



numtest1=numtest[0:20000]

numtest2=numtest[20000:]
# impute catgorical test data and convert                                                                                                                                                   

encode_testdata1 = pd.DataFrame(np.round(imputer.fit_transform(cattest1)),columns = cattest.columns)

encode_testdata2 = pd.DataFrame(np.round(imputer.fit_transform(cattest2)),columns = cattest.columns)

# impute catgorical train data and convert                                                                                                                                                   

encode_data1 = pd.DataFrame(np.round(imputer.fit_transform(cattrain1)),columns = cattrain.columns)

encode_data2 = pd.DataFrame(np.round(imputer.fit_transform(cattrain2)),columns = cattrain.columns)

encode_data3 = pd.DataFrame(np.round(imputer.fit_transform(cattrain3)),columns = cattrain.columns)

encode_data4 = pd.DataFrame(np.round(imputer.fit_transform(cattrain4)),columns = cattrain.columns)

encode_data5 = pd.DataFrame(np.round(imputer.fit_transform(cattrain5)),columns = cattrain.columns)
cattrainfill=pd.concat([encode_data1,encode_data2,encode_data3,encode_data4,encode_data5])

cattestfill=pd.concat([encode_testdata1,encode_testdata2])
#impute numerical test data

encode_testdatanum = pd.DataFrame(np.round(imputer.fit_transform(numtest1)),columns = numtest.columns)

encode_testdatanum2 = pd.DataFrame(np.round(imputer.fit_transform(numtest2)),columns = numtest.columns)

#impute numerical train data

encode_datanum1 = pd.DataFrame(np.round(imputer.fit_transform(numtrain1)),columns = numtrain.columns)

encode_datanum2 = pd.DataFrame(np.round(imputer.fit_transform(numtrain2)),columns = numtrain.columns)

encode_datanum3 = pd.DataFrame(np.round(imputer.fit_transform(numtrain3)),columns = numtrain.columns)

encode_datanum4 = pd.DataFrame(np.round(imputer.fit_transform(numtrain4)),columns = numtrain.columns)

encode_datanum5 = pd.DataFrame(np.round(imputer.fit_transform(numtrain5)),columns = numtrain.columns)
numtrainfill=pd.concat([encode_datanum1,encode_datanum2,encode_datanum3,encode_datanum4,encode_datanum5])

numtestfill=pd.concat([encode_testdatanum,encode_testdatanum2])
trainv6=pd.concat([numtrainfill,cattrainfill],axis=1,join='inner')

testv6=pd.concat([numtestfill,cattestfill],axis=1,join='inner')
y=trainv6['hospital_death']

trainv7=trainv6.drop(['hospital_death'], axis=1)
# Split into training and validation set

x_train, x_val, y_train, y_val = train_test_split(trainv7, y, test_size = 0.25, random_state = 1)
#Model building

d_train = lgb.Dataset(x_train, label=y_train)

params = {}

params['learning_rate'] = 0.003

params['boosting_type'] = 'gbdt'

params['objective'] = 'binary'

params['metric'] = 'binary_logloss'

params['sub_feature'] = 0.5

params['num_leaves'] = 100

params['min_data'] = 50

params['max_depth'] = 10

clf = lgb.train(params, d_train, 100)
#Prediction

y_pred=clf.predict(x_val)

y_pred1=np.round(y_pred)
#Measure accuracy

#Confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_val, y_pred1)

print (cm)

#Accuracy

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_pred1,y_val)

print (accuracy)
#Prediction on Test variables

pred_on_test=clf.predict(testv6)
solution.hospital_death = pred_on_test

solution.to_csv("submissionlgbm.csv", index=0)