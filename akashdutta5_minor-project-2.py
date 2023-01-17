!pip install scikit-learn --upgrade
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory







# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-01.csv")

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-02.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-03.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-04.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-05.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-06.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-07.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-08.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-09.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-10.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-11.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-12.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-13.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-14.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-15.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-16.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-17.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-18.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-19.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-20.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-21.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-22.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-23.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-24.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-25.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-26.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-27.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-28.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-29.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-30.csv"))

df = df.append(pd.read_csv("/kaggle/input/backblaze-q4-data/data_Q4_2019/2019-10-31.csv"))



df.info()
df.isnull().sum()
df = df.drop(["smart_2_normalized" , "smart_2_raw" , "smart_8_normalized" , "smart_8_raw" , "smart_189_normalized" , "smart_189_raw" , "smart_191_normalized" , "smart_191_raw" , "smart_196_normalized" , "smart_196_raw" , "smart_200_normalized" , "smart_200_raw" , "date" , "smart_11_normalized" , "smart_11_raw" , "smart_13_normalized" , "smart_13_raw" , "smart_15_normalized" , "smart_15_raw" , "smart_16_normalized" , "smart_16_raw" , "smart_17_normalized" , "smart_17_raw" , "smart_18_normalized" , "smart_18_raw" , "smart_22_normalized" , "smart_22_raw" , "smart_23_normalized" , "smart_23_raw" , "smart_24_normalized" , "smart_24_raw" , "smart_168_normalized" , "smart_168_raw" , "smart_170_normalized" , "smart_170_raw" , "smart_173_normalized" , "smart_173_raw" , "smart_174_normalized" , "smart_174_raw" , "smart_177_normalized" , "smart_177_raw" , "smart_179_normalized" , "smart_179_raw" , "smart_181_normalized" , "smart_181_raw" , "smart_182_normalized" , "smart_182_raw" , "smart_183_normalized" , "smart_183_raw" , "smart_184_normalized" , "smart_184_raw" , "smart_201_normalized" , "smart_201_raw" , "smart_218_normalized" , "smart_218_raw" , "smart_220_normalized" , "smart_220_raw" , "smart_222_normalized" , "smart_222_raw" , "smart_223_normalized" , "smart_223_raw" , "smart_224_normalized" , "smart_224_raw" , "smart_225_normalized" , "smart_225_raw" , "smart_226_normalized" , "smart_226_raw" , "smart_231_normalized" , "smart_231_raw" , "smart_232_normalized" , "smart_232_raw" , "smart_233_normalized" , "smart_233_raw" , "smart_235_normalized" , "smart_235_raw" , "smart_250_normalized" , "smart_250_raw" , "smart_251_normalized" , "smart_251_raw" , "smart_252_normalized" , "smart_252_raw" , "smart_254_normalized" , "smart_254_raw" , "smart_255_normalized" , "smart_255_raw" ] , axis = 1)
df.info()
df.isnull().sum()
mean1 = df.smart_1_normalized.mean()

print(mean1)

mean1r = df.smart_1_raw.mean()

print(mean1r)

df.smart_1_normalized.fillna(mean1 , inplace = True)

df.smart_1_raw.fillna(mean1r , inplace = True)



mean3 = df.smart_3_normalized.mean()

print(mean3)

mean3r = df.smart_3_raw.mean()

print(mean3r)

df.smart_3_normalized.fillna(mean3 , inplace = True)

df.smart_3_raw.fillna(mean3r , inplace = True)



mean4 = df.smart_4_normalized.mean()

print(mean4)

mean4r = df.smart_4_raw.mean()

print(mean4r)

df.smart_4_normalized.fillna(mean4 , inplace = True)

df.smart_4_raw.fillna(mean4r , inplace = True)



mean5 = df.smart_5_normalized.mean()

print(mean5)

mean5r = df.smart_5_raw.mean()

print(mean5r)

df.smart_5_normalized.fillna(mean5 , inplace = True)

df.smart_5_raw.fillna(mean5r , inplace = True)



mean7 = df.smart_7_normalized.mean()

print(mean7)

mean7r = df.smart_7_raw.mean()

print(mean7r)

df.smart_7_normalized.fillna(mean7 , inplace = True)

df.smart_7_raw.fillna(mean7r , inplace = True)



mean9 = df.smart_9_normalized.mean()

print(mean9)

mean9r = df.smart_9_raw.mean()

print(mean9r)

df.smart_9_normalized.fillna(mean9 , inplace = True)

df.smart_9_raw.fillna(mean9r , inplace = True)



mean10 = df.smart_10_normalized.mean()

print(mean10)

mean10r = df.smart_10_raw.mean()

print(mean10r)

df.smart_10_normalized.fillna(mean10 , inplace = True)

df.smart_10_raw.fillna(mean10r , inplace = True)



mean12 = df.smart_12_normalized.mean()

print(mean12)

mean12r = df.smart_12_raw.mean()

print(mean12r)

df.smart_12_normalized.fillna(mean12 , inplace = True)

df.smart_12_raw.fillna(mean12r , inplace = True)



mean187 = df.smart_187_normalized.mean()

print(mean187)

mean187r = df.smart_187_raw.mean()

print(mean187r)

df.smart_187_normalized.fillna(mean187 , inplace = True)

df.smart_187_raw.fillna(mean187r , inplace = True)



mean188 = df.smart_188_normalized.mean()

print(mean188)

mean188r = df.smart_188_raw.mean()

print(mean188r)

df.smart_188_normalized.fillna(mean188 , inplace = True)

df.smart_188_raw.fillna(mean188r , inplace = True)



mean190 = df.smart_190_normalized.mean()

print(mean190)

mean190r = df.smart_190_raw.mean()

print(mean190r)

df.smart_190_normalized.fillna(mean190 , inplace = True)

df.smart_190_raw.fillna(mean190r , inplace = True)



mean192 = df.smart_192_normalized.mean()

print(mean192)

mean192r = df.smart_192_raw.mean()

print(mean192r)

df.smart_192_normalized.fillna(mean192 , inplace = True)

df.smart_192_raw.fillna(mean192r , inplace = True)



mean193 = df.smart_193_normalized.mean()

print(mean193)

mean193r = df.smart_193_raw.mean()

print(mean193r)

df.smart_193_normalized.fillna(mean193 , inplace = True)

df.smart_193_raw.fillna(mean193r , inplace = True)



mean194 = df.smart_194_normalized.mean()

print(mean194)

mean194r = df.smart_194_raw.mean()

print(mean194r)

df.smart_194_normalized.fillna(mean194 , inplace = True)

df.smart_194_raw.fillna(mean194r , inplace = True)



mean195 = df.smart_195_normalized.mean()

print(mean195)

mean195r = df.smart_195_raw.mean()

print(mean195r)

df.smart_195_normalized.fillna(mean195 , inplace = True)

df.smart_195_raw.fillna(mean195r , inplace = True)



mean197 = df.smart_197_normalized.mean()

print(mean197)

mean197r = df.smart_197_raw.mean()

print(mean197r)

df.smart_197_normalized.fillna(mean197 , inplace = True)

df.smart_197_raw.fillna(mean197r , inplace = True)



mean198 = df.smart_198_normalized.mean()

print(mean198)

mean198r = df.smart_198_raw.mean()

print(mean198r)

df.smart_198_normalized.fillna(mean198 , inplace = True)

df.smart_198_raw.fillna(mean198r , inplace = True)



mean199 = df.smart_199_normalized.mean()

print(mean199)

mean199r = df.smart_199_raw.mean()

print(mean199r)

df.smart_199_normalized.fillna(mean199 , inplace = True)

df.smart_199_raw.fillna(mean199r , inplace = True)



mean240 = df.smart_240_normalized.mean()

print(mean240)

mean240r = df.smart_240_raw.mean()

print(mean240r)

df.smart_240_normalized.fillna(mean240 , inplace = True)

df.smart_240_raw.fillna(mean240r , inplace = True)



mean241 = df.smart_241_normalized.mean()

print(mean241)

mean241r = df.smart_241_raw.mean()

print(mean241r)

df.smart_241_normalized.fillna(mean241 , inplace = True)

df.smart_241_raw.fillna(mean241r , inplace = True)



mean242 = df.smart_242_normalized.mean()

print(mean242)

mean242r = df.smart_242_raw.mean()

print(mean242r)

df.smart_242_normalized.fillna(mean242 , inplace = True)

df.smart_242_raw.fillna(mean242r , inplace = True)
from sklearn.preprocessing import LabelEncoder

label=LabelEncoder()

df['serial_number']=label.fit_transform(df['serial_number'])

df['model']=label.fit_transform(df['model'])
a= len(df[df['failure'] == 0] )

print ("Amount of Non Failure = " , a)

b = len(df[df['failure'] == 1])

print ("Amount of Failure  = " ,b )
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot('failure', data=df)

plt.ylabel("Frequency")

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.utils import resample

from imblearn.over_sampling import SMOTE



# Separate input features and target

Y = df.failure

X = df.drop(['failure',], axis=1)



# setting up testing and training sets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2727)



sm = SMOTE(random_state=2727)

X_train, Y_train = sm.fit_sample(X_train, Y_train)
X_train = pd.DataFrame(data=X_train)

X_train.columns = ['serial_number','model','capacity_bytes','smart_1_normalized','smart_1_raw',

                   'smart_3_normalized','smart_3_raw','smart_4_normalized','smart_4_raw','smart_5_normalized',

                   'smart_5_raw','smart_7_normalized','smart_7_raw','smart_9_normalized','smart_9_raw',

                   'smart_10_normalized','smart_10_raw','smart_12_normalized','smart_12_raw','smart_187_normalized',

                   'smart_187_raw','smart_188_normalized','smart_188_raw','smart_190_normalized','smart_190_raw',

                   'smart_192_normalized','smart_192_raw','smart_193_normalized','smart_193_raw',

                   'smart_194_normalized','smart_194_raw','smart_195_normalized','smart_195_raw',

                   'smart_197_normalized','smart_197_raw','smart_198_normalized','smart_198_raw',

                   'smart_199_normalized','smart_199_raw','smart_240_normalized','smart_240_raw',

                   'smart_241_normalized','smart_241_raw','smart_242_normalized','smart_242_raw']

Y_train = pd.DataFrame(data = Y_train)

Y_train.columns = ['failure']
sns.countplot('failure', data=Y_train)

plt.ylabel("Frequency")

plt.show()
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

smote = XGBClassifier()

smote.fit(X_train, Y_train)



# Predict on test

smote_pred = smote.predict(X_test)

# predict probabilities

probs = smote.predict_proba(X_test)

# keep probabilities for the positive outcome only

probs = probs[:, 1]
accuracy = accuracy_score(Y_test, smote_pred)

print("Test Accuracy is {:.2f}%".format(accuracy * 100.0))