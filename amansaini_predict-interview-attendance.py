# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Interview.csv')
df.info()
#to see whole list of Attributes
pd.set_option('display.max_columns', 500)
#dropping unnecessary colomns
df = df.drop(['Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27'], axis=1)

df = df.drop(df.tail(1).index)

from collections import defaultdict
#rename column names to simplified names
column_names = defaultdict(str)
for column in df.columns:
    column_names[column] = ''
new_cols = ['Date', 'Client', 'Industry', 'Location', 'Position', 'Skillset', 'Interview_Type', 'ID', 'Gender', 'Candidate_Loc', 'Job_Location', 'Venue', 'Native_Loc', 'Permission', 'Hope', '3_hour_call', 'Alt_Number', 'Resume_Printout', 'Clarify_Venue', 'Share_Letter', 'Expected', 'Attendance', 'Marital_Status']
for idx,key in enumerate(column_names):
    column_names[key] = new_cols[idx]

df = df.rename(columns=column_names)

df.head()
%matplotlib inline
sns.countplot(df['Hope'])
sns.countplot(df['Expected'])
#lets automate some plotting
fig,axes = plt.subplots(len(df.columns),figsize=(10,5*len(df.columns)))
plt.tight_layout()
for idx,col in enumerate(df.columns):
    sns.countplot(df[col],ax=axes[idx])
df.Hope = df.Hope.fillna('unsure')
for i,v in enumerate(df.Hope):
    value = v.lower()
    if value == 'unsure' or value == 'not sure' or value == 'cant say' or value == 'nan' or value == 'na':
        df.Hope.iloc[i] = 'no'
    else:
        df.Hope.iloc[i] = value
df.Hope.unique()
df.Permission = df.Permission.fillna('no')
for i,v in enumerate(df.Permission):
    value = v.lower()
    if value == 'not yet' or value == 'na':
        df.Permission.iloc[i] = 'no'
    elif value == 'yet to confirm':
        df.Permission.iloc[i] = 'yes'
    else:
        df.Permission.iloc[i] = value
df.Permission.unique()
df['3_hour_call'] = df['3_hour_call'].fillna('no')
for i,v in enumerate(df['3_hour_call']):
    value = v.lower()
    if value == 'no dont' or value == 'na':
        df['3_hour_call'].iloc[i] = 'no'
    else:
         df['3_hour_call'].iloc[i] = value
df['3_hour_call'].unique()
df.Alt_Number = df.Alt_Number.fillna('no')
for i,v in enumerate(df.Alt_Number):
    value = v.lower()
    if value == 'no i have only thi number' or value == 'na':
        df.Alt_Number.iloc[i] = 'no'
    else:
         df.Alt_Number.iloc[i] = value
df.Alt_Number.unique()
df.Attendance = df.Attendance.fillna('na')
for i,v in enumerate(df.Attendance):
    value = v.lower()
    if value == 'no ' or value == 'na':
        df.Attendance.iloc[i] = 'no'
    elif value == 'yes ':
        df.Attendance.iloc[i] = 'yes'
    else:
        df.Attendance.iloc[i] = value
        
df['Attendance'].unique()
df.Clarify_Venue.unique()
df.Share_Letter.unique()
df.Resume_Printout = df.Resume_Printout.fillna('NA')
for i,v in enumerate(df.Resume_Printout):
    value = v.lower()
    if value == 'no- will take it soon' or value == 'not yet' or value == 'na':
        df.Resume_Printout.iloc[i] = 'no'
    else:
        df.Resume_Printout.iloc[i] = value

print(df.Resume_Printout.unique())
        
df.Clarify_Venue = df.Clarify_Venue.fillna('na')
for i,v in enumerate(df.Clarify_Venue):
    value = v.lower()
    if value == 'no- i need to check' or value == 'na':
        df.Clarify_Venue.iloc[i] = 'no'
    else:
        df.Clarify_Venue.iloc[i] = value
        
print(df.Clarify_Venue.unique())

df.Share_Letter = df.Share_Letter.fillna('na')
for i,v in enumerate(df.Share_Letter):
    value = v.lower()
    if value == 'havent checked':
        df.Share_Letter.iloc[i] = 'no'
    elif value == 'need to check':
        df.Share_Letter.iloc[i] = 'no'
    elif value == 'not sure':
        df.Share_Letter.iloc[i] = 'no'
    elif value == 'yet to check':
        df.Share_Letter.iloc[i] = 'no'
    elif value == 'not yet' or value == 'na':
        df.Share_Letter.iloc[i] = 'no'
    else:
        df.Share_Letter.iloc[i] = value
        
print(df.Share_Letter.unique())                
df.Expected = df.Expected.fillna('uncertain')

for i,v in enumerate(df.Expected):
    value = v.lower()
    if value == '11:00 am':
        df.Expected.iloc[i] = 'yes'
    elif value == '10.30 am':
        df.Expected.iloc[i] = 'yes'
    elif value == 'uncertain':
        df.Expected.iloc[i] = 'no'
    else:
        df.Expected.iloc[i] = value
print(df.Expected.unique())
df = df.drop(['Share_Letter'],axis=1)
for col in df.columns:
    print(df[col].unique())
from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(df['Attendance'])
df.Attendance = encoder.fit_transform(df['Attendance'])
for col in df.drop(['Attendance'],axis=1).columns :
    encoder.fit(df[col])
    df[col] = encoder.transform(df[col])
df.head()
model = Sequential()
model.add(Dense(12, input_dim=21, activation='relu'))

model.add(Dense(8,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(df.loc[:,df.columns != 'Attendance'],df.Attendance,epochs=100,batch_size=10)
