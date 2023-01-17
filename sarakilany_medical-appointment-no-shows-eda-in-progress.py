# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import datetime as dt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import data

df = pd.read_csv('../input/noshowappointments/KaggleV2-May-2016.csv')

df.head()
df.info()
#converting  ScheduledDay  and AppointmentDay to dates

df[['AppointmentDay','ScheduledDay']]=df[['AppointmentDay','ScheduledDay']].astype('datetime64[ns]')
plt.figure(figsize=(10,10));

df.hist();
y=df['No-show'].copy()

X=df.drop(columns='No-show', axis=0)

X.head()
#dropping ids

X.drop(columns=['PatientId','AppointmentID'], inplace=True)

X.head()
#finding the years in which appointments were in

pd.DatetimeIndex(X['ScheduledDay']).year.unique()

#separating the year

X['year']=pd.DatetimeIndex(X['ScheduledDay']).year
X.head()
#extracting the day of the appointment and the days between the two dates

X['DaysbtwSchApp'] = (X['AppointmentDay']-X['ScheduledDay']).astype(str).apply(lambda x :x.rsplit('d')[0]).astype(int)

X.loc[X['DaysbtwSchApp'] <0,'DaysbtwSchApp']= 0

X['AppDay'] = X['AppointmentDay'].apply(lambda x: dt.datetime.strftime(x, '%A'))

X.head()
X.head()