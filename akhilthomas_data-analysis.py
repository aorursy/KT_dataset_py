# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

database = '../input/wildlife-strikes/database.csv'
df = pd.read_csv(database, low_memory=False)
df.head()
df.columns
df.info()
df['Operator']= df['Operator'].replace('UNKNOWN',np.nan)
df['Operator ID'] = df['Operator ID'].replace('UNK',np.nan)
df['Operator'].value_counts().head(10).plot(kind = 'barh')
df = df.replace('UNKNOWN',np.nan)
df.groupby('Incident Year').size().plot()
df.groupby('Incident Month').size().plot()

df_2 = df[['Aircraft Damage', 'Radome Strike', 'Radome Damage',

       'Windshield Strike', 'Windshield Damage', 'Nose Strike', 'Nose Damage',

       'Engine1 Strike', 'Engine1 Damage', 'Engine2 Strike', 'Engine2 Damage',

       'Engine3 Strike', 'Engine3 Damage', 'Engine4 Strike', 'Engine4 Damage',

       'Engine Ingested', 'Propeller Strike', 'Propeller Damage',

       'Wing or Rotor Strike', 'Wing or Rotor Damage', 'Fuselage Strike',

       'Fuselage Damage', 'Landing Gear Strike', 'Landing Gear Damage',

       'Tail Strike', 'Tail Damage', 'Lights Strike', 'Lights Damage',

       'Other Strike', 'Other Damage']]
df_2['Aircraft Damage'] = df_2['Aircraft Damage'].replace(0,np.nan)
df_2.head()
df_2.dropna().head()
df_2.sum()
a={'random' : 1583/18761,

'windshield' : 1037/24189,

'nose' : 1145/21305,

'engine_1' : 2515/9681,

'engine_2' : 2023/7864,

'engine_3' : 169/554,

'engine_4' : 74/334,

'propellor' : 579/3497,

'wing_or_rotor' : 4180/20746,

'fuselage' :824/17881,

'landing_gear' :  1012/8051,

'tail' : 727/1956,

'lights' :734/1038,

'other' : 1565/15796}
maximium = max(a,key=a.get)
print(maximium)
df_3=df[['Species ID','Aircraft Damage']]
df_3['Aircraft Damage'] = df_3['Aircraft Damage'].replace(0,np.nan)
df_3.dropna().head()
df_3['Species ID'].value_counts(ascending=False).head(15).plot(kind='barh', title='Species Involved in Aircraft Damage',)

plt.show()