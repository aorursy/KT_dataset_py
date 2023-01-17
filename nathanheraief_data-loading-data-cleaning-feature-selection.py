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
df = pd.read_csv("/kaggle/input/cs-challenge/training_set.csv")
df.info()
df.isna().sum().sum()
nan_number = df.isna().sum().sum()

print(nan_number)
nan_number*100/(432170 *79)
(100*df.isna().sum()/432170).sort_values(ascending=False).head(20)
df = df.sort_values('Date_time')

df = df.fillna(method='ffill')
df
df.isna().sum().sum()
import matplotlib.pyplot as plt

import seaborn as sns
#Using Pearson Correlation

plt.figure(figsize=(12,10))

cor = df.corr()

sns.heatmap(cor)

plt.show()
#Correlation with output variable

cor_target = abs(cor["TARGET"])

#Selecting highly correlated features

relevant_features = cor_target[cor_target>0.5]

relevant_features
df_clean = df[relevant_features.index]

df_clean
Y = df_clean['TARGET']

X = df_clean.drop('TARGET', axis=1)
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=404)
len(X_train) + len(X_test) == len(X)
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, Y_train)



Y_pred = reg.predict(X_test)
hat = pd.DataFrame([Y_pred, Y_test]).T

hat.columns = ['Prediction' ,'Verité']

hat['erreur'] = abs(hat.iloc[:,0] - hat.iloc[:,1])

hat
from sklearn.metrics import mean_absolute_error



mean_absolute_error(Y_test, Y_pred)
df_test = pd.read_csv('/kaggle/input/cs-challenge/test_set.csv')

ids = df_test['ID']
df_test = df_test[['Generator_converter_speed', 'Generator_converter_speed_min',

       'Generator_converter_speed_max', 'Generator_speed',

       'Generator_speed_min', 'Generator_speed_max',

       'Gearbox_bearing_1_temperature', 'Gearbox_bearing_1_temperature_min',

       'Gearbox_bearing_1_temperature_max', 'Gearbox_bearing_2_temperature',

       'Gearbox_bearing_2_temperature_min',

       'Gearbox_bearing_2_temperature_max', 'Rotor_speed', 'Rotor_speed_min','Rotor_speed_max','Date_time']]
df_test = df_test.sort_values('Date_time')

df_test = df_test.drop('Date_time', axis=1)

df_test = df_test.fillna(method='ffill')
df_test.isna().sum().sum()
X_real_world = StandardScaler().fit_transform(df_test)
prediction = reg.predict(X_real_world)
results = pd.DataFrame()

results['ID'] = ids

results['TARGET'] = prediction

results.to_csv('linear_results.csv', index=False)
results