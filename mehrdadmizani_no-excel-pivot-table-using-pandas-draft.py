import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/train.csv")

df.head(3)
df.describe()
df.info()
#Columns of the dataframe

df.columns
df.pivot_table(index=['Sex'])
df_simple = df[['Survived','Sex']]
df_simple.pivot_table(index=['Sex'])
df_simple.pivot_table(values=['Survived'], index=['Sex'], aggfunc=np.mean)
df_simple['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
pvt_sex = df_simple.pivot_table(values=['Survived'],\

                                index=['Sex'],aggfunc=np.mean)

pvt_sex 
df_simple['Age'] = df['Age'] 
df_simple['Age'] = df_simple['Age'] < 18

df_simple['Age'] = df_simple['Age'].map({True:'child',False:'adult'})

pvt_sex_age = df_simple.pivot_table(values=['Survived'], index=['Sex','Age'],aggfunc=np.mean)

pvt_sex_age
df_simple['Pclass'] = df['Pclass']

pvt_sex_pclass = df_simple.pivot_table(values=['Survived'], index=['Sex','Pclass'],aggfunc=np.mean)

pvt_sex_pclass 
max_fare = np.max(df['Fare'])

max_fare
fare_bins = [0,10,20,30,513]

fare_labels = ['<10','10-20','20-30','+30']

df_simple['Fare'] = pd.cut(df['Fare'],fare_bins, labels = fare_labels, include_lowest=True)

pvt_sex_pclass_fare =  df_simple.pivot_table(values=['Survived'], \

                                     index=['Sex','Pclass','Fare'],aggfunc=np.mean)

pvt_sex_pclass_fare