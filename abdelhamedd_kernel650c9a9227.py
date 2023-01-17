import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv('../input/datacsv/exams.csv')

data.head(10)
listt = data.columns

for i in listt :

    print(data[i].unique())

print(data.describe())
data.isnull().sum()
p = sns.countplot(x="math score", data = data , palette="muted")

_ = plt.setp(p.get_xticklabels(),rotation=90) 
data['Math_PassStatus'] = np.where( data['math score'] < 40 , 'F', 'P')

data.Math_PassStatus.value_counts()
p = sns.countplot(x = 'reading score' , data = data ,palette = "muted")

_ = plt.setp(p.get_xticklabels() , rotation = 90)

data['p_f_reading'] = np.where(data["reading score"] < 40 , 'f' , 'P' )

data['p_f_reading'].value_counts()
p = sns.countplot(x = 'parental level of education', data = data, hue = 'p_f_reading', palette='bright')



p = sns.countplot(x="writing score", data = data, palette="muted")

_ = plt.setp(p.get_xticklabels(), rotation=90) 
data['Writing_PassStatus'] = np.where(data['writing score']<40, 'F', 'P')

data.Writing_PassStatus.value_counts()
p = sns.countplot(x='parental level of education', data = data, hue='Writing_PassStatus', palette='bright')

_ = plt.setp(p.get_xticklabels(), rotation=90) 
#and  data['p_f_reading'] == 'p' and data['Writing_PassStatus'] == 'p'

#x = (data['Math_PassStatus'] == 'P').sum()

x = data['Math_PassStatus'] == 'P'

y = data['p_f_reading'] == 'P' 

z = data['Writing_PassStatus'] == 'P'

#data["all_pass"].value_counts()

#x = data[ data['Math_PassStatus'] == 'P' or x ]

#x

#data["all_p"] = data[ data['Math_PassStatus'] == 'P']

#data["all_p"]
data["all_pass"] = ( ( x & y ) & z )

data["all_pass"] = data.apply( lambda x : 'P' if x["all_pass"] == 1 else 'f' , axis = 1 )

p = sns.countplot(x = 'parental level of education' , data = data  , hue = "all_pass" , palette='bright' )

_ = plt.setp(p.get_xticklabels(), rotation=90) 