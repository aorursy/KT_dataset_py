# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

test = pd.read_csv("../input/cat-in-the-dat-ii/test.csv")

train = pd.read_csv("../input/cat-in-the-dat-ii/train.csv")
test.head(2)
test.shape
train.head(2)
train.shape
#find and handle missing data.

print(test.isnull().sum())
a=test.fillna(0)

a.head(2)
#verify no data is missing and lenghts match

a.info()
import matplotlib.pyplot as plt 

import numpy as np 

import seaborn as sns



sns.countplot(a['nom_3']);
%matplotlib inline



temperature_count = a['ord_2'].value_counts()

sns.set(style="darkgrid")

sns.barplot(temperature_count.index, temperature_count.values, alpha=0.9)

plt.title('Frequency of temperatures')

plt.ylabel('Occurrences', fontsize=12)

plt.xlabel('Temperature', fontsize=12)

plt.show()
#Drop float columns for logistic regression



dropd=a.drop(['bin_0','bin_1','bin_2','ord_0','day', 'month'], axis=1)

#how many categories in each column

#id is still numerical, others are 

dropd.nunique()
#visualize some columns

categorical_features = ['bin_4',"nom_1", "ord_3"]

fig, ax = plt.subplots(1, len(categorical_features))

for i, categorical_feature in enumerate(dropd[categorical_features]):

    dropd[categorical_feature].value_counts().plot("bar", ax=ax[i]).set_title(categorical_feature)

fig.show()
##labelEncoding

#It convert the data in machine-readable data, but it assigns a unique number(from 0 on) to each class of data. 

#This may lead to priority issues in training of data sets:

#that is, a label with high value may be considered to have high priority than a label having lower value.



from sklearn.preprocessing import LabelEncoder



#It requires the category column to be of ‘category’ datatype. By default, a non-numerical column is of ‘object’ type.

#So, it must be changed to ‘category’ type before running it.



dropd['bin_3'] = dropd['bin_3'].astype('category')# Assigning numerical values and storing in another column

dropd['bin_3_Cat'] = dropd['bin_3'].cat.codes

dropd.head(2)

dropd['bin_3'].unique()
#plot

dropd.hist(column='bin_3_Cat', color='green')
#Go back to dataframe: a

#Label-encode the rest of the columns.

a['bin_3'] = a['bin_3'].astype('category')# Assigning numerical values and storing in another column

a['bin_3_Cat'] = a['bin_3'].cat.codes



a['bin_4'] = a['bin_4'].astype('category')# Assigning numerical values and storing in another column

a['bin_4_Cat'] = a['bin_4'].cat.codes



a['nom_0'] = a['nom_0'].astype('category')

a['nom_0_Cat'] = a['nom_0'].cat.codes



a['nom_1'] = a['nom_1'].astype('category')

a['nom_1_Cat'] = a['nom_1'].cat.codes



a['nom_2'] = a['nom_2'].astype('category')

a['nom_2_Cat'] = a['nom_2'].cat.codes



a['nom_3'] = a['nom_3'].astype('category')

a['nom_3_Cat'] = a['nom_3'].cat.codes

a['nom_4'] = a['nom_4'].astype('category')

a['nom_4_Cat'] = a['nom_4'].cat.codes

a['nom_5'] = a['nom_5'].astype('category')

a['nom_5_Cat'] = a['nom_5'].cat.codes

a['nom_6'] = a['nom_6'].astype('category')

a['nom_6_Cat'] = a['nom_6'].cat.codes

a['nom_7'] = a['nom_7'].astype('category')

a['nom_7_Cat'] = a['nom_7'].cat.codes

a['nom_8'] = a['nom_8'].astype('category')

a['nom_8_Cat'] = a['nom_8'].cat.codes

a['nom_9'] = a['nom_9'].astype('category')

a['nom_9_Cat'] = a['nom_9'].cat.codes





a['ord_1'] = a['ord_1'].astype('category')

a['ord_1_Cat'] = a['ord_1'].cat.codes

a['ord_2'] = a['ord_2'].astype('category')

a['ord_2_Cat'] = a['ord_2'].cat.codes

a['ord_3'] = a['ord_3'].astype('category')

a['ord_3_Cat'] = a['ord_3'].cat.codes

a['ord_4'] = a['ord_4'].astype('category')

a['ord_4_Cat'] = a['ord_4'].cat.codes

a['ord_5'] = a['ord_5'].astype('category')

a['ord_5_Cat'] = a['ord_5'].cat.codes



a.head(2)
##Data cleaning

#remove the old, non-numerical columns.

Ints=a.drop(['bin_3','bin_4','nom_0','nom_1','nom_2','nom_3','nom_4','nom_5','nom_6','nom_7','nom_8','nom_9','ord_1', 'ord_2', 'ord_3', 'ord_4','ord_5' ], axis=1)

Ints.head(2)
##Data cleaning



#convert the data to int

Ints.astype('int')
##One-hot encode categories with less than 10 unique entries.

#these are:

#'bin_3_Cat','bin_4_Cat','nom_0_Cat','nom_1_Cat','nom_2_Cat','nom_3_Cat','nom_4_Cat','ord_1_Cat','ord_2_Cat'



from sklearn.preprocessing import OneHotEncoder

 

# creating instance of one-hot-encoder

enc = OneHotEncoder(handle_unknown='ignore')

# passing the label-encoded values 



enc_df = pd.DataFrame(enc.fit_transform(Ints[['bin_3_Cat']]).toarray()) 





# merge with the dataframe on key values

Ints = Ints.join(enc_df)

Ints.head(2)
Ints=Ints.rename(columns={0: "b3_zero", 1: "b3_False", 2: 'b3_True'}) #the original categories 0,F,T

#drop old column

Ints=Ints.drop(['bin_3_Cat'], axis=1)

Ints.head(2)   
# creating instance of one-hot-encoder

enc = OneHotEncoder(handle_unknown='ignore')

# passing the label-encoded values 



enc_df = pd.DataFrame(enc.fit_transform(Ints[['bin_4_Cat']]).toarray()) 



# merge with the dataframe on key values

Ints = Ints.join(enc_df)

Ints=Ints.rename(columns={0: "b4_zero", 1: "b4_NO", 2: 'b4_YES'}) #the original categories 0,N,Y

#drop old columns

Ints=Ints.drop(['bin_4_Cat' ], axis=1)

  
# creating instance of one-hot-encoder

enc = OneHotEncoder(handle_unknown='ignore')

# passing the label-encoded values 



enc_df = pd.DataFrame(enc.fit_transform(Ints[['nom_0_Cat']]).toarray()) 



# merge with the dataframe on key values

Ints = Ints.join(enc_df)

Ints=Ints.rename(columns={0: "n0_zero", 1: "n0_Blue", 2: 'n0_Green', 3: 'n0Red'}) #the original categories Blue, Red, Green, 0

#drop old column

Ints=Ints.drop(['nom_0_Cat' ], axis=1)

# creating instance of one-hot-encoder

enc = OneHotEncoder(handle_unknown='ignore')

# passing the label-encoded values 



enc_df = pd.DataFrame(enc.fit_transform(Ints[['nom_1_Cat']]).toarray()) 



# merge with the dataframe on key values

Ints = Ints.join(enc_df)
Ints=Ints.rename(columns={0: "n1_zero", 1: "n1_Circl", 2: 'n1_PolyG', 3: 'n1_Sqr', 4:'n1_Star', 5: 'n1_TrapZd', 6: 'n1_Triangl'}) # Polygon, Circle, Star, Trapezoid, Triangle, Square, 0 

#drop old column

Ints=Ints.drop(['nom_1_Cat' ], axis=1)

# creating instance of one-hot-encoder

enc = OneHotEncoder(handle_unknown='ignore')

# passing the label-encoded values 



enc_df = pd.DataFrame(enc.fit_transform(Ints[['nom_2_Cat']]).toarray()) 



# merge with the dataframe on key values

Ints = Ints.join(enc_df)
Ints=Ints.rename(columns={0: "n2_zero", 1: "n2_Axolotl", 2: 'n2_CAT', 3: 'n2_Dog', 4:'n2_Hamster', 5: 'n2_Lion', 6: 'n2_Snake'}) # Axolotl, Lion, 0, Dog, Hamster, Snake, Cat 

#drop old column

Ints=Ints.drop(['nom_2_Cat' ], axis=1)

Ints.head(2)
# creating instance of one-hot-encoder

enc = OneHotEncoder(handle_unknown='ignore')

# passing the label-encoded values 



enc_df = pd.DataFrame(enc.fit_transform(Ints[['nom_3_Cat']]).toarray()) 



# merge with the dataframe on key values

Ints = Ints.join(enc_df)

Ints.head(2)
#nom_3   #results in: Finland, Russia, Costa Rica, India, China, Canada, 0

Ints=Ints.rename(columns={0: "n3_zero", 1: "n3_Canada", 2: 'n3_China', 3: 'n3_Costa Rica', 4:'n3_Finland', 5: 'n3_India', 6: 'n3_Russia'})  

#drop old column

Ints=Ints.drop(['nom_3_Cat' ], axis=1)

# creating instance of one-hot-encoder

enc = OneHotEncoder(handle_unknown='ignore')

# passing the label-encoded values 



enc_df = pd.DataFrame(enc.fit_transform(Ints[['nom_4_Cat']]).toarray()) 



# merge with the dataframe on key values

Ints = Ints.join(enc_df)

#nom_4   #results in: Piano, Bassoon, Theremin, Oboe, 0

Ints=Ints.rename(columns={0: "n4_zero", 1: "n4_Bassoon", 2: 'n4_Oboe', 3: 'n4_Piano', 4:'n4_Theremin'})  

#drop old column

Ints=Ints.drop(['nom_4_Cat' ], axis=1)

# creating instance of one-hot-encoder

enc = OneHotEncoder(handle_unknown='ignore')

# passing the label-encoded values 



enc_df = pd.DataFrame(enc.fit_transform(Ints[['ord_1_Cat']]).toarray()) 



# merge with the dataframe on key values

Ints = Ints.join(enc_df)

 # Novice, Expert, Contributor, Grandmaster, Master, 0

Ints=Ints.rename(columns={0: "o1_zero", 1: "o1_Contribtr", 2: 'o1_Exprt', 3: 'o1_GrndMstr', 4:'o1_Master', 5:'o1_Novice'})  

#drop old column

Ints=Ints.drop(['ord_1_Cat' ], axis=1)

# creating instance of one-hot-encoder

enc = OneHotEncoder(handle_unknown='ignore')

# passing the label-encoded values 



enc_df = pd.DataFrame(enc.fit_transform(Ints[['ord_2_Cat']]).toarray()) 



# merge with the dataframe on key values

Ints = Ints.join(enc_df)

#Boiling Hot, Cold, Warm, Hot, Lava Hot, Freezing, 0

Ints=Ints.rename(columns={0: "o2_zero", 1: "o2_Boiling Hot", 2: 'o2_Cold', 3: 'o2_Freezing', 4:'o2_Hot', 5:'o2_Lava Hot', 6: 'o2_Warm'})  

#drop old column

Ints=Ints.drop(['ord_2_Cat' ], axis=1)

#there dont seem to be correlation among the large set of columns.

Ints.corr()
a['nom_5'].value_counts()

#I repeated this for the rest of the columns
##Binning

#For continuous variables, like 'id', you can use optional “bins” argument to separate the values

example=Ints['id'].value_counts(bins=5)

example
#Add a columns based on other column



# it’s each value will be calculated based on other columns in each row i.e.

Ints = Ints.assign(o5_duos_zeros = lambda x: (x['ord_5_Cat']==0 )) #kept 0s separate

Ints = Ints.assign(o5_duos1 = lambda x: (x['ord_5_Cat']>0 )<= 38)

Ints = Ints.assign(o5_duos2 = lambda x: (x['ord_5_Cat']>39 )<= 76)

Ints = Ints.assign(o5_duos3 = lambda x: (x['ord_5_Cat']>77 )<= 114)

Ints = Ints.assign(o5_duos4 = lambda x: (x['ord_5_Cat']>115 )<= 152)

Ints = Ints.assign(o5_duos5 = lambda x: (x['ord_5_Cat']>153 )<= 190)

#drop old column

Ints=Ints.drop(['ord_5_Cat'], axis=1)

Ints.head(2)
#convert boolean to int

Ints['o5_duos_zeros'] = (Ints['o5_duos_zeros']).astype(int)

Ints['o5_duos1'] = (Ints['o5_duos1']).astype(int)

Ints['o5_duos2'] = (Ints['o5_duos2']).astype(int)

Ints['o5_duos3'] = (Ints['o5_duos3']).astype(int)

Ints['o5_duos4'] = (Ints['o5_duos4']).astype(int)

Ints['o5_duos5'] = (Ints['o5_duos5']).astype(int)

#verify

Ints.head(2)

#repeat with ord_3_Cat and ord_4_Cat

Ints = Ints.assign(o3_zeros = lambda x: (x['ord_3_Cat']==0 )) #kept 0s separate

Ints = Ints.assign(o3_lowCase1 = lambda x: (x['ord_3_Cat']>0 )<= 5)

Ints = Ints.assign(o3_lowCase2 = lambda x: (x['ord_3_Cat']>6 )<= 10)

Ints = Ints.assign(o3_lowCase3 = lambda x: (x['ord_3_Cat']>11 )<= 15)

Ints = Ints.assign(o3_lowCase4 = lambda x: (x['ord_3_Cat']>16 )<= 25)

#drop old column

Ints=Ints.drop(['ord_3_Cat'], axis=1)

#convert boolean to int

Ints['o3_zeros'] = (Ints['o3_zeros']).astype(int)

Ints['o3_lowCase1'] = (Ints['o3_lowCase1']).astype(int)

Ints['o3_lowCase2'] = (Ints['o3_lowCase2']).astype(int)

Ints['o3_lowCase3'] = (Ints['o3_lowCase3']).astype(int)

Ints['o3_lowCase4'] = (Ints['o3_lowCase4']).astype(int)

Ints = Ints.assign(o4_zeros = lambda x: (x['ord_4_Cat']==0 )) #kept 0s separate

Ints = Ints.assign(o4_UprCase1 = lambda x: (x['ord_4_Cat']>0 )<= 5)

Ints = Ints.assign(o4_UprCase2 = lambda x: (x['ord_4_Cat']>6 )<= 10)

Ints = Ints.assign(o4_UprCase3 = lambda x: (x['ord_4_Cat']>11 )<= 15)

Ints = Ints.assign(o4_UprCase4 = lambda x: (x['ord_4_Cat']>16 )<= 20)

Ints = Ints.assign(o4_UprCase5 = lambda x: (x['ord_4_Cat']>21 )<= 25)

#drop old column

Ints=Ints.drop(['ord_4_Cat'], axis=1)

#convert boolean to int

Ints['o4_zeros'] = (Ints['o4_zeros']).astype(int)

Ints['o4_UprCase1'] = (Ints['o4_UprCase1']).astype(int)

Ints['o4_UprCase2'] = (Ints['o4_UprCase2']).astype(int)

Ints['o4_UprCase3'] = (Ints['o4_UprCase3']).astype(int)

Ints['o4_UprCase4'] = (Ints['o4_UprCase4']).astype(int)

Ints['o4_UprCase5'] = (Ints['o4_UprCase5']).astype(int)

#Repeat for num_5 to 9, which are random strings series with no obvious distinction among the values

#these are large

#num_5 has 1219 labels

Ints = Ints.assign(n5_zeros = lambda x: (x['nom_5_Cat']==0 )) #kept 0s separate

Ints = Ints.assign(n5_serie1 = lambda x: (x['nom_5_Cat']>0 )<= 243)

Ints = Ints.assign(n5_serie2 = lambda x: (x['nom_5_Cat']>244 )<= 486)

Ints = Ints.assign(n5_serie3 = lambda x: (x['nom_5_Cat']>487 )<= 730)

Ints = Ints.assign(n5_serie4 = lambda x: (x['nom_5_Cat']>731 )<= 975)

Ints = Ints.assign(n5_serie5 = lambda x: (x['nom_5_Cat']>976 )<= 1219)

#drop old column

Ints=Ints.drop(['nom_5_Cat'], axis=1)

#convert boolean to int

Ints['n5_zeros'] = (Ints['n5_zeros']).astype(int)

Ints['n5_serie1'] = (Ints['n5_serie1']).astype(int)

Ints['n5_serie2'] = (Ints['n5_serie2']).astype(int)

Ints['n5_serie3'] = (Ints['n5_serie3']).astype(int)

Ints['n5_serie4'] = (Ints['n5_serie4']).astype(int)

Ints['n5_serie5'] = (Ints['n5_serie5']).astype(int)

#nom_6 has 1517 labels

Ints = Ints.assign(n6_zeros = lambda x: (x['nom_6_Cat']==0 )) #kept 0s separate

Ints = Ints.assign(n6_serie1 = lambda x: (x['nom_6_Cat']>0 )<= 303)

Ints = Ints.assign(n6_serie2 = lambda x: (x['nom_6_Cat']>304 )<= 607)

Ints = Ints.assign(n6_serie3 = lambda x: (x['nom_6_Cat']>608 )<= 909)

Ints = Ints.assign(n6_serie4 = lambda x: (x['nom_6_Cat']>910 )<= 1212)

Ints = Ints.assign(n6_serie5 = lambda x: (x['nom_6_Cat']>1213 )<= 1517)

#drop old column

Ints=Ints.drop(['nom_6_Cat'], axis=1)

#convert boolean to int

Ints['n6_zeros'] = (Ints['n6_zeros']).astype(int)

Ints['n6_serie1'] = (Ints['n6_serie1']).astype(int)

Ints['n6_serie2'] = (Ints['n6_serie2']).astype(int)

Ints['n6_serie3'] = (Ints['n6_serie3']).astype(int)

Ints['n6_serie4'] = (Ints['n6_serie4']).astype(int)

Ints['n6_serie5'] = (Ints['n6_serie5']).astype(int)

#nom_7 and nom_8 have 222 labels each

Ints = Ints.assign(n7_zeros = lambda x: (x['nom_7_Cat']==0 )) #kept 0s separate

Ints = Ints.assign(n7_serie1 = lambda x: (x['nom_7_Cat']>0 )<= 74)

Ints = Ints.assign(n7_serie2 = lambda x: (x['nom_7_Cat']>75 )<= 147)

Ints = Ints.assign(n7_serie3 = lambda x: (x['nom_7_Cat']>148 )<= 222)



#drop old column

Ints=Ints.drop(['nom_7_Cat'], axis=1)

#convert boolean to int

Ints['n7_zeros'] = (Ints['n7_zeros']).astype(int)

Ints['n7_serie1'] = (Ints['n7_serie1']).astype(int)

Ints['n7_serie2'] = (Ints['n7_serie2']).astype(int)

Ints['n7_serie3'] = (Ints['n7_serie3']).astype(int)



#for nom_8

Ints = Ints.assign(n8_zeros = lambda x: (x['nom_8_Cat']==0 )) #kept 0s separate

Ints = Ints.assign(n8_serie1 = lambda x: (x['nom_8_Cat']>0 )<= 74)

Ints = Ints.assign(n8_serie2 = lambda x: (x['nom_8_Cat']>75 )<= 147)

Ints = Ints.assign(n8_serie3 = lambda x: (x['nom_8_Cat']>148 )<= 222)



#drop old column

Ints=Ints.drop(['nom_8_Cat'], axis=1)

#convert boolean to int

Ints['n8_zeros'] = (Ints['n8_zeros']).astype(int)

Ints['n8_serie1'] = (Ints['n8_serie1']).astype(int)

Ints['n8_serie2'] = (Ints['n8_serie2']).astype(int)

Ints['n8_serie3'] = (Ints['n8_serie3']).astype(int)

#nom_9 has 2216

#for nom_9

Ints = Ints.assign(n9_zeros = lambda x: (x['nom_9_Cat']==0 )) #kept 0s separate

Ints = Ints.assign(n9_serie1 = lambda x: (x['nom_9_Cat']>0 )<= 554)

Ints = Ints.assign(n9_serie2 = lambda x: (x['nom_9_Cat']>555 )<= 1108)

Ints = Ints.assign(n9_serie3 = lambda x: (x['nom_9_Cat']>148 )<= 1662)

Ints = Ints.assign(n9_serie4 = lambda x: (x['nom_9_Cat']>1663 )<= 2216)



#drop old column

Ints=Ints.drop(['nom_9_Cat'], axis=1)

#convert boolean to int

Ints['n9_zeros'] = (Ints['n9_zeros']).astype(int)

Ints['n9_serie1'] = (Ints['n9_serie1']).astype(int)

Ints['n9_serie2'] = (Ints['n9_serie2']).astype(int)

Ints['n9_serie3'] = (Ints['n9_serie3']).astype(int)

Ints['n9_serie4'] = (Ints['n9_serie4']).astype(int)
# creating instance of one-hot-encoder

enc = OneHotEncoder(handle_unknown='ignore')

# passing the label-encoded values 



enc_df = pd.DataFrame(enc.fit_transform(Ints[['ord_0']]).toarray()) 

# merge with the dataframe on key values

Ints = Ints.join(enc_df)

Ints=Ints.rename(columns={0: "o0_zero", 1: "o0_1", 2: 'o0_2', 3: 'o0_3'}) # 1,2,3,0 

#drop old column

Ints=Ints.drop(['ord_0'], axis=1)
#day had 0-7 and month and 0-12



#for day, I separate weekday (Monday-Thurd) and weekends (Fri-Sun)

Ints = Ints.assign(day_zeros = lambda x: (x['day']==0 )) #kept 0s separate

Ints = Ints.assign(Weekday = lambda x: (x['day']>0 )<= 4)

Ints = Ints.assign(WkEnd = lambda x: (x['day']>5 )<= 7)



#drop old column

Ints=Ints.drop(['day'], axis=1)

#convert boolean to int

Ints['day_zeros'] = (Ints['day_zeros']).astype(float)

Ints['Weekday'] = (Ints['Weekday']).astype(float)

Ints['WkEnd'] = (Ints['WkEnd']).astype(float)



#verify

Ints.head(2)
#for month, I separated as winter spring summer and fall

Ints = Ints.assign(month_zeros = lambda x: (x['month']==0 )) #kept 0s separate

Ints = Ints.assign(wintr = lambda x: (x['month']>0 )<= 3)

Ints = Ints.assign(sprg = lambda x: (x['month']>4 )<= 6)

Ints = Ints.assign(summr = lambda x: (x['month']>7 )<= 9)

Ints = Ints.assign(fall = lambda x: (x['month']>10 )<= 12)



#drop old column

Ints=Ints.drop(['month'], axis=1)

#convert boolean to int

Ints['month_zeros'] = (Ints['month_zeros']).astype(int)

Ints['wintr'] = (Ints['wintr']).astype(int)

Ints['sprg'] = (Ints['sprg']).astype(int)

Ints['summr'] = (Ints['summr']).astype(int)

Ints['fall'] = (Ints['fall']).astype(int)



#verify

Ints.head(2)
#Pepare a new dataframe without 'zero' columns (these represent NaN values in the original data)

Binaries=Ints.drop(['b4_zero', 'day_zeros', 'b3_zero', 'month_zeros', 'n5_zeros', 'n6_zeros', 'n7_zeros', 'n8_zeros', 'n9_zeros' ], axis=1)

#drop the rest of zeros to 'clean' data 

Binary1=Binaries.drop(['n0_zero', 'n1_zero', 'n2_zero', 'n3_zero', 'n4_zero', 'o1_zero', 'o2_zero', 'o5_duos_zeros', 'o3_zeros', 'o4_zeros', 'o0_zero' ], axis=1)

Binary1.head(2)
##Data cleaning

#even the columns to int

Binary1.astype('int')
#the 'id' column remains a series of continuous data from 600000 to 999999 as identifying rows for target and prediction

#the rest is BINARY (0/1)



from sklearn.linear_model import *

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



#Rename the column 

Binary1=Binary1.rename(columns={'n2_CAT': "CAT"})

Binary1=Binary1.rename(columns={'b4_YES': "YES"})
# roc curve

from sklearn.datasets import make_classification

from sklearn.metrics import roc_curve

from matplotlib import pyplot



X=Binary1['YES']

y=Binary1['CAT']
# generate 2 class dataset

X, y = make_classification(n_samples=2000, n_classes=2, random_state=1)



# split into train/test sets

trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)

# fit a model

model = LogisticRegression()

model.fit(trainX, trainy)
print(model.predict(testX)) ##array



df = pd.DataFrame(model.predict(testX))

print(df)


# predict probabilities

probs = model.predict_proba(testX)

# keep probabilities for the positive outcome only

probs1 = probs[:, 1]

# calculate roc curve

fpr, tpr, thresholds = roc_curve(testy, probs1)

# plot no skill

pyplot.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

pyplot.plot(fpr, tpr)

# show the plot

pyplot.show()
#The integrated area under the ROC curve(AUC) measures the skill of the model across all evaluated thresholds.

from sklearn.metrics import roc_auc_score

# calculate log-loss/AUC

auc = roc_auc_score(testy, probs1)

print(auc)
##Run model on full dataset for prediction

# generate 2 class dataset

X1=Binary1['YES']

y1=Binary1['CAT']



X1, y1 = make_classification(n_samples=400000, n_classes=2, random_state=1)



# fit the defined model

model.fit(X1, y1)
df1 = pd.DataFrame(model.predict(X1))

df1.head(3)
df2 = pd.DataFrame(model.predict_proba(X1))

df2.head(2)
df3 = model.predict_proba(X1)

# keep probabilities for the positive outcome only

df4 = df3[:, 1]



df4 = pd.DataFrame(df4)

df4.head(2)
ids=Binary1[['id']]

ids.head(2)
#join two dataframes

answerSubm = pd.concat([ids, df4], axis=1)

#rename 0 to target



answerSubm=answerSubm.rename(columns={0: "target"})



answerSubm
answerSubm.to_csv('answerSubm.csv',index=False)