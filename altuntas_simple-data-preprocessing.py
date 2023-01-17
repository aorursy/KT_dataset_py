import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import MinMaxScaler     #1. Rescale Data

from sklearn.preprocessing import Binarizer        #2. Binarize Data (Make Binary)

from sklearn.preprocessing import StandardScaler   #3. Standardize Data



import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))



names = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Class']



df = pd.read_csv("../input/pima-indians-diabetes.csv",names=names)

df.head()
df.isnull().sum() #we dont have any missing value I'll show later what we can do with missing values
#1. Rescale Data: It's make our data's values on the same scales.

#Because many algorithms doesn't work when your data have different scales.



array = df.values

scaler = MinMaxScaler(feature_range=(0, 1)) #Making a scaler between 0 and 1

rescalled_df = scaler.fit_transform(array)



#Now we should convert this array to dataset, as most of us know first we write pd.DataFrame than we give names than we choose the column.

rescalled_df = pd.DataFrame(rescalled_df)

rescalled_df.columns = names

rescalled_df.head() 
#2. Binarize Data (Make Binary): It's make our data's 1 or 0, ıt's called binarizing your data or threshold your data.

#I take a little look at this and it's usefull for making threshold pictures



array = df.values

binarizer = Binarizer(threshold=0.0).fit(array)#We are giving array to Binarizer now Binarizer can know which values should 0 which values should be 1

binarized_df = binarizer.transform(array)#then we transform that values 



#Now we should convert this array to dataset, as most of us know first we write pd.DataFrame than we give names than we choose the column.

binarized_df = pd.DataFrame(binarized_df)

binarized_df.columns = names

binarized_df.head()
#3. Standardize Data:It's find which values are to bigger than and to smaller than the avarage.

#then to bigger number's being bigger than 1 to smaller number's being smaller tahn 0



array = df.values

scaler = StandardScaler().fit(array)  #giving array to StandardScaler and it's finding avarage than making magic :)

standard_df = scaler.transform(array)



#Now we should convert this array to dataset, as most of us know first we write pd.DataFrame than we give names than we choose the column.

standard_df = pd.DataFrame(standard_df)

standard_df.columns = names

standard_df.head()
# 4-Outlier detection

def find_outlier(x):

    q1 = np.percentile(x, 25)

    q3 = np.percentile(x, 75)

    iqr = q3-q1

    floor = q1 - 1.5*iqr

    ceiling= q3 + 1.5*iqr

    outlier_indices = list(x.index[(x<floor)|(x>ceiling)])

    outlier_values = list(x[(x<floor)|(x>ceiling)])

    return outlier_indices, outlier_values

outlier_age = np.sort(find_outlier(df.Age)) #These values are the outlier indices and values

print(outlier_age)



outlier_df = df.drop(outlier_age[0])
#5-Interaction among features

#here we make combo with featrues/columns, function take one column and multipcation with another one

from itertools import combinations

from sklearn.preprocessing import PolynomialFeatures

   

def add_interations(df):

    # Get features/columns names

    combos = list(combinations(list(df.columns), 2))

    colnames = list(df.columns) + ['_'.join(x) for x in combos]

        

    # Find interations

    poly = PolynomialFeatures(interaction_only=True, include_bias=False)

    df = poly.fit_transform(df)

    df = pd.DataFrame(df)

    df.columns = colnames

        

    return df

    

df_combo = add_interations(df)

print(df_combo.shape)

df_combo.head()

#Now lets take a look what actually we did



#0-Countplot of pure data

plt.figure(figsize=(8,8))

sns.countplot(x=df.Age)

plt.xticks(rotation=90)

plt.show()



#1-Countplot of resacalled data

plt.figure(figsize=(8,8))

sns.countplot(x=rescalled_df.Age)

plt.xticks(rotation=90)

plt.show()



#2-Countplot of binarized data

plt.figure(figsize=(8,8))

sns.countplot(x=binarized_df.Age)

plt.xticks(rotation=90)

plt.show()



#3-Countplot of standardize Data

plt.figure(figsize=(8,8))

sns.countplot(x=standard_df.Age)

plt.xticks(rotation=90)

plt.show()
# Not much different after drop outlier_age but if you look at carefully you can see the different 

sns.jointplot(x=df.Age.unique(),y=df.Age.value_counts(),kind='reg')

plt.show()



sns.jointplot(x=outlier_df.Age.unique(),y=outlier_df.Age.value_counts(),kind='reg')

plt.show()