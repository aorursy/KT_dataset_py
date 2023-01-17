from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os 

import pandas as pd 

import seaborn as sns

sns.set(rc={'figure.figsize':(10,10)}) 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
raw = pd.read_csv("/kaggle/input/buildingdatagenomeproject2/hotwater_cleaned.csv", index_col = "timestamp", parse_dates = True)

raw.head()
raw.index.dtype
raw.shape
import missingno as msno 

  

msno.matrix(raw) 
raw.dtypes
#dropping columns that have more than 75% of values as 0

clean = raw.copy()



for col in clean.columns:

    count = 0

    for row in clean.index:

        if clean.loc[row, col] == 0.0:

                count += 1

        per = (count/(clean.shape[0])) * 100

    if per > 75.0:

        clean = clean.drop(col, axis = 1)
clean.head()
#separating building type depending on column name

#education

edu = pd.DataFrame()

education = [col for col in clean.columns if 'education' in col]

edu[education] = clean[education]



office = pd.DataFrame()

office_col= [col for col in clean.columns if 'office' in col]

office[office_col] = clean[office_col]



rel = pd.DataFrame()

religion = [col for col in clean.columns if 'religion' in col]

rel[religion] = clean[religion]



ass = pd.DataFrame()

assembly = [col for col in clean.columns if 'assembly' in col]

ass[assembly] = clean[assembly]



lodge = pd.DataFrame()

lodging = [col for col in clean.columns if 'lodging' in col]

lodge[lodging] = clean[lodging]



unknown = pd.DataFrame()

un = [col for col in clean.columns if 'unknown' in col]

unknown[un] = clean[un]



#there are other building types like retail, science, health, public but these are in the minority.
edu.head()
types = [edu, office, lodge, rel, ass, unknown]
#checking for outliers
from scipy import stats



def outliers_col(df):

    for column in df:

        if df[column].dtype != np.object:

            n_outliers = len(df[(np.abs(stats.zscore(df[column])) > 3)& \

                  (df[column].notnull())

                 ])

            print("{} | {} | {}".format(

                df[column].name,

                n_outliers,

                df[column].dtype

        ))



outliers_col(clean)
def plotting(list):

    for i in list:

        i = i.iloc[:50,:10]

        i.plot()

plotting(types)
edu.plot()
edu.resample("D", axis = 0).sum()

edu.plot()
#function shows the percentage of missing values and type of the values

def missing_data(data):

    percent = (data.isnull().sum() / data.isnull().count())

    x = pd.concat([percent], axis=1, keys=['Percentage_of_Missing_Values'])

    type = []

    

    for col in data.columns:

        dtype = str(data[col].dtype)

        type.append(dtype)

    x['Data Type'] = type

    

    return(np.transpose(x))
missing = missing_data(clean)

missing
#removing columns that have more than 50% missing values

for col in clean.columns:

    if missing.loc["Percentage_of_Missing_Values", col] >= .5:

        clean = clean.drop(col, axis = 1)
clean.shape
msno.matrix(clean)
#interpolating

clean = clean.interpolate(method='slinear')
msno.matrix(clean)
#few missing values

#back propagation fill

clean = clean.fillna(method='bfill')



#forward propagation fill 

clean = clean.fillna(method='ffill') 
msno.matrix(clean)
edu, office, lodge, rel, ass, unknown

#edu.to_csv("education_hot_water.csv")

#office.to_csv("office_hot_water.csv")

#lodge.to_csv("lodge_hot_water.csv")

#rel.to_csv("rel_hot_water.csv")

#ass.to_csv("ass_hot_water.csv")

#unknown.to_csv("unknown_hot_water.csv")

clean.to_csv("hot_water_cleaned.csv")