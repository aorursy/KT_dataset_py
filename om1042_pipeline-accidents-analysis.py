# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/database.csv')



print(df.head())
sns.countplot(df['Operator Name'])
sns.countplot(df['Pipeline Location'])
sns.countplot(df['Pipeline Type'])
df_PipeLine_Type_All_Cost = df[['Pipeline Type','All Costs']]

#print(df_PipeLine_Type_All_Cost)



#sns.pairplot(df_PipeLine_Type_All_Cost);



#sns.stripplot(x='Pipeline Type', y="All Costs", data=df_PipeLine_Type_All_Cost)



#sns.boxplot(x='Pipeline Type', y="All Costs",  data=df_PipeLine_Type_All_Cost)



sns.barplot(x='Pipeline Type', y="All Costs",  data=df_PipeLine_Type_All_Cost)
group_by_Pipeline_Type = df_PipeLine_Type_All_Cost.groupby('Pipeline Type').aggregate(np.sum)

print(group_by_Pipeline_Type)



print('\n After converting dataframe to matrix \n')

df_group_by_Pipeline_Type = pd.DataFrame(group_by_Pipeline_Type).reset_index()

print(df_group_by_Pipeline_Type)
sns.stripplot(x='Pipeline Type', y="All Costs", data=df_group_by_Pipeline_Type)
df_accident_location =df[['Operator ID','Pipeline Type',"All Costs",'Accident Latitude','Accident Longitude']]

print(df_accident_location.head())
#sns.pairplot(df_accident_location);
step = 0.2

to_bin = lambda x: np.floor(x / step) * step

df["latbin"] = df.Latitude.map(to_bin)

df["lonbin"] = df.Longitude.map(to_bin)

groups = df.groupby(("latbin", "lonbin"))
step = 0.2

to_bin = lambda x: np.floor(x / step) * step

df_accident_location["latbin"] = df_accident_location['Accident Latitude'].map(to_bin)

df_accident_location["lonbin"] = df_accident_location['Accident Longitude'].map(to_bin)

groups =pd.DataFrame(df_accident_location.groupby(("latbin", "lonbin")).aggregate(np.sum)).reset_index()

print(groups)

g = sns.JointGrid(x="latbin", y="lonbin", data=groups)

g = g.plot(sns.regplot, sns.distplot)