# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/crimes-in-boston/crime.csv",encoding = "ISO-8859-1")

df.head()
df.info()
df.describe().transpose()
fig=plt.figure(figsize=(10,8))

sns.heatmap(df.isnull(),cmap="viridis",yticklabels=False,cbar=False)
#inspecting shooting column before dropping
df["SHOOTING"].unique()
df[df["SHOOTING"]=="Y"]
def c(x):

    if x=="Y":

        return "Y"

    else:

        return "N"

        



df["SHOOTING"]=df["SHOOTING"].apply(lambda x:c(x))
fig=plt.figure(figsize=(7,4))

sns.countplot(x="SHOOTING",data=df)
df["SHOOTING"].value_counts()
df["OFFENSE_CODE"].unique()
len(df["OFFENSE_CODE"].unique())
print(df["OFFENSE_CODE"].value_counts().head(10))
print(df["OFFENSE_CODE"].value_counts().head(10).plot(kind="bar"))
#Now lets see about OFFENSE_CODE_GROUP
df["OFFENSE_CODE_GROUP"].nunique()
fig=plt.figure(figsize=(20,8))

sns.countplot(x="OFFENSE_CODE_GROUP",data=df,)

plt.tight_layout()

plt.xticks(rotation=90)
df["OFFENSE_CODE_GROUP"].value_counts().head(10)
df["OFFENSE_CODE_GROUP"].value_counts().head(10).plot(kind="bar",cmap="ocean")
df["DISTRICT"].nunique()
df["DISTRICT"].unique()
df["DISTRICT"].value_counts().head(5)
fig=plt.figure(figsize=(9,4))

sns.countplot(x="DISTRICT",data=df,palette="spring")
# now lets analyse how each distric fares 
od=df["OFFENSE_CODE_GROUP"].value_counts().head(5).index
fig=plt.figure(figsize=(20,7))



sns.countplot(x="DISTRICT",data=df,hue="OFFENSE_CODE_GROUP",hue_order=od)
od2=df["SHOOTING"].value_counts().index[1]

fig=plt.figure(figsize=(20,7))

cp=sns.countplot(x="SHOOTING",data=df[df["SHOOTING"]=="Y"],hue="DISTRICT")

### District with high chances of shooting happening
df["DISTRICT"].unique()

li=['D14', 'C11', 'D4', 'B3', 'B2', 'C6', 'A1', 'E5', 'A7', 'E13',

       'E18', 'A15']
for i in li:

    a=int((len(df[(df["SHOOTING"]=="Y") & (df["DISTRICT"]== i)].index)/len(df[(df["SHOOTING"]=="Y")].index))*100)

    print("The chance of Shooting happening in the District {} is almost {} %".format(i,a))
df.groupby("YEAR").count()["OFFENSE_CODE_GROUP"]
a=len(df[df["YEAR"]==2015].index)/201

print("Number of crimes in the year 2015 per day on an average is {}".format(a))

b=len(df[df["YEAR"]==2016].index)/366

print("Number of crimes in the year 2016 per day on an average is {}".format(b))

c=len(df[df["YEAR"]==2017].index)/365

print("Number of crimes in the year 2017 per day on an average is {}".format(c))

d=len(df[df["YEAR"]==2018].index)/246

print("Number of crimes in the year 2018 per day on an average is {}".format(d))
fig=plt.figure(figsize=(20,7))

sns.countplot(x="DISTRICT",data=df,hue="YEAR",palette="mako")
df["OFFENSE_CODE_GROUP"].value_counts().tail(10)
df["MONTH"].value_counts()
sns.countplot("MONTH",data=df)
fig=plt.figure(figsize=(20,7))

sns.countplot("MONTH",data=df,hue="OFFENSE_CODE_GROUP",hue_order=od)
fig=plt.figure(figsize=(20,7))

sns.countplot(x="HOUR",data=df,hue="DAY_OF_WEEK",palette="viridis")
df["DAY_OF_WEEK"].value_counts()
fig=plt.figure(figsize=(20,7))

sns.countplot(x="DAY_OF_WEEK",data=df,palette="summer")

df["UCR_PART"].nunique()
fig=plt.figure(figsize=(20,7))

sns.countplot(x="UCR_PART",data=df)
df["STREET"].fillna("Unknown", inplace = True) 
fig=plt.figure(figsize=(20,7))

sns.countplot(y='STREET',order=df.STREET.value_counts().head(10).index,data=df)
### 