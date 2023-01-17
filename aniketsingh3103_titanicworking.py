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
import matplotlib.pyplot as plt

import seaborn as sns
data1 = pd.read_csv("/kaggle/input/titanic/train.csv")

data2 = pd.read_csv("/kaggle/input/titanic/test.csv")

print(data2.describe())
data = data1.append(data2)

len(data)
data['Age'].fillna(0,inplace=True)

sns.set()

f,ax=plt.subplots(figsize=(20,5))

sns.distplot(data['Age'],kde=False,bins=50,color="red")

plt.xlabel("Age")

plt.title("Age disturbation")

plt.ylabel("Count")

plt.show()
data['Pclass'].value_counts().plot(kind="barh")

plt.show()
data[data["Survived"]==1].groupby("Sex")["Survived"].count().plot(kind="barh")

plt.show()
data[data["Cabin"].notna()]["Cabin"].value_counts()

data.fillna({"Cabin":"G6","Embarked":"S"},inplace=True)
#pd.set_option("display.max_rows",None)

#data

data['Age']=data['Age'].astype(int)

data.head(20)
#sns.set()

f,ax=plt.subplots(figsize=(20,5))

sns.distplot(data['Fare'],kde=False,bins=100,color="Green")

#plt.xlim(0,300)

plt.xlabel("Fare")

plt.ylabel("Count")

plt.ylim(0,100)

plt.show()
sns.set_style("whitegrid")

f,ax=plt.subplots(1,2,figsize=(30,10))



male_survived_data = data[(data["Survived"]==1) & ( data["Sex"]=="male")]["Fare"]

female_survived_data = data[(data["Survived"]==1) & ( data["Sex"]=="female")]["Fare"]



male_survived_data_norm = (male_survived_data - male_survived_data.min())/(male_survived_data.max() - male_survived_data.min())

female_survived_data_norm = (female_survived_data-female_survived_data.min())/(female_survived_data.max()-female_survived_data.min())



ax[0].plot(male_survived_data_norm,color="green",alpha=1)

ax[0].plot(female_survived_data_norm,alpha=1,color="orange")



ax[1].plot(male_survived_data,color="green",alpha=1)

ax[1].plot(female_survived_data,alpha=1,color="orange")



ax[0].set_title("Normalized Data")

ax[1].set_title("NON-Normalized Data")

ax[0].set_xlabel("Fare")

ax[0].set_ylabel("People Survived")

ax[1].set_xlabel("Fare")

ax[1].set_ylabel("People Survived")

#plt.xlim(0,1000)

#plt.ylim(0,500)

plt.show()

#data[data["Survived"]==0]["Fare"].count()
data[data["Survived"]==1]["Fare"].max()

data.sample(10)

male_survived_data = (data[(data["Survived"]==1) & ( data["Sex"]=="male")]["Fare"]).max()

male_survived_data
f,ax=plt.subplots(figsize=(28,10))

sns.distplot(data[(data["Survived"]==1) & (data["Sex"]=="male")]["Fare"],color="green",kde=False,hist_kws=dict(alpha=1),bins=100)

sns.distplot(data[(data["Survived"]==1) & (data["Sex"]=="female")]["Fare"],color="blue",kde=False,hist_kws=dict(alpha=1),bins=100)

sns.distplot(data[(data["Survived"]==0) & (data["Sex"]=="male")]["Fare"],color="green",kde=False,hist_kws=dict(alpha=1),bins=100)

sns.distplot(data[data["Survived"]==0]["Fare"],color="red",kde=False,hist_kws=dict(alpha=0.4),bins=100)

plt.title("FARE VS SURVIVED COUNT")

plt.xlabel("FARE")

plt.ylabel("PEOPLE SURVIVED")

plt.xticks(np.arange(0,data["Fare"].max(),step=10))

plt.xlim(0,200)

plt.show()