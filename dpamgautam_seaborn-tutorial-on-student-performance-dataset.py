# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/StudentsPerformance.csv')
data.head(5)
data.tail(5)
data.sample(5)
data.info()
data.describe()
data.dtypes
data.corr()
data.isnull().sum()
for i,col in enumerate(data.columns):

    print(i+1, ". column is : ", col)
#rename columns

data.rename(columns=({'gender':'Gender','race/ethnicity':'Race/Ethnicity'

                     ,'parental level of education':'Parental_Level_of_Education'

                     ,'lunch':'Lunch','test preparation course':'Test_Preparation_Course'

                      ,'math score':'Math_Score','reading score':'Reading_Score'

                     ,'writing score':'Writing_Score'}),inplace=True)
for i,col in enumerate(data.columns):

    print(i+1, ". column is : ", col)
data['Gender'].value_counts()
data['Gender'].unique()
# Gender bar plot



sns.set(style='whitegrid')

ax = sns.barplot(x=data['Gender'].value_counts().index, y=data['Gender'].value_counts().values, palette='Blues_d', hue=['female','male'])

plt.legend(loc=8)

plt.xlabel("Gender")

plt.ylabel('Frequency')

plt.title('show of gender bar plot')

plt.show()
plt.figure(figsize=(7,7))

ax = sns.barplot(x=data["Race/Ethnicity"].value_counts().index, y=data["Race/Ethnicity"].value_counts().values, palette=sns.cubehelix_palette(120))

plt.xlabel("Race/Ethnicity")

plt.ylabel("Frequency")

plt.title("Race/Ethnicity bar plot")

plt.show()
sns.barplot(x="Parental_Level_of_Education", y='Writing_Score', data=data, hue="Gender" )

plt.xticks(rotation=45)

plt.show()
sns.barplot(x="Parental_Level_of_Education", y='Reading_Score', data=data, hue="Gender" )

plt.xticks(rotation=45)

plt.show()
sns.barplot(x="Parental_Level_of_Education", y='Math_Score', data=data, hue="Gender" )

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(10,10))

sns.catplot(x="Gender", y="Math_Score", hue="Parental_Level_of_Education", data=data, kind='bar', height=4, aspect=1.5)

plt.show()
labels = data["Race/Ethnicity"].value_counts().index

values = data["Race/Ethnicity"].value_counts().values

colors = ["red","blue","green","yellow","brown"]

explode = [0,0,0.1,0,0]



plt.figure(figsize=(7,7))

plt.pie(values, labels=labels, colors=colors, explode=explode, autopct="%1.1f%%")

plt.title("Race/Ethnicity analysis")

plt.show()
data.groupby("Race/Ethnicity")["Reading_Score"].mean()
sns.kdeplot(data["Math_Score"])

plt.xlabel("Values")

plt.ylabel("Frequency")

plt.title("Math score kde plot")

plt.show()
sns.kdeplot(data["Reading_Score"],data["Writing_Score"])

plt.show()
data.head(5)
sns.heatmap(data.corr())

plt.show()
sns.heatmap(data.corr(), annot=True)

plt.show()
sns.pairplot(data)

plt.show()
sns.pairplot(data, diag_kind="kde")

plt.show()
sns.pairplot(data, kind="reg")

plt.show()
sns.countplot(data["Race/Ethnicity"])

plt.show()
sns.countplot(data["Gender"])

plt.show()
sns.countplot(data["Math_Score"])

plt.show()