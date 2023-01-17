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

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df=pd.read_csv("../input/cardiogoodfitness/CardioGoodFitness.csv")

df.head()
df.info()
df.describe()
df.duplicated().sum()
#count of male and female

count_M=df[df['Gender']=='Male'].count()

male_count=count_M['Gender']

count_F=df[df['Gender']=='Female'].count()



Female_count=count_F['Gender']
#percent of male and female

male_percentage=(male_count/df.shape[0])*100

female_percentage=(Female_count/df.shape[0])*100



print(male_percentage)

print(female_percentage)
pieLabels=['Male','Female']

percentage=[57.77,42.22]

plt.pie(percentage,labels=pieLabels,autopct='%1.2f',startangle=90)

plt.legend()
df.hist(figsize=(20,10))
sns.boxplot(x="Gender",y="Age",data=df)
sns.pairplot(data=df)
sns.countplot(x='Product',hue='Gender',data=df)
sns.countplot(x='Product',hue='MaritalStatus',data=df)
df['Age'].std()
sns.distplot(df['Age'])
df.hist(by='Gender',column='Income');
df.hist(by='Gender',column='Miles');
corr=df.corr()

sns.heatmap(corr,annot=True);
from sklearn.linear_model import LinearRegression

x = df[['Usage','Fitness']]

y = df['Miles']

lm=LinearRegression()



# Train the model using the training sets

lm.fit(x,y)
lm.coef_
lm.intercept_
#prediction miles==-56.74+usage*20.21+fitnesss*27.20