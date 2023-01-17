# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns; sns.set()

# Disabling warnings
import warnings
warnings.simplefilter("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/insurance/insurance.csv")
df=data.copy()
display(df.head())
display(df.tail())
df.info()
sex = df.groupby('sex').size()
print(sex)
smoker = df.groupby('smoker').size()
print(smoker)
region = df.groupby('region').size()
print(region)
#The data is very much balanced between sex and region. 
#On the other hand, non-smokers outnumber the smokers.
df.describe()
df.isnull().sum() #There is not NaN value
region_list=list(df['region'].unique())
region_list  #There are 4 region in data.
children_list=list(df['children'].unique())
children_list #There are 6 differents options of children.
df.corr()  # Prints correlation for the numerical columns.

f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.2f',ax=ax)
plt.show()
sns.scatterplot(x="bmi", y="charges", data=df) 
#BMI ile ucret arasindaki ilk gorunum.
sns.set(style = "ticks")
sns.pairplot(df, hue = "smoker")
sns.scatterplot(x="bmi", y="charges", data=df, hue='smoker')
sns.lmplot(x = "bmi", y = "charges", hue="smoker",data = df);
df.groupby('smoker')[['charges','bmi']].corr() 
#sigara icenlerin koralasyonunun daha yuksek oldugunu goruyoruz
## check the distribution of charges
distPlot = sns.distplot(df['charges'])
plt.title("Distirbution of Charges")
plt.show(distPlot)
# 1) Charges Between Gender
meanGender = data.groupby(by = "sex")["charges"].mean()
print(meanGender)
sns.violinplot(x = "sex", y = "charges", data = df);
# 2) Charges between Smokers and non-Smokers
meanSmoker = data.groupby(by = "smoker")["charges"].mean()
print(meanSmoker)
print(meanSmoker["yes"] - meanSmoker["no"])
sns.violinplot(x = "smoker", y = "charges", data = df);
#3)Charges Among Regions
meanRegion = data.groupby(by = "region")["charges"].mean()
print(meanRegion)
sns.violinplot(x = "region", y = "charges", data = df);
labels = df.groupby('region').mean().index
colors = ['grey','blue','red','yellow']
explode = [0.02,0.02,0.2,0.02]
sizes = df.groupby('region').sum()['charges']
plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%')
plt.title('Region - Charges',color = 'blue',fontsize = 15);
#4)The following shows the relationship of the medical charges to other
#numerical variables.
pairPlot = sns.pairplot(df)
sns.set(style = "ticks")
sns.pairplot(df, hue = "smoker")
#Creating another column containing bins of charges
df['charges_bins'] = pd.cut(df['charges'], bins=[0, 15000, 30000, 45000, 60000, 75000])

df.head()
#Creating a countplot based on the amount of charges
plt.figure(figsize=(12,4))
sns.countplot(x='charges_bins',data=df) 
plt.title('Number of pepople paying x amount\n for each charges category', size='23')
plt.xticks(rotation='25')
plt.ylabel('Count',size=18)
plt.xlabel('Charges',size=18)
plt.show()
#Making bins for the ages
df['age_bins'] = pd.cut(df['age'], bins = [0, 20, 35, 50, 70])

#Creating boxplots based on the amount of different age categories
plt.figure(figsize=(12,4))
sns.boxplot(x='age_bins', y='charges', data=df) 
plt.title('Charges according to age categories', size='23')
plt.xticks(rotation='25')
plt.grid(True)
plt.ylabel('Charges',size=18)
plt.xlabel('Age',size=18)
plt.show()
#Countplot for different 'number of children' categories
plt.figure(figsize=(12,4))
sns.countplot(x='children', data=df) 
plt.title('Number of pepople having x children', size='23')
plt.ylabel('Count',size=18)
plt.xlabel('Number of children',size=18)
plt.show()
#Charges according to number of children
#Creating a violinplot for each category
plt.figure(figsize=(12,4))
sns.violinplot(x='children', y='charges', data=df, hue='sex')
plt.title('Charges according to number of children', size='23')
plt.ylabel('Charges',size=18)
plt.xlabel('Number of children',size=18)
plt.show()
sns.lineplot(x="age", y='bmi',data = df);
plt.figure(figsize=(10,8))
sns.scatterplot(x='age',y='bmi', data=df);
sns.lmplot(x = "age", y = "bmi",data = df);
plt.figure(figsize=(40,54))
sns.catplot(x='sex', y="bmi", kind="box",data=df);
plt.xticks(rotation=30);
plt.figure(figsize=(40,54))
sns.catplot(x='children', y="bmi", kind="box",hue="sex",data=df);

