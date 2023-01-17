import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pylab as pl

import pandas as pd

import matplotlib.pyplot as plt 

%matplotlib inline

import seaborn as sns

from sklearn.utils import shuffle

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.model_selection import cross_val_score, GridSearchCV

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



y_2015 = pd.read_csv('/kaggle/input/world-happiness-report/2015.csv')

y_2016 = pd.read_csv('/kaggle/input/world-happiness-report/2016.csv')

y_2017 = pd.read_csv('/kaggle/input/world-happiness-report/2017.csv')

y_2018 = pd.read_csv('/kaggle/input/world-happiness-report/2018.csv')

y_2019 = pd.read_csv('/kaggle/input/world-happiness-report/2019.csv')

y_2020 = pd.read_csv('/kaggle/input/world-happiness-report/2020.csv')



Data= y_2015.append([y_2015,y_2016,y_2017,y_2018, y_2019,y_2020])

x = Data.iloc[:, [3]].values
Data.head()
print("Any missing sample in train set:",Data.isnull().values.any(), "\n")

Data = Data.replace([np.inf, -np.inf], np.nan)

Data =Data.fillna(0)

Data
Data['Happiness_Rank'] = Data['Happiness Rank']

Data
Happiness_c = Data.sort_values(by='Happiness_Rank', ascending=False)[:100]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Happiness_c.Country, x=Happiness_c.Happiness_Rank)

plt.xticks()

plt.xlabel('Happiness_Rank')

plt.ylabel('Country')

plt.title('The Most Happiness Country by Happiness_Rank ')

plt.show()
Data['Happiness_Score'] = Data['Happiness Score']

Data
Happiness_c= Data.sort_values(by='Happiness_Score', ascending=False)[:100]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Happiness_c.Country, x=Happiness_c.Happiness_Rank)

plt.xticks()

plt.xlabel('Happiness_Score')

plt.ylabel('Country')

plt.title('The Most Happiness Country by Happiness_Score ')

plt.show()
Data['Economy_GDPperCapita'] = Data['Economy (GDP per Capita)']

Data
GDP = Data.sort_values(by='Economy_GDPperCapita', ascending=False)[:100]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=GDP.Country, x=GDP.Happiness_Rank)

plt.xticks()

plt.xlabel('Economy_GDPperCapita')

plt.ylabel('Country')

plt.title('Economy (GDP per Capita) each Country')

plt.show()
Fam_c = Data.sort_values(by='Family', ascending=False)[:100]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Fam_c.Country, x=Fam_c.Family)

plt.xticks()

plt.xlabel('Family')

plt.ylabel('Country')

plt.title('The Most Happiness Country by Family')

plt.show()
Freedom_c = Data.sort_values(by='Freedom', ascending=False)[:100]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Freedom_c.Country, x=Freedom_c.Freedom)

plt.xticks()

plt.xlabel('Freedom')

plt.ylabel('Country')

plt.title('The Most Happiness Country by Freedom')

plt.show()
Data['Trust'] = Data['Trust (Government Corruption)']

Data
Trust_c = Data.sort_values(by='Trust', ascending=False)[:100]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Trust_c.Country, x=Trust_c.Trust)

plt.xticks()

plt.xlabel('Trust')

plt.ylabel('Country')

plt.title('The Most Happiness Country by Trust')

plt.show()
Data['Health'] = Data['Health (Life Expectancy)']

Data
Health_c = Data.sort_values(by='Health', ascending=False)[:100]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Health_c.Country, x=Health_c.Health)

plt.xticks()

plt.xlabel('Health')

plt.ylabel('Country')

plt.title('The Most Happiness Country by Health')

plt.show()
# here we are comparing the Trust with each Country

# first group the Country and get max,min and avg of Trust

display(Data[["Country","Trust",]].groupby(["Country"]).agg(["max",'mean',"min"]).style.background_gradient(cmap="Blues"))



# here we are ploting these values using lineplot

Data[["Country","Trust",]].groupby(["Country"]).agg(["max",'mean',"min"]).plot(kind="line",color =["red","black","blue"])

plt.title("Trust (Government Corruption) each Country (max,mean,min)", fontsize=20)

plt.ylabel("Trust",fontsize=15)

plt.xlabel(" ")

plt.show()
# here we are comparing the Freedom with each Country

# first group the Country and get max,min and avg of Freedom

display(Data[["Country","Freedom",]].groupby(["Country"]).agg(["max",'mean',"min"]).style.background_gradient(cmap="BuPu_r"))
# here we are ploting these values using lineplot

Data[["Country","Freedom",]].groupby(["Country"]).agg(["max",'mean',"min"]).plot(kind="line",color =["red","black","blue"])

plt.title("Freedom  each Country (max,mean,min)", fontsize=20)

plt.ylabel("Freedom",fontsize=15)

plt.xlabel(" ")

plt.show()
# here we are comparing the Health with each Country

# first group the Country and get max,min and avg of Health

display(Data[["Country","Health",]].groupby(["Country"]).agg(["max",'mean',"min"]).style.background_gradient(cmap="Oranges"))
# here we are ploting these values using lineplot

Data[["Country","Health",]].groupby(["Country"]).agg(["max",'mean',"min"]).plot(kind="line",color =["red","black","blue"])

plt.title("Health  each Country (max,mean,min)", fontsize=20)

plt.ylabel("Health",fontsize=15)

plt.xlabel(" ")

plt.show()


# here we are comparing the Economy_GDPperCapita with each Country

# first group the Country and get max,min and avg of Economy_GDPperCapita

display(Data[["Country","Economy_GDPperCapita",]].groupby(["Country"]).agg(["max",'mean',"min"]).style.background_gradient(cmap="CMRmap_r"))
# here we are ploting these values using lineplot

Data[["Country","Economy_GDPperCapita",]].groupby(["Country"]).agg(["max",'mean',"min"]).plot(kind="line",color =["red","black","blue"])

plt.title("Economy GDP per Capita  each Country (max,mean,min)", fontsize=20)

plt.ylabel("Economy GDP per Capita",fontsize=15)

plt.xlabel(" ")

plt.show()


# here we are comparing the Family with each Country

# first group the Country and get max,min and avg of Family

display(Data[["Country","Family",]].groupby(["Country"]).agg(["max",'mean',"min"]).style.background_gradient(cmap="GnBu"))
# here we are ploting these values using lineplot

Data[["Country","Family",]].groupby(["Country"]).agg(["max",'mean',"min"]).plot(kind="line",color =["red","black","blue"])

plt.title("Family  each Country (max,mean,min)", fontsize=20)

plt.ylabel("Family",fontsize=15)

plt.xlabel(" ")

plt.show()