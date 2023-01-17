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
        
train=pd.read_csv('../input/real-estate/train.csv')
test=pd.read_csv('../input/real-estate/test.csv')

train.head(10)

# %% [code]
print(train.isna().sum())
print(test.isna().sum())

# %% [code]
train.drop(['BLOCKID'],axis=1,inplace=True)
test.drop(['BLOCKID'],axis=1,inplace=True)

# %% [code]
#Gives each columns with sum of null values

def Cal_Null_Value(data):
    Null_Variables=[]
    c=0
    for i in data.columns:
         if data[i].isna().any()==True:
            print(i,end=' ')
            print(data[i].isna().sum())
            Null_Variables.append(i)
            c=c+1
    print('Numbers of columns having Null values = ',c) #Prints total columns having null values    
    print(Null_Variables)
    return Null_Variables

# %% [code]
Null_val_Colm=Cal_Null_Value(train)

# %% [code]
Cal_Null_Value(test)

# %% [code]
from sklearn.impute import SimpleImputer
imp=SimpleImputer(strategy='mean')
#train['rent_mean']=imp.fit_transform(train[['rent_mean']]).ravel()
for i in Null_val_Colm:
    
    train[i]=imp.fit_transform(train[[i]]).ravel()
    test[i]=imp.fit_transform(test[[i]]).ravel()

# %% [code]
print(train.isna().sum())
print(test.isna().sum())

# %% [code]
train.describe()

# %% [code]
#Exploring the top 2,500 locations where the percentage of households with a 
#second mortgage is the highest and percent ownership is above 10 percent

# %% [code]
y=train.sort_values(by='second_mortgage',ascending=False)

# %% [code]
Percent_ownership=train[train['home_equity']>0.10]

# %% [code]
Top_2500_Location=Percent_ownership.sort_values(by='second_mortgage',ascending=False).head(2500)

# %% [code]
Top_2500_Location=Top_2500_Location[['state','city','state_ab','place','lat','lng']]

# %% [code]
Top_2500_Location.head()

# %% [code]
Top_2500_Location.info()

# %% [code]
#Visualizing using geo-map.
import geopandas as gpd

# %% [code]
gdf = gpd.GeoDataFrame(Top_2500_Location, geometry=gpd.points_from_xy(x=Top_2500_Location.lng, y=Top_2500_Location.lat))

# %% [code]
gdf

# %% [code]
Top_2500_Location['city'].nunique

# %% [code]
import geopandas as gpd
import matplotlib.pyplot as plt

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# We restrict to South America.
ax = world[world.name=='United States of America'].plot(
    color='white', edgecolor='black',figsize=(40,10))

# We can now plot our ``GeoDataFrame``.
gdf.plot(ax=ax,column='state',markersize=8,figsize=(40,10))

plt.show()

# %% [code]
#Creating Bad debt
Bad_Debt=train['second_mortgage']+train['home_equity']-train['home_equity_second_mortgage']
Bad_Debt_test=test['second_mortgage']+test['home_equity']-test['home_equity_second_mortgage']

# %% [code]
#Adding column to dataframe
train['Bad_Debt']=Bad_Debt
test['Bad_Debt']=Bad_Debt_test

# %% [code]
train['debt'].mean()*100

# %% [code]
train['Bad_Debt'].mean()*100

# %% [code]
#Pie charts to show overall debt and bad debt

import matplotlib.pyplot as plt

# Data to plot
labels = 'Debt', 'Bad_Debt'
sizes = [train['debt'].mean()*100, train['Bad_Debt'].mean()*100]
colors = [ 'lightskyblue','lightcoral']
explode = (0.1, 0)  # explode 1st slice

# Plot
plt.pie(sizes,explode=explode,labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()

# %% [code]
#Box and whisker plot and analyze the distribution for 2nd mortgage, home equity, good debt,

# %% [code]
import seaborn as sns
ax = sns.boxplot(train['second_mortgage'])

# %% [code]
df = train[['debt','home_equity','second_mortgage','Bad_Debt']]
df.plot.box(grid='True')

# %% [code]
#Creating a collated income distribution chart for family income, house hold income.

# %% [code]
plt.hist(train['hi_mean'])
plt.show()

# %% [code]
plt.hist(train['family_mean'])
plt.show()

# %% [code]
#Using pop and ALand variables to create a new field called population density
Pop_Density=train['pop']/train['ALand']
Pop_Density_test=test['pop']/test['ALand']

# %% [code]
train['Pop_Density']=Pop_Density#Adding to dataframe
test['Pop_Density']=Pop_Density_test

# %% [code]
train['Pop_Density']

# %% [code]
plt.hist(train['Pop_Density'])
plt.show()

# %% [code]
#Using male_age_median, female_age_median, male_pop, and female_pop to create a new field called median age 
Age=train[['male_age_median','female_age_median','male_pop','female_pop']]
Age_test=test[['male_age_median','female_age_median','male_pop','female_pop']]

# %% [code]
x=Age['male_pop']+Age['female_pop']
y=Age['male_age_median']+Age['female_age_median']
median_age=x/y

x_test=Age_test['male_pop']+Age_test['female_pop']
y_test=Age_test['male_age_median']+Age_test['female_age_median']
median_age_test=x_test/y_test

# %% [code]
#Adding to dataframe
train['median_age']=median_age
test['median_age']=median_age_test

# %% [code]
train['median_age']

# %% [code]
plt.hist(train['median_age'])
plt.show()

# %% [code]
#Creating bins for population into a new variable by selecting appropriate class interval 
#so that the number of categories don’t exceed 5 for the ease of analysis.

# %% [code]
pd.cut(train['pop'],5,labels=False).unique()

# %% [code]
#Analyzing the married, separated, and divorced population for these population brackets

# %% [code]
plt.scatter(pd.cut(train['pop'],5,labels=False),train['married'])

# %% [code]
plt.scatter(pd.cut(train['pop'],5,labels=False),train['separated'])

# %% [code]
plt.scatter(pd.cut(train['pop'],5,labels=False),train['divorced'])

# %% [code]
#Visualizing using appropriate chart type

# %% [code]
sns.boxplot(pd.cut(train['pop'],5,labels=False),train['divorced'])

# %% [code]
sns.boxplot(pd.cut(train['pop'],5,labels=False),train['married'])

# %% [code]
df = train[['married','divorced']]
df.plot.box(grid='True')

# %% [code]
train['state_ab'].unique()

# %% [code]
# Rent as a percentage of income for different states.

# %% [code]
sns.set(rc={'figure.figsize':(15.7,10.27)})

sns.barplot(x='state_ab', y="rent_mean",data=train)
#HI,CA,DC,MD has the top rent as a percentage of income

# %% [code]
#Performing correlation analysis for all the relevant variables by creating a heatmap.

# %% [code]
sns.set(rc={'figure.figsize':(8.7,5.27)})

sns.heatmap(data=train[['hc_mortgage_mean','ALand','pop','rent_mean','hi_mean','hc_mean','family_mean','hs_degree','debt','home_equity']].corr(),annot=True)
#rent_mean,hi_mean,hc_mean,family_mean has a good co relation with the target varianle i.e-hc_mortagage_mean

# %% [code]
#Now Finding the LATENT VARIABLES

# %% [code]
from sklearn.decomposition import PCA

# %% [code]
pca=PCA(n_components=1)

# %% [code]
Highschool_graduation_rates=pca.fit_transform(train[['hs_degree','hs_degree_male','hs_degree_female']])
Highschool_graduation_rates_test=pca.fit_transform(test[['hs_degree','hs_degree_male','hs_degree_female']])
train['Highschool_graduation_rates']=Highschool_graduation_rates
test['Highschool_graduation_rates']=Highschool_graduation_rates_test

# %% [code]
Highschool_graduation_rates

# %% [code]
sns.pairplot(train[['hs_degree','hs_degree_male','hs_degree_female']])

# %% [code]
sns.heatmap(train[['hs_degree','hs_degree_male','hs_degree_female']].corr())

# %% [code]
Median_population_age=pca.fit_transform(train[['male_age_mean','male_age_median','male_age_stdev','female_age_mean','female_age_median','female_age_stdev']])
Median_population_age_test=pca.fit_transform(test[['male_age_mean','male_age_median','male_age_stdev','female_age_mean','female_age_median','female_age_stdev']])
train['Median_population_age']=Median_population_age
test['Median_population_age']=Median_population_age_test


# %% [code]
Median_population_age

# %% [code]
sns.pairplot(train[['male_age_mean','male_age_median','male_age_stdev','female_age_mean','female_age_median','female_age_stdev']])

# %% [code]
sns.heatmap(train[['male_age_mean','male_age_median','male_age_stdev','female_age_mean','female_age_median','female_age_stdev']].corr())

# %% [code]
Second_mortgage_statistics=pca.fit_transform(train[['home_equity_second_mortgage','second_mortgage']])
Second_mortgage_statistics_test=pca.fit_transform(test[['home_equity_second_mortgage','second_mortgage']])
train['Second_mortgage_statistics']=Second_mortgage_statistics
test['Second_mortgage_statistics']=Second_mortgage_statistics_test


# %% [code]
Second_mortgage_statistics

# %% [code]
sns.pairplot(train[['home_equity_second_mortgage','second_mortgage']])

# %% [code]
Bad_debt_expense=pca.fit_transform(train[['debt','debt_cdf','Bad_Debt']])
Bad_debt_expense_test=pca.fit_transform(test[['debt','debt_cdf','Bad_Debt']])
train['Bad_debt_expense']=Bad_debt_expense
test['Bad_debt_expense']=Bad_debt_expense_test

# %% [code]
Bad_debt_expense

# %% [code]
train.isna().any().sum()

# %% [code]
#Removing some columns which are alreadt used to create new variables and which do not have reasonable impact in resutls
train=train.drop(['primary',
                   'state_ab','SUMLEVEL','UID',
                   'hs_degree','hs_degree_male',
                   'hs_degree_female',
                   'male_age_mean','male_age_median','male_age_stdev',
                   'female_age_mean',
                   'female_age_median','female_age_stdev',
                   'home_equity_second_mortgage','second_mortgage','debt',
                   'debt_cdf',
                   'Bad_Debt','male_pop','female_pop'],axis=1)

test=test.drop(['primary',
                   'state_ab','SUMLEVEL','UID',
                   'hs_degree','hs_degree_male',
                   'hs_degree_female',
                   'male_age_mean','male_age_median','male_age_stdev',
                   'female_age_mean',
                   'female_age_median','female_age_stdev',
                   'home_equity_second_mortgage','second_mortgage','debt',
                   'debt_cdf',
                   'Bad_Debt','male_pop','female_pop'],axis=1)

# %% [code]
train.head()

# %% [code]
from sklearn.preprocessing import LabelEncoder

# %% [code]
lb=LabelEncoder()

# %% [code]
train['state']=lb.fit_transform(train['state'])
train['city']=lb.fit_transform(train['city'])
train['place']=lb.fit_transform(train['place'])
train['type']=lb.fit_transform(train['type'])

test['state']=lb.fit_transform(test['state'])
test['city']=lb.fit_transform(test['city'])
test['place']=lb.fit_transform(test['place'])
test['type']=lb.fit_transform(test['type'])


# %% [code]
train.head()

# %% [code]
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

# %% [code]
train1=scaler.fit_transform(train)
train= pd.DataFrame(train1, columns = train.columns)

# %% [code]
test1=scaler.fit_transform(test)
test= pd.DataFrame(test1, columns = test.columns)

# %% [code]
train.head()

# %% [code]
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# %% [code]
x=train.drop(['hc_mortgage_mean'],axis=1)

# %% [code]


# %% [code]
y=train['hc_mortgage_mean']

# %% [code]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# %% [code]
from sklearn.linear_model import LinearRegression
lr=LinearRegression()

# %% [code]
lr.fit(X_train,y_train)

# %% [code]
ypredict=lr.predict(X_test)

# %% [code]
r2_score(ypredict,y_test)


x=test.drop(['hc_mortgage_mean'],axis=1)
y=test['hc_mortgage_mean']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
lr.fit(X_train,y_train)
ypredict_test=lr.predict(X_test)

# %% [code]
print(r2_score(ypredict_test,y_test))
