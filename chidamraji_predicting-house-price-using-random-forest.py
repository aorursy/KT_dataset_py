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
# Goal of this kernel:
# 1. Understanding the features better
# 2. Feature Engineering
# 3. Model building
#Loading the data from the training set
import pandas as pd

df = pd.read_csv('../input/train.csv')
df.head()
#1. Understanding the features better
# Before I started with feature engineering, I researched about this project to understand the outliers.
# I stumbled upon Dean De Cock's (who provided the Ames Housing Dataset to Kaggle) project re Ames Housing Dataset
# (http://jse.amstat.org/v19n3/decock.pdf). The pdf states that there are 5 data points that can be considered as
#as outliers, including 'GrLivArea' > 4000 sq.ft. 

#To check for outliers
import matplotlib.pyplot as plt
%matplotlib inline
fig, ax = plt.subplots()
plt.scatter(df['GrLivArea'], df['SalePrice'])
ax.set_xlabel("Above grade living area square feet")
ax.set_ylabel("SalePrice");

#I have removed the four points (as can be seen from the plot) that are 
#above 4000 sq.ft. and the one point that is above 3600 sq.ft.

df[df['GrLivArea']>3600]#there are two points above 3600 and less than 4000, but removed the point (out of the two) 
#that's with SalePrice=625000 on top of the four points that are above 4000 sq.ft.

#To remove the outliers
df.drop(df.index[[523,691,1182,1298, 1169]], inplace=True)
df=df.reset_index(drop=True)#here drop=True means to drop the alreading existing index column
#Visualizations to understand the features
# How to go about the same? As there are many features. Did some research and figured the primary features that homebuyers
# look for and that can be categorized on a high level under: must have and nice to have features. The features
#can further be divided into: living (space, bedroom, bathrooms, number of rooms), garage, area, location,
#amenities, paved street, height of basement, number of fire

#There seem to exist other significant factors over HalfBath affecting the SalePrice. Majority of the houses seemed to have
# 0 or 1 HalfBath. So having 2 HalfBaths looks like nice to have and not must have.TotRmsAbvGrd, GarageArea(although clustered and therefore might exhibit a non-linear trend), 
# TotalBsmtSF, and GrLivArea seem significant factors affecting the  'SalePrice'. It hard to interpret 'LotArea' as it is on 
# a completely different scale so let us visualize it against the 'SalePrice' separately.
import seaborn as sns
ax=sns.pairplot(df, x_vars=['LotArea','TotRmsAbvGrd','HalfBath','GarageArea','TotalBsmtSF','GrLivArea'], y_vars=['SalePrice'], palette="husl");


#To visualize 'YearBuilt', 'YearRemodAdd','MoSold', 'YrSold' Vs. 'SalePrice'. 
#Interestingly, the houses that were built in 2000 seem to have an increasing trend with 'SalePrice'. 
#Also the houses that are recently remodeled seem to have affected the 'SalePrice' positively. 

#There is no clear negative trend in 'SalePrice' because of the 2008 financial crisis. Just to verify this claim we 
#shall get the count on the houses sold in each year. It is a known fact that monnths play an important role in 
#affecting the selling of a house. Although there is no clear dividing or surprising trend amongst the months 
#against 'SalePrice', let us take a closer look to ensure the same.

ax=sns.pairplot(df, x_vars=['YearBuilt', 'YearRemodAdd','MoSold', 'YrSold'], y_vars=['SalePrice'], palette="husl")
#Ought to express months in words
import calendar
df['MoSold'] = df['MoSold'].apply(lambda x: calendar.month_abbr[x])


#'MoSold' Vs. 'SalePrice. After 400000 in every month the values get sparser meaning fewer values or outliers
ax = sns.boxplot(x='MoSold', y='SalePrice', data=df, linewidth=2.5);
                 
#Let's visualize 'Neighborhood' Vs. 'SalePrice'. Residential Low Density is prevalent in every neighborhood. Although the
# data points above 400000 look a lot like outliers with re to months they appear within range values depending on
# neighborhoods, so planning to have the points around.

# Low Density Residential Zoning Definition. "Low density residential zones" are locations intended for housing that 
# include a lot of open space. These zones are meant for a small number of residential homes, and exclude large 
# industries, apartment complexes, and other large structures.

plt.figure(figsize=(20,10))
ax = sns.boxplot(y='SalePrice', x='Neighborhood', data=df, linewidth=2.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=-80);


#Plotted to understand any unusual highs and lows in neighborhoods with re to the year the houses were sold. But ended up
# say no clear pattern 
plt.figure(figsize=(20,10))
ax = sns.boxplot(y='SalePrice', x='Neighborhood', hue=df['YrSold'], data=df, linewidth=2.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=-80);
#'SaleCondition' Vs. 'SalePrice'

plt.figure(figsize=(20,10))
ax = sns.boxplot(y='SalePrice', x='SaleCondition', hue=df['YrSold'],data=df, linewidth=2.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=-80);




#'KitchenAbvGr', 'KitchenQual', 'SalePrice'
#It seems like kitchen at excellent quality sold at higher selling price

plt.figure(figsize=(20,10))
ax = sns.boxplot(y='SalePrice', x='KitchenQual', hue=df['KitchenAbvGr'], data=df, linewidth=2.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=-80);



#GarageCars Vs. SalePrice
#Surprisingly, GarageCars proved to be a signficant factor (this will come up during model building)

plt.figure(figsize=(20,10))
ax = sns.boxplot(y='SalePrice', x='GarageCars', data=df, linewidth=2.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=-80);



#There is no clear pattern re whether the 2008 financial crisis was a major cause in affecting how home buyers chose which
#house to buy and in which neighborhood. Mostly Single-family Detached houses were sold in the low density residential zone 
# which offers more open space. Not sure whether it was super cheap to buy houses in such zones as it varies from place to place and therefore from 
# state to state. It could mean that we are dealing with completely different set of homebuyers who had money to buy house
# (may be the house prices were cheap and so home buyers took advantage of the situation)
# and as the matter of fact most of the sale conditions is normal

df.groupby('YrSold')['BldgType'].value_counts()
#'Condition1' Vs. 'SalePrice'

# Although proximity of houses to amenities, school, and parks matter and therefore affect the SalePrice positively, not 
# much useful insights could be driven from the following plot. Need more info on proximity: Adjacent to arterial street
# and Adjacent to feeder street as it turns out both the options can be an advantage or disadvantage depending 
# on how well they are designed in spite of next to heavy traffic zone.



plt.figure(figsize=(20,10))
ax = sns.swarmplot(x='Condition1', y='SalePrice', data=df, linewidth=2.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=-80);
#LotShape Vs. SalePrice
#But LotShape did not prove to be a strong signal as I built models with and without it.
plt.figure(figsize=(20,10))
ax = sns.pointplot(x='LotShape', y='SalePrice', data=df, linewidth=2.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=-80);
#MasVnrType Vs. SalePrice

plt.figure(figsize=(20,10))
ax = sns.boxplot(x='MasVnrType', y='SalePrice', data=df, linewidth=2.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=-80);


#BuiltIn and Attchd were the two favorable garage choices

plt.figure(figsize=(20,10))
ax = sns.pointplot(x='GarageType', y='SalePrice', data=df, linewidth=2.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=-80);



#Looks like average garage condition was good enough to attract potential home buyers. Not a big deciding factor.

plt.figure(figsize=(20,10))
ax = sns.stripplot(x='GarageCond', y='SalePrice', data=df, linewidth=2.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=-80);
#KitchenAbvGr: Number of kitchens. As can be seen even one kitchen at an excellent quality would suffice to win the heart
#of home buyers.

plt.figure(figsize=(20,10))
ax = sns.boxplot(x='KitchenAbvGr', y=df['SalePrice'], data=df, hue='KitchenQual', linewidth=2.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=-80);

#Number of full bathrooms seemed to have influenced, but more importantly the height of basement what mattered the most!

plt.figure(figsize=(20,10))
ax = sns.boxplot(x='BsmtQual', y=df['SalePrice'], data=df, hue='BsmtFullBath',linewidth=2.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=-80);

  
#Fullbath Vs. SalePrice
plt.figure(figsize=(20,10))
ax = sns.boxplot(x='FullBath', y=df['SalePrice'], data=df, linewidth=2.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=-80);



#Now that we have a glimpse of how major features react with the target variable: 'SalePrice', let's move on to the step:2
#2. Feature Engineering

#To find the age of a building by subtracting 'YearBuilt' from 'YrSold' 
df['Age_building']=(df['YrSold']-df['YearBuilt'])

#Age of the building Vs. SalePrice. Clearly newly built buidlings are sold at higher price
#compared to the majority of the old ones

plt.figure(figsize=(20,10))
ax = sns.swarmplot(x='Age_building', y='SalePrice', data=df, linewidth=2.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=-80);
#To find the number of years since a building had been remodeled before the time it was sold by subtracting 'YearRemodAdd' from
#'YrSold'
df['Since_remodeled']=(df['YrSold']-df['YearRemodAdd'])


#'Since_remodeled' Vs. 'SalePrice'. It becomes clear again that recently remodeled buildings selling at higher price
import numpy as np
plt.figure(figsize=(20,10))
ax = sns.boxplot(x='Since_remodeled', y='SalePrice', data=df, linewidth=2.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=-80);

medians = df.groupby(['Since_remodeled'])['SalePrice'].median().values
median_labels = [str(np.round(s, 2)) for s in medians]

pos = range(len(medians))
for tick,label in zip(pos,ax.get_xticklabels()):
    ax.text(pos[tick], medians[tick] + 0.8, median_labels[tick], horizontalalignment='right', size='x-small', color='r', weight='bold')
#Filling in the missing values in 'GarageYrBlt'
df.GarageYrBlt = df.GarageYrBlt.fillna(df['GarageYrBlt'].median())
#GarageYrBlt Vs. SalePrice 
#Similarly (similar trends that existed in 'Since_remodeled' and 'Age_building' also existed in 'Age_garage') 
#let's get the info re when the garage was built
df['Age_garage']=(df['YrSold']-df['GarageYrBlt'])




#To create a new column: Season to include the seasonal changes affecting the SalePrice
df['Season']=df['MoSold'].copy()
#To group months according to season: summer, winter, autumn, and spring

df.loc[(df['MoSold']=='Jun') | (df['MoSold']=='Jul') | (df['MoSold']=='Aug'),'Season']='Summer'
df.loc[(df['MoSold']=='Mar') | (df['MoSold']=='Apr') | (df['MoSold']=='May'),'Season']= 'Spring'
df.loc[(df['MoSold']=='Sep') | (df['MoSold']=='Oct') | (df['MoSold']=='Nov'),'Season']='Autumn'
df.loc[(df['MoSold']=='Jan') | (df['MoSold']=='Feb') | (df['MoSold']=='Dec'),'Season']='Winter'



#It makes more sense in terms of seasonal changes with autumn steadily declining and summer doing better. Also from 2007 to 2008
#the SalePrice declined in autumn and winter.
plt.figure(figsize=(20,10))
ax = sns.pointplot(x='YrSold', y='SalePrice', hue='Season', data=df, linewidth=2.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=-80);
#Next let's group the neighborhoods according to maximum SalePrice
def neighborhood_convert(d):
    from collections import defaultdict
    d2=defaultdict(list)
    for k, v in d.items():
        if v<=200000:
            d2['g1'].append(k)     
        elif (v>200000) & (v<=300000):
            d2['g2'].append(k)
        elif (v>300000) & (v<=400000):
            d2['g3'].append(k)
        else:
            d2['g4'].append(k)
            
    return d2
            
#To get the output of the function neighborhood_convert in the cell above
neighborhood_convert(dict(df.groupby('Neighborhood')['SalePrice'].max()))
#The following is not the smartest way to categorize the neighborhoods, but however it works :)

df.loc[(df['Neighborhood']=='IDOTRR')|(df['Neighborhood']=='MeadowV')|(df['Neighborhood']=='SWISU')|(df['Neighborhood']=='BrDale')|(df['Neighborhood']=='NPkVill')|(df['Neighborhood']=='Blueste')|(df['Neighborhood']=='Sawyer'),'Neighborhood']='g' 
    
df.loc[(df['Neighborhood']=='Blmngtn')|(df['Neighborhood']=='Mitchel')|(df['Neighborhood']=='BrkSide')|(df['Neighborhood']=='NWAmes'),'Neighborhood']='h' 

df.loc[(df['Neighborhood']=='Veenker')|(df['Neighborhood']=='SawyerW')|(df['Neighborhood']=='ClearCr')|(df['Neighborhood']=='Edwards')|(df['Neighborhood']=='Gilbert')|(df['Neighborhood']=='NAmes')|(df['Neighborhood']=='Timber')|(df['Neighborhood']=='Crawfor'),'Neighborhood']='i' 

df.loc[(df['Neighborhood']=='NoRidge')|(df['Neighborhood']=='NridgHt')|(df['Neighborhood']=='StoneBr')|(df['Neighborhood']=='OldTown')|(df['Neighborhood']=='Somerst')|(df['Neighborhood']=='CollgCr'),'Neighborhood']='j' 
         
#Let's check the unique values in the column: Neighborhood
df['Neighborhood'].unique()
#Filling in the missing values in GarageType with BuiltIn as it is a popular choice but there are not many values from 
#that category
df.GarageType = df.GarageType.fillna('BuiltIn')
#BsmtQual has NaN values

df['BsmtQual'].value_counts(dropna=False)
#Let's categorize the NaN values as 'missing'
df.loc[df['BsmtQual'].isnull(), 'BsmtQual'] = 'missing'
# Because the median SalePrice for the missing category is 101800 and is close to median SalePrice of 'Fair' 
#category in 'BsmtQual'. I am converting missing to Fa

df.loc[df['BsmtQual']=='missing', 'BsmtQual'] = 'Fa'



#To get value_counts() of 'MasVnrType'

df['MasVnrType'].value_counts(dropna=False)
        
#Converting the few missing values in 'MasVnrType' to 'Stone', the popular option
df.loc[df['MasVnrType'].isnull(), 'MasVnrType'] = 'Stone'
#Given the context: Approx. Ames averages 29 inches of snow per year. The US average is 26 inches of snow per year.
#Converting the 'Detchd' option alone to negative value as obviously it will be inconvenient for people to access 
#their cars during winter. The idea is to capture the additional inconvenience compared to the builtin and attached
#garage options.Using frequency encoding to convert all the categorical values.

def convert_garagetype(d, x, y, col_name):
    import pandas as pd
    for ele in range(len(x)):
        if x[ele]=='Detchd':
            if x[ele] in y:
                x[ele]=-y[x[ele]]
                
        else:
            if x[ele] in y:
                x[ele]=y[x[ele]]
        
    d.drop(col_name, axis=1)
    d[col_name]=pd.Series(x)
    return d[col_name].head()

convert_garagetype(df, df['GarageType'].tolist(), dict(df['GarageType'].value_counts()/len(df)), 'GarageType')
#Convert 'MasVnrType' into numerical: None means no veneer type at all

def convert_MasVnrType(d, x, y, col_name):
    import pandas as pd
    for ele in range(len(x)):
        if x[ele]=='None':
            if x[ele] in y:
                x[ele]=-y[x[ele]]
                
        else:
            if x[ele] in y:
                x[ele]=y[x[ele]]
        
    d.drop(col_name, axis=1)
    d[col_name]=pd.Series(x)

convert_MasVnrType(df, df['MasVnrType'].tolist(), dict(df['MasVnrType'].value_counts()/len(df)), 'MasVnrType')
#Converting the 'Abnorml' option in SaleCondition to negative value
def convert_SaleCondition(d, x, y, col_name):
    import pandas as pd
    for ele in range(len(x)):
        if x[ele]=='Abnorml':
            if x[ele] in y:
                x[ele]=-y[x[ele]]
                
        else:
            if x[ele] in y:
                x[ele]=y[x[ele]]
        
    d.drop(col_name, axis=1)
    d[col_name]=pd.Series(x)

convert_SaleCondition(df, df['SaleCondition'].tolist(), dict(df['SaleCondition'].value_counts()/len(df)), 'SaleCondition')
#'BsmtUnfSF' has zero and other values. Zero means there are no unfinished square feet left. Therefore negating 
#the values, except zero, in 'BsmtUnfSF'

def convert_BsmtUnfSF(d, x, col_name):
    import pandas as pd
    for ele in range(len(x)):
        if x[ele]!=0:
            x[ele]=-x[ele]
        
    d.drop(col_name, axis=1)
    d[col_name]=pd.Series(x)

convert_BsmtUnfSF(df, df['BsmtUnfSF'].tolist(), 'BsmtUnfSF')
#'Functional' has minor damages to moderate, major, and severe damages. So negating the values of moderate, major 1 & 2,
#and severe damages

def convert_Functional(d, x, y, col_name):
    import pandas as pd
    for ele in range(len(x)):
        if (x[ele]=='Mod') | (x[ele]=='Maj1') | (x[ele]=='Maj2') | (x[ele]=='Sev'):
            if x[ele] in y:
                x[ele]=-y[x[ele]]        
        else:
            if x[ele] in y:
                x[ele]=y[x[ele]]       
    d.drop(col_name, axis=1)
    d[col_name]=pd.Series(x)

convert_Functional(df, df['Functional'].tolist(), dict(df['Functional'].value_counts()/len(df)), 'Functional')
#Convert the following categorical values using frequency encoding that is done by calculating: 
#number of times a value has occurred in a categorical column/length of the categorical column

def convert_categorical(d, x, y, col_name):
    import pandas as pd
    for ele in range(len(x)):
        if x[ele] in y:
            x[ele]=y[x[ele]]         
    d.drop(col_name, axis=1)
    d[col_name]=pd.Series(x)

convert_categorical(df, df['KitchenQual'].tolist(), dict(df['KitchenQual'].value_counts()/len(df)), 'KitchenQual')

convert_categorical(df, df['HouseStyle'].tolist(), dict(df['HouseStyle'].value_counts()/len(df)), 'HouseStyle')

convert_categorical(df, df['BsmtQual'].tolist(), dict(df['BsmtQual'].value_counts()/len(df)), 'BsmtQual')

convert_categorical(df, df['Electrical'].tolist(), dict(df['Electrical'].value_counts()/len(df)), 'Electrical')

convert_categorical(df, df['MoSold'].tolist(), dict(df['MoSold'].value_counts()/len(df)), 'MoSold')

convert_categorical(df, df['Season'].tolist(), dict(df['Season'].value_counts()/len(df)), 'Season')

convert_categorical(df, df['Neighborhood'].tolist(), dict(df['Neighborhood'].value_counts()/len(df)), 'Neighborhood')

convert_categorical(df, df['BldgType'].tolist(), dict(df['BldgType'].value_counts()/len(df)), 'BldgType')

convert_categorical(df, df['SaleType'].tolist(), dict(df['SaleType'].value_counts()/len(df)), 'SaleType')

convert_categorical(df, df['MSZoning'].tolist(), dict(df['MSZoning'].value_counts()/len(df)), 'MSZoning')

convert_categorical(df, df['YrSold'].tolist(), dict(df['YrSold'].value_counts()/len(df)), 'YrSold')



#Select the necessary 32 columns:['LotArea', 'Neighborhood', 'GarageCars','MasVnrType', 'BsmtQual', 'TotalBsmtSF', '1stFlrSF','2ndFlrSF', 'GrLivArea', 'FullBath', 'HalfBath', 'TotRmsAbvGrd','GarageType', 'GarageArea', 'WoodDeckSF', 'MoSold', 'SaleType','SaleCondition','Age_building', 'Since_remodeled', 'Age_garage', 'BldgType', 'MSZoning', 'Season', 'KitchenQual', 'BedroomAbvGr', 'Functional', 'BsmtUnfSF','BsmtFullBath', 'KitchenAbvGr','YrSold','SalePrice']

cols = [col for col in df.columns if col in ['LotArea', 'Neighborhood', 'GarageCars','MasVnrType', 'BsmtQual', 'TotalBsmtSF', '1stFlrSF','2ndFlrSF', 'GrLivArea', 'FullBath', 'HalfBath', 'TotRmsAbvGrd','GarageType', 'GarageArea', 'WoodDeckSF', 'MoSold', 'SaleType','SaleCondition','Age_building', 'Since_remodeled', 'Age_garage', 'BldgType', 'MSZoning', 'Season', 'KitchenQual', 'BedroomAbvGr', 'Functional', 'BsmtUnfSF','BsmtFullBath', 'KitchenAbvGr','YrSold','SalePrice']]
df1 = df[cols]
#To assign target variable 'y'
y=df1['SalePrice'].values
#To assign feature set 'X'
df1=df1.drop('SalePrice', axis=1)
X=df1.values
#We have the feature set and target variable ready. Time to move on to building the model
#3. Model Building
#I chose a RandomForestRegressor to predict SalePrice

from sklearn.ensemble import RandomForestRegressor
RFR1 = RandomForestRegressor(random_state=42)


#Setting the parameters to grid search for best parameters
from sklearn.model_selection import GridSearchCV
parameters ={'n_estimators':[100, 150, 200, 250], 'min_samples_split':[2, 3, 4, 5], 'min_samples_leaf':[1, 2, 3], 'max_features':[10, 11, 12, 13]}
#GridSearchCV
from sklearn.metrics import mean_squared_error
clf_rfr = GridSearchCV(RFR1, parameters, scoring='neg_mean_squared_error', cv=10)
#Fitting X and y
clf_rfr.fit(X, y)
# The final step is to check the clf_rfr.best_score_ and once satisfied predict X_test and submit the predictions. 
# So far we have analyzed, visualized, and selected the necessary features and went ahead to build a random forest
# regressor to predict the SalePrice. Further improvements can be made in the model performance by feature 
# engineering in a better and diverse way to decorrelate the individual trees in the model thereby improving its
# performance. Especially I ought to capture the seasonal changes in a better way to say the least: for example the
# SalePrice increased in Autumn from 2006 to 2007 but decreased from 2007 to 2008. Wish to capture such changes and
# encode appropriately. Could encode the column: Neighborhood better with information on crime rate, safety, amenities, 
# commute, and school access, etc., and so the column: Condition1. Also could use different encoding techniques to 
# have a diverse set of features improving the model performance.