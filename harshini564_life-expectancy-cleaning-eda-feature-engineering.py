# Import libraries

import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

from sqlalchemy import create_engine

import warnings



warnings.filterwarnings('ignore')
# Load dataset

le = pd.read_csv('../input/Life Expectancy Data.csv', delimiter=',')

le.dataframeName = 'Life Expectancy Data.csv'
# First 5 rows of the dataset.

le.head()
# Description and context of the Life Expectancy (WHO) dataset can be found here.

  ## https://www.kaggle.com/kumarajarshi/life-expectancy-who
# Renaming some column names as they contain trailing spaces.

le.rename(columns={" BMI ":"BMI","Life expectancy ":"Life_Expectancy","Adult Mortality":"Adult_Mortality",

                   "infant deaths":"Infant_Deaths","percentage expenditure":"Percentage_Exp","Hepatitis B":"HepatitisB",

                  "Measles ":"Measles"," BMI ":"BMI","under-five deaths ":"Under_Five_Deaths","Diphtheria ":"Diphtheria",

                  " HIV/AIDS":"HIV/AIDS"," thinness  1-19 years":"thinness_1to19_years"," thinness 5-9 years":"thinness_5to9_years","Income composition of resources":"Income_Comp_Of_Resources",

                   "Total expenditure":"Tot_Exp"},inplace=True)

# Verify descriptive statistics

le.describe()
# Verifying whether data for each variable is according to its dataype or not.

le.info()
# Identify percentage of null values in each column.

le.isnull().sum()*100/le.isnull().count()
country_list = le.Country.unique()

fill_list = ['Life_Expectancy','Adult_Mortality','Alcohol','HepatitisB','BMI','Polio','Tot_Exp','Diphtheria','GDP','Population','thinness_1to19_years','thinness_5to9_years','Income_Comp_Of_Resources','Schooling']

# Treat null values using interpolation.

for country in country_list:

    le.loc[le['Country'] == country,fill_list] = le.loc[le['Country'] == country,fill_list].interpolate()

    

# Drop remaining null values after interpolation.

le.dropna(inplace=True)
# Verifying null-values after applying above methods.

le.isnull().sum()
# Create a dictionary of columns.

col_dict = {'Life_Expectancy':1,'Adult_Mortality':2,'Infant_Deaths':3,'Alcohol':4,'Percentage_Exp':5,'HepatitisB':6,'Measles':7,'BMI':8,'Under_Five_Deaths':9,'Polio':10,'Tot_Exp':11,'Diphtheria':12,'HIV/AIDS':13,'GDP':14,'Population':15,'thinness_1to19_years':16,'thinness_5to9_years':17,'Income_Comp_Of_Resources':18,'Schooling':19}



# Detect outliers in each variable using box plots.

plt.figure(figsize=(20,30))



for variable,i in col_dict.items():

                     plt.subplot(5,4,i)

                     plt.boxplot(le[variable],whis=1.5)

                     plt.title(variable)



plt.show()
# Calculate number of outliers and its percentage in each variable using Tukey's method.



for variable in col_dict.keys():

    q75, q25 = np.percentile(le[variable], [75 ,25])

    iqr = q75 - q25



    min_val = q25 - (iqr*1.5)

    max_val = q75 + (iqr*1.5)

    print("Number of outliers and percentage of it in {} : {} and {}".format(variable,

                                                                             len((np.where((le[variable] > max_val) | 

                                                                                           (le[variable] < min_val))[0])),len((np.where((le[variable] > max_val) | 

                                                                                           (le[variable] < min_val))[0]))*100/1987))
# Removing Outliers in the variables using Winsorization technique.

# Winsorize Life_Expectancy

from scipy.stats.mstats import winsorize

plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_Life_Expectancy = le['Life_Expectancy']

plt.boxplot(original_Life_Expectancy)

plt.title("original_Life_Expectancy")



plt.subplot(1,2,2)

winsorized_Life_Expectancy = winsorize(le['Life_Expectancy'],(0.01,0))

plt.boxplot(winsorized_Life_Expectancy)

plt.title("winsorized_Life_Expectancy")



plt.show()
# Winsorize Adult_Mortality

from scipy.stats.mstats import winsorize

plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_Adult_Mortality = le['Adult_Mortality']

plt.boxplot(original_Adult_Mortality)

plt.title("original_Adult_Mortality")



plt.subplot(1,2,2)

winsorized_Adult_Mortality = winsorize(le['Adult_Mortality'],(0,0.03))

plt.boxplot(winsorized_Adult_Mortality)

plt.title("winsorized_Adult_Mortality")



plt.show()
# Winsorize Infant_Deaths

from scipy.stats.mstats import winsorize

plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_Infant_Deaths = le['Infant_Deaths']

plt.boxplot(original_Infant_Deaths)

plt.title("original_Infant_Deaths")



plt.subplot(1,2,2)

winsorized_Infant_Deaths = winsorize(le['Infant_Deaths'],(0,0.10))

plt.boxplot(winsorized_Infant_Deaths)

plt.title("winsorized_Infant_Deaths")



plt.show()
# Winsorize Alcohol

from scipy.stats.mstats import winsorize

plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_Alcohol = le['Alcohol']

plt.boxplot(original_Alcohol)

plt.title("original_Alcohol")



plt.subplot(1,2,2)

winsorized_Alcohol = winsorize(le['Alcohol'],(0,0.01))

plt.boxplot(winsorized_Alcohol)

plt.title("winsorized_Alcohol")



plt.show()
# Winsorize Percentage_Exp

from scipy.stats.mstats import winsorize

plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_Percentage_Exp = le['Percentage_Exp']

plt.boxplot(original_Percentage_Exp)

plt.title("original_Percentage_Exp")



plt.subplot(1,2,2)

winsorized_Percentage_Exp = winsorize(le['Percentage_Exp'],(0,0.12))

plt.boxplot(winsorized_Percentage_Exp)

plt.title("winsorized_Percentage_Exp")



plt.show()
# Winsorize HepatitisB

from scipy.stats.mstats import winsorize

plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_HepatitisB = le['HepatitisB']

plt.boxplot(original_HepatitisB)

plt.title("original_HepatitisB")



plt.subplot(1,2,2)

winsorized_HepatitisB = winsorize(le['HepatitisB'],(0.11,0))

plt.boxplot(winsorized_HepatitisB)

plt.title("winsorized_HepatitisB")



plt.show()
# Winsorize Measles

from scipy.stats.mstats import winsorize

plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_Measles = le['Measles']

plt.boxplot(original_Measles)

plt.title("original_Measles")



plt.subplot(1,2,2)

winsorized_Measles = winsorize(le['Measles'],(0,0.19))

plt.boxplot(winsorized_Measles)

plt.title("winsorized_Measles")



plt.show()
le = le.drop('Measles',axis=1)
# Winsorize Under_Five_Deaths

from scipy.stats.mstats import winsorize

plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_Under_Five_Deaths = le['Under_Five_Deaths']

plt.boxplot(original_Under_Five_Deaths)

plt.title("original_Under_Five_Deaths")



plt.subplot(1,2,2)

winsorized_Under_Five_Deaths = winsorize(le['Under_Five_Deaths'],(0,0.12))

plt.boxplot(winsorized_Under_Five_Deaths)

plt.title("winsorized_Under_Five_Deaths")



plt.show()
# Winsorize Polio

from scipy.stats.mstats import winsorize

plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_Polio = le['Polio']

plt.boxplot(original_Polio)

plt.title("original_Polio")



plt.subplot(1,2,2)

winsorized_Polio = winsorize(le['Polio'],(0.09,0))

plt.boxplot(winsorized_Polio)

plt.title("winsorized_Polio")



plt.show()
# Winsorize Tot_Exp

from scipy.stats.mstats import winsorize

plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_Tot_Exp = le['Tot_Exp']

plt.boxplot(original_Tot_Exp)

plt.title("original_Tot_Exp")



plt.subplot(1,2,2)

winsorized_Tot_Exp = winsorize(le['Tot_Exp'],(0,0.01))

plt.boxplot(winsorized_Tot_Exp)

plt.title("winsorized_Tot_Exp")



plt.show()
# Winsorize Diphtheria

from scipy.stats.mstats import winsorize

plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_Diphtheria = le['Diphtheria']

plt.boxplot(original_Diphtheria)

plt.title("original_Diphtheria")



plt.subplot(1,2,2)

winsorized_Diphtheria = winsorize(le['Diphtheria'],(0.10,0))

plt.boxplot(winsorized_Diphtheria)

plt.title("winsorized_Diphtheria")



plt.show()
# Winsorize HIV/AIDS

from scipy.stats.mstats import winsorize

plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_HIV = le['HIV/AIDS']

plt.boxplot(original_HIV)

plt.title("original_HIV")



plt.subplot(1,2,2)

winsorized_HIV = winsorize(le['HIV/AIDS'],(0,0.16))

plt.boxplot(winsorized_HIV)

plt.title("winsorized_HIV")



plt.show()
# Winsorize GDP

from scipy.stats.mstats import winsorize

plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_GDP = le['GDP']

plt.boxplot(original_GDP)

plt.title("original_GDP")



plt.subplot(1,2,2)

winsorized_GDP = winsorize(le['GDP'],(0,0.13))

plt.boxplot(winsorized_GDP)

plt.title("winsorized_GDP")



plt.show()
# Winsorize Population

from scipy.stats.mstats import winsorize

plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_Population = le['Population']

plt.boxplot(original_Population)

plt.title("original_Population")



plt.subplot(1,2,2)

winsorized_Population = winsorize(le['Population'],(0,0.14))

plt.boxplot(winsorized_Population)

plt.title("winsorized_Population")



plt.show()
# Winsorize thinness_1to19_years

from scipy.stats.mstats import winsorize

plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_thinness_1to19_years = le['thinness_1to19_years']

plt.boxplot(original_thinness_1to19_years)

plt.title("original_thinness_1to19_years")



plt.subplot(1,2,2)

winsorized_thinness_1to19_years = winsorize(le['thinness_1to19_years'],(0,0.04))

plt.boxplot(winsorized_thinness_1to19_years)

plt.title("winsorized_thinness_1to19_years")



plt.show()
# Winsorize thinness_5to9_years

from scipy.stats.mstats import winsorize

plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_thinness_5to9_years = le['thinness_5to9_years']

plt.boxplot(original_thinness_5to9_years)

plt.title("original_thinness_5to9_years")



plt.subplot(1,2,2)

winsorized_thinness_5to9_years = winsorize(le['thinness_5to9_years'],(0,0.04))

plt.boxplot(winsorized_thinness_5to9_years)

plt.title("winsorized_thinness_5to9_years")



plt.show()
# Winsorize Income_Comp_Of_Resources

from scipy.stats.mstats import winsorize

plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_Income_Comp_Of_Resources = le['Income_Comp_Of_Resources']

plt.boxplot(original_Income_Comp_Of_Resources)

plt.title("original_Income_Comp_Of_Resources")



plt.subplot(1,2,2)

winsorized_Income_Comp_Of_Resources = winsorize(le['Income_Comp_Of_Resources'],(0.05,0))

plt.boxplot(winsorized_Income_Comp_Of_Resources)

plt.title("winsorized_Income_Comp_Of_Resources")



plt.show()
# Winsorize Schooling

from scipy.stats.mstats import winsorize

plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_Schooling = le['Schooling']

plt.boxplot(original_Schooling)

plt.title("original_Schooling")



plt.subplot(1,2,2)

winsorized_Schooling = winsorize(le['Schooling'],(0.02,0.01))

plt.boxplot(winsorized_Schooling)

plt.title("winsorized_Schooling")



plt.show()
# Check number of Outliers after Winsorization for each variable.

win_list = [winsorized_Life_Expectancy,winsorized_Adult_Mortality,winsorized_Infant_Deaths,winsorized_Alcohol,

            winsorized_Percentage_Exp,winsorized_HepatitisB,winsorized_Under_Five_Deaths,winsorized_Polio,winsorized_Tot_Exp,winsorized_Diphtheria,winsorized_HIV,winsorized_GDP,winsorized_Population,winsorized_thinness_1to19_years,winsorized_thinness_5to9_years,winsorized_Income_Comp_Of_Resources,winsorized_Schooling]



for variable in win_list:

    q75, q25 = np.percentile(variable, [75 ,25])

    iqr = q75 - q25



    min_val = q25 - (iqr*1.5)

    max_val = q75 + (iqr*1.5)

    

    print("Number of outliers after winsorization : {}".format(len(np.where((variable > max_val) | (variable < min_val))[0])))

    

    
# Adding winsorized variables to the data frame.

le['winsorized_Life_Expectancy'] = winsorized_Life_Expectancy

le['winsorized_Adult_Mortality'] = winsorized_Adult_Mortality

le['winsorized_Infant_Deaths'] = winsorized_Infant_Deaths

le['winsorized_Alcohol'] = winsorized_Alcohol

le['winsorized_Percentage_Exp'] = winsorized_Percentage_Exp

le['winsorized_HepatitisB'] = winsorized_HepatitisB

le['winsorized_Under_Five_Deaths'] = winsorized_Under_Five_Deaths

le['winsorized_Polio'] = winsorized_Polio

le['winsorized_Tot_Exp'] = winsorized_Tot_Exp

le['winsorized_Diphtheria'] = winsorized_Diphtheria

le['winsorized_HIV'] = winsorized_HIV

le['winsorized_GDP'] = winsorized_GDP

le['winsorized_Population'] = winsorized_Population

le['winsorized_thinness_1to19_years'] = winsorized_thinness_1to19_years

le['winsorized_thinness_5to9_years'] = winsorized_thinness_5to9_years

le['winsorized_Income_Comp_Of_Resources'] = winsorized_Income_Comp_Of_Resources

le['winsorized_Schooling'] = winsorized_Schooling

# Descriptive statistics of continuous variables.

le.describe()
# Distribution of each numerical variable.

all_col = ['Life_Expectancy','winsorized_Life_Expectancy','Adult_Mortality','winsorized_Adult_Mortality','Infant_Deaths',

         'winsorized_Infant_Deaths','Alcohol','winsorized_Alcohol','Percentage_Exp','winsorized_Percentage_Exp','HepatitisB',

         'winsorized_HepatitisB','Under_Five_Deaths','winsorized_Under_Five_Deaths','Polio','winsorized_Polio','Tot_Exp',

         'winsorized_Tot_Exp','Diphtheria','winsorized_Diphtheria','HIV/AIDS','winsorized_HIV','GDP','winsorized_GDP',

         'Population','winsorized_Population','thinness_1to19_years','winsorized_thinness_1to19_years','thinness_5to9_years',

         'winsorized_thinness_5to9_years','Income_Comp_Of_Resources','winsorized_Income_Comp_Of_Resources',

         'Schooling','winsorized_Schooling']



plt.figure(figsize=(15,75))



for i in range(len(all_col)):

    plt.subplot(18,2,i+1)

    plt.hist(le[all_col[i]])

    plt.title(all_col[i])



plt.show()



    
# Descriptive statistics of categorical variables.

le.describe(include=['O'])
# Life_Expectancy w.r.t Status using bar plot.

plt.figure(figsize=(6,6))

plt.bar(le.groupby('Status')['Status'].count().index,le.groupby('Status')['winsorized_Life_Expectancy'].mean())

plt.xlabel("Status",fontsize=12)

plt.ylabel("Avg Life_Expectancy",fontsize=12)

plt.title("Life_Expectancy w.r.t Status")

plt.show()
# Life_Expectancy w.r.t Country using bar plot.

le_country = le.groupby('Country')['winsorized_Life_Expectancy'].mean()

le_country.plot(kind='bar', figsize=(50,15), fontsize=25)

plt.title("Life_Expectancy w.r.t Country",fontsize=40)

plt.xlabel("Country",fontsize=35)

plt.ylabel("Avg Life_Expectancy",fontsize=35)

plt.show()
# Life_Expectancy w.r.t Year using bar plot.

plt.figure(figsize=(7,5))

plt.bar(le.groupby('Year')['Year'].count().index,le.groupby('Year')['winsorized_Life_Expectancy'].mean(),color='pink',alpha=0.65)

plt.xlabel("Year",fontsize=12)

plt.ylabel("Avg Life_Expectancy",fontsize=12)

plt.title("Life_Expectancy w.r.t Year")

plt.show()
# Scatter plot between the target variable(winsorized_Life_Expectancy) and all continuous variables.

plt.figure(figsize=(18,40))



plt.subplot(6,3,1)

plt.scatter(le["winsorized_Life_Expectancy"], le["winsorized_Adult_Mortality"])

plt.title("LifeExpectancy vs AdultMortality")



plt.subplot(6,3,2)

plt.scatter(le["winsorized_Life_Expectancy"], le["winsorized_Infant_Deaths"])

plt.title("LifeExpectancy vs Infant_Deaths")



plt.subplot(6,3,3)

plt.scatter(le["winsorized_Life_Expectancy"], le["winsorized_Alcohol"])

plt.title("LifeExpectancy vs Alcohol")



plt.subplot(6,3,4)

plt.scatter(le["winsorized_Life_Expectancy"], le["winsorized_Percentage_Exp"])

plt.title("LifeExpectancy vs Percentage_Exp")



plt.subplot(6,3,5)

plt.scatter(le["winsorized_Life_Expectancy"], le["winsorized_HepatitisB"])

plt.title("LifeExpectancy vs HepatitisB")



plt.subplot(6,3,6)

plt.scatter(le["winsorized_Life_Expectancy"], le["winsorized_Under_Five_Deaths"])

plt.title("LifeExpectancy vs Under_Five_Deaths")



plt.subplot(6,3,7)

plt.scatter(le["winsorized_Life_Expectancy"], le["winsorized_Polio"])

plt.title("LifeExpectancy vs Polio")



plt.subplot(6,3,8)

plt.scatter(le["winsorized_Life_Expectancy"], le["winsorized_Tot_Exp"])

plt.title("LifeExpectancy vs Tot_Exp")



plt.subplot(6,3,9)

plt.scatter(le["winsorized_Life_Expectancy"], le["winsorized_Diphtheria"])

plt.title("LifeExpectancy vs Diphtheria")



plt.subplot(6,3,10)

plt.scatter(le["winsorized_Life_Expectancy"], le["winsorized_HIV"])

plt.title("LifeExpectancy vs HIV")



plt.subplot(6,3,11)

plt.scatter(le["winsorized_Life_Expectancy"], le["winsorized_GDP"])

plt.title("LifeExpectancy vs GDP")



plt.subplot(6,3,12)

plt.scatter(le["winsorized_Life_Expectancy"], le["winsorized_Population"])

plt.title("LifeExpectancy vs Population")



plt.subplot(6,3,13)

plt.scatter(le["winsorized_Life_Expectancy"], le["winsorized_thinness_1to19_years"])

plt.title("LifeExpectancy vs thinness_1to19_years")



plt.subplot(6,3,14)

plt.scatter(le["winsorized_Life_Expectancy"], le["winsorized_thinness_5to9_years"])

plt.title("LifeExpectancy vs thinness_5to9_years")



plt.subplot(6,3,15)

plt.scatter(le["winsorized_Life_Expectancy"], le["winsorized_Income_Comp_Of_Resources"])

plt.title("LifeExpectancy vs Income_Comp_Of_Resources")



plt.subplot(6,3,16)

plt.scatter(le["winsorized_Life_Expectancy"], le["winsorized_Schooling"])

plt.title("LifeExpectancy vs Schooling")





plt.show()
plt.figure(figsize=(20,8))



plt.subplot(1,3,1)

plt.scatter(le["winsorized_Schooling"], le["winsorized_Adult_Mortality"])

plt.title("Schooling vs AdultMortality")



plt.subplot(1,3,2)

plt.scatter(le["winsorized_Schooling"], le["winsorized_Income_Comp_Of_Resources"])

plt.title("Schooling vs Income_Comp_Of_Resources")



plt.subplot(1,3,3)

plt.scatter(le["winsorized_Adult_Mortality"], le["winsorized_Income_Comp_Of_Resources"])

plt.title("AdultMortality vs Income_Comp_Of_Resources")



plt.show()
# Correlation of winsorized variables

le_win = le.iloc[:,21:]

le_win['Country'] = le['Country']

le_win['Year'] = le['Year']

le_win['Status'] = le['Status']

le_win_num = le_win.iloc[:,:-3]

cormat = le_win_num.corr()
# Using heatmap to observe correlations.

import seaborn as sns



plt.figure(figsize=(15,15))

sns.heatmap(cormat, square=True, annot=True, linewidths=.5)

plt.title("Correlation matrix among winsorized variables")

plt.show()
round(le[['Status','Life_Expectancy']].groupby(['Status']).mean(),2)
# Finding the significance of difference of Average_Life_Expectancy between Developed and Developing countries using 

# t-test

import scipy.stats as stats

stats.ttest_ind(le.loc[le['Status']=='Developed','Life_Expectancy'],le.loc[le['Status']=='Developing','Life_Expectancy'])
# Create a data frame with features.

feature_df = le[['Status','winsorized_Life_Expectancy','winsorized_Income_Comp_Of_Resources','winsorized_HIV','winsorized_Adult_Mortality']]

# Convert categorical values to numerical values using one-hot encoding for 'Status' feature.

feature_df = pd.concat([feature_df,pd.get_dummies(feature_df['Status'],drop_first=True)],axis=1)

final = feature_df.drop('Status',axis=1)

final.head()