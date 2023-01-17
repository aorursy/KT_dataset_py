import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

import statsmodels.formula.api as smf

%matplotlib inline
#read in the data-set 

covid = pd.read_csv("../input/covid-hackathon/COVID19 Hackathon Dataset.csv")

covid.columns
#Changing column names to remove duplicate name for Labour Force and shorten some names 



covid.columns = ['Date', 'Continent', 'Countries', 'New_Cases', 'New_Deaths',

       'Total_Cases', 'Total_Deaths', 'Weekly_Cases', 'Weekly_Deaths',

       'Population_Density', 'Fertility_Rate', 'GDP_PPP', 'Corruption',

       'Government_Effectiveness', 'Political_Stability', 'Rule_of_Law',

       'Government_Healthcare_Spend', 'Urban_Population_percent', 'Smoking_Prevalence',

       'Tourism', 'Women_In_Parliament', 'Obesity_Rate', 'RunTot_New_Cases',

       'RunTot_New_Deaths', 'Country', 'Country_Code', 'Year',

       'Diabetes_Prevelance', 'Labour_Force_Total',

       'Population', 'Population_Aged_65',

       'Urban_Population', 'Death_Rate_Per_1000_2017',

       'Air_pollution',

       'Population_exposed_to_Polution',

       'Hospital_Beds_Year', 'Hospital_Beds']

covid.columns
covid.describe()
#check for nulls 

covid.isnull().sum() 
#using the group by function to only take the max of the column - short cut to summarise the data

covid_countries_max = covid.groupby(["Countries","Continent"]).max()

covid_countries_max.describe()
#group by creates an index using countries & continent and will take the max value for each variable. 

covid_countries_max.head()
covid_countries_max.isnull().sum()

#We can see the urban population metric given is not complete (Missing for 62 countries and Smoking Data is missing for 42 Countries)
#List of columns I am no longer interested in after grouping the data 

remove_list = [ 'New_Cases', 'New_Deaths','Weekly_Cases', 

               'Weekly_Deaths','RunTot_New_Deaths','Country_Code','RunTot_New_Cases','RunTot_New_Deaths', 'Hospital_Beds_Year', 'Year']
#removing countries with less that 1000 total cases



covid_over_thousand = covid_countries_max[covid_countries_max['Total_Cases'] > 1000].copy()



#dropping the daily/ weekly cases variables and unwanted columns 



covid_over_thousand.drop(remove_list, axis=1, inplace=True)



#this is removing the coutries / continent index and creating a new index but will keep the columns. 



covid_over_thousand.reset_index(inplace=True)
#creating new columns per 1000 (have shortened to cap because I am lazy)



covid_over_thousand["deaths_per_capita"] = (covid_over_thousand["Total_Deaths"] / covid_over_thousand['Population']) * 1000

covid_over_thousand["urban_cap"] = (covid_over_thousand["Urban_Population"] / covid_over_thousand['Population'])  * 1000

covid_over_thousand["labour_cap"] = (covid_over_thousand["Labour_Force_Total"] / covid_over_thousand['Population'])  * 1000

covid_over_thousand["tourism_cap"] = (covid_over_thousand["Tourism"] / covid_over_thousand['Population'])  * 1000

covid_over_thousand['infection_cap'] = (covid_over_thousand["Total_Cases"] / covid_over_thousand['Population']) *1000 

covid_over_thousand['death_case_ratio'] = (covid_over_thousand["Total_Deaths"] / covid_over_thousand['Total_Cases']) 



#using correlation matrix visulisation from this example https://www.kaggle.com/pragyanbo/performing-multiple-regression-using-python



pd.options.display.float_format = '{:,.4f}'.format

corr = covid_over_thousand.corr()

corr[np.abs(corr) < 0.25] = 0

plt.figure(figsize=(20,20))

sns.heatmap(corr, annot=True, cmap='YlGnBu')

plt.show()
#quick scatterplot example using seaborn (sns)



sns.scatterplot(data=covid_over_thousand, x="deaths_per_capita", y="Air_pollution")
#this creates a dummy variable based on continent



data_for_model=pd.get_dummies(covid_over_thousand, columns=['Continent'])



model_variation_all_fields = [

'Population_Density', 

'Fertility_Rate', 

'GDP_PPP', 

'Corruption',

'Government_Effectiveness', 

'Political_Stability', 

'Rule_of_Law',

'Government_Healthcare_Spend',

'Smoking_Prevalence', 

'Women_In_Parliament', 

'Obesity_Rate',

'Diabetes_Prevelance', 

'Population_Aged_65', 

'Urban_Population', 

'Death_Rate_Per_1000_2017',

'Air_pollution', 

'Population_exposed_to_Polution', 

'Hospital_Beds',

'urban_cap',

'labour_cap',

"tourism_cap",

'infection_cap',

'Continent_Africa',

'Continent_Asia',

'Continent_Europe',

'Continent_North America',

'Continent_Oceania',

'Continent_South America'

]
#removing nulls from the data - as this particular model can't handle missing values 



data_model_remove_nulls = data_for_model.dropna()

data_model_remove_nulls.isnull().sum()
#now we have 84 coutries in the data set 

data_model_remove_nulls.describe()
y = data_model_remove_nulls['deaths_per_capita']

X = data_model_remove_nulls[model_variation_all_fields] #you can change the list name here to quickly switch between model variations



X_constant = sm.add_constant(X)
model = sm.OLS(y, X_constant)

lin_reg = model.fit()



lin_reg.summary()