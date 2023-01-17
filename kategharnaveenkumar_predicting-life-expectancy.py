# Import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats.mstats import winsorize

import scipy.stats as stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
life = pd.read_csv(r'../input/Life Expectancy Data.csv')
# First 5 rows of the dataset

life.head()
life.size
life.shape
life.columns
life.isnull().sum()
country_list = life.Country.unique()

fill_list = ['Country', 'Year', 'Status', 'Life expectancy ', 'Adult Mortality',

       'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B',

       'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure',

       'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',

       ' thinness  1-19 years', ' thinness 5-9 years',

       'Income composition of resources', 'Schooling']
for country in country_list:

    life.loc[life['Country'] == country,fill_list] = life.loc[life['Country'] == country,fill_list].interpolate()

life.dropna(inplace=True)
#Verifying for null-values after removing 

life.isna().sum()
# Create a dictionary of columns

col_dict = {'Life expectancy ':1 , 'Adult Mortality':2 ,

        'Alcohol':3 , 'percentage expenditure': 4, 'Hepatitis B': 5,

       'Measles ' : 6, ' BMI ': 7, 'under-five deaths ' : 8, 'Polio' : 9, 'Total expenditure' :10,

       'Diphtheria ':11, ' HIV/AIDS':12, 'GDP':13, 'Population' :14,

       ' thinness  1-19 years' :15, ' thinness 5-9 years' :16,

       'Income composition of resources' : 17, 'Schooling' :18, 'infant deaths':19}
plt.figure(figsize=(20,30))



for variable,i in col_dict.items():

                     plt.subplot(5,4,i)

                     plt.boxplot(life[variable],whis=1.5)

                     plt.title(variable)



plt.show()
for variable in col_dict.keys():

    q75, q25 = np.percentile(life[variable], [75 ,25])

    iqr = q75 - q25

    min_val = q25 - (iqr*1.5)

    max_val = q75 + (iqr*1.5)

    print("Number of outliers and percentage of it in {} : {} and {}".format(variable,

                                                                             len((np.where((life[variable] > max_val) | (life[variable] < min_val))[0])),

                                                                             len((np.where((life[variable] > max_val) | (life[variable] < min_val))[0]))*100/1987))
plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_Life_Expectancy = life['Life expectancy ']

plt.boxplot(original_Life_Expectancy)

plt.title("original_Life_Expectancy")



plt.subplot(1,2,2)

winsorized_Life_Expectancy = winsorize(life['Life expectancy '],(0.01,0))

plt.boxplot(winsorized_Life_Expectancy)

plt.title("winsorized_Life_Expectancy")



plt.show()
plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_Adult_Mortality = life['Adult Mortality']

plt.boxplot(original_Adult_Mortality)

plt.title("original_Adult_Mortality")



plt.subplot(1,2,2)

winsorized_Adult_Mortality = winsorize(life['Adult Mortality'],(0,0.03))

plt.boxplot(winsorized_Adult_Mortality)

plt.title("winsorized_Adult_Mortality")



plt.show()
plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_Infant_Deaths = life['infant deaths']

plt.boxplot(original_Infant_Deaths)

plt.title("original_Infant_Deaths")



plt.subplot(1,2,2)

winsorized_Infant_Deaths = winsorize(life['infant deaths'],(0,0.10))

plt.boxplot(winsorized_Infant_Deaths)

plt.title("winsorized_Infant_Deaths")



plt.show()
plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_Alcohol = life['Alcohol']

plt.boxplot(original_Alcohol)

plt.title("original_Alcohol")



plt.subplot(1,2,2)

winsorized_Alcohol = winsorize(life['Alcohol'],(0,0.01))

plt.boxplot(winsorized_Alcohol)

plt.title("winsorized_Alcohol")



plt.show()
plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_Percentage_Exp = life['percentage expenditure']

plt.boxplot(original_Percentage_Exp)

plt.title("original_Percentage_Exp")



plt.subplot(1,2,2)

winsorized_Percentage_Exp = winsorize(life['percentage expenditure'],(0,0.12))

plt.boxplot(winsorized_Percentage_Exp)

plt.title("winsorized_Percentage_Exp")



plt.show()

plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_HepatitisB = life['Hepatitis B']

plt.boxplot(original_HepatitisB)

plt.title("original_HepatitisB")



plt.subplot(1,2,2)

winsorized_HepatitisB = winsorize(life['Hepatitis B'],(0.11,0))

plt.boxplot(winsorized_HepatitisB)

plt.title("winsorized_HepatitisB")



plt.show()
plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_Measles = life['Measles ']

plt.boxplot(original_Measles)

plt.title("original_Measles")



plt.subplot(1,2,2)

winsorized_Measles = winsorize(life['Measles '],(0,0.19))

plt.boxplot(winsorized_Measles)

plt.title("winsorized_Measles")



plt.show()
plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_Under_Five_Deaths = life['under-five deaths ']

plt.boxplot(original_Under_Five_Deaths)

plt.title("original_Under_Five_Deaths")



plt.subplot(1,2,2)

winsorized_Under_Five_Deaths = winsorize(life['under-five deaths '],(0,0.12))

plt.boxplot(winsorized_Under_Five_Deaths)

plt.title("winsorized_Under_Five_Deaths")



plt.show()
plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_Polio = life['Polio']

plt.boxplot(original_Polio)

plt.title("original_Polio")



plt.subplot(1,2,2)

winsorized_Polio = winsorize(life['Polio'],(0.09,0))

plt.boxplot(winsorized_Polio)

plt.title("winsorized_Polio")



plt.show()
plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_Tot_Exp = life['Total expenditure']

plt.boxplot(original_Tot_Exp)

plt.title("original_Tot_Exp")



plt.subplot(1,2,2)

winsorized_Tot_Exp = winsorize(life['Total expenditure'],(0,0.01))

plt.boxplot(winsorized_Tot_Exp)

plt.title("winsorized_Tot_Exp")



plt.show()
plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_Diphtheria = life['Diphtheria ']

plt.boxplot(original_Diphtheria)

plt.title("original_Diphtheria")



plt.subplot(1,2,2)

winsorized_Diphtheria = winsorize(life['Diphtheria '],(0.10,0))

plt.boxplot(winsorized_Diphtheria)

plt.title("winsorized_Diphtheria")



plt.show()
plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_HIV = life[' HIV/AIDS']

plt.boxplot(original_HIV)

plt.title("original_HIV")



plt.subplot(1,2,2)

winsorized_HIV = winsorize(life[' HIV/AIDS'],(0,0.16))

plt.boxplot(winsorized_HIV)

plt.title("winsorized_HIV")



plt.show()
plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_GDP = life['GDP']

plt.boxplot(original_GDP)

plt.title("original_GDP")



plt.subplot(1,2,2)

winsorized_GDP = winsorize(life['GDP'],(0,0.13))

plt.boxplot(winsorized_GDP)

plt.title("winsorized_GDP")



plt.show()
plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_Population = life['Population']

plt.boxplot(original_Population)

plt.title("original_Population")



plt.subplot(1,2,2)

winsorized_Population = winsorize(life['Population'],(0,0.14))

plt.boxplot(winsorized_Population)

plt.title("winsorized_Population")



plt.show()
plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_thinness_1to19_years = life[' thinness  1-19 years']

plt.boxplot(original_thinness_1to19_years)

plt.title("original_thinness_1to19_years")



plt.subplot(1,2,2)

winsorized_thinness_1to19_years = winsorize(life[' thinness  1-19 years'],(0,0.04))

plt.boxplot(winsorized_thinness_1to19_years)

plt.title("winsorized_thinness_1to19_years")



plt.show()
plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_thinness_5to9_years = life[' thinness 5-9 years']

plt.boxplot(original_thinness_5to9_years)

plt.title("original_thinness_5to9_years")



plt.subplot(1,2,2)

winsorized_thinness_5to9_years = winsorize(life[' thinness 5-9 years'],(0,0.04))

plt.boxplot(winsorized_thinness_5to9_years)

plt.title("winsorized_thinness_5to9_years")



plt.show()
plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_Income_Comp_Of_Resources = life['Income composition of resources']

plt.boxplot(original_Income_Comp_Of_Resources)

plt.title("original_Income_Comp_Of_Resources")



plt.subplot(1,2,2)

winsorized_Income_Comp_Of_Resources = winsorize(life['Income composition of resources'],(0.05,0))

plt.boxplot(winsorized_Income_Comp_Of_Resources)

plt.title("winsorized_Income_Comp_Of_Resources")



plt.show()
plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

original_Schooling = life['Schooling']

plt.boxplot(original_Schooling)

plt.title("original_Schooling")



plt.subplot(1,2,2)

winsorized_Schooling = winsorize(life['Schooling'],(0.02,0.01))

plt.boxplot(winsorized_Schooling)

plt.title("winsorized_Schooling")



plt.show()
winsorized_list = [winsorized_Life_Expectancy,winsorized_Adult_Mortality,winsorized_Alcohol,winsorized_Measles,winsorized_Infant_Deaths,

            winsorized_Percentage_Exp,winsorized_HepatitisB,winsorized_Under_Five_Deaths,winsorized_Polio,winsorized_Tot_Exp,winsorized_Diphtheria,winsorized_HIV,winsorized_GDP,winsorized_Population,winsorized_thinness_1to19_years,winsorized_thinness_5to9_years,winsorized_Income_Comp_Of_Resources,winsorized_Schooling]



for variable in winsorized_list:

    q75, q25 = np.percentile(variable, [75 ,25])

    iqr = q75 - q25



    min_val = q25 - (iqr*1.5)

    max_val = q75 + (iqr*1.5)

    

    print("Number of outliers after winsorization   : {} ".format(len(np.where((variable > max_val) | (variable < min_val))[0])))
life['winsorized_Life_Expectancy'] = winsorized_Life_Expectancy

life['winsorized_Adult_Mortality'] = winsorized_Adult_Mortality

life['winsorized_Infant_Deaths'] = winsorized_Infant_Deaths

life['winsorized_Alcohol'] = winsorized_Alcohol

life['winsorized_Percentage_Exp'] = winsorized_Percentage_Exp

life['winsorized_HepatitisB'] = winsorized_HepatitisB

life['winsorized_Under_Five_Deaths'] = winsorized_Under_Five_Deaths

life['winsorized_Polio'] = winsorized_Polio

life['winsorized_Tot_Exp'] = winsorized_Tot_Exp

life['winsorized_Diphtheria'] = winsorized_Diphtheria

life['winsorized_HIV'] = winsorized_HIV

life['winsorized_GDP'] = winsorized_GDP

life['winsorized_Population'] = winsorized_Population

life['winsorized_thinness_1to19_years'] = winsorized_thinness_1to19_years

life['winsorized_thinness_5to9_years'] = winsorized_thinness_5to9_years

life['winsorized_Income_Comp_Of_Resources'] = winsorized_Income_Comp_Of_Resources

life['winsorized_Schooling'] = winsorized_Schooling

life['winsorized_Measles'] = winsorized_Measles
life.head()
life.size
life.shape
life.describe()


all_col = ['Life expectancy ','winsorized_Life_Expectancy','Adult Mortality','winsorized_Adult_Mortality','infant deaths',

         'winsorized_Infant_Deaths','Alcohol','winsorized_Alcohol','percentage expenditure','winsorized_Percentage_Exp','Hepatitis B',

         'winsorized_HepatitisB','under-five deaths ','winsorized_Under_Five_Deaths','Polio','winsorized_Polio','Total expenditure',

         'winsorized_Tot_Exp','Diphtheria ','winsorized_Diphtheria',' HIV/AIDS','winsorized_HIV','GDP','winsorized_GDP',

         'Population','winsorized_Population',' thinness  1-19 years','winsorized_thinness_1to19_years',' thinness 5-9 years',

         'winsorized_thinness_5to9_years','Income composition of resources','winsorized_Income_Comp_Of_Resources',

         'Schooling','winsorized_Schooling','Measles ','winsorized_Measles','GDP','winsorized_GDP']



plt.figure(figsize=(15,75))



for i in range(len(all_col)):

    plt.subplot(19,2,i+1)

    plt.hist(life[all_col[i]])

    plt.title(all_col[i])



plt.show()

life.describe(include= 'O')
plt.figure(figsize=(6,6))

plt.bar(life.groupby('Status')['Status'].count().index,life.groupby('Status')['winsorized_Life_Expectancy'].mean())

plt.xlabel("Status",fontsize=12)

plt.ylabel("Avg Life_Expectancy",fontsize=12)

plt.title("Life_Expectancy w.r.t Status")

plt.show()

le_country = life.groupby('Country')['winsorized_Life_Expectancy'].mean().sort_values(ascending=True)

le_country.plot(kind='bar', figsize=(50,15), fontsize=25)

plt.title("Life_Expectancy w.r.t Country",fontsize=40)

plt.xlabel("Country",fontsize=35)

plt.ylabel("Avg Life_Expectancy",fontsize=35)

plt.show()
le_country = life.groupby('Country')['winsorized_GDP'].mean().sort_values(ascending=True)

le_country.plot(kind='bar', figsize=(50,15), fontsize=25)

plt.title("Average GDP w.r.t Country",fontsize=40)

plt.xlabel("Country",fontsize=35)

plt.ylabel("Avg GDP",fontsize=35)

plt.show()
le_country = life.groupby('Country')['winsorized_Adult_Mortality'].mean().sort_values(ascending=True)

le_country.plot(kind='bar', figsize=(50,15), fontsize=25)

plt.title("Adult_Mortality w.r.t Country",fontsize=40)

plt.xlabel("Country",fontsize=35)

plt.ylabel("Avg Adult Mortality",fontsize=35)

plt.show()
le_country = life.groupby('Country')['winsorized_Alcohol'].mean().sort_values(ascending=True)

le_country.plot(kind='bar', figsize=(50,15), fontsize=25)

plt.title("Alcohol w.r.t Country",fontsize=40)

plt.xlabel("Country",fontsize=35)

plt.ylabel("Avg Alcohol Comsumption",fontsize=35)

plt.show()
le_country = life.groupby('Country')['winsorized_Diphtheria'].mean().sort_values(ascending=True)

le_country.plot(kind='bar', figsize=(50,15), fontsize=25)

plt.title("Diphtheria w.r.t Country",fontsize=40)

plt.xlabel("Country",fontsize=35)

plt.ylabel("Avg Diphtheria",fontsize=35)

plt.show()
le_country = life.groupby('Country')['winsorized_HepatitisB'].mean().sort_values(ascending=True)

le_country.plot(kind='bar', figsize=(50,15), fontsize=25)

plt.title("HepatitisB w.r.t Country",fontsize=40)

plt.xlabel("Country",fontsize=35)

plt.ylabel("Avg HepatitisB",fontsize=35)

plt.show()
le_country = life.groupby('Country')['winsorized_HIV'].mean().sort_values(ascending=True)

le_country.plot(kind='bar', figsize=(50,15), fontsize=25)

plt.title("HIV w.r.t Country",fontsize=40)

plt.xlabel("Country",fontsize=35)

plt.ylabel("Avg HIV cases",fontsize=35)

plt.show()
le_country = life.groupby('Country')['winsorized_Income_Comp_Of_Resources'].mean().sort_values(ascending=True)

le_country.plot(kind='bar', figsize=(50,15), fontsize=25)

plt.title("Income Composition of Resources w.r.t Country",fontsize=40)

plt.xlabel("Country",fontsize=35)

plt.ylabel("Avg income composition of resourses",fontsize=35)

plt.show()
le_country = life.groupby('Country')['winsorized_Infant_Deaths'].mean().sort_values(ascending=True)

le_country.plot(kind='bar', figsize=(50,15), fontsize=25)

plt.title("Infant Deaths w.r.t Country",fontsize=40)

plt.xlabel("Country",fontsize=35)

plt.ylabel("Avg Infant Deaths",fontsize=35)

plt.show()
le_country = life.groupby('Country')['winsorized_Measles'].mean().sort_values(ascending=True)

le_country.plot(kind='bar', figsize=(50,15), fontsize=25)

plt.title("Measles w.r.t Country",fontsize=40)

plt.xlabel("Country",fontsize=35)

plt.ylabel("Avg Measles cases",fontsize=35)

plt.show()
le_country = life.groupby('Country')['winsorized_Percentage_Exp'].mean().sort_values(ascending=True)

le_country.plot(kind='bar', figsize=(50,15), fontsize=25)

plt.title("Percentage Expenditure w.r.t Country",fontsize=40)

plt.xlabel("Country",fontsize=35)

plt.ylabel("Avg percentage expenditure",fontsize=35)

plt.show()
le_country = life.groupby('Country')['winsorized_Polio'].mean().sort_values(ascending=True)

le_country.plot(kind='bar', figsize=(50,15), fontsize=25)

plt.title("Polio w.r.t Country",fontsize=40)

plt.xlabel("Country",fontsize=35)

plt.ylabel("Avg Polio Cases",fontsize=35)

plt.show()
le_country = life.groupby('Country')['winsorized_Population'].mean().sort_values(ascending=True)

le_country.plot(kind='bar', figsize=(50,15), fontsize=25)

plt.title("Population w.r.t Country",fontsize=40)

plt.xlabel("Country",fontsize=35)

plt.ylabel("Avg Population",fontsize=35)

plt.show()
le_country = life.groupby('Country')['winsorized_Schooling'].mean().sort_values(ascending=True)

le_country.plot(kind='bar', figsize=(50,15), fontsize=25)

plt.title("Schooling w.r.t Country",fontsize=40)

plt.xlabel("Country",fontsize=35)

plt.ylabel("Avg Schooling",fontsize=35)

plt.show()
le_country = life.groupby('Country')['winsorized_thinness_1to19_years'].mean().sort_values(ascending=True)

le_country.plot(kind='bar', figsize=(50,15), fontsize=25)

plt.title(" Thinness 1to19 years w.r.t Country",fontsize=40)

plt.xlabel("Country",fontsize=35)

plt.ylabel("Avg Thinness 1 to 19 Years",fontsize=35)

plt.show()
le_country = life.groupby('Country')['winsorized_thinness_5to9_years'].mean().sort_values(ascending=True)

le_country.plot(kind='bar', figsize=(50,15), fontsize=25)

plt.title("Thinness 5 to 9 years w.r.t Country",fontsize=40)

plt.xlabel("Country",fontsize=35)

plt.ylabel("Avg thinness 5 to 9 years ",fontsize=35)

plt.show()
le_country = life.groupby('Country')['winsorized_Tot_Exp'].mean().sort_values(ascending=True)

le_country.plot(kind='bar', figsize=(50,15), fontsize=25)

plt.title(" Total Expenditure w.r.t Country",fontsize=40)

plt.xlabel("Country",fontsize=35)

plt.ylabel("Avg Total Expenditure",fontsize=35)

plt.show()
le_country = life.groupby('Country')['winsorized_Under_Five_Deaths'].mean().sort_values(ascending=True)

le_country.plot(kind='bar', figsize=(50,15), fontsize=25)

plt.title(" Under five Deaths w.r.t Country",fontsize=40)

plt.xlabel("Country",fontsize=35)

plt.ylabel("Avg under 5 deaths",fontsize=35)

plt.show()
plt.figure(figsize=(7,5))

plt.bar(life.groupby('Year')['Year'].count().index,life.groupby('Year')['winsorized_Life_Expectancy'].mean(),color='green',alpha=0.65)

plt.xlabel("Year",fontsize=12)

plt.ylabel("Avg Life_Expectancy",fontsize=12)

plt.title("Life_Expectancy w.r.t Year")

plt.show()
corr= life.corr()

sns.heatmap(corr)
plt.figure(figsize=(18,40))



plt.subplot(6,3,1)

plt.scatter(life["winsorized_Life_Expectancy"], life["winsorized_Adult_Mortality"])

plt.title("LifeExpectancy vs AdultMortality")



plt.subplot(6,3,2)

plt.scatter(life["winsorized_Life_Expectancy"], life["winsorized_Infant_Deaths"])

plt.title("LifeExpectancy vs Infant_Deaths")



plt.subplot(6,3,3)

plt.scatter(life["winsorized_Life_Expectancy"], life["winsorized_Alcohol"])

plt.title("LifeExpectancy vs Alcohol")



plt.subplot(6,3,4)

plt.scatter(life["winsorized_Life_Expectancy"], life["winsorized_Percentage_Exp"])

plt.title("LifeExpectancy vs Percentage_Exp")



plt.subplot(6,3,5)

plt.scatter(life["winsorized_Life_Expectancy"], life["winsorized_HepatitisB"])

plt.title("LifeExpectancy vs HepatitisB")



plt.subplot(6,3,6)

plt.scatter(life["winsorized_Life_Expectancy"], life["winsorized_Under_Five_Deaths"])

plt.title("LifeExpectancy vs Under_Five_Deaths")



plt.subplot(6,3,7)

plt.scatter(life["winsorized_Life_Expectancy"], life["winsorized_Polio"])

plt.title("LifeExpectancy vs Polio")



plt.subplot(6,3,8)

plt.scatter(life["winsorized_Life_Expectancy"], life["winsorized_Tot_Exp"])

plt.title("LifeExpectancy vs Tot_Exp")



plt.subplot(6,3,9)

plt.scatter(life["winsorized_Life_Expectancy"], life["winsorized_Diphtheria"])

plt.title("LifeExpectancy vs Diphtheria")



plt.subplot(6,3,10)

plt.scatter(life["winsorized_Life_Expectancy"], life["winsorized_HIV"])

plt.title("LifeExpectancy vs HIV")



plt.subplot(6,3,11)

plt.scatter(life["winsorized_Life_Expectancy"], life["winsorized_GDP"])

plt.title("LifeExpectancy vs GDP")



plt.subplot(6,3,12)

plt.scatter(life["winsorized_Life_Expectancy"], life["winsorized_Population"])

plt.title("LifeExpectancy vs Population")



plt.subplot(6,3,13)

plt.scatter(life["winsorized_Life_Expectancy"], life["winsorized_thinness_1to19_years"])

plt.title("LifeExpectancy vs thinness_1to19_years")



plt.subplot(6,3,14)

plt.scatter(life["winsorized_Life_Expectancy"], life["winsorized_thinness_5to9_years"])

plt.title("LifeExpectancy vs thinness_5to9_years")



plt.subplot(6,3,15)

plt.scatter(life["winsorized_Life_Expectancy"], life["winsorized_Income_Comp_Of_Resources"])

plt.title("LifeExpectancy vs Income_Comp_Of_Resources")



plt.subplot(6,3,16)

plt.scatter(life["winsorized_Life_Expectancy"], life["winsorized_Schooling"])

plt.title("LifeExpectancy vs Schooling")





plt.show()
round(life[['Status','Life expectancy ']].groupby(['Status']).mean(),2)
stats.ttest_ind(life.loc[life['Status']=='Developed','Life expectancy '],life.loc[life['Status']=='Developing','Life expectancy '])
life.columns
life.shape
life.head()
new_life=pd.DataFrame(data=life,columns=['Country', 'Year','Status',' BMI ',

       'winsorized_Life_Expectancy', 'winsorized_Adult_Mortality',

       'winsorized_Infant_Deaths', 'winsorized_Alcohol',

       'winsorized_Percentage_Exp', 'winsorized_HepatitisB',

       'winsorized_Under_Five_Deaths', 'winsorized_Polio',

       'winsorized_Tot_Exp', 'winsorized_Diphtheria', 'winsorized_HIV',

       'winsorized_GDP', 'winsorized_Population',

       'winsorized_thinness_1to19_years', 'winsorized_thinness_5to9_years',

       'winsorized_Income_Comp_Of_Resources', 'winsorized_Schooling',

       'winsorized_Measles'])
new_life.shape
new_life.head()
new_life.rename(columns={

    'winsorized_Life_Expectancy':'Life_Expectancy', 'winsorized_Adult_Mortality':'Adult_Mortality',

       'winsorized_Infant_Deaths':'Infant_Deaths', 'winsorized_Alcohol':'Alcohol',

       'winsorized_Percentage_Exp':'Percentage_Expenditure', 'winsorized_HepatitisB':'HepatitisB',

       'winsorized_Under_Five_Deaths':'Under_Five_Deaths', 'winsorized_Polio':'Polio',

       'winsorized_Tot_Exp':'Total_Expenditure', 'winsorized_Diphtheria':'Diphtheria', 'winsorized_HIV':'HIV',

       'winsorized_GDP':'GDP', 'winsorized_Population':'Population',

       'winsorized_thinness_1to19_years':'thinness_1to19_years', 'winsorized_thinness_5to9_years':'thinness_5to9_years',

       'winsorized_Income_Comp_Of_Resources':'Income_Comp_Of_Resources', 'winsorized_Schooling':'Schooling',

       'winsorized_Measles':'Measles'

},inplace=True)
new_life.head()
new_life.columns
dummies=pd.get_dummies(new_life.Status)

dummies
merged=pd.concat([new_life,dummies],axis='columns')

merged
final=merged.drop(['Status','Developed'],axis='columns')

final
final.columns
X=final.drop(['Life_Expectancy','Country'],axis='columns')

Y=pd.DataFrame(data=final,columns=['Life_Expectancy'])
X.head()
Y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 42)
print(X_train.shape)

print(Y_train.shape)

print(X_test.shape)

print(Y_test.shape)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
model = LinearRegression(fit_intercept=True, normalize=True).fit(X_train, Y_train)

predictions= model.predict(X_test)

len(predictions)
model.score(X_train, Y_train)
r2_score(predictions, Y_test)
mean_squared_error(predictions, Y_test)
mean_absolute_error(predictions, Y_test)