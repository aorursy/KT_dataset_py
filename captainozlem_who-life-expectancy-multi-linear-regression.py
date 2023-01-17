import numpy as np

import pandas as pd

import statsmodels.api as sm

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression



import seaborn as sns

sns.set()
raw_data = pd.read_csv('../input/life-expectancy-who/Life Expectancy Data.csv')

raw_data.head()
raw_data.describe(include='all')
raw_data.isnull().sum()
# Checking why and which countries has null value for alcohol

null_alcohol = raw_data[raw_data["Alcohol"].isnull()]

#alcohol_na = raw_data.query('Alcohol == 0')

#alcohol_na

null_alcohol
null_bmi = raw_data[raw_data[" BMI "].isnull()]

null_bmi
## Turns out only Sudan and South Sudan do not report the BMI. We can use imputation for Monaco and San Marino from

## previous years because only one year missing from those countries. I will drop Sudan and South Sudan from the data
## Question: Does Life Expectancy have positive or negative relationship with drinking alcohol?

## Data is missing for almost every country in 2015, so I will drop the 2015 from the data

is_2015 = raw_data[raw_data["Year"]==2015].index

is_2015

data_wo_2015 = raw_data.drop(is_2015)

data_wo_2015
## South Sudan does not have any Alcohol data for taking mean and fill the null spaces, so I will drop South Sudan completely

is_s_sudan = data_wo_2015[data_wo_2015["Country"]=="South Sudan"].index

is_s_sudan

data_alcohol = data_wo_2015.drop(is_s_sudan)

data_alcohol
data_alcohol.isnull().sum()
data_1 = data_alcohol[data_alcohol['Life expectancy '].isnull()].index

data_1
data_2 = data_alcohol.drop(data_1) 

data_2
na_bmi = data_2[data_2[" BMI "].isnull()].index

na_bmi
data_3 = data_2.drop(na_bmi)

data_3
data_3.isnull().sum()
data_4 = data_3[data_3['Alcohol'].isnull()].index

data_4
data_clean = data_3.drop(data_4)

data_clean.isnull().sum()
## Dropping multiple columns at the same time.
to_drop = ["Hepatitis B", "Polio", "Total expenditure", "Diphtheria ", "GDP", "Population", "Income composition of resources","Schooling"]

data_clean.drop(to_drop, inplace=True, axis=1)



#passing in the inplace parameter as True and the axis parameter as 1. This tells Pandas that we want the changes to be made directly in our object and that it should look for the values to be dropped in the columns of the object.
#include='all' shows all the data not only numerical

data_clean.describe(include='all')
data_clean.isnull().sum()
sns.distplot(data_clean["Alcohol"])
sns.distplot(data_clean[" BMI "])
# Based on the PDF, BMI is bimodel distrubition
sns.distplot(data_clean[' HIV/AIDS'])
sns.distplot(data_clean['percentage expenditure'])
sns.distplot(data_clean[' thinness  1-19 years'])
round(data_clean[['Status','Life expectancy ']].groupby('Status').mean(),2)
sns.distplot(data_clean['Life expectancy '])
data_clean["Country"].unique()
# Transform to categorical data to numerical data, 1 stands for "Developed countries, and 0 for "developing countries

data_clean["Status"] = data_clean["Status"].map({'Developed':1,'Developing':0})
data_clean['Status'].unique()

plt.figure(figsize=(15,35))



plt.subplot(6,3,1)

plt.scatter(data_clean['Status'], data_clean["Life expectancy "])

plt.title("Life Expectancy vs Status")

plt.xlabel('Developed or Developing status')

plt.ylabel('Age (yrs)')



plt.subplot(6,3,2)

plt.scatter(data_clean['Alcohol'], data_clean["Life expectancy "])

plt.title("Life Expectancy vs Alcohol")

plt.xlabel('Litres')

plt.ylabel('Age (yrs)')



plt.subplot(6,3,3)

plt.scatter(data_clean[' BMI '], data_clean["Life expectancy "])

plt.title("Life Expectancy vs BMI")

plt.xlabel('BMI')

plt.ylabel('Age (yrs)')



plt.subplot(6,3,4)

plt.scatter(data_clean[' HIV/AIDS'], data_clean["Life expectancy "])

plt.title("Life Expectancy vs HIV/AIDS")

plt.xlabel('Deaths per 1000 live births')

plt.ylabel('Age (yrs)')



plt.subplot(6,3,5)

plt.scatter(data_clean['percentage expenditure'], data_clean["Life expectancy "])

plt.title("Life Expectancy vs Percentage Expenditure")

plt.xlabel('% of total government expenditure')

plt.ylabel('Age (yrs)')



plt.subplot(6,3,6)

plt.scatter(data_clean[' thinness  1-19 years'], data_clean["Life expectancy "])

plt.title("Life Expectancy vs Teenage Thinness")

plt.xlabel('%')

plt.ylabel('Age (yrs)')



plt.show()
#It did not work but, box coxis a tranformation technic for the data not normaly distributed

# Try to transform Box-Cox, because the data is not normaly distributed



#from scipy import stats



# get values from our data_clean columns

#t_alcohol = np.asarray(data_clean['Alcohol'].values)



#tranfrom values and store as "d_t_"

#d_t_alcohol = stats.boxcox(t_alcohol)[0]



#plot the transformed data

#plt.hist(d_t_alcohol,bins=10)

#plt.show()
# I replace 0 with 0.01 in data_clean['percentage expenditure'] for better log transformation

data_clean['percentage expenditure'] = data_clean['percentage expenditure'].mask(data_clean['percentage expenditure']==0, 0.01)

data_clean['percentage expenditure']
# Just because boxcox did not work, i will transform the x axis for hiv and percent expenditure, that are close to be linear relationship

log_hiv = np.log(data_clean[' HIV/AIDS'])

data_clean['log_hiv'] = log_hiv



log_expenditure = np.log(data_clean['percentage expenditure'])

data_clean['log_expenditure'] = log_expenditure



data_clean
sns.distplot(data_clean['log_hiv'])
sns.distplot(data_clean['log_expenditure'])
plt.figure(figsize=(15,35))



plt.subplot(6,3,1)

plt.scatter(data_clean['Status'], data_clean["Life expectancy "])

plt.title("Life Expectancy vs Status")

plt.xlabel('Developed or Developing status')

plt.ylabel('Age (yrs)')



plt.subplot(6,3,2)

plt.scatter(data_clean['Alcohol'], data_clean["Life expectancy "])

plt.title("Life Expectancy vs Alcohol")

plt.xlabel('Litres')

plt.ylabel('Age (yrs)')



plt.subplot(6,3,3)

plt.scatter(data_clean[' BMI '], data_clean["Life expectancy "])

plt.title("Life Expectancy vs BMI")

plt.xlabel('BMI')

plt.ylabel('Age (yrs)')



plt.subplot(6,3,4)

plt.scatter(data_clean['log_hiv'], data_clean["Life expectancy "])

plt.title("Life Expectancy vs log HIV/AIDS")

plt.xlabel('Deaths per 10 live births')

plt.ylabel('Age (yrs)')



plt.subplot(6,3,5)

plt.scatter(data_clean['log_expenditure'], data_clean["Life expectancy "])

plt.title("Life Expectancy vs Log Transformed Percentage Expenditure")

plt.xlabel('log of total government expenditure')

plt.ylabel('Age (yrs)')



plt.subplot(6,3,6)

plt.scatter(data_clean[' thinness  1-19 years'], data_clean["Life expectancy "])

plt.title("Life Expectancy vs Teenage Thinness")

plt.xlabel('%')

plt.ylabel('Age (yrs)')



plt.show()
# Dropping the useless or pre-transformed data

drop_axis = [' HIV/AIDS','percentage expenditure']

data_clean.drop(drop_axis, inplace=True, axis=1)
#I do not need to write (include='all'), because I already transformed categorical data to numerical

data_clean.describe()
# One of the best ways to check for multicollinearity is VIF(variance inflation factor)

data_clean.columns.values
from statsmodels.stats.outliers_influence import variance_inflation_factor

var = data_clean[['Status','Alcohol',' BMI ', ' thinness  1-19 years','log_hiv', 'log_expenditure']]

vif = pd.DataFrame()

vif["VIF"] = [variance_inflation_factor(var.values,i) for i in range(var.shape[1])]

vif['features'] = var.columns
#VIF = 1: no multicollinearity

# 1<VIF < 5: perfectly okay

#10 < VIF : unacceptable (there is no upperlimit and different sources are saying different numbers, but for acceptance try to keep under 5)



vif
# all my data is VIF<5, which is awesome because I do not need to remove any column
# if we have N categories for a feature, we have to create N-1 dummies

data_w_dummies = pd.get_dummies(data_clean, drop_first=True)
data_w_dummies.head()
data_w_dummies.columns.values
cols =['Life expectancy ', 'Status',

       'Alcohol', ' BMI ',

       ' thinness  1-19 years',

        'log_hiv', 'log_expenditure',

       'Country_Albania', 'Country_Algeria', 'Country_Angola',

       'Country_Antigua and Barbuda', 'Country_Argentina',

       'Country_Armenia', 'Country_Australia', 'Country_Austria',

       'Country_Azerbaijan', 'Country_Bahamas', 'Country_Bahrain',

       'Country_Bangladesh', 'Country_Barbados', 'Country_Belarus',

       'Country_Belgium', 'Country_Belize', 'Country_Benin',

       'Country_Bhutan', 'Country_Bolivia (Plurinational State of)',

       'Country_Bosnia and Herzegovina', 'Country_Botswana',

       'Country_Brazil', 'Country_Brunei Darussalam', 'Country_Bulgaria',

       'Country_Burkina Faso', 'Country_Burundi', 'Country_Cabo Verde',

       'Country_Cambodia', 'Country_Cameroon', 'Country_Canada',

       'Country_Central African Republic', 'Country_Chad',

       'Country_Chile', 'Country_China', 'Country_Colombia',

       'Country_Comoros', 'Country_Congo', 'Country_Costa Rica',

       'Country_Croatia', 'Country_Cuba', 'Country_Cyprus',

       'Country_Czechia', "Country_Côte d'Ivoire",

       "Country_Democratic People's Republic of Korea",

       'Country_Democratic Republic of the Congo', 'Country_Denmark',

       'Country_Djibouti', 'Country_Dominican Republic',

       'Country_Ecuador', 'Country_Egypt', 'Country_El Salvador',

       'Country_Equatorial Guinea', 'Country_Eritrea', 'Country_Estonia',

       'Country_Ethiopia', 'Country_Fiji', 'Country_Finland',

       'Country_France', 'Country_Gabon', 'Country_Gambia',

       'Country_Georgia', 'Country_Germany', 'Country_Ghana',

       'Country_Greece', 'Country_Grenada', 'Country_Guatemala',

       'Country_Guinea', 'Country_Guinea-Bissau', 'Country_Guyana',

       'Country_Haiti', 'Country_Honduras', 'Country_Hungary',

       'Country_Iceland', 'Country_India', 'Country_Indonesia',

       'Country_Iran (Islamic Republic of)', 'Country_Iraq',

       'Country_Ireland', 'Country_Israel', 'Country_Italy',

       'Country_Jamaica', 'Country_Japan', 'Country_Jordan',

       'Country_Kazakhstan', 'Country_Kenya', 'Country_Kiribati',

       'Country_Kuwait', 'Country_Kyrgyzstan',

       "Country_Lao People's Democratic Republic", 'Country_Latvia',

       'Country_Lebanon', 'Country_Lesotho', 'Country_Liberia',

       'Country_Libya', 'Country_Lithuania', 'Country_Luxembourg',

       'Country_Madagascar', 'Country_Malawi', 'Country_Malaysia',

       'Country_Maldives', 'Country_Mali', 'Country_Malta',

       'Country_Mauritania', 'Country_Mauritius', 'Country_Mexico',

       'Country_Micronesia (Federated States of)', 'Country_Mongolia',

       'Country_Montenegro', 'Country_Morocco', 'Country_Mozambique',

       'Country_Myanmar', 'Country_Namibia', 'Country_Nepal',

       'Country_Netherlands', 'Country_New Zealand', 'Country_Nicaragua',

       'Country_Niger', 'Country_Nigeria', 'Country_Norway',

       'Country_Oman', 'Country_Pakistan', 'Country_Panama',

       'Country_Papua New Guinea', 'Country_Paraguay', 'Country_Peru',

       'Country_Philippines', 'Country_Poland', 'Country_Portugal',

       'Country_Qatar', 'Country_Republic of Korea',

       'Country_Republic of Moldova', 'Country_Romania',

       'Country_Russian Federation', 'Country_Rwanda',

       'Country_Saint Lucia', 'Country_Saint Vincent and the Grenadines',

       'Country_Samoa', 'Country_Sao Tome and Principe',

       'Country_Saudi Arabia', 'Country_Senegal', 'Country_Serbia',

       'Country_Seychelles', 'Country_Sierra Leone', 'Country_Singapore',

       'Country_Slovakia', 'Country_Slovenia', 'Country_Solomon Islands',

       'Country_Somalia', 'Country_South Africa', 'Country_Spain',

       'Country_Sri Lanka', 'Country_Suriname', 'Country_Swaziland',

       'Country_Sweden', 'Country_Switzerland',

       'Country_Syrian Arab Republic', 'Country_Tajikistan',

       'Country_Thailand',

       'Country_The former Yugoslav republic of Macedonia',

       'Country_Timor-Leste', 'Country_Togo', 'Country_Tonga',

       'Country_Trinidad and Tobago', 'Country_Tunisia', 'Country_Turkey',

       'Country_Turkmenistan', 'Country_Uganda', 'Country_Ukraine',

       'Country_United Arab Emirates',

       'Country_United Kingdom of Great Britain and Northern Ireland',

       'Country_United Republic of Tanzania',

       'Country_United States of America', 'Country_Uruguay',

       'Country_Uzbekistan', 'Country_Vanuatu',

       'Country_Venezuela (Bolivarian Republic of)', 'Country_Viet Nam',

       'Country_Yemen', 'Country_Zambia', 'Country_Zimbabwe']
data_preprocessed = data_w_dummies[cols]

data_preprocessed.head()
targets = data_preprocessed['Life expectancy ']

inputs = data_preprocessed.drop(['Life expectancy '], axis=1)
## Scaling dummies are not actually recommended, but so far I do not know the selective scaling so 

## only for this example I will go for it



from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)



#test_size= 0.2 means 20/80 split between test and train and random_state makes fix randomize for future tests
reg = LinearRegression()

reg.fit(x_train, y_train)
y_hat = reg.predict(x_train)
plt.scatter(y_train, y_hat)

plt.xlabel('Targets (y_train)', size =18)

plt.ylabel('Predictions (y_hat)', size=18)

plt.xlim(35,90)

plt.ylim(35,90)

plt.show()
## Residual plot

## Residual = Differences between the targets and the predictions 

## The residuals are estimates of the errors and expected to behave normality and homoscedasticity



sns.distplot(y_train - y_hat)

plt.title("Residuals PDF", size =18)

  
reg.score(x_train, y_train)
def adj_r2(x,y):

    r2 = reg.score(x,y)

    n = x.shape[0]

    p = x.shape[1]

    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)

    return adjusted_r2
adj_r2(x_train,y_train)
## bias == intercept (ML term)



reg.intercept_
## weight == coefficient (ML term) Bigger weight is bigger impact

reg.coef_
## For readablity, I will make summary table

reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])

reg_summary["Weights"] = reg.coef_

reg_summary.head(20)
## Finding the benchmark - based on the dummies which one is 1 - in our case benchmark is Afghanistan

data_clean['Country'].unique()
y_hat_test = reg.predict(x_test)
## alpha gives opactiy which can used for seeing which area has more density,

## the more saturated the color, the higher the concentration

plt.scatter(y_test, y_hat_test, alpha=0.2)

plt.xlabel("Targets (y_test)", size=18)

plt.ylabel("Predictions (y-hat_test)", size=18)

plt.xlim(35,90)

plt.ylim(35,90)

plt.show()
## df_pf == DataFrame Performance for seeing how accurate is our predictions



df_pf = pd.DataFrame(y_hat_test, columns=['Prediction'])

df_pf

df_pf["Target"] = y_test

df_pf
## We need to reset the indexes, because or the random_scale

y_test = y_test.reset_index(drop=True)

y_test.head()
df_pf["Target"] = y_test

df_pf
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']
## Whether an observation is off by +1% or -1% is mostly irrelevant

df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf
df_pf.describe()
## checking all of the data

pd.options.display.max_rows = 999

pd.set_option('display.float_format', lambda x: '%.2f' %x)

df_pf.sort_values(by=['Difference%'])