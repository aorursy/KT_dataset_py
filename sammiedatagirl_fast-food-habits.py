import pandas as pd
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import seaborn as sns
filename = "../input/FastFood_Habits_Questionnaire.csv"
#select a random sample from your dataset csv
n  = sum(1 for line in open(filename)) - 1 #number of records in a file minus the header
s = 405 #sample size(385) + 20(number of nan values expected to be in the dataset)
skip = sorted (rand.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
df = pd.read_csv(filename, skiprows = skip, skip_blank_lines = True)
len(df)
df.columns
df.rename(columns = {'What age group do you fall under?': 'Age Group', 
                  'Which is your gender?':'Gender', 
                  'What would you say accurately matches your personality in a group discussion?': 'Personality type',
                     'Choose all the options that are close to where you reside.': 'Location',
                  'How often do you eat from fast foods each month?': 'FastFood_Frequency',
                  'What does your purchase mostly consist of?  [Nigerian dishes]': 'Purchase_Nigerian_Dishes',
                  'What does your purchase mostly consist of?  [Burger]':'Purchase_Burger',
                  'What does your purchase mostly consist of?  [Pizza]': 'Purchase_Pizza',
                  'What does your purchase mostly consist of?  [Small chops]':'Purchase_Small_Chops',
                  'What does your purchase mostly consist of?  [Shawarma]': 'Purchase_Shawarma',
                  'What do you look for when choosing a fast food restaurant?':'FastFood_Choice',
                  'Which of the following have you been to at least twice in the last 2 months': 'FastFood_Visited',
                  'When it comes to ordering, which do you prefer the most?':'Ordering_Type',
                  'What is the best way to reach you for special offers/Discounts?':'Preffered_Channel',
                  'What about a fast food restaurant endears you to them? [Their special offers]':'Special_Offers',
                  'What about a fast food restaurant endears you to them? [The quality]':'Quality',
                  'What about a fast food restaurant endears you to them? [Value for your money]':'VFM', 
                  'What about a fast food restaurant endears you to them? [Excellent service]':'Excellent_Service', 
                  'What about a fast food restaurant endears you to them? [Convenience]':'Convenience',
                  'What about a fast food restaurant endears you to them? [Price]':'Price',
                  'What about a fast food restaurant endears you to them? [Social media savviness]':'SM_Savviness',
                  'What do you consider yourself to be the most?':'FoodieType',
                 }, inplace = True)
df.columns
#find columns that have missing data
df.isnull().sum()
#find the number of null values in data
df.isnull().sum().sum()
#drop null values
df = df.dropna()

#check if null values still exist
print (df.isnull().sum())
print (df.isnull().sum().sum())
#385 which is the sample size needed
len(df)
#Check the data types stored in each column
df.info()
#The data has some columns which have values that we need to transform to categoricals for further processing
#list the unique values in a column 
df['Purchase_Nigerian_Dishes'].unique()

#list the unique values in a column
df['Special_Offers'].unique()
#change strongly disagrre to strongly disagree
cleanup_nums = {'Special_Offers': {'Stongly Disagrre': 'Strongly Disagree'},
               'Quality' : {'Stongly Disagrre': 'Strongly Disagree'},
            'VFM' : {'Stongly Disagrre': 'Strongly Disagree'},
            'Excellent_Service': {'Stongly Disagrre': 'Strongly Disagree'},
                'Convenience': {'Stongly Disagrre': 'Strongly Disagree'},
       'Price': {'Stongly Disagrre': 'Strongly Disagree'}, 
                'SM_Savviness': {'Stongly Disagrre': 'Strongly Disagree'}}
df.replace(cleanup_nums, inplace = True)
#Check that it has changed
df['Special_Offers'].unique()
#change certain columns to category 
#change column types from object to category

for col in  ['Purchase_Nigerian_Dishes', 'Purchase_Burger', 'Purchase_Pizza',
       'Purchase_Small_Chops', 'Purchase_Shawarma',
          'Special_Offers', 'Quality', 'VFM', 'Excellent_Service', 'Convenience',
       'Price', 'SM_Savviness',]:
    df[col] = df[col].astype('category')
print(df.info())
df.columns
#reorder categroris to  High Unlikely < Likely < Most likely
columns = ['Purchase_Nigerian_Dishes', 'Purchase_Burger', 'Purchase_Pizza',
       'Purchase_Small_Chops', 'Purchase_Shawarma']
for col in columns:
    df[col].cat.reorder_categories([ 'Highly Unlikely','Likely','Most likely'], inplace = True)
print( df[columns[1]])
#reorder categories to Stongly Disagree < Indifferent < Agree < Strongly Agree
columns = ['Special_Offers', 'Quality', 'VFM', 'Excellent_Service', 'Convenience',
       'Price', 'SM_Savviness']
for col in columns:
    df[col].cat.reorder_categories([ 'Strongly Disagree', 'Indifferent' , 'Agree' , 'Strongly Agree'], inplace = True)
print(df[columns[0]])
# Also change Gender to categorical where 1 for Male, 0 for Female using LabelEncoder
import sklearn
from sklearn.preprocessing import LabelEncoder

var_mod = ['Gender']
le = LabelEncoder()


for i in var_mod:
      df[i] = le.fit_transform(df[i]) #changes categorical values into numerical eg Yes = 0, No = 1
    
df.dtypes
var_mod = ['Gender']
le = LabelEncoder()


for i in var_mod:
      dfcopy[i] = le.fit_transform(df[i]) #changes categorical values into numerical eg Yes = 0, No = 1
    
dfcopy.dtypes
#Other columns such as Age group can be preprocessed as well 
df['Age Group'].value_counts()
#by grouping 18 - 25 into 25 - 34 and 21 and 16 to 18 - 24
cleanup_nums = {'18 - 25': '25 - 34',
               '21':'18 - 24',
               '16':'18 - 24'}
df.replace(cleanup_nums, inplace = True)
df['Age Group'].value_counts()
df.dtypes
df.columns
#About to experiment so make a copy of dataframe so that we don't mess up our original dataframe as we experiment

dfcopy = df.copy()
dfcopy.head(10)
dfcopy.columns
#In order to run correlation on the coulms we need to code the categories 
#Numerical coding of Categories 1
dfcopy['Purchase_Shawarma_CAT'] = dfcopy['Purchase_Shawarma'].cat.codes
dfcopy['Purchase_Small_Chops_CAT'] = dfcopy['Purchase_Small_Chops'].cat.codes
dfcopy['Purchase_Pizza_CAT'] = dfcopy['Purchase_Pizza'].cat.codes
dfcopy['Purchase_Burger_CAT'] = dfcopy['Purchase_Burger'].cat.codes
dfcopy['Purchase_Nigerian_Dishes_CAT'] = dfcopy['Purchase_Nigerian_Dishes'].cat.codes


dfcopy[['Purchase_Shawarma','Purchase_Shawarma_CAT','Purchase_Small_Chops','Purchase_Small_Chops_CAT',
        'Purchase_Pizza','Purchase_Pizza_CAT' ,'Purchase_Burger','Purchase_Burger_CAT',
        'Purchase_Nigerian_Dishes','Purchase_Nigerian_Dishes_CAT']]

#the result of the numerical coding should give us : Highly Unlikely = 0 Likely = 1 Most Likely = 2
#Numerical coding of Categories 2

dfcopy['Special_Offers_CAT'] = dfcopy['Special_Offers'].cat.codes
dfcopy['Quality_CAT'] = dfcopy['Quality'].cat.codes
dfcopy['VFM_CAT'] = dfcopy['VFM'].cat.codes
dfcopy['Excellent_Service_CAT'] = dfcopy['Excellent_Service'].cat.codes
dfcopy['Convenience_CAT'] = dfcopy['Convenience'].cat.codes
dfcopy['Price_CAT'] = dfcopy['Price'].cat.codes
dfcopy['SM_Savviness_CAT'] = dfcopy['SM_Savviness'].cat.codes

dfcopy[['Special_Offers','Special_Offers_CAT', 'Quality', 'Quality_CAT',
        'VFM', 'VFM_CAT', 'Excellent_Service','Excellent_Service_CAT',
         'Convenience','Convenience_CAT',
        'Price', 'Price_CAT', 'SM_Savviness','SM_Savviness_CAT']]
dfcopy.columns
#correlation on Gendr and Fast Food Choice purchase probabily
#Hypothesis : Boys are more likely to buy Nigerian Dishes(NGD) as Fast Foods than Girls 
#because Girls are more likely to cook their own NGD and wont want to buy what they have already at home


print('Correlation between Gender and Nigeria Dishes Purchase is %s '% dfcopy['Gender'].corr(dfcopy['Purchase_Nigerian_Dishes_CAT'],  method = 'spearman'))
print('Correlation between Gender and Burger Purchase is %s '% dfcopy['Gender'].corr(dfcopy['Purchase_Burger_CAT'],  method = 'spearman'))
print('Correlation between Gender and Pizza Purchase is %s '% dfcopy['Gender'].corr(dfcopy['Purchase_Pizza_CAT'], method = 'spearman'))
print('Correlation between Gender and Small Chops Purchase is %s '% dfcopy['Gender'].corr(dfcopy['Purchase_Small_Chops_CAT'], method = 'spearman'))
print('Correlation between Gender and Shawarma Purchase is %s '% dfcopy['Gender'].corr(dfcopy['Purchase_Shawarma_CAT'], method = 'spearman'))
#correlation
#Hypothes Girls care more about Quality than Guys 

print('Correlation between Gender and Quality is %s '% dfcopy['Gender'].corr(dfcopy['Quality_CAT']))
print('Correlation between Gender and Value for Money %s '% dfcopy['Gender'].corr(dfcopy['VFM_CAT']))
print('Correlation between Gender and Excellent service is %s '% dfcopy['Gender'].corr(dfcopy['Excellent_Service_CAT']))
print('Correlation between Gender and Convenience is %s '% dfcopy['Gender'].corr(dfcopy['Convenience_CAT']))
print('Correlation between Gender and Price is %s '% dfcopy['Gender'].corr(dfcopy['Price_CAT' ]))
print('Correlation between Gender and Social media savviness is %s '% dfcopy['Gender'].corr(dfcopy['SM_Savviness_CAT']))
                                                                                           
#However there was a bit of correlation between people who valued Excellent Service and Quality 
print('Correlation between Excellent Service and Quality is %s '% dfcopy['Excellent_Service_CAT'].corr(dfcopy['Quality_CAT']))
print('Correlation between Excellent Service and Convenience is %s '% dfcopy['Excellent_Service_CAT'].corr(dfcopy['Convenience_CAT']))
                                                                                           
#Plot a correlation table between Fast Food Traits to explore more relationships
dfcorrelation = dfcopy[['Special_Offers_CAT', 'Quality_CAT','VFM_CAT', 'Excellent_Service_CAT','Convenience_CAT',
       'Price_CAT', 'SM_Savviness_CAT']]
dfcorrelation.corr()
print(dfcopy['Special_Offers'].unique())
print(dfcopy['Special_Offers_CAT'].unique())
#Plot some boxplots to visualize relationships
dfcopy.boxplot( 'SM_Savviness_CAT', 'Age Group', figsize = (15,5));
dfcopy.boxplot( 'Special_Offers_CAT', 'Age Group', figsize = (15,5));
dfcopy.boxplot( 'Convenience_CAT', 'Age Group', figsize = (15,5));
dfcopy.boxplot( 'Price_CAT', 'Age Group', figsize = (15,5));
dfcopy['Personality type'].value_counts()
dfcopy.boxplot( 'Special_Offers_CAT', 'Personality type', figsize = (25,10));
dfcopy.boxplot( 'VFM_CAT', 'Excellent_Service_CAT', figsize = (25,10));
