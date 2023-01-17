!pip install ppscore
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import OneHotEncoder

from sklearn import preprocessing

import random

import math

import ppscore as pps

def import_csv(filename): 

    df = pd.read_csv("/kaggle/input/iiitb-ai511ml2020-assignment-1/Assignment" + filename)

    print(filename + ' loaded...')

    print(filename + ' shape: ',df.shape)

    return df



#pass df.T for better view

def display_all(df) :

    with pd.option_context("display.max_rows", 1000, "display.max_columns", 20): 

        display(df)
alcdata = import_csv("/alcoholism/student-mat.csv")

fifadata = import_csv("/fifa18/data.csv")

accidata1 = import_csv("/accidents/accidents_2005_to_2007.csv")

accidata2 = import_csv("/accidents/accidents_2009_to_2011.csv")

accidata3 = import_csv("/accidents/accidents_2012_to_2014.csv")
#common space for alcohol data



#G1 and G2 are weighted as 0.25 as they are period marks. G3 is given 0.5 weight as it represents final marks.

def addGrades(df):

    df['Grades'] = 0.25 * df['G1'] + 0.25 * df['G2'] + 0.5 * df['G3']

    return df

#enter code/answer in this cell. You can add more code/markdown cells below for your answer.

alcdata = addGrades(alcdata)

alc_corr_data = alcdata.corr()['Grades']

plt.figure(figsize=(30,10))

plt.xticks(rotation=90)

plt.rc('xtick', labelsize=20) 

plt.rc('ytick', labelsize=20) 

sns.lineplot(data=alc_corr_data)

plt.show()
predictors_df = pps.predictors(alcdata, y='Grades')
plt.figure(figsize=(35,8))

plt.xticks(rotation=90)

plt.rc('xtick', labelsize=15) 

plt.rc('ytick', labelsize=15) 

sns.barplot(data=predictors_df, x="x", y="ppscore")
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 



plt.figure(figsize=(20,20))

plt.rc('xtick', labelsize=10) 

plt.rc('ytick', labelsize=10) 

sns.heatmap(alcdata.select_dtypes(include=['int64','float64']).corr(), cmap="YlGnBu",cbar_kws={"aspect": 40}, annot=True)

#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 

plt.figure(figsize=(25,7))

plt.xticks(rotation=0)

plt.rc('xtick', labelsize=15) 

plt.rc('ytick', labelsize=15) 

alc_grades_fam = pd.DataFrame(alcdata[['Grades', 'famrel']])

#convert grades to same range as family relation

alc_grades_fam['Grades'] = ((alc_grades_fam['Grades'] - alc_grades_fam['Grades'].min()) * 4 )/20 + 1

sns.lineplot(data=alc_grades_fam)

#could have used scatter plot here but it would make no sense as it would be too messy. Still is
#randomly see variation with 1/8th sample size

random.seed(10)

plt.figure(figsize=(25,7))

plt.xticks(rotation=0)

plt.rc('xtick', labelsize=15) 

plt.rc('ytick', labelsize=15) 

sns.scatterplot(data=(alc_grades_fam.sample(frac=1/8)))
#Mean and confidence of grades with each family relation

plt.figure(figsize=(15,4))

plt.xticks(rotation=0)

plt.rc('xtick', labelsize=15) 

plt.rc('ytick', labelsize=15) 

sns.lineplot(data=alcdata, x= 'famrel', y = 'Grades')

#Passing the entire dataset in long-form mode will aggregate over repeated values (each famrel) to show the mean and 95% confidence interval
alcdata_dummy = pd.get_dummies(alcdata)

plt.figure(figsize=(15,4))

plt.xticks(rotation=0)

plt.rc('xtick', labelsize=15) 

plt.rc('ytick', labelsize=15) 

sns.lineplot(data=alcdata_dummy, x= 'Pstatus_T', y = 'Grades')
plt.figure(figsize=(15,4))

plt.xticks(rotation=0)

plt.rc('xtick', labelsize=15) 

plt.rc('ytick', labelsize=15) 

sns.lineplot(data=alcdata_dummy, x= 'Pstatus_A', y = 'Grades')
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 

num_alcdata = alcdata.select_dtypes(include=['int64','float64'])

dropCol = ['famrel', 'failures', 'Medu', 'Fedu', 'traveltime', 'studytime', 'freetime', 'goout', 'Dalc', 'Walc', 'health']

num_alcdata = num_alcdata.drop(dropCol, axis = 1)



print("Skew of continuous columns:\n\n", num_alcdata.skew())





sns.pairplot(num_alcdata)

alcdata['absences'].unique()
num_alcdata['absences'] = np.log2(alcdata['absences']+1)

plt.figure(figsize=(15,4))

plt.xticks(rotation=0)

plt.rc('xtick', labelsize=15) 

plt.rc('ytick', labelsize=15) 

sns.distplot(num_alcdata['absences'], kde_kws={'bw':0.1})

print(num_alcdata['absences'].skew())
#fifadata common space

def clean_currency(df, feature):

    ans = []

    for e in df[feature]:

        e = e.replace('€', '')

        if 'K' in e:

            e = float(e.replace('K', ''))*1000

        elif 'M' in e:

            e = float(e.replace('M',''))*1000000

        ans.append(float(e))

    df[feature] = ans

    return df



display_all(fifadata.T)
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 

'''first = set()

last = set()

for e in fifadata['Wage']:

    first.add(e[0])

    last.add(e[-1])

print(first)         #The first position only has € which can be removed

print(last)          #The last position only has {'0', 'K'} which can be removed

'''

df = fifadata.copy()

df = clean_currency(df, 'Wage')

df = clean_currency(df, 'Value')



plt.figure(figsize=(4,4))

plt.rc('xtick', labelsize=10) 

plt.rc('ytick', labelsize=10) 

sns.heatmap(df[['Wage','Value']].corr(), cmap="YlGnBu",cbar_kws={"aspect": 40}, annot=True)



df['ValuePerPotential'] = df['Value'] / df['Potential']

df['WagePerPotential'] = df['Wage'] / df['Potential']



print("Most economical club in terms of Value: ", df.groupby('Club').mean()['ValuePerPotential'].idxmin())

print("Most economical club in terms of Wage: ", df.groupby('Club').mean()['WagePerPotential'].idxmin())

#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 

plt.figure(figsize=(15,4))

plt.xticks(rotation=0)

plt.rc('xtick', labelsize=15) 

plt.rc('ytick', labelsize=15) 

sns.boxplot(data=df, x= 'Age')
plt.figure(figsize=(15,4))

plt.xticks(rotation=0)

plt.rc('xtick', labelsize=15) 

plt.rc('ytick', labelsize=15) 

sns.boxplot(data=df, x= 'Age', y = 'Potential')



plt.figure(figsize=(15,4))

plt.xticks(rotation=0)

plt.rc('xtick', labelsize=15) 

plt.rc('ytick', labelsize=15) 

sns.lineplot(data=df, x= 'Age', y = 'Potential')
plt.figure(figsize=(15,4))

plt.xticks(rotation=0)

plt.rc('xtick', labelsize=15) 

plt.rc('ytick', labelsize=15) 

sns.lineplot(data=df, x= 'Age', y = 'Value')
plt.figure(figsize=(15,4))

plt.xticks(rotation=0)

plt.rc('xtick', labelsize=15) 

plt.rc('ytick', labelsize=15) 

sns.lineplot(data=df, x= 'Age', y = 'SprintSpeed')
#predict for cleaned values of Value

predictors_df = pps.predictors(df, y='Potential')
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 

print(predictors_df)

plt.figure(figsize=(35,8))

plt.xticks(rotation=90)

plt.rc('xtick', labelsize=15) 

plt.rc('ytick', labelsize=15) 

sns.barplot(data=predictors_df, x="x", y="ppscore")
print(predictors_df.iloc[8])

print("\n\n",predictors_df.iloc[9])
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 

predictors_df = pps.predictors(df, y='Wage')
print(predictors_df)

plt.figure(figsize=(35,8))

plt.xticks(rotation=90)

plt.rc('xtick', labelsize=15) 

plt.rc('ytick', labelsize=15) 

sns.barplot(data=predictors_df, x="x", y="ppscore")
predictors_df.iloc[1]
#enter code/answer in this cell. You can add more code/markdown cells below for your answer.

#only plotting distribution for 10 clubs

plt.figure(figsize=(20,8))

plt.xticks(rotation=0)

plt.rc('xtick', labelsize=20) 

plt.rc('ytick', labelsize=20)

count = 10

for e in df[['Club', 'Age']].groupby(by='Club'):

    if count == 0: 

        break

    sns.distplot(e[1]['Age'], hist=False, label=e[0])

    count = count - 1
#Summary of age distribution for all clubs

#display_all(df[['Club', 'Age']].groupby(by='Club').describe())
#Considering young means less than or equal to 22 years

display_all(df[['Age','Club']].loc[df['Age'] <= 22].groupby(by='Club').count().max())

display_all(df[['Age','Club']].loc[df['Age'] <= 22].groupby(by='Club').count().idxmax())

#common space for accident data

df.dtypes

#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 

accidata = pd.concat([accidata1,accidata2,accidata3])

df = accidata.copy()

df.shape
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 

plt.figure(figsize=(10,4))

plt.xticks(rotation=0)

plt.rc('xtick', labelsize=10) 

plt.rc('ytick', labelsize=10)

plotdata = df[['Day_of_Week', 'Number_of_Casualties']].groupby('Day_of_Week').sum().sort_values(by='Number_of_Casualties', ascending = False).reset_index();

sns.barplot(x= 'Day_of_Week', y = 'Number_of_Casualties', data=plotdata, order=plotdata['Day_of_Week'])

#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 

print('Min speed limits\n', df[['Speed_limit','Day_of_Week']].groupby(by='Day_of_Week').min().reset_index())

print("\n\nMax speed limits\n", df[['Speed_limit','Day_of_Week']].groupby(by='Day_of_Week').max().reset_index())
print("Junction_Detail null values: ", df['Junction_Detail'].isna().sum())

#print("\nDuplicate index \n", df.groupby('Accident_Index').count())

print("\n2nd_Road_Class with -1 values: ", df[df['2nd_Road_Class'] == -1]['2nd_Road_Class'].count())
plt.figure(figsize=(20,20))

plt.rc('xtick', labelsize=10) 

plt.rc('ytick', labelsize=10) 

sns.heatmap(df.corr(), cmap="YlGnBu",cbar_kws={"aspect": 40}, annot=True)
df = accidata.copy()

#feature engineering to create epoch timestamp

df['epoch'] = pd.to_datetime(

    pd.to_datetime(df['Date']).apply(str).str[:10] 

    + ' ' 

    + pd.to_datetime(df['Time']).apply(str).str[11:]).astype('int64')//1e9



to_drop = ['Junction_Detail', 'Accident_Index', 'Local_Authority_(District)',

           'Location_Easting_OSGR', 'Location_Northing_OSGR', '2nd_Road_Class', 'Time', 'Date', 'Year' ]

df = df.drop(to_drop,1)
#enter code/answer in this cell. You can add more code/markdown cells below for your answer.

df_t = df.copy()

cols = ['Weather_Conditions', 'Light_Conditions', 'Number_of_Casualties']



#weather preferred

weather_p= ['Raining without high winds', 'Fine without high winds','Snowing without high winds',

                     'Fine with high winds' ]

#weather not preferred

weather_np = ['Raining with high winds', 'Fog or mist','Snowing with high winds']

#weather other

weather_o = ['Unknown', 'Other']



light_p = ['Daylight: Street light present', 'Darkness: Street lights present and lit', ]

light_np = ['Darkness: Street lights present but unlit', 'Darkeness: No street lighting','Darkness: Street lighting unknown']



df_t.loc[df_t['Weather_Conditions'].isin(weather_p), 'Weather_Conditions' ] = 'p'

df_t.loc[df_t['Weather_Conditions'].isin(weather_np), 'Weather_Conditions' ] = 'np'

df_t.loc[df_t['Weather_Conditions'].isin(weather_o), 'Weather_Conditions' ] = 'o'

df_t.loc[df_t['Weather_Conditions'].isna(), 'Weather_Conditions' ] = 'o'

df_t.loc[df_t['Light_Conditions'].isin(light_p), 'Light_Conditions' ] = 'p'

df_t.loc[df_t['Light_Conditions'].isin(light_np), 'Light_Conditions' ] = 'np'



pd.get_dummies(df_t[['Weather_Conditions', 'Light_Conditions', 'Accident_Severity']]).corr()['Accident_Severity']



#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 

sns.pairplot(

            df.select_dtypes(include = ['int64', 'float64'])

             .dropna(subset = ['Longitude', 'Latitude'])

             .sample(1000)[['Longitude', 'Latitude','Urban_or_Rural_Area','Accident_Severity']]

            )

#Accident_Severity

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_validate

from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing



cols_to_use = ['Number_of_Casualties', 'Number_of_Vehicles', 'Longitude','Weather_Conditions_Raining with high winds',

       'Weather_Conditions_Snowing with high winds']



def preprocess(df):

    #most preprocessing is already done in above cells

    df = df.dropna(subset = ['Longitude', 'Latitude'])

    return df



def get_X_y(df):

    X = preprocess(df)

    

    #To use classification for one vs rest

    y = pd.get_dummies(X['Accident_Severity'].apply(str), prefix='Severity')

    

    to_drop = ['Local_Authority_(Highway)', 'Accident_Severity', 'LSOA_of_Accident_Location']

    X = pd.get_dummies(X.drop(to_drop,1))

    

    return X, y

    

X, y = get_X_y(df)

scores = []

for i in range(1,4):

    column_to_predict = 'Severity_' + str(i)

    scores.append(cross_validate(

        LogisticRegression(),

        preprocessing.scale(X[cols_to_use]),

        y[column_to_predict],

        cv=5, scoring='f1',

        return_train_score =True))
for i in range(len(scores)):

    print('F1 score for Accident Severity ' + str(i+1) + ': ', scores[i]["test_score"].mean())
severity1 = df[ df['Accident_Severity'] == 1]

severity2 = df[ df['Accident_Severity'] == 2]

severity3 = df[ df['Accident_Severity'] == 3]



fig , (a1,a2,a3) = plt.subplots(1, 3, figsize=(15,8))

x = 'Longitude'

y = 'Latitude'

s= .01

a=.3



severity1.plot(kind='scatter', x=x, y =y, color='red', s=s, alpha=a, subplots=True, ax=a1)

a1.set_title("Accident_Severity_1")

a1.set_facecolor('white')



severity2.plot(kind='scatter', x=x,y =y, color='red', s=s, alpha=a, subplots=True, ax=a2)

a2.set_title("Accident_Severity_2")

a2.set_facecolor('white')



severity3.plot(kind='scatter', x=x,y =y, color='red', s=s, alpha=a, subplots=True, ax=a3)

a3.set_title("Accident_Severity_3")

a3.set_facecolor('white')



plt.rc('xtick', labelsize=15) 

plt.rc('ytick', labelsize=15) 

fig.show()



#Referrence : https://www.kaggle.com/yesterdog/
print(severity1['Number_of_Casualties'].sum()/severity1.shape[0])

print(severity2['Number_of_Casualties'].sum()/severity2.shape[0])

print(severity3['Number_of_Casualties'].sum()/severity3.shape[0])