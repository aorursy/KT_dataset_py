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



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
df_train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv")

df_train.head()
df_test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/test.csv", parse_dates=["Date"])

df_test.head()
df_train.info()
# Countries with top 10 population

population_wise = df_train.Population.groupby(df_train["Country_Region"]).max().sort_values(ascending= False)

top10_population = pd.DataFrame()

top10_population["Population"] = population_wise.iloc[:10]

top10_population["Country"] = population_wise.index[:10]

top10_population  
# Top 10 countries with highest poplulation graph

plt.figure(figsize =(10,10))

plt.subplot(2,1,1)

sns.barplot(x="Country", y= "Population", data=top10_population )

plt.title("Top 10 countries based on poplulation");

plt.xlabel("Country")

plt.ylabel("Population");
df1 = df_train[df_train["Target"]== "ConfirmedCases"]

len(df1)
# To 10 countries with most confirmed covid cases

df2 = df1.TargetValue.groupby(df1["Country_Region"]).sum().sort_values(ascending = False)

df_confirmed = pd.DataFrame()

df_confirmed["ConfirmedCases"] = df2[:10]

df_confirmed["Country"]= df2.index[:10]

df_confirmed  
# Top 10 countries with most covid fatalities

df3 = df_train[df_train["Target"]=="Fatalities"]

df4 = df3.TargetValue.groupby(df3["Country_Region"]).sum().sort_values(ascending= False)

df_deaths= pd.DataFrame()

df_deaths["Deaths"] = df4[:10]

df_deaths["Country"]= df4.index[:10]

df_deaths
# plot confirmed cases and deaths due to covid 19 - top 10

plt.figure(figsize=(10,10))

plt.subplot(2,1,1)

sns.barplot(x= "Country", y= "ConfirmedCases", data= df_confirmed);

plt.title("Top 10 countries with most confirmed covid 19 cases")

plt.xlabel("Country")

plt.ylabel("Number of confirmed covid 19 cases");

plt.subplot(2,1,2)

sns.barplot(x= "Country", y= "Deaths", data= df_deaths);

plt.title("Top 10 countries with most deaths due to covid 19")

plt.xlabel("Country")

plt.ylabel("Number of deaths");
listMaxConfirmed10 = []

for country in df_confirmed["Country"]:

  listMaxConfirmed10.append(country)



listMaxConfirmed10
# Confirmed cases VS fatalities all around the world

df5 = df_train.TargetValue.groupby(df_train["Target"]).sum()

labels = [df5.index[0], df5.index[1]]

sizes= [df5[0], df5[1]]

explode= (0, 0.2)

plt.figure(figsize=(5,5))

plt.pie(sizes, explode = explode, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')

plt.axis('equal')

plt.show();
# In India Confirmed Vs Deaths due to Covid 19 until June 10, 2020

df6 = df_train[df_train["Country_Region"]== "India"].TargetValue.groupby(df_train["Target"]).sum()

labels = [df6.index[0], df6.index[1]]

sizes= [df6[0], df6[1]]

explode= (0, 0.2)

plt.figure(figsize=(5,5))

plt.pie(sizes, explode = explode, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')

plt.axis('equal')

plt.title(" Covid 19 Confirmed cases Vs Fatalities in India")

plt.show();
df_input = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv", parse_dates=["Date"])

#Parsing Date column from object to DateTime type

df_input.info()
df_traincopy = df_input

df_traincopy.head(2)
df_traincopy["Month"] = pd.DatetimeIndex(df_traincopy["Date"]).month

df_traincopy["Day"] = pd.DatetimeIndex(df_traincopy['Date']).day

df_traincopy[:2]
df_trainCopy_confirmed = df_traincopy[df_traincopy["Target"]== "ConfirmedCases"]

df_trainCopy_confirmed[:2]
df6 = pd.DataFrame()

for country in listMaxConfirmed10:

  df6 = df6.append(df_trainCopy_confirmed[df_trainCopy_confirmed["Country_Region"]== country])
df_trend= pd.DataFrame()

df7 = pd.DataFrame()

df8 = pd.DataFrame()

for country in listMaxConfirmed10:

  df7 = df6[df6["Country_Region"]== country]

  df7= df7.TargetValue.groupby(df7["Month"]).sum()

  df8["Month"]= df7.index 

  df8["Cases"]= df7

  df8["Country"]= country

  df_trend = df_trend.append(df8)
df_trend[:3]
# plotting the trend of Covid 19 contraction in top 10 countries with most confirmed cases

px.line(df_trend, x = "Month", y="Cases", color="Country", hover_name="Country",title= "Preogression of Covid-19")
df_traincopy.drop(columns=["Id", "County", "Province_State", "Date"], axis = 1, inplace = True)

df_traincopy.head(2)
# Finding string columns in features dataset

for label, content in df_traincopy.items():

  if pd.api.types.is_string_dtype(content):

    print(label)
# Turning string feature columns into categories

for label, content in df_traincopy.items():

  if pd.api.types.is_string_dtype(content):

    df_traincopy[label] = content.astype('category').cat.as_ordered()
df_traincopy.info()
#Turning category into numbers in features columns

for label, content in df_traincopy.items():

  if not pd.api.types.is_numeric_dtype(content):

    df_traincopy[label]= pd.Categorical(content).codes+1 
df_traincopy.head(2)
# Features 

df_X = df_traincopy.drop(['TargetValue'], axis = 1)

# Target

df_Y = df_traincopy["TargetValue"]
# Creating training and validation set

np.random.seed(42)

X_train, X_val, y_train, y_val = train_test_split(df_X, df_Y, test_size = 0.2)
# importing and training model

np.random.seed(42)



reg = RandomForestRegressor(n_jobs= -1)

reg.fit(X_train, y_train)
score = reg.score(X_val,y_val)

score
# Adding day and month column in test data

df_testCopy = df_test.copy()

df_testCopy["Month"] = pd.DatetimeIndex(df_test["Date"]).month

df_testCopy["Day"] = pd.DatetimeIndex(df_test['Date']).day

df_testCopy.head(2)
df_testCopy.drop( ["ForecastId", "County", "Province_State", "Date"], axis = 1, inplace = True)

df_testCopy.head(2)
# Converting string columns in test data to categorical

for label, content in df_testCopy.items():

  if pd.api.types.is_string_dtype(content):

    df_testCopy[label] = content.astype('category').cat.as_ordered()
# Converting category to numbers

for label,content in df_testCopy.items():

  if not pd.api.types.is_numeric_dtype(content):

    df_testCopy[label] = pd.Categorical(content).codes+1
df_testCopy.head(2)
# Predictiong for test data

predictions = reg.predict(df_testCopy)
list_pred = [int(x) for x in predictions]
pred_df = pd.DataFrame({'ForecastID':df_test.ForecastId,'Predictions': list_pred })

pred_df[:2]
Q05 = pred_df.groupby('ForecastID')['Predictions'].quantile(q=0.05).reset_index()

Q50 = pred_df.groupby('ForecastID')['Predictions'].quantile(q=0.5).reset_index()

Q95 = pred_df.groupby('ForecastID')['Predictions'].quantile(q=0.95).reset_index()



Q05.columns=['number','0.05']

Q50.columns=['number','0.5']

Q95.columns=['number','0.95']
concat_df = pd.concat([Q05, Q50['0.5'], Q95['0.95']], axis=1)

concat_df.head(4)
sub_df = pd.melt(concat_df, id_vars=['number'], value_vars=['0.05','0.5','0.95'])

sub_df.head()
sub_df["ForecastId_Quantile"] = sub_df["number"].astype(str)+"_"+ sub_df["variable"]

sub_df["TargetValue"] = sub_df["value"]

sub_df[:2]
sub_df = sub_df[["ForecastId_Quantile", "TargetValue"]]

sub_df[:2]
sub_df.reset_index(drop = True, inplace=True)

sub_df.head(2)
submission_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/submission.csv")
sub_df.to_csv("submission.csv", index = False)
sub_df.head(10)