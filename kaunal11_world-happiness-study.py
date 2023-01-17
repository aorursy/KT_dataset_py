# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Import Dependencies

%matplotlib inline



# Start Python Imports

import math, time, random, datetime



# Data Manipulation

import numpy as np

import pandas as pd



# Visualization 

import matplotlib.pyplot as plt

from matplotlib import rcParams

import missingno as msno

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go



# Preprocessing

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize



# Machine learning

# import catboost

# from sklearn.model_selection import train_test_split

# from sklearn import model_selection, tree, preprocessing, metrics, linear_model

# from sklearn.svm import LinearSVC

# from sklearn.ensemble import GradientBoostingClassifier

# from sklearn.neighbors import KNeighborsClassifier

# from sklearn.naive_bayes import GaussianNB

# from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier

# from sklearn.tree import DecisionTreeClassifier

# from catboost import CatBoostClassifier, Pool, cv



# Let's ignore warnings for now

import warnings

warnings.filterwarnings('ignore')





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import Datasets



df_2015=pd.read_csv('../input/world-happiness-report/2015.csv')

df_2016=pd.read_csv('../input/world-happiness-report/2016.csv')

df_2017=pd.read_csv('../input/world-happiness-report/2017.csv')

df_2018=pd.read_csv('../input/world-happiness-report/2018.csv')

df_2019=pd.read_csv('../input/world-happiness-report/2019.csv')

df_2020=pd.read_csv('../input/world-happiness-report/2020.csv')
# Verifying



df_2020.head()
# Checking for Null values



df_2020.isnull().sum()
# Visualizing the happiest countries (Top 10)

sns.barplot(x="Ladder score", y="Country name", data=df_2020.sort_values(by='Ladder score',ascending=False).head(10)).set_title('Top 10 happiest countries')
# Visualizing the unhappiest countries (Top 10)

sns.barplot(x="Ladder score", y="Country name", data=df_2020.sort_values(by='Ladder score').head(10)).set_title('Least 10 happiest countries')
# Showing the happiness trend by regions



sns.barplot(x="Ladder score", y=df_2020.groupby(['Regional indicator']).mean().sort_values(by='Ladder score',ascending=False).index, data=df_2020.groupby(['Regional indicator']).mean().sort_values(by='Ladder score',ascending=False)).set_title('Happiness by Region')
# Dropping a few columns that are not needed



df_2020_analysis=df_2020.drop(['Regional indicator','Standard error of ladder score','upperwhisker','lowerwhisker',

                               'Explained by: Log GDP per capita','Explained by: Social support',

                               'Explained by: Healthy life expectancy','Explained by: Freedom to make life choices',

                               'Explained by: Generosity','Explained by: Perceptions of corruption'],axis=1)

df_2020_analysis.head()
# Checking correlation between various parameters



rcParams["figure.figsize"] = 20,10

plt.title("Corellation between different features")

sns.heatmap(df_2020_analysis.corr(),annot=True,cmap="YlGnBu")
# Creating a new Dataframe to track the Happiness scores for the top 10 and bottom 10 countries from 2020 and

# seeing how they fared in previous years



df=pd.DataFrame(df_2020.sort_values(by='Ladder score',ascending=False)['Country name'].head(10))

# Getting Happiness scores across years for the 10 Happiest countries in 2020



score_2015=[]

score_2016=[]

score_2017=[]

score_2018=[]

score_2019=[]

score_2020=[]

for country in df['Country name']:

    score_2015.append(list(df_2015[df_2015['Country']==country]['Happiness Score']))

    score_2016.append(list(df_2016[df_2016['Country']==country]['Happiness Score']))

    score_2017.append(list(df_2017[df_2017['Country']==country]['Happiness.Score']))

    score_2018.append(list(df_2018[df_2018['Country or region']==country]['Score']))

    score_2019.append(list(df_2019[df_2019['Country or region']==country]['Score']))

    score_2020.append(list(df_2020[df_2020['Country name']==country]['Ladder score']))



df['2015']=pd.DataFrame(score_2015)

df['2016']=pd.DataFrame(score_2016)

df['2017']=pd.DataFrame(score_2017)

df['2018']=pd.DataFrame(score_2018)

df['2019']=pd.DataFrame(score_2019)

df['2020']=pd.DataFrame(score_2020)

df=df.transpose()

df.columns=['Finland','Denmark','Switzerland','Iceland','Norway','Netherlands','Sweden','New Zealand','Austria','Luxembourg']

df.drop(['Country name'],inplace=True)

df.reset_index(drop=True, inplace=True)

df['Year']=['2015','2016','2017','2018','2019','2020']
df
# multiple line plot

num=0

for column in df.drop('Year', axis=1):

    

    num+=1

    plt.plot(df['Year'], df[column], linewidth=1, alpha=0.9, label=column,marker='o')

    

# Add legend

plt.legend(loc=2, ncol=2)

 

# Add titles

plt.title("Trend of happiness index for happiest countries", loc='center', fontsize=12, fontweight=0, color='orange')

plt.xlabel("Year")

plt.ylabel("Happiness Score")
