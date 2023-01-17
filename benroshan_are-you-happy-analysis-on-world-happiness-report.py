#load packages

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib as mpl

import seaborn as sns

import matplotlib.pylab as pylab

import numpy as np

import sklearn

import warnings

warnings.filterwarnings('ignore')

print('-'*25)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
#Configure Visualization Defaults

#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8
year_2015=pd.read_csv("../input/world-happiness/2015.csv")

year_2016=pd.read_csv("../input/world-happiness/2016.csv")

year_2017=pd.read_csv("../input/world-happiness/2017.csv")

year_2018=pd.read_csv("../input/world-happiness/2018.csv")

year_2019=pd.read_csv("../input/world-happiness/2019.csv")
year_2015.head(3)
year_2016_copy=year_2016.copy()

year_2016.head(3)
year_2017.head(3)
year_2018.head(3)
year_2019_copy=year_2019.copy()

year_2019.head(3)
#Looking at the datatypes of each factor

year_2019.dtypes
year_2019.describe()
print('Data columns with null values:',year_2019.isnull().sum(), sep = '\n')
year_2019_columns=['Generosity','Perceptions of corruption']

for i in year_2019_columns:

    plt.boxplot(year_2019[i])

    plt.show()
Q1 = year_2019['Generosity'].quantile(0.25)

Q3 = year_2019['Generosity'].quantile(0.75)

IQR = Q3 - Q1    #IQR is interquartile range. 



filter = (year_2019['Generosity'] >= Q1 - 1.5 * IQR) & (year_2019['Generosity'] <= Q3 + 1.2 *IQR)

year_2019_1=year_2019.loc[filter]



Q1 = year_2019_1['Perceptions of corruption'].quantile(0.25)

Q3 = year_2019_1['Perceptions of corruption'].quantile(0.75)

IQR = Q3 - Q1    #IQR is interquartile range. 



filter = (year_2019_1['Perceptions of corruption'] >= Q1 - 1.5 * IQR) & (year_2019_1['Perceptions of corruption'] <= Q3 + 0.6 *IQR)

cleaned_2019=year_2019_1.loc[filter]
year_2019_columns=['Generosity','Perceptions of corruption']

for i in year_2019_columns:

    plt.boxplot(cleaned_2019[i])

    plt.show()
# 5a. Happiness score(year-wise)

data_1=[year_2019['Country or region'],year_2015['Happiness Score'],year_2016['Happiness Score'],year_2017['Happiness.Score'],

       year_2018['Score'],year_2019['Score']]

headers_1=["Country","2015","2016","2017","2018","2019"]

score = pd.concat(data_1, axis=1, keys=headers_1,join='inner')



# 5b. GDP per capita score (year-wise)

data_2=[year_2019['Country or region'],year_2015['Economy (GDP per Capita)'],year_2016['Economy (GDP per Capita)'],year_2017['Economy..GDP.per.Capita.'],

       year_2018['GDP per capita'],year_2019['GDP per capita']]

gdp = pd.concat(data_2, axis=1, keys=headers_1,join='inner')



# 5c. Health Life Expectancy score (year-wise)

data_3=[year_2019['Country or region'],year_2015['Health (Life Expectancy)'],year_2016['Health (Life Expectancy)'],year_2017['Health..Life.Expectancy.'],

       year_2018['Healthy life expectancy'],year_2019['Healthy life expectancy']]

life_exp = pd.concat(data_3, axis=1, keys=headers_1,join='inner')



# 5d. Freedom score (year-wise)

data_4=[year_2019['Country or region'],year_2015['Freedom'],year_2016['Freedom'],year_2017['Freedom'],

       year_2018['Freedom to make life choices'],year_2019['Freedom to make life choices']]

freedom = pd.concat(data_4, axis=1, keys=headers_1,join='inner')



# 5d. Generosity score (year-wise)

data_5=[year_2019['Country or region'],year_2015['Generosity'],year_2016['Generosity'],year_2017['Generosity'],

       year_2018['Generosity'],year_2019['Generosity']]

generosity = pd.concat(data_5, axis=1, keys=headers_1,join='inner')
gdp.head()
# Year 2019

year_2019.columns = ["rank","region","score",

                  "gdp_per_capita","social_support","healthy_life_expectancy",

                 "freedom_to_life_choice","generosity","corruption_perceptions"]



# Year 2018

year_2018.columns = ["rank","region","score",

                  "gdp_per_capita","social_support","healthy_life_expectancy",

                 "freedom_to_life_choice","generosity","corruption_perceptions"]



pd.set_option('display.width', 500)

pd.set_option('display.expand_frame_repr', False)



# Year 2017

year_2017.drop(["Whisker.high","Whisker.low",

            "Family","Dystopia.Residual"],axis=1,inplace=True)



year_2017.columns =  ["region","rank","score",

                  "gdp_per_capita","healthy_life_expectancy",

                 "freedom_to_life_choice","generosity","corruption_perceptions"]



# Year 2016

year_2016.drop(['Region','Lower Confidence Interval','Upper Confidence Interval',

            "Family",'Dystopia Residual'],axis=1,inplace=True)



year_2016.columns = ["region","rank","score",

                  "gdp_per_capita","healthy_life_expectancy",

                 "freedom_to_life_choice","corruption_perceptions","generosity"]



# Year 2015

year_2015.drop(["Region",'Standard Error', 'Family', 'Dystopia Residual'],axis=1,inplace=True)

year_2015.columns = ["region", "rank", "score", 

                     "gdp_per_capita","healthy_life_expectancy", 

                     "freedom_to_life_choice", "corruption_perceptions", "generosity"]



#Adding year to all the dataframe

year_2015["year"] = 2015

year_2016["year"] = 2016

year_2017["year"] = 2017

year_2018["year"] = 2018

year_2019["year"] = 2019



final_happy = year_2015.append([year_2016,year_2017,year_2018,year_2019])

final_happy=final_happy.drop(['freedom_to_life_choice','freedom_to_life_choice','social_support'],axis=1)

final_happy.dropna(inplace=True)

final_happy.head()
sns.set(font_scale = 2)

plt.style.use('seaborn-white')

labels = ['Finland','Denmark','Norway','Iceland','Netherlands','Switzerland','Sweden','New Zealand','Canada','Austria']

top_10=score.head(10)

ax=top_10.plot.bar(rot=0)

ax.set_xticklabels(labels)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=40)

fig = plt.gcf()

fig.set_size_inches(40,10)
sns.set(font_scale = 2)

plt.style.use('seaborn-white')

labels_tail = ['Zimbabwe','Haiti','Botswana','Syria','Malawi','Yemen','Rwanda','Tanzania','Afghanistan','Central African Republic	']

top_10=score.tail(10)

ax=top_10.plot.bar(rot=0)

ax.set_xticklabels(labels_tail)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=40)

fig = plt.gcf()

fig.set_size_inches(40,10)
gdp_10=gdp.head(10)

plt.plot( 'Country', '2015', data=gdp_10, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)

plt.plot( 'Country', '2016', data=gdp_10, marker='X', color='black', markersize=20, linewidth=2)

plt.plot( 'Country', '2017', data=gdp_10, marker='o', color='crimson', markersize=12, linewidth=2, linestyle='solid', label="2017")

plt.plot( 'Country', '2018', data=gdp_10, marker='X', color='orangered', markersize=20, linewidth=2, linestyle='dashed', label="2018")

plt.plot( 'Country', '2019', data=gdp_10, marker='o', color='olive', markersize=15, linewidth=8, linestyle='solid', label="2019")

plt.legend()

fig = plt.gcf()

fig.set_size_inches(40,10)
ax=sns.boxplot(y='healthy_life_expectancy', x='region', 

                 data=final_happy, 

                 palette="BrBG",

                 dodge=False)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=7,rotation=90, ha="right")

fig = plt.gcf()

fig.set_size_inches(15,10)
plt.figure(figsize = (15, 7))

plt.style.use('seaborn-white')

plt.subplot(2,2,1)

top_10_corrupt=final_happy[['region', 'corruption_perceptions']].sort_values(by = 'corruption_perceptions',ascending = False).head(32)

ax=sns.barplot(x="region", y="corruption_perceptions", data=top_10_corrupt, palette="Greens")

ax.set_xticklabels(ax.get_xticklabels(),fontsize=11,rotation=40, ha="right")

ax.set_title('Top 10 Corruption free Countries',fontsize= 22)

ax.set_xlabel('Countries',fontsize = 20) 

ax.set_ylabel('Corruption rate', fontsize = 20)



plt.subplot(2,2,2)

bot_10_corrupt=final_happy[['region', 'corruption_perceptions']].sort_values(by = 'corruption_perceptions',ascending = True).head(32)

ax=sns.barplot(x="region", y="corruption_perceptions", data=bot_10_corrupt,palette="Greys")

ax.set_xticklabels(ax.get_xticklabels(),fontsize=11, rotation=40, ha="right")

ax.set_title('Bottom 10 Corruption free Countries',fontsize= 22)

ax.set_xlabel('Countries',fontsize = 20) 

ax.set_ylabel('Corruption rate', fontsize = 20)
plt.figure(figsize = (15, 7))

plt.style.use('seaborn-white')

plt.subplot(2,2,1)

top_10_Generous=final_happy[['region', 'generosity']].sort_values(by = 'generosity',ascending = False).head(24)

ax=sns.barplot(x="region", y="generosity", data=top_10_Generous, palette="Greens")

ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40, ha="right")

ax.set_title('Top 10 Generous Countries',fontsize = 22)

ax.set_xlabel('Countries',fontsize = 20) 

ax.set_ylabel('Generosity rate', fontsize = 20)





plt.subplot(2,2,2)

bot_10_Generous=final_happy[['region', 'generosity']].sort_values(by = 'generosity',ascending = True).head(24)

ax=sns.barplot(x="region", y="generosity", data=bot_10_Generous,palette="Greys")

ax.set_xticklabels(ax.get_xticklabels(),fontsize=11, rotation=40, ha="right")

ax.set_title('Bottom 10 Generous Countries',fontsize = 22)

ax.set_xlabel('Countries',fontsize = 20) 

ax.set_ylabel('Generosity rate', fontsize = 20)
ax=sns.barplot(x="Region", y="Happiness Score", data=year_2016_copy, palette="magma")

ax.set_xticklabels(ax.get_xticklabels(),fontsize=11, rotation=40, ha="right")

fig = plt.gcf()

fig.set_size_inches(10,10)
sns.set(font_scale = 2)

sns.pairplot(year_2019_copy,size=5,corner=True,kind="reg")
#Inspired from: https://www.kaggle.com/mshinde10/predicting-world-happiness

import plotly.graph_objs as go

from plotly.offline import iplot



map_happy = dict(type = 'choropleth', 

           locations = year_2019_copy['Country or region'],

           locationmode = 'country names',

           z = year_2019_copy['Score'], 

           text = year_2019_copy['Country or region'],

           colorbar = {'title':'Happiness score'})



layout = dict(title = 'Happiness Score across the World', 

              geo = dict(showframe = False, projection = {'type': 'equirectangular'}))



choromap3 = go.Figure(data = [map_happy], layout=layout)

iplot(choromap3)
# Dataframe for regression model

reg_model=year_2019_copy.drop(['Overall rank','Country or region'],axis=1)

# Outcome and Explanatory variables

x = reg_model.drop(['Score'],axis=1)

y = reg_model['Score']
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(x,y)

print("Regression coefficients:",reg.coef_)

print("Regression intercept:",reg.intercept_)

print("R square value:",reg.score(x,y))
r2 = reg.score(x,y)

n = x.shape[0]

p = x.shape[1]



adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)

print("Adjusted R square value:",adjusted_r2)
from sklearn.feature_selection import f_regression

f_regression(x,y)

p_values = f_regression(x,y)[1]

reg_summary = pd.DataFrame(data = x.columns.values, columns=['Features'])

reg_summary ['Coefficients'] = reg.coef_

reg_summary ['p-values'] = p_values.round(3)

reg_summary
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

score=r2_score(y_test,y_pred)

print("R2 score:",score)
import statsmodels.api as sm

x1 = sm.add_constant(x)

results = sm.OLS(y,x1).fit()

results.summary()