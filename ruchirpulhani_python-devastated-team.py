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
#load the data
train = pd.read_csv("/kaggle/input/train-1/Cleared_data_03.06.2020.csv")
train

#head of the data 
train.head()
# It shows that our data set was last updated on 27th May , 2020 
train.tail()
# importing important models
import pandas as pd
import numpy as np
import warnings
import scipy
from datetime import timedelta

# Forceasting with decompasable model
from pylab import rcParams
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# For marchine Learning Approach
from statsmodels.tsa.tsatools import lagmat
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')

warnings.filterwarnings('ignore')
train.describe()
#total counts of available data ( region wise)
train["Region"].value_counts()
# Chart showing the amount of data entries for each state in Brazil 
# For example for the state SP we have > 17500 data entries because it has many provinces and thus the entries are more . 
# RO-TO - North Region , MA - BA ( Nordeste) , MG - SP ( Sudeste) , PR - DF ( Sul)
plt.figure(figsize=(8,5))
sns.countplot(x="state",data = train)
plt.xticks(rotation = "vertical")
# scatter diagram depicting the cases accumulated during various weeks of the pandemic
# For example , It shows that in the 22nd week ( 24th May to 27th May) 
import matplotlib.pyplot as plt
%matplotlib inline
train.plot(kind = "scatter", x = "cases accumulated", y = "week of epedamic")
plt.show()
# Plotting accumulated cases with the week number and displayed according to regions
sns.FacetGrid(train,size=5, hue="Region").map(plt.scatter, "week of epedamic", "cases accumulated").add_legend()
# Define names for the state abbreviations
state_names = {'AC':'Acre',
               'AL':'Alagoas',
               'AP':'Amapá',
               'AM':'Amazonas',
               'BA': 'Bahia',
               'CE':'Ceará',
               'DF':'Distrito Federal',
               'ES':'Espírito Santo',
               'GO':'Goiás',
               'MA':'Maranhão',
               'MT': 'Mato Grosso',
               'MS':'Mato Grosso do Sul',
               'MG':'Minas Gerais',
               'PA':'Pará',
               'PB':'Paraíba',
               'PR':'Paraná',
               'PE':'Pernambuco',
               'PI':'Piauí',
               'RJ':'Rio de Janeiro',
               'RN':'Rio Grande do Norte',
               'RS':'Rio Grande do Sul',
               'RO':'Rondônia',
               'RR':'Roraima',
               'SC':'Santa Catarina',
               'SP':'São Paulo',
               'SE':'Sergipe',
               'TO':'Tocantins',
              }
# add a column with state names
train['state names'] = train['state'].map(state_names)
#Display head of the data , the coloumn has been added to the right end 
train.head()
#Let's first just take the slice of the dataframe that represents the most recent covid-19 numbers
# to see some information about the current situation in the country.

#To get the most recent data ]we load recent data set



recent = pd.read_csv("/kaggle/input/recentt/Recent.csv",encoding = 'cp1252')
recent.head()
# Overview of the data for the date , 27th May , 2020
recent
#No. of cases on the 27th of May , 2020
recent['New cases'].sum()
recent['New death'].sum()
#I will do an analysis of the data here, separating them by region of the country.
byregion = recent.groupby(['Region']).sum()
#Which region has more confirmed cases?
# Highest number of cases region wise on the 27th of May , 2020
byregion[byregion['New cases'] == byregion['New cases'].max()]['New cases']
# Sorting the data to see cases in Patrocínio Monte Carmelo Health region on 27th May , 2020
recent.loc[recent['name health region'].str.contains('Patrocínio Monte Carmelo',case=False,regex=False)==True]
# Region with the highest number of death counts on the 27th May , 2020
byregion[byregion['New death'] == byregion['New death'].max()]['New death']
byregion_overall = train.groupby(['Region']).sum()
# Pie chart to show Number of confirmed cases of covid-19 in Brazil by region
plt.figure(figsize=(12,6))
values = byregion_overall['New cases']
labels = byregion_overall.reset_index()['Region']
plt.pie(values, labels= values)
plt.title('Number of confirmed cases of covid-19 in Brazil by region')
plt.legend(labels,loc=3, bbox_to_anchor=(1, -0.2, 0.5, 1))
#Pie chart to show the number of new confirmed cases of covid-19 on 27th May ,2020 in Brazil by region
plt.figure(figsize=(12,6))
values = byregion['New cases']
labels = byregion.reset_index()['Region']
plt.pie(values, labels= values)
plt.title('Number of new confirmed cases of covid-19 on 27th May ,2020 in Brazil by region')
plt.legend(labels,loc=3, bbox_to_anchor=(1, -0.2, 0.5, 1))
# Pie chart showing Distribution of confirmed cases of covid-19 in Brazil in percentage
plt.figure(figsize=(12,6))
values = byregion_overall['New cases']
labels = byregion_overall.reset_index()['Region']
plt.pie(values, labels=labels, autopct='%1.1f%%',
counterclock=False, pctdistance=0.6, labeldistance=1.2)
plt.title('Distribution of confirmed cases of covid-19 in Brazil in percentage')
plt.figure(figsize=(12,6))
values = byregion_overall['New death']
labels = byregion_overall.reset_index()['Region']
plt.pie(values, labels= values)
plt.title('Número de mortes por covid-19 no Brasil por região')
plt.legend(labels,loc=3, bbox_to_anchor=(1, -0.2, 0.5, 1))
# Bar chart region wise - total cases
plt.figure(figsize=(12,6))
plt.title('Number of confirmed cases of covid-19 in Brazil by Region')
sns.barplot(x='Region', y='New cases', data=byregion_overall.reset_index(), palette='summer')
# Bar chart - Total deaths region wise
plt.figure(figsize=(12,6))
plt.title('Number of confirmed deaths of covid-19 in Brazil by Region')
sns.barplot(x='Region', y='New death', data=byregion_overall.reset_index(), palette='summer')
recents_northeast = recent[recent['Region'] == 'Nordeste'].copy()
recents_northeast
recents_northeast[recents_northeast['New cases'] == recents_northeast['New cases'].max()]['state']
recents_northeast['New cases'].max()
train.info()
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
%matplotlib inline
plt.style.use('ggplot')
sns.set_style('whitegrid')
recents_northeast.head()
Northeast_recent = recents_northeast.groupby(['state']).sum()
plt.figure(figsize=(10,8))
values = Northeast_recent['New cases']
labels = Northeast_recent.reset_index()['state']
plt.pie(values, labels= values)
plt.title('Number of confirmed cases of covid-19 in the northeast region')
plt.legend(labels,loc=3, bbox_to_anchor=(1, -0.2, 0.5, 1))
def customized_bin(column, cuttingpoints, custom_labels):
    min_val = column.min()
    max_val = column.max()
    
    breaking_points = [min_val] + cuttingpoints + [max_val]
    print(breaking_points)
    
    colBinned = pd.cut(column, bins=breaking_points, labels=custom_labels, include_lowest=True)
    return colBinned

## call the function ##
cuttingpoints = [5, 20, 30]
custom_labels = ["low", "medium", "high", "very high"]
train["NewCasesBinned"] = customized_bin(train["New cases"], cuttingpoints, custom_labels)

## see output ##
train
print(pd.value_counts(train["NewCasesBinned"], sort=False))
## replacing information ##
def custom_coding(column, dictionary):
    column_coded = pd.Series(column, copy=True)
    for key, value in dictionary.items():
        column_coded.replace(key, value, inplace=True)
    
    return column_coded

## code NewCasesBinned - 
train["NewCases_coded"] = custom_coding(train["NewCasesBinned"], {"low":0,"medium":0 ,"high":1, "very high":1,})

train
train["New cases"].hist(bins=10)
recent.boxplot(column="cases accumulated", figsize=(15,8))
recent.describe()
train["NewCases_coded"].value_counts(ascending=True)
train["NewCasesBinned"].value_counts(ascending=True)
train["NewCasesBinned"].value_counts(ascending=True, normalize=True)
def customized_bin(column, cuttingpoints, custom_labels):
    min_val = column.min()
    max_val = column.max()
    
    breaking_points = [min_val] + cuttingpoints + [max_val]
    print(breaking_points)
    
    colBinned = pd.cut(column, bins=breaking_points, labels=custom_labels, include_lowest=True)
    return colBinned

## call the function ##
cuttingpoints = [2, 5, 10]
custom_labels = ["low", "medium", "high", "very high"]
recent["NewCasesBinned"] = customized_bin(recent["New cases"], cuttingpoints, custom_labels)

## replacing information ##
def custom_coding(column, dictionary):
    column_coded = pd.Series(column, copy=True)
    for key, value in dictionary.items():
        column_coded.replace(key, value, inplace=True)
    
    return column_coded

## code NewCasesBinned - 
recent["NewCases_coded"] = custom_coding(recent["NewCasesBinned"], {"low":0,"medium":0 ,"high":1, "very high":1,})

recent

temp1 = pd.crosstab([recent["Region"], recent["state"]], recent["NewCases_coded"])
print(temp1)
temp1.plot(kind="bar", stacked=True, color=["orange", "grey"], grid=True, figsize=(12,6))
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
def classification_model(model, data, predictors, outcome, kfoldnumber):
    ## fit data
    model.fit(data[predictors], data[outcome])
    ## predict train-data
    predictvalues = model.predict(data[predictors])
    ## accuracy
    accuracy = metrics.accuracy_score(predictvalues, data[outcome])
    print("Accuracy: %s" % "{0:.3%}".format(accuracy))
    ##
    ## k-fold cross-validation
    kfold = KFold(n_splits=kfoldnumber)
    error =  []
    ##
    for train, test in kfold.split(data):
        #print("------ run ------")
        #print("traindata")
        #print(train)
        #print("testdata")
        #print(test)
        ##
        ## filter training data
        train = data[predictors].iloc[train,:]
        target= data[outcome].iloc[train]
        ##
        #print("Traindata")
        #print(train_data)
        #print("TrainTarget")
        #print(train_target)
        ##
        ## fit data
        model.fit(train_data, train_target)
        ##
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    ##
    print("Cross Validation Score: %s" % "{0:.3%}".format(np.mean(error)))
    ##
    model.fit(data[predictors], data[outcome])
        
recent.rename(columns={"":"NewdeathYesNo"},inplace=True)
outcome_var = "New cases"
predictor_var = ["NewCases_coded"]
model = LogisticRegression(solver="lbfgs")
##
classification_model(model, recent, predictor_var, outcome_var, 5)
def customized_bin(column, cuttingpoints, custom_labels):
    min_val = column.min()
    max_val = column.max()
    
    breaking_points = [min_val] + cuttingpoints + [max_val]
    print(breaking_points)
    
    colBinned = pd.cut(column, bins=breaking_points, labels=custom_labels, include_lowest=True)
    return colBinned

## call the function ##
cuttingpoints = [0.5]
custom_labels = ["No","Yes"]
recent["NewCasesYesNo"] = customized_bin(recent["New cases"], cuttingpoints, custom_labels)

## see output ##
recent
## replacing information ##
def custom_coding(column, dictionary):
    column_coded = pd.Series(column, copy=True)
    for key, value in dictionary.items():
        column_coded.replace(key, value, inplace=True)
    
    return column_coded

recent["New_cases-code"] = custom_coding(recent["NewCases-Yes/No"], {"No":0,"Yes":1})
recent
temp1 = pd.crosstab([recent["NewCasesBinned"], recent["Region"]], recent["NewCases-Yes/No"])
print(temp1)
temp1.plot(kind="bar", stacked=True, color=["orange", "grey"], grid=True, figsize=(12,6))
def classification_model(model, data, predictors, outcome, kfoldnumber):
    ## fit data
    model.fit(data[predictors], data[outcome])
    ## predict train-data
    predictvalues = model.predict(data[predictors])
    ## accuracy
    accuracy = metrics.accuracy_score(predictvalues, data[outcome])
    print("Accuracy: %s" % "{0:.3%}".format(accuracy))
    ##
    ## k-fold cross-validation
    kfold = KFold(n_splits=kfoldnumber)
    error =  []
    ##
    for train, test in kfold.split(data):
        #print("------ run ------")
        #print("traindata")
        #print(train)
        #print("testdata")
        #print(test)
        ##
        ## filter training data
        train = data[predictors].iloc[train,:]
        target= data[outcome].iloc[train]
        ##
        #print("Traindata")
        #print(train_data)
        #print("TrainTarget")
        #print(train_target)
        ##
        ## fit data
        model.fit(train_data, train_target)
        ##
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    ##
    print("Cross Validation Score: %s" % "{0:.3%}".format(np.mean(error)))
    ##
    model.fit(data[predictors], data[outcome])
        
def customized_bin(column, cuttingpoints, custom_labels):
    min_val = column.min()
    max_val = column.max()
    
    breaking_points = [min_val] + cuttingpoints + [max_val]
    print(breaking_points)
    
    colBinned = pd.cut(column, bins=breaking_points, labels=custom_labels, include_lowest=True)
    return colBinned

## call the function ##
cuttingpoints = [.5]
custom_labels = ["No","Yes"]
recent["NewDeaths-Yes/No"] = customized_bin(recent["New death"], cuttingpoints, custom_labels)

## see output ##
recent
## replacing information ##
def custom_coding(column, dictionary):
    column_coded = pd.Series(column, copy=True)
    for key, value in dictionary.items():
        column_coded.replace(key, value, inplace=True)
    
    return column_coded
## code LoanStatus - Y > 1, N > 0, yes > 1, Yes > 1, ...
recent["New__Deaths-code"] = custom_coding(recent["NewDeaths-Yes/No"], {"No":0,"Yes":1})
recent
def classification_model(model, data, predictors, outcome, kfoldnumber):
    ## fit data
    model.fit(data[predictors], data[outcome])
    ## predict train-data
    predictvalues = model.predict(data[predictors])
    ## accuracy
    accuracy = metrics.accuracy_score(predictvalues, data[outcome])
    print("Accuracy: %s" % "{0:.3%}".format(accuracy))
    ##
    ## k-fold cross-validation
    kfold = KFold(n_splits=kfoldnumber)
    error =  []
    ##
    for train, test in kfold.split(data):
        #print("------ run ------")
        #print("traindata")
        #print(train)
        #print("testdata")
        #print(test)
        ##
        ## filter training data
        abc = data[predictors].iloc[train,:]
        target = data[outcome].iloc[train]
        ##
        #print("Traindata")
        #print(train_data)
        #print("TrainTarget")
        #print(train_target)
        ##
        ## fit data
        model.fit(abc, target)
        ##
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    ##
    print("Cross Validation Score: %s" % "{0:.3%}".format(np.mean(error)))
    ##
    model.fit(data[predictors], data[outcome])
recent.rename(columns={"NewDeaths-Yes/No":"NewdeathYesNo"},inplace=True)
outcome_var = "NewdeathYesNo"
predictor_var = ["New cases"]
model = LogisticRegression(solver="lbfgs")
##
classification_model(model,recent, predictor_var, outcome_var, 5)
def classification_model(model, data, predictors, outcome, kfoldnumber):
    ## fit data
    model.fit(data[predictors], data[outcome])
    ## predict train-data
    predictvalues = model.predict(data[predictors])
    ## accuracy
    accuracy = metrics.accuracy_score(predictvalues, data[outcome])
    print("Accuracy: %s" % "{0:.3%}".format(accuracy))
    ##
    ## k-fold cross-validation
    kfold = KFold(n_splits=kfoldnumber)
    error =  []
    ##
    for train, test in kfold.split(data):
        #print("------ run ------")
        #print("traindata")
        #print(train)
        #print("testdata")
        #print(test)
        ##
        ## filter training data
        train = data[predictors].iloc[train,:]
        target= data[outcome].iloc[train]
        ##
        #print("Traindata")
        #print(train_data)
        #print("TrainTarget")
        #print(train_target)
        ##
        ## fit data
        model.fit(train_data, train_target)
        ##
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    ##
    print("Cross Validation Score: %s" % "{0:.3%}".format(np.mean(error)))
    ##
    model.fit(data[predictors], data[outcome])
outcome_var = "NewdeathYesNo"
predictor_var = ["New cases"]
model = LogisticRegression(solver="lbfgs")
##
classification_model(model,recent, predictor_var, outcome_var, 5)
recent_filteredNORTE = recent[recent['Region'] =='Norte']
recent_filteredNORTE

def classification_model(model, data, predictors, outcome, kfoldnumber):
    ## fit data
    model.fit(data[predictors], data[outcome])
    ## predict train-data
    predictvalues = model.predict(data[predictors])
    ## accuracy
    accuracy = metrics.accuracy_score(predictvalues, data[outcome])
    print("Accuracy: %s" % "{0:.3%}".format(accuracy))
    ##
    ## k-fold cross-validation
    kfold = KFold(n_splits=kfoldnumber)
    error =  []
    ##
    for train, test in kfold.split(data):
        #print("------ run ------")
        #print("traindata")
        #print(train)
        #print("testdata")
        #print(test)
        ##
        ## filter training data
        abc = data[predictors].iloc[train,:]
        target = data[outcome].iloc[train]
        ##
        #print("Traindata")
        #print(train_data)
        #print("TrainTarget")
        #print(train_target)
        ##
        ## fit data
        model.fit(abc, target)
        ##
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    ##
    print("Cross Validation Score: %s" % "{0:.3%}".format(np.mean(error)))
    ##
    model.fit(data[predictors], data[outcome])
recent_filtered.rename(columns={"NewDeaths-Yes/No":"NewdeathYesNo"},inplace=True)
outcome_var = "NewdeathYesNo"
predictor_var = ["New cases"]
model = LogisticRegression(solver="lbfgs")
##
classification_model(model,recent_filteredNORTE, predictor_var, outcome_var, 5)
