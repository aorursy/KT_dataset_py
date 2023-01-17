# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt

# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

#Modelling

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#importing all the required ML packages
from sklearn.linear_model import LogisticRegression #logistic regression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
ByPlatform = pd.read_csv('../input/By-Platform.csv')
ByCountry = pd.read_csv('../input/By-Country.csv')
ByAge = pd.read_csv('../input/By-Age.csv')
ByPlatform.head(2)
ByCountry.head(5)
ByAge.head(5)
ByPlatform.shape
ByPlatform.columns
ByPlatform.describe()
ByPlatform.columns
ByPlatform = ByPlatform.rename(columns= {'Reporting Starts':'ReportingStarts', 'Reporting Ends':'ReportingEnds', 'Ad Set':'AdSet', 
                                         'Result Rate':'ResultRate', 'Result Indicator':'ResultIndicator',
                                         'Cost per Results':'CostPerResults', 'Budget Type':'BudgetType',
                                         'Link Clicks':'LinkClicks', 'CPC (Link) (EUR)':'CPC_Link', 'CPC (All) (EUR)':'CPC_All',
                                         'Cost per 1,000 People Reached (EUR)':'CostPerThousandPeopleReached', 'CTR (All)':'CTR_All',
                                         'Amount Spent Today (EUR)':'AmountSpentToday_EUR',
                                         'Add to Cart (Facebook Pixel)':'AddtoCart_FPix',
                                         'Cost per Add To Cart (Facebook Pixel) (EUR)':'CostPerAddToCart_FPix',
                                         'Initiate Checkout (Facebook Pixel)':'InitiateCheckout_FPix',
                                         'Cost per Initiate Checkout (Facebook Pixel) (EUR)':'CostPerInitiateCheckout_FPix',
                                         'Purchase (Facebook Pixel)':'Purchase_FPix', 
                                         'Cost per Purchase (Facebook Pixel) (EUR)':'CostPerPurchase_FPix',
                                         'Amount Spent (EUR)':'AmountSpent_EUR', 
                                         'Purchase Conversion Value (Facebook Pixel)':'PurchaseConversionValue_FPix'})
ByPlatform.info()
f,ax = plt.subplots(figsize=(17, 17))
sns.heatmap(ByPlatform.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax ,cmap="Blues")
plt.show()
plt.rcParams['figure.figsize']=25,12  #to adjust the plot size

df = ByPlatform.drop(['ReportingStarts', 'ReportingEnds', 'AdSet', 
                      'Delivery','CostPerInitiateCheckout_FPix','BudgetType',
                      'ResultIndicator', 'Ends', 'Starts', 'CPC_Link', 'CPC_All', 
                      'CostPerThousandPeopleReached', 'CostPerAddToCart_FPix', 
                      'CostPerPurchase_FPix', 'Frequency','CTR_All', 'AmountSpentToday_EUR',
                     'ResultRate', 'CostPerResults','Reach', 'Purchase_FPix'], axis=1)
sns.boxplot(data=df) 
plt.ylim(0, 800)

plt.show()
#prepare data frame
dfcut = df.iloc[:, :500]
Ndf = dfcut.iloc[:1000,:]

# Creating trace1
trace1 = go.Scatter(y = Ndf.AddtoCart_FPix,
                    x = Ndf.index,
                    mode = "lines+markers",
                    name = "AddtoCart",
                    marker = dict(color = 'blue'))
# Creating trace2
trace2 = go.Scatter(y = Ndf.InitiateCheckout_FPix,
                    x = Ndf.index,
                    mode = "lines",
                    name = "InitiateCheckout",
                    marker = dict(color = 'red'))

dataS = [trace1, trace2]
layout = dict(title = 'AddtoCart_FPix and InitiateCheckout_FPix of the Ad Sets',
              xaxis= dict(title= 'Index',ticklen= 5,zeroline= False)
             )
fig = dict(data = dataS, layout = None)
iplot(fig)
# create trace 1 that is 3d scatter
df2 = ByPlatform.drop(['ReportingStarts', 'ReportingEnds', 'AdSet', 
                      'Delivery','CostPerInitiateCheckout_FPix','BudgetType',
                      'ResultIndicator', 'Ends', 'Starts', 'CPC_Link', 'CPC_All', 
                      'CostPerThousandPeopleReached', 'CostPerAddToCart_FPix', 
                      'CostPerPurchase_FPix', 'Frequency','CTR_All', 'AmountSpentToday_EUR',
                     'ResultRate', 'CostPerResults'], axis=1)
df3 = df2.iloc[:, :500]

#prepare data frame
Ndf2 = df3.iloc[:1800,:]

trace1 = go.Scatter3d(
                    y = Ndf2.Reach,
                    x = Ndf2.Purchase_FPix,
                    z = Ndf2.index,
                    mode = "markers",
                    name = "Reach",
                    marker=dict(size=10,color='rgb(255,0,0)'))

Data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    )
    
)
fig = go.Figure(data=Data, layout=layout)

iplot(fig)
#Plotting Link Clicks against Cost per Click
#It is expected to get smaller cost per clich while number of clicks increases. 

fig,axes = plt.subplots(1,1,figsize=(8,6),sharey=True)
axes.scatter(ByPlatform.CPC_Link.values,ByPlatform.LinkClicks.values)
axes.set_xlabel("CPC_Link")
axes.set_ylabel("LinkClicks")
plt.xlim(0, 2)
plt.show()
fig, axarr = plt.subplots(2, 2, figsize=(10, 10))
#AmountSpentToday
ByPlatform.groupby(["Platform"])["AmountSpentToday_EUR"].count().sort_values().plot(kind="barh",
title="Amount Spend Today per Platform", ax=axarr[0][0], color=sns.color_palette("BrBG", 7));
#Reach
ByPlatform.groupby(["Platform"])["Reach"].count().sort_values().plot(kind="barh",
title="Reach", ax=axarr[0][1],color=sns.color_palette("hls", 8) );
#Purchase
ByPlatform.groupby(["Platform"])["Purchase_FPix"].count().sort_values().plot(kind="barh",
title="Purchase per Platform",ax=axarr[1][0],color=sns.color_palette("Set2"));
#Link Clicked
ByPlatform.groupby(["Platform"])["LinkClicks"].count().sort_values().plot(kind="barh",
title="Link Clicks",ax=axarr[1][1],color=sns.color_palette("BuGn_r"));

plt.subplots(figsize = (10,5))
plt.title('Frequency / Platform')
sns.violinplot(x='Platform',data=ByPlatform, y = "Frequency")
plt.show()
#let's compare the Reach and InitiateCheckout stats by taking platforms into consideration.

sns.lmplot(x='Reach', y='InitiateCheckout_FPix', data=df2, height=4, aspect=2, fit_reg=True, 
           hue='Platform')
plt.show()
#check the missing columns
ByPlatform.isnull().values.any()
#Let's see if there is columns with at least 20% missing values and delete them if so. 
threshold = 0.2

ByPlatform = ByPlatform.drop(ByPlatform.std()[ByPlatform.std() < threshold].index.values, axis=1)

ByPlatform.shape
ByPlatformVars = ByPlatform.columns
data_types = {Var: ByPlatform[Var].dtype for Var in ByPlatformVars}

for Var in ByPlatformVars:
    if data_types[Var] == int:
        x = ByPlatform[Var].astype(float)
        ByPlatform.loc[:, Var] = x
        data_types[Var] = x.dtype
    elif data_types[Var] != float:
        x = ByPlatform[Var].astype('category')
        ByPlatform.loc[:, Var] = x
        data_types[Var] = x.dtype

data_types
float_ByPlatformVars = [Var for Var in ByPlatformVars
                     if data_types[Var] == float]
float_ByPlatformVars
float_x_means = ByPlatform.mean()

for Var in float_ByPlatformVars:
    x = ByPlatform[Var]
    isThereMissing = x.isnull()
    if isThereMissing.sum() > 0:
        ByPlatform.loc[isThereMissing.tolist(), Var] = float_x_means[Var]   
ByPlatform[float_ByPlatformVars].isnull().sum()
np.allclose(ByPlatform.mean(), float_x_means)
#Let's see the number of categories of each categorical feature:

ByPlatformVars = ByPlatform.columns

categorical_ByPlatformVars = [Var for Var in ByPlatformVars
                           if data_types[Var] != float]

categorical_levels = ByPlatform[categorical_ByPlatformVars].apply(lambda col: len(col.cat.categories))

categorical_levels
collapsed_categories = {}
removed_categorical_ByPlatformVars = set()

for Vars in categorical_ByPlatformVars:
    
    isTheremissing_value = ByPlatform[Vars].isnull()
    if isTheremissing_value.sum() > 0:
        ByPlatform[Vars].cat.add_categories('unknown', inplace=True)
        ByPlatform.loc[isTheremissing_value.tolist(), Vars] = 'unknown'
ByPlatform[categorical_ByPlatformVars].isnull().sum()
ByPlatform.info()
ByPlatform.columns
#Let's build a model with only a some featuresthat are int type
#selecting multiple features
#Purchase_FPix is prediction Target = y
ByPlatformData_features = ['Reach', 'Results', 
                           'CTR_All', 'AddtoCart_FPix','AmountSpent_EUR']
DatatoModel = ByPlatform[ByPlatformData_features].reindex()
DatatoModel.head()
from sklearn.tree import DecisionTreeRegressor
y = ByPlatform['Purchase_FPix']  #prediction target
# Define model. Specify a number for random_state to ensure same results each run
ModelData = DecisionTreeRegressor(random_state=1)

# Fit model
ModelData.fit(DatatoModel, y) 
print("Making predictions for the following 5 reports:")
print(DatatoModel.head())
print("The predictions are")
print(ModelData.predict(DatatoModel.head()))
df.groupby('Platform').apply(np.mean)
df2.columns
df2['Platform'].value_counts()
sns.countplot(x='Platform', data = df2, palette = 'hls')
plt.show()
NewData = ByPlatform.copy()
NewData['Platform'].replace(['Facebook','Instagram','Audience Network'],[1,0,-1],inplace=True)
NewData['Platform'].tail(10)
#we'll build a model with the data that categorical features droped.
#We select multiple features

features = ['Reach', 'Results','AddtoCart_FPix','AmountSpent_EUR','Platform']
LegModData = NewData[features].reindex()
LegModData.head()
#LegModData['Platform'] = LegModData_target

decisiontree = DecisionTreeClassifier() # defining  new object
train = LegModData[50:]   # seperated the first 50 rows are as test rest as train dataset
test = LegModData[:50]

x_train = LegModData.drop('Platform', axis=1) # x_train as Platform feature droped
y_train = LegModData['Platform']  # y_train is the Platform value

x_test = test.drop('Platform', axis=1) # same thing for test dataset
y_test = test['Platform']

decisiontree.fit(x_train, y_train) # model decisiontree fit to x_train, y_train values

pred = decisiontree.predict(x_test) # make prediction to the model about x_test and keep in pred 

print("accuracy:", accuracy_score(y_test, pred)) #to test the accuracy of the model
