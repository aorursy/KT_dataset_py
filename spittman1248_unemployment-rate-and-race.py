
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as np
import sys
import matplotlib 
import numpy as np
import seaborn as sns
from subprocess import check_output
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

%matplotlib inline


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Location of file
Location = '../input/acs2015_census_tract_data.csv'

df = pd.read_csv(Location)

df.info()
#In case we wanted to just see the data types of each column
#df.dtypes;
#Drop unnecessary columns and Data that has nan values
df = df.drop(['CensusTract','County'], 1)
df=df.dropna()
df = df.reset_index(drop = True)
df.shape
df.head()
#Convert percent of each race into total number of people with this ethnicity for each county
#Also find the total number of people that are self-employed, unemployed, etc... for each county

headnames = list(df)
X=headnames[4:10]
X.extend(headnames[len(headnames)-5:len(headnames)])

for i in X:
    df[i] = df['TotalPop'] * df[i]/100   

    #We won't be doing further data cleaning on our data in this tutorial.

#Just to double check our X here

X
#Group Data By State

df = df.groupby('State', as_index=False).sum()

#creates new column of df with the fraction of each gender in each state 
df['M_share'] = df.Men/df.TotalPop 
df['F_share'] = df.Women/df.TotalPop 

#creates new column of df with the fraction of unemployed people in each state
df['Unemployment_Rate'] = df.Unemployment/df.TotalPop 

#creates now column of df with the fraction of each race in each state 

df['White Fraction'] = df.White/df.TotalPop
df['Black Fraction'] = df.Black/df.TotalPop

df['Asian Fraction']=df.Asian/df.TotalPop
df['Hispanic Fraction']=df.Hispanic/df.TotalPop

df['Native Fraction']=df.Native/df.TotalPop

df['Pacific Fraction']=df.Asian/df.TotalPop

#Since I am interested in just the 50 US states and DC, I eliminate Puerto Rico from the dataset
df=df[~(df['State']=='Puerto Rico')]


df = df.reset_index(drop = True)
#Let's see what we are working with now

df.head()
#Let's look at the top and bottom 10 states for the fraction of unemployment

#For those that have latex, this is a nice way to print out labels
#plt.rc('text', usetex=True)

plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 14})

sorted_df = df.sort_values(['Unemployment_Rate'], ascending = [True])

plt.figure(figsize = (10,10))

plt.subplot(2,1,1)
plt.barh(range(10),sorted_df.tail(10).Unemployment_Rate)
plt.yticks(range(10),sorted_df.tail(10).State, fontsize = 10)
plt.plot([1,1],[0,10], '--',color = 'r')
plt.title('Top 10 Unemployment Rates By State 2011-2015')
plt.xlim([0,0.11])


plt.subplot(2,1,2)
plt.barh(range(10),sorted_df.head(10).Unemployment_Rate)
plt.yticks(range(10),sorted_df.head(10).State, fontsize = 10)
plt.plot([1,1],[0,10], '--',color = 'r')
plt.title('Lowest 10 Unemployment Rates By State 2011-2015')
plt.xlim([0,0.11])
!pip install plotly
import plotly.plotly as py


#df2 has all the state abbreviations that we need for the color map
df2 = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv')
df=df[~(df['State']=='District of Columbia')]

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df2['code'],
        z = df['Unemployment_Rate'].astype(float),
        locationmode = 'USA-states',
        text = df2['code'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Unemployment Rate")
        ) ]

layout = dict(
          geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename ='Unemployment Rate' )

#Pretty Plot will show!

plt.figure(figsize = (10,10))
f=df[df.columns[37:]].corr()
ax=sns.heatmap(f, annot=True)

new_df= pd.DataFrame()

new_df=df[df.columns[38:]]


model = LinearRegression()
X = new_df


#StandardScalar() standardizes features by removing the mean and scaling to unit variance
X_std = StandardScaler().fit_transform(X)
#X_std=X
y = df['Unemployment_Rate']

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)

model.fit(X_train,y_train)

plt.figure(figsize = (9,8))
plt.barh(range(X.shape[1]),model.coef_)
plt.xlabel('Coefficient')

plt.yticks(range(X.shape[1]),list(new_df), fontsize = 12)
plt.title('Regression Coefficients')

plt.show()

print('R^2 on training...',model.score(X_train,y_train))
print('R^2 on test...',model.score(X_test,y_test))

print('Model Coefficients',model.coef_)
print('Model Intercept',model.intercept_)
#Just to Doublecheck the Machine Learning Score, which we see doesn't have as good as R^2 score as the Machine Learnning One

import statsmodels.api as sm # import statsmodels 

X = X_std## X usually means our input variables (or independent variables)
#X=new_df
y = df["Unemployment_Rate"] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
model.summary()

