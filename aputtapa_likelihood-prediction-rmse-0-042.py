# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots



import seaborn as sns



# Modeling packages as needed



from sklearn.linear_model import Ridge,RidgeCV



from sklearn.metrics import mean_squared_error



from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV



import xgboost as xgb



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Let's take the latest file for now;



data = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
data.shape
data.head()
# Any missing values?



data.describe()
# distribution of variables in the data

sns.distplot(data['Chance of Admit '])
sq = np.power(data['Chance of Admit '],2)



sns.distplot(sq)



# Looks good now; it has 2 modes but that shouldn't be a problem for now
# Let's create a grid and look at the scatters and distributions of all other variables in the data;

sns.pairplot(data)



plt.show()
# Let's look at a box plot of admit column by research



fig = go.Figure()

# Use x instead of y argument for horizontal plot

fig.add_trace(go.Box(x= data['Chance of Admit '][data['Research'] == 0],name="Research = 0"))

fig.add_trace(go.Box(x=data['Chance of Admit '][data['Research'] == 1],name="Research = 1"))



fig.update_layout(title='Box by Research',template='plotly_white')



fig.show()
pd.crosstab(data['LOR '],1)
data[['LOR ','Chance of Admit ']].groupby('LOR ').mean()



#  Clearly noticeable, let's look at a boxplot
fig = go.Figure()



unq = list(data.sort_values('LOR ')['LOR '].drop_duplicates())



for i in unq:

    

    fig.add_trace(go.Box(y = data['Chance of Admit '][data['LOR '] == i],name="LOR = "+str(i)))





fig.update_layout(title='Box by LOR score',template='plotly_white')



fig.show()
# University Rating by Chance of Admit 



data[['University Rating','Chance of Admit ']].groupby('University Rating').mean()
fig = go.Figure()



unq = list(data.sort_values('University Rating')['University Rating'].drop_duplicates())



for i in unq:

    

    fig.add_trace(go.Box(y = data['Chance of Admit '][data['University Rating'] == i],name="University Rating = "+str(i)))





fig.update_layout(title='Box by University Rating score',template='plotly_white')



fig.show()
#Calculate the correlations between the independent and dependent variables



corr = data.corr()

fig, ax = plt.subplots(figsize=(8, 8))

colormap = sns.diverging_palette(220, 10, as_cmap=True)

dropSelf = np.zeros_like(corr)

dropSelf[np.triu_indices_from(dropSelf)] = True

colormap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=colormap, linewidths=.5, annot=True, fmt=".2f", mask=dropSelf)

plt.title('Correlation Plots')

plt.show()
#  I want to try ridge as it reduces standard errors;



# split the data into train and test



data_for_model = data.drop(['Serial No.'],axis = 1)



data_for_model['dependent'] =  np.power(data_for_model['Chance of Admit '],2)



n = data_for_model.shape[0]



train = data_for_model[0:(n-100)]



test = data_for_model[400:n]



print(train.shape)

print(test.shape)
train.columns
x = train.drop(['Chance of Admit ','dependent'],axis = 1)

y = train['dependent']



xt = test.drop(['Chance of Admit ','dependent'],axis = 1)

yt = test['dependent']
ridgecv = RidgeCV(alphas = 10**np.linspace(10,-2,100)*0.5, scoring = 'neg_mean_squared_error', normalize = True)



ridgecv.fit(x, y)

ridgecv.alpha_
ridge = Ridge(alpha = ridgecv.alpha_, normalize = True)

ridge.fit(x, y)





m1 = mean_squared_error(yt, ridge.predict(xt))



print(m1)

print(np.sqrt(m1))



#  This would be WRT the transformed variable
plt.plot(np.sqrt(yt), np.sqrt(ridge.predict(xt)), '*')



# The predictions look good
# Calculate the RMSE on the actual variable by taking the square root as yt is a squared variable



m2 = mean_squared_error(np.sqrt(yt), np.sqrt(ridge.predict(xt)))



np.sqrt(m2)



print("MSE: {0:.5f}\nRMSE: {1:.5f}".format(m2, np.sqrt(m2)))