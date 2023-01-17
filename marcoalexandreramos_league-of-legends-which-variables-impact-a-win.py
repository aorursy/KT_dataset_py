#importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
#importing and defining Data Frame

df = pd.read_csv("../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")
# cheking info of the df

def data_inv(df):

    print('dataframe: ',df.shape[0])

    print('dataset variables: ',df.shape[1])

    print('-'*10)

    print('dateset columns: \n')

    print(df.columns)

    print('-'*10)

    print('data-type of each column: \n')

    print(df.dtypes)

    print('-'*10)

    print('missing rows in each column: \n')

    c=df.isnull().sum()

    print(c[c>0])

data_inv(df)
# checking the df:

df.head()
# creating a copy/checkpoint before deleting unnecessary columns

df_1 = df.copy()
#dropping columns

df_1 = df_1.drop(['blueGoldDiff', 'blueExperienceDiff','redGoldDiff',

       'redExperienceDiff','gameId'], axis=1)
#check number of diferent values

df_1.nunique()
#the probability of blue team winning is inversely correlated with red team, so the task is to analyse the Blue Team.

df_blue = df_1.drop(['redWardsPlaced','redWardsDestroyed',

       'redFirstBlood', 'redKills', 'redDeaths', 'redAssists',

       'redEliteMonsters', 'redDragons', 'redHeralds', 'redTowersDestroyed',

       'redTotalGold', 'redAvgLevel', 'redTotalExperience',

       'redTotalMinionsKilled', 'redTotalJungleMinionsKilled', 'redCSPerMin', 'redGoldPerMin'], axis=1)
x = df_blue['blueWins']

y = df_blue['blueTotalGold']

plt.bar(x, y)

plt.xticks(range(0,2))

plt.show()
# Total Gold vs Total minions killed, as we can see more minions killed doesn't equate more gold 

x1 = df_blue['blueTotalMinionsKilled']

y1 = df_blue['blueTotalGold']

plt.scatter(x1, y1)

plt.show()
# correlation between the variables,

# To avoid Multicollinearity, the independent variables must not be over 0,7 correlation or the regression output will 

        # be erroneous, for example: Blue Kills is highly correlated with Blue Assists, and one must be omitted from the model

corr = df_blue.corr()
plt.figure(figsize=(16,10))

sns.heatmap(corr, annot =  True)
# creating an input table with only the independent variables, ommiting the correlating ones,

# creating the target variable = Blue Wins

df_blue.columns
unscaled_inputs = df_blue.filter(['blueWardsPlaced', 'blueWardsDestroyed', 'blueFirstBlood',

       'blueKills', 'blueDeaths','blueEliteMonsters','blueHeralds', 'blueTowersDestroyed','blueAvgLevel','blueTotalMinionsKilled', 'blueTotalJungleMinionsKilled'], axis=1)

target = df_blue.filter(['blueWins'])
# import the libraries needed to create the Custom Scaler

# note that all of them are a part of the sklearn package

# moreover, one of them is actually the StandardScaler module, 

# so you can imagine that the Custom Scaler is build on it



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import StandardScaler



# create the Custom Scaler class



class CustomScaler(BaseEstimator,TransformerMixin): 

    

    # init or what information we need to declare a CustomScaler object

    # and what is calculated/declared as we do

    

    def __init__(self,columns,copy=True,with_mean=True,with_std=True):

        

        # scaler is nothing but a Standard Scaler object

        self.scaler = StandardScaler(copy,with_mean,with_std)

        # with some columns 'twist'

        self.columns = columns

        self.mean_ = None

        self.var_ = None

        

    

    # the fit method, which, again based on StandardScale

    

    def fit(self, X, y=None):

        self.scaler.fit(X[self.columns], y)

        self.mean_ = np.mean(X[self.columns])

        self.var_ = np.var(X[self.columns])

        return self

    

    # the transform method which does the actual scaling



    def transform(self, X, y=None, copy=None):

        

        # record the initial order of the columns

        init_col_order = X.columns

        

        # scale all features that you chose when creating the instance of the class

        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)

        

        # declare a variable containing all information that was not scaled

        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]

        

        # return a data frame which contains all scaled features and all 'not scaled' features

        # use the original order (that you recorded in the beginning)

        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]
# categorical columns to omit

columns_to_omit = ['blueFirstBlood']
# create the columns to scale, based on the columns to omit

# use list comprehension to iterate over the list

columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]
blue_scaler = CustomScaler(columns_to_scale)
blue_scaler.fit(unscaled_inputs)
scaled_inputs = blue_scaler.transform(unscaled_inputs)

scaled_inputs
from sklearn.model_selection import train_test_split
train_test_split(scaled_inputs, target)
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, target, train_size=0.8, random_state=20)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
reg = LogisticRegression()
reg.fit(x_train, y_train)
# Regression score

reg.score(x_train, y_train)
# The Intercept

intercept = reg.intercept_

intercept
# Creating a Summary Table to visualize the Variable and respective Coefficients and Odds Ratio

variables = unscaled_inputs.columns.values

variables
summary_table = pd.DataFrame(columns=['Variables'], data = variables)

summary_table['Coef'] = np.transpose(reg.coef_)

# add the intercept at index 0

summary_table.index = summary_table.index + 1

summary_table.loc[0] = ['Intercept', reg.intercept_[0]]

# calculate the Odds Ratio and add to the table

summary_table['Odds Ratio'] = np.exp(summary_table.Coef)
summary_table.sort_values(by=['Odds Ratio'], ascending=False)
import statsmodels.api as sm

x = sm.add_constant(x_train)

logit_model=sm.Logit(y_train,x)

result=logit_model.fit()

print(result.summary())
# testing the data is important to evalute the accuracy on a dataset that the model has never seen, to see if it's Overfitting

  # a test score 10% below the training reveals an overfitting

reg.score(x_test, y_test)
predicted_prob = reg.predict_proba(x_test)

predicted_prob[:,1]
df_blue['predicted'] = reg.predict_proba(scaled_inputs)[:,1]
df_blue