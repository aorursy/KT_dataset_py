#import necessary modules

import pandas as pd



life_data = pd.read_csv("../input/life-expectancy-who/Life Expectancy Data.csv")

life_data.head()          #View top 5 rows
#Lets look at the columns

life_data.columns
#Remove Leading spaces : use lstrip() method

life_data.columns =  [names.lstrip() for names in life_data.columns]



#Remove Trailing spaces : use lstrip() method

life_data.columns =  [names.rstrip() for names in life_data.columns]



#Capliatize column name, making them consistent



life_data.columns = [names.capitalize() for names in life_data.columns]
#Lets view our Columns

life_data.columns
life_data.info()
life_data.rename(columns={'thinness_1-19_years':'thinness_10-19_years'}, inplace=True)
life_data.describe()
#import 

%matplotlib inline    



import matplotlib.pyplot as plt

#creating histogram for each numeric attribute

life_data.hist(bins = 50,

               figsize = (20,15))

plt.show()
life_data["Country"].value_counts()
life_data["Status"].value_counts()
#Copy the test data

life_copy = life_data.copy()
plt.figure(figsize=(15,10))



for i,column in enumerate(['Adult mortality', 'Infant deaths', 'Bmi', 'Under-five deaths', 'Gdp', 'Population'],start=1):

    plt.subplot(2, 3,i)

    life_copy.boxplot(column)
#import

import numpy as np
#Adult mortality rates lower than the 5th percentile

mortality_less_5_per = np.percentile(life_copy["Adult mortality"].dropna(),5) 

life_copy["Adult mortality"] = life_copy.apply(lambda x: np.nan if x["Adult mortality"] < mortality_less_5_per else x["Adult mortality"], axis=1)

#Remove Infant deaths of 0

life_copy["Infant deaths"] = life_copy["Infant deaths"].replace(0,np.nan)
#Remove the invalid BMI

life_copy["Bmi"] =life_copy.apply(lambda x : np.nan if (x["Bmi"] <10 or x["Bmi"] >50) else x["Bmi"],axis =1)
#Remove Under five deaths

life_copy["Under-five deaths"] =life_copy["Under-five deaths"].replace(0,np.nan)
def count_null(df):

    df_cols = list(df.columns)

    cols_total_count = len(df_cols)

    cols_count = 0

    

    for loc,col in enumerate(df_cols):

        null_count = df[col].isnull().sum()                                  #total null values

        total_count = df[col].isnull().count()                               #Total rows

        percent_null = round(null_count/total_count*100, 2)                  #Percentage null 

      

        if null_count > 0:

            cols_count += 1

            print('[iloc = {}] {} has {} null values: {}% null'.format(loc, col, null_count, percent_null))

    

    cols_percent_null = round(cols_count/cols_total_count*100, 2)

    print('Out of {} total columns, {} contain null values; {}% columns contain null values.'.format(cols_total_count, cols_count, cols_percent_null))
count_null(life_copy)
life_copy.drop(columns='Bmi', inplace=True)
imputed_data = []



for year in list(life_copy.Year.unique()):

    year_data = life_copy[life_copy.Year == year].copy()

    

    for col in list(year_data.columns)[3:]:

        year_data[col] = year_data[col].fillna(year_data[col].dropna().mean()).copy()



    imputed_data.append(year_data)

df = pd.concat(imputed_data).copy()
count_null(df)
life_numeric_data = df.drop(columns=["Year","Country","Status"])
%matplotlib inline



def plot_numeric_data(data):

    i = 0

    for col in data.columns:

        i += 1

        plt.subplot(9, 4, i)

        plt.boxplot(data[col])

        plt.title('{} boxplot'.format(col))

        i += 1

        plt.subplot(9, 4, i)

        plt.hist(data[col])

        plt.title('{} histogram'.format(col))

        

    plt.show()

plt.figure(figsize=(15,40))

plot_numeric_data(life_numeric_data)
def outlier_count(col, data=df):

    

    print("\n"+15*'-' + col + 15*'-'+"\n")

    

    q75, q25 = np.percentile(data[col], [75, 25])

    iqr = q75 - q25

    min_val = q25 - (iqr*1.5)

    max_val = q75 + (iqr*1.5)

    outlier_count = len(np.where((data[col] > max_val) | (data[col] < min_val))[0])

    outlier_percent = round(outlier_count/len(data[col])*100, 2)

    print('Number of outliers: {}'.format(outlier_count))

    print('Percent of data that is outlier: {}%'.format(outlier_percent))
cont_vars = list(life_numeric_data)

for col in cont_vars:

    outlier_count(col)
from scipy.stats.mstats import winsorize



def test_wins(col, lower_limit=0, upper_limit=0, show_plot=True):

    wins_data = winsorize(df[col], limits=(lower_limit, upper_limit))

    wins_dict[col] = wins_data

    if show_plot == True:

        plt.figure(figsize=(15,5))

        plt.subplot(121)

        plt.boxplot(df[col])

        plt.title('original {}'.format(col))

        plt.subplot(122)

        plt.boxplot(wins_data)

        plt.title('wins=({},{}) {}'.format(lower_limit, upper_limit, col))

        plt.show()
wins_dict = {}

test_wins(cont_vars[0], lower_limit=.01, show_plot=True)

test_wins(cont_vars[1], upper_limit=.04, show_plot=False)

test_wins(cont_vars[2], upper_limit=.05, show_plot=False)

test_wins(cont_vars[3], upper_limit=.0025, show_plot=False)

test_wins(cont_vars[4], upper_limit=.135, show_plot=False)

test_wins(cont_vars[5], lower_limit=.1, show_plot=False)

test_wins(cont_vars[6], upper_limit=.19, show_plot=False)

test_wins(cont_vars[7], upper_limit=.05, show_plot=False)

test_wins(cont_vars[8], lower_limit=.1, show_plot=False)

test_wins(cont_vars[9], upper_limit=.02, show_plot=False)

test_wins(cont_vars[10], lower_limit=.105, show_plot=False)

test_wins(cont_vars[11], upper_limit=.185, show_plot=False)

test_wins(cont_vars[12], upper_limit=.105, show_plot=False)

test_wins(cont_vars[13], upper_limit=.07, show_plot=False)

test_wins(cont_vars[14], upper_limit=.035, show_plot=False)

test_wins(cont_vars[15], upper_limit=.035, show_plot=False)

test_wins(cont_vars[16], lower_limit=.05, show_plot=False)

test_wins(cont_vars[17], lower_limit=.025, upper_limit=.005, show_plot=False)
plt.figure(figsize=(15,5))



for i, col in enumerate(cont_vars, 1):

    plt.subplot(2, 9, i)

    plt.boxplot(wins_dict[col])



    plt.tight_layout()

plt.show()
#A new dataframe with the winsorized data 

wins_df = df.iloc[:, 0:3]

for col in cont_vars:

    wins_df[col] = wins_dict[col]
dataset = wins_df.drop(columns= ["Year","Country"],axis = True)
#Dealing with Categorical data
status = pd.get_dummies(dataset.Status)

dataset = pd.concat([dataset, status], axis = 1)

dataset= dataset.drop(['Status'], axis=1)
dataset.columns
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(dataset.drop(columns = ["Life expectancy"],axis = 1),

                                                 dataset["Life expectancy"],

                                                 test_size = 0.2,

                                                 random_state = 42)
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()

X_train_scaled = std_scaler.fit_transform(X_train)

#import necessary modules

from sklearn.linear_model import LinearRegression



linear_regressor = LinearRegression()



linear_regressor.fit(X_train_scaled,y_train)

from sklearn.metrics import r2_score



#Make predictions

y_pred = linear_regressor.predict(X_train_scaled)



#Calculating RMSE

linear_r2_score = r2_score(y_train,y_pred)



print(linear_r2_score)
from sklearn.model_selection import cross_val_score

from sklearn.metrics import make_scorer

scoring = make_scorer(r2_score)



linear_scores = cross_val_score(linear_regressor,X_train_scaled,y_train,

                       scoring = scoring,cv=10)

linear_scores
#import necessary modules

from sklearn.tree import DecisionTreeRegressor



tree_reg = DecisionTreeRegressor()



tree_reg.fit(X_train_scaled,y_train)

#Make predictions

y_pred = tree_reg.predict(X_train_scaled)



#Calculating RMSE

tree_r2_score = r2_score(y_train,y_pred)



print(tree_r2_score)
from sklearn.metrics import make_scorer

scoring = make_scorer(r2_score)

scores = cross_val_score(tree_reg,X_train_scaled,y_train,

                       scoring = scoring,cv=10)

scores
#RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import make_scorer

socre = make_scorer("r2_score")

forest_reg = RandomForestRegressor()



forest_reg.fit(X_train_scaled,y_train)



#Make predictions

y_pred = forest_reg.predict(X_train_scaled)

#Calculating RMSE

forest_r2_score = r2_score(y_train,y_pred)



print(forest_r2_score)
forest_score = cross_val_score(forest_reg, X_train_scaled,y_train,

                              scoring=scoring,cv=10)



forest_score
X_test_scaled = std_scaler.fit_transform(X_test)

y_pred = forest_reg.predict(X_test_scaled)



#Calculating RMSE

tree_r2_score = r2_score(y_test,y_pred)



print("R^2 score: %.2f"%tree_r2_score)

