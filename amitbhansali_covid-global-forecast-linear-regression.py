import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

import time

from datetime import datetime

from scipy import integrate, optimize

import warnings

warnings.filterwarnings('ignore')



# ML libraries

import lightgbm as lgb

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn import linear_model

from sklearn.metrics import mean_squared_error
submission_example = pd.read_csv("../input/covid19-global-forecasting-week-3/submission.csv")

test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")

train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")

display(train.head(5))

display(test.head(5))



print("Train Data:")

print("Dates go from day", max(train['Date']), "to day", min(train['Date']), ", a total of", train['Date'].nunique(), "days")

print("Test Data:")

print("Dates go from day", max(test['Date']), "to day", min(test['Date']), ", a total of", test['Date'].nunique(), "days")

total_data = pd.concat([train,test],axis=0,sort=False) # join train and test

total_data.isna().sum() # verifying na
confirmed_total_date_India = train[train['Country_Region']=='India'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_India = train[train['Country_Region']=='India'].groupby(['Date']).agg({'Fatalities':['sum']})



plt.figure(figsize=(17,10))

plt.subplot(2, 2, 1)

confirmed_total_date_India.plot(ax=plt.gca(), title='India Confirmed')

plt.ylabel("Confirmed infection cases", size=13)



plt.subplot(2, 2, 2)

fatalities_total_date_India.plot(ax=plt.gca(), title='India Fatalities')

plt.ylabel("Fatalities cases", size=13)









confirmed_total_date_UK = train[train['Country_Region']=='United Kingdom'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_UK = train[train['Country_Region']=='United Kingdom'].groupby(['Date']).agg({'Fatalities':['sum']})



plt.figure(figsize=(17,10))

plt.subplot(2, 2, 1)

confirmed_total_date_UK.plot(ax=plt.gca(), title='UK Confirmed')

plt.ylabel("Confirmed infection cases", size=13)



plt.subplot(2, 2, 2)

fatalities_total_date_UK.plot(ax=plt.gca(), title='UK Fatalities')

plt.ylabel("Fatalities cases", size=13)

# Load countries data file

world_population = pd.read_csv("/kaggle/input/population-by-country-2020/population_by_country_2020.csv")



# Select desired columns and rename some of them

world_population = world_population[['Country (or dependency)', 'Density (P/Km²)']]

world_population.columns = ['Country (or dependency)',  'Density', ]



# Replace United States by US

world_population.loc[world_population['Country (or dependency)']=='United States', 'Country (or dependency)'] = 'US'



print("Cleaned country details dataset")

#display(world_population)



# Now join the dataset to our previous DataFrame and clean missings (not match in left join)- label encode cities

print("Joined dataset")

total_data = total_data.merge(world_population, left_on='Country_Region', right_on='Country (or dependency)', how='left')

#train['Density'] = train[[ 'Density']].fillna(0)

#display(train)



print("Encoded dataset")

# Label encode countries and provinces. Save dictionary for exploration purposes

# all_data.drop('Country (or dependency)', inplace=True, axis=1)

le = preprocessing.LabelEncoder()

total_data['Country_Code'] = le.fit_transform(total_data['Country_Region'])



number_c = total_data['Country_Code']

countries = le.inverse_transform(total_data['Country_Code'])

country_dict = dict(zip(countries, number_c)) 



# Create date columns

total_data['Date'] = pd.to_datetime(total_data['Date'])

le = preprocessing.LabelEncoder()

total_data['Day_num'] = le.fit_transform(total_data.Date)



# Fill null values given that we merged train-test datasets

total_data['ConfirmedCases'].fillna(0, inplace=True)

total_data['Fatalities'].fillna(0, inplace=True)

total_data['Id'].fillna(-1, inplace=True)

total_data[['ConfirmedCases', 'Fatalities']] = total_data[['ConfirmedCases', 'Fatalities']].astype('float64')

#total_data['FatalitiesDensity'] = total_data.Fatalities * total_data.Density

total_data[['ConfirmedCases', 'Fatalities']] = total_data[['ConfirmedCases', 'Fatalities']].apply(lambda x: np.log1p(x))



train = total_data[total_data.Id != -1]

train = train[train.Day_num >= 39]

test =  total_data[total_data.Id == -1]  



display(train)


# Linear regression model

def lin_reg(X_train, Y_train, X_test):

    # Create linear regression object

    regr = linear_model.LinearRegression()



    # Train the model using the training sets

    regr.fit(X_train, Y_train)



    # Make predictions using the testing set

    y_pred = regr.predict(X_test)

    

    return regr, y_pred





# Submission function

def get_submission(df, target1, target2):

    

    prediction_1 = df[target1]

    prediction_2 = df[target2]



    # Submit predictions

    prediction_1 = [int(item) for item in list(map(round, prediction_1))]

    prediction_2 = [int(item) for item in list(map(round, prediction_2))]

    

    submission = pd.DataFrame({

        "ForecastId": df['ForecastId'].astype('int32'), 

        "ConfirmedCases": prediction_1, 

        "Fatalities": prediction_2

    })

    submission.to_csv('submission.csv', index=False)
def plot_linreg_basic_country(train,test, country_name):

      

    train = train[train['Country_Code']==country_dict[country_name]]

    test = test[test['Country_Code']==country_dict[country_name]]

    

    X_train_1 = train[['Day_num']]

    Y_train_1 = train[['Fatalities']]

    X_test_1 = test[['Day_num']]

    

    X_train_2 = train[['Day_num']]

    Y_train_2 = train[['Fatalities']]

    X_test_2 = test[['Day_num']]

 

    model, pred = lin_reg(X_train_1, Y_train_1, X_test_1)

    

    # Plot results

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))



    ax1.plot(train[['Day_num']], np.expm1(train[['Fatalities']]))

    ax1.plot(test[['Day_num']], np.expm1(pred))

    

    ax1.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

    ax1.set_ylabel("Confirmed Cases")

    plt.suptitle(("ConfirmedCases predictions based on Log-Lineal Regression for "+country_name))
# Filter Spain, run the Linear Regression workflow

country_name = "India"

plot_linreg_basic_country(train, test, country_name)
def linreg_basic_all_countries(train, test):

    



    # Set the dataframe where we will update the predictions

    data_pred = test[['Country_Code', 'Province_State', 'Day_num', 'ForecastId', 'Date', 'Country_Region']]

    data_pred['Predicted_ConfirmedCases'] = [0]*len(data_pred)

    data_pred['Predicted_Fatalities'] = [0]*len(data_pred)

  

    print("Currently running Linear Regression for all countries")



    # Main loop for countries

    for country_name in data_pred['Country_Code'].unique():

       



        train_data = train[train['Country_Code']== country_name]

        test_data = test[test['Country_Code']== country_name]

  

        X_train_1 = train_data[['Day_num']]

        Y_train_1 = train_data[['ConfirmedCases']]

        X_test_1 = test_data[['Day_num']]

        

        X_train_2 = train_data[['Day_num']]

        Y_train_2 = train_data[['Fatalities']]

        X_test_2 = test_data[['Day_num']]

    

        model_1, pred_1 = lin_reg(X_train_1, Y_train_1, X_test_1)

        model_2, pred_2 = lin_reg(X_train_2, Y_train_2, X_test_2)



        data_pred.loc[(data_pred['Country_Code']==country_name), 'Predicted_ConfirmedCases'] = pred_1

        data_pred.loc[(data_pred['Country_Code']==country_name), 'Predicted_Fatalities'] = pred_2



    # Apply exponential transf. and clean potential infinites due to final numerical precision

    data_pred[['Predicted_ConfirmedCases', 'Predicted_Fatalities']] = data_pred[['Predicted_ConfirmedCases', 'Predicted_Fatalities']].apply(lambda x: np.expm1(x))

    data_pred.replace([np.inf, -np.inf], 0, inplace=True) 

    

    return data_pred







data_pred = linreg_basic_all_countries(train, test)

display (data_pred)

get_submission(data_pred, 'Predicted_ConfirmedCases', 'Predicted_Fatalities')

import plotly.express as px



data_pred_plot = data_pred[(data_pred['Day_num']>73)& (data_pred['Day_num']<100) & (data_pred['Country_Region'] == 'India') ]



#data_pred_plot = data_pred[(data_pred['Day_num']>73)& (data_pred['Day_num']<80)  ]



temp = data_pred_plot.groupby(['Date', 'Country_Region'])['Predicted_ConfirmedCases'].sum().reset_index().sort_values('Predicted_ConfirmedCases', ascending=False)



fig = px.line(temp, x="Date", y="Predicted_ConfirmedCases", color='Country_Region', title='Cases Spread', height=600)

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()



#================================



temp = data_pred_plot.groupby(['Date', 'Country_Region'])['Predicted_Fatalities'].sum().reset_index().sort_values('Predicted_Fatalities', ascending=False)



fig = px.line(temp, x="Date", y="Predicted_Fatalities", color='Country_Region', title='Deaths', height=600)

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()