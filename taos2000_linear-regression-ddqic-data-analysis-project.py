import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from datetime import datetime
from sklearn.model_selection import cross_val_score,StratifiedKFold, KFold
from sklearn.metrics import accuracy_score,mean_absolute_error
#import data
df_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
#df_confirmed
df_confirmed_Canada = df_confirmed.loc[df_confirmed['Country/Region'] == 'Canada'].reset_index(drop = True)
#df_confirmed_Canada
Canada_disease_trend = pd.DataFrame(np.zeros((1, df_confirmed_Canada.shape[1])), columns = df_confirmed_Canada.columns)
#Canada_disease_trend
df_confirmed_Canada = df_confirmed_Canada.append(df_confirmed_Canada.sum(axis=0).to_frame().transpose(), ignore_index = True)
df_confirmed_Canada
df_confirmed_Canada['Province/State'][len(df_confirmed_Canada['Province/State'])-1] = 'total'
#df_confirmed_Canada
df_confirmed_Canada.drop(['Country/Region', 'Lat', 'Long'], axis = 1, inplace = True)
df_confirmed_Canada
df_covid19 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv")
#df_covid19
df_covid19_Canada = df_covid19.loc[df_covid19['Country_Region'] == 'Canada'].reset_index(drop = True)
df_covid19_Canada
province_list = df_confirmed_Canada['Province/State'].tolist()
#province_list
def date_converter(date_list):
    #input: list of date, each one must between 1 day: 1999/09/09, 1999/09/10
    #output: 0, 1, 2, ... or 55,56,..., choose 1/22/20 as 0.
    #https://www.google.com/search?q=python+calculate+time+difference&rlz=1C1CHBF_enCA882CA882&oq=python+calculate+time+difference&aqs=chrome..69i57j0l7.4637j0j7&sourceid=chrome&ie=UTF-8
    #print(date_list)
    date_format = "%m/%d/%Y"
    start_date = datetime.strptime(date_list[0], date_format)
    i = 0 
    while i < len(date_list):
        date = datetime.strptime(date_list[i], date_format)
        date_list[i] = (date - start_date).days
        i+=1
    return date_list
def linear_regression(region_name, data, estimate_date):
    #this fuction apply linear regression to our datasets
    #input: region_name: province name or total
    #data: our train dataset
    #estimate_date: number of days to estimate
    #output: solution of approximation for next X days
    
    #choose province and get data
    row = data[data['Province/State'] == region_name]
    row_column = row.columns.tolist()[1:]
    row = row.values.tolist()[0][1:]
    
    #change format of date
    i = 0 
    while i < len(row_column):
        row_column[i] = row_column[i][:-2] + '20' + row_column[i][-2:]
        i+=1
    row_column = date_converter(row_column)
    
    #start linear regression
    #print(row)
    #print(row_column)
    X = pd.DataFrame(row_column)
    y = pd.Series(row)
    print('********')
    #print(X,y)
    print(X.values.tolist())
    X_list = X.values.tolist()
    k = []
    for i in X_list:
        k.append(i[0])
    print(k)
    df_merge_col = pd.DataFrame({'Date': k,
                           'Number of cases': y.tolist()})
    df_merge_col.to_csv("../../kaggle/working/Confirm_datasets_clean.csv", index=False)
    #model
    #try model
    lr =  LinearRegression(copy_X = False, n_jobs = -1)
    lr.fit(X,y)
    submit = lr.predict(estimate_date)
    plt.scatter(estimate_date[0].tolist(), submit,  color='black')
    plt.plot(estimate_date[0].tolist(), submit, color='blue', linewidth=3)
    plt.show()
    altogether_X = pd.concat([X,estimate_date], axis = 0 ) 
    altogether_y = pd.concat([y,pd.Series(submit)], axis = 0 )
    plt.scatter(altogether_X, altogether_y,  color='black')
    plt.plot(altogether_X, altogether_y, color='blue', linewidth=3)
    plt.show()
    
    
    
    #StratifiedKFold
    N = 5
    confirm = pd.DataFrame(np.zeros((len(test_data), N)), columns=['Fold_{}'.format(i) for i in range(1, N + 1)])
    auc_score_total = 0
    num = 0
    skf = StratifiedKFold(n_splits=N, random_state=5)
    for train_index, test_index in skf.split(X, y):
        num +=1
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        lr.fit(X_train, y_train)
        
        confirm.loc[:, 'Fold_{}'.format(num)] = lr.predict(estimate_date)
        prediction = lr.predict(X_test)
        #auc score
        auc_score = mean_absolute_error(y_test, prediction)
        auc_score_total += auc_score
        print("MAE score: ", auc_score)
    print(auc_score_total/5)
    auc_score_total = 0
    
    #submit
    submit = confirm.sum(axis=1) / N
    print(estimate_date[0])
    submission = pd.DataFrame({'Date': estimate_date[0].tolist(),
                           'Number of cases': submit})
    submission.to_csv("../../kaggle/working/Confirm_prediction.csv", index=False)
    print(submission.head())
    
    #plots
    plt.scatter(estimate_date[0].tolist(), submit,  color='black')
    plt.plot(estimate_date[0].tolist(), submit, color='blue', linewidth=3)
    plt.show()
    
    altogether_X = pd.concat([X,estimate_date], axis = 0 ) 
    altogether_y = pd.concat([y,submit], axis = 0 )
    plt.scatter(altogether_X, altogether_y,  color='black')
    plt.plot(altogether_X, altogether_y, color='blue', linewidth=3)
    plt.show()
#user_input = input('choose country')
#user's choice
#now default: choose last one, total
user_input = province_list[-1]
test_data = []
#0 means 1/22/2020, 161 means 6/30/2020, 223 means 8/31/2020
for i in range(161,223):
    test_data.append(i)
test_data = pd.DataFrame(test_data)
linear_regression(user_input, df_confirmed_Canada,test_data)
