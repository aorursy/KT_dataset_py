import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import datetime as dt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RepeatedKFold 

from sklearn.tree import DecisionTreeRegressor  

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVR
data = pd.read_csv("../input/spanish-high-speed-rail-system-ticket-pricing/renfe.csv")
data.info()
# Travel ticket breakdowns for PONFERRADA - MADRID route 

data.groupby(['origin',"destination","start_date","end_date","train_type","train_class","fare"]).count().loc["PONFERRADA"].groupby(["train_type","train_class","fare"]).count().head(5)
#Finding the columns that have null values in their rows

data.columns[data.isnull().any()]
#Finding percentage of null values comparing to entire dataframe

total_row_count = data['insert_date'].count()

print("Total rows:",total_row_count)

Null_values=data[['price','train_class','fare']].isnull().sum() / total_row_count *100

Null_values.to_frame().rename(index={"price":"Price","train_class":"Train Class","fare":"Fare"},columns={0:"Percentage of null values"})
#Removing null values from dataframe

data = data.dropna()

data.columns[data.isnull().any()]

print("Total rows without null values:",data["insert_date"].count())
#Converting string date values into datetime values

data['start_date'] = pd.to_datetime(data['start_date'])

data['end_date'] = pd.to_datetime(data['end_date'])
#Converting distance values in hours into seconds

data['Duration'] = data['end_date'] - data['start_date']

data['Duration'] = data['Duration'].dt.total_seconds()

data['Date'] = data['start_date'].dt.date
# Calculating mean of price and duration based on the grouped indexes

data_one_train_per_day = data.groupby(["Date","train_class","train_type","fare","origin","destination"])['price','Duration'].mean().reset_index()

number_of_days_per_each_class = data_one_train_per_day.groupby(["train_class","train_type","fare","origin","destination"]).count().reset_index();number_of_days_per_each_class.head()
# Eliminating the rows whose Date values is smaller than 90 (below 90 sample is not sufficient) and removing 3 columns

row_deduction_data = number_of_days_per_each_class[number_of_days_per_each_class["Date"]>=90]

row_deduction_data = row_deduction_data.drop(columns=['Date','price','Duration'])
# Merging our deducted data  

data_one_train_per_day = pd.merge(data_one_train_per_day, row_deduction_data, on=["train_class","train_type","fare","origin","destination"])
train_categories = data_one_train_per_day.groupby(["train_class","train_type","fare","origin","destination"]).count().reset_index() 

data_one_train_per_day['Ori_Des'] = data_one_train_per_day['origin'] + '_' + data_one_train_per_day['destination']
#Showing that the price is not stationary but it is moving - Just for one unique combination

visualiz_data = data[(data['origin']=="BARCELONA") & (data['destination']=="MADRID") & (data['train_type']=="AVE")

& (data['train_class']=="Turista")

& (data['fare']=="Flexible")]

sns.lineplot(x="Date", y="price", data=visualiz_data)
#Number of trains each day between cities

x_values = np.arange(len(data_one_train_per_day.Ori_Des.value_counts().values))

y_values = data_one_train_per_day.Ori_Des.value_counts().values

city_names = data_one_train_per_day.Ori_Des.value_counts().index

df = pd.DataFrame({'Number of train journeys per route': y_values},index=city_names)

df.plot.barh()
main_data_without_outlier = pd.DataFrame()

for i in range(len(train_categories)):

    train_class=train_categories.iloc[i].train_class

    train_type=train_categories.iloc[i].train_type

    fare=train_categories.iloc[i].fare

    origin=train_categories.iloc[i].origin

    destination=train_categories.iloc[i].destination

    

    outlier_detection_data = data_one_train_per_day[(data_one_train_per_day['train_class']==train_class) &

                          (data_one_train_per_day['train_type']==train_type) &

                          (data_one_train_per_day['fare']==fare) &

                          (data_one_train_per_day['origin']==origin) &

                          (data_one_train_per_day['destination']==destination)].copy()

    

    mean = outlier_detection_data['price'].mean()

    std = outlier_detection_data['price'].std()

    outlier_detection_data['IsOutlier'] = np.where((outlier_detection_data['price'] < (mean + 2*std)) 

                                                   & (outlier_detection_data['price'] > (mean - 2*std))

                                                   , 0, 1)

    

    outlier_detection_data['price'] = np.where(outlier_detection_data['IsOutlier']==1 , mean , outlier_detection_data['price'])

    main_data_without_outlier = main_data_without_outlier.append(outlier_detection_data)
moving_average_parameter = 5

main_data_moving_ave = pd.DataFrame()



for i in range(len(train_categories)):

    train_class=train_categories.iloc[i].train_class

    train_type=train_categories.iloc[i].train_type

    fare=train_categories.iloc[i].fare

    origin=train_categories.iloc[i].origin

    destination=train_categories.iloc[i].destination



    ma_data = main_data_without_outlier[(main_data_without_outlier['train_class']==train_class) &

                          (main_data_without_outlier['train_type']==train_type) &

                          (main_data_without_outlier['fare']==fare) &

                          (main_data_without_outlier['origin']==origin) &

                          (main_data_without_outlier['destination']==destination)].copy()

    

    ma_data = ma_data.sort_values('Date')

    ma_data['MovingAverage'] = ma_data.price.rolling(window=moving_average_parameter).sum()

    ma_data['MovingAverage'] = ma_data['MovingAverage'] - ma_data['price']

    ma_data['MovingAverage'] = ma_data['MovingAverage'] / (moving_average_parameter-1)

    main_data_moving_ave = main_data_moving_ave.append(ma_data)
#Moving average vs actual price over time

ma_data_graph = main_data_moving_ave[main_data_moving_ave['MovingAverage'].notnull()]

ma_data_graph = ma_data_graph[(ma_data_graph['origin']=="BARCELONA") & (ma_data_graph['destination']=="MADRID") & (ma_data_graph['train_type']=="AVE")

& (ma_data_graph['train_class']=="Turista")

& (ma_data_graph['fare']=="Flexible")]



plt.plot( 'Date', 'price', data=ma_data_graph, markerfacecolor='blue', color='skyblue', linewidth=1)

plt.plot( 'Date', 'MovingAverage', data=ma_data_graph, color='olive', linewidth=1)
#Finding mean absolute percentage error for predicted output with moving average model

def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

ma_actual_values = main_data_moving_ave[main_data_moving_ave['MovingAverage'].notnull()].price

ma_actual_pred = main_data_moving_ave[main_data_moving_ave['MovingAverage'].notnull()].MovingAverage

mean_absolute_percentage_error(ma_actual_values,ma_actual_pred)
#Make the categorical values binary/numerical for linear regression

main_data_lm = main_data_moving_ave.copy()

dummy_columns = pd.get_dummies(main_data_moving_ave[['fare','train_type','train_class','origin','destination']])

main_data_lm = pd.concat([main_data_lm, dummy_columns], axis=1, ignore_index=False)

main_data_lm['Duration_for_model'] = main_data_lm['Duration']
X = main_data_lm.iloc[:,11:].values

Y = main_data_lm.iloc[:,6].values

# TRAIN/TEST SPLIT

X_train,X_test,Y_train,Y_test,duration_train,duration_test = train_test_split(X,Y,main_data_moving_ave["Duration"],test_size=0.2,random_state=5)

# MODEL FITTING

model = LinearRegression()

model = model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

#R-squared

test_score = model.score(X_test, Y_test);test_score
#Finding mean absolute percentage error for predicted output with linear regression model

errors = abs(y_pred - Y_test)

np.mean(errors)
main_data_lm = main_data_moving_ave.copy()

dummy_columns = pd.get_dummies(main_data_moving_ave[['fare','train_type','train_class','origin','destination']])

main_data_lm = pd.concat([main_data_lm, dummy_columns], axis=1, ignore_index=False)

main_data_lm['Duration_for_model'] = main_data_lm['Duration']
X = main_data_lm.iloc[:,11:].values

Y = main_data_lm.iloc[:,6].values

# TRAIN/TEST SPLIT

X_train,X_test,Y_train,Y_test,duration_train,duration_test = train_test_split(X,Y,main_data_moving_ave["Duration"],test_size=0.2,random_state=5)

# MODEL FITTING

regressor = SVR()

regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)



errors = abs(y_pred - Y_test)

np.mean(errors)
main_data_rf = main_data_moving_ave.copy()

dummy_columns = pd.get_dummies(main_data_moving_ave[['fare','train_type','train_class','origin','destination']])

main_data_rf = pd.concat([main_data_rf, dummy_columns], axis=1, ignore_index=False)



X = main_data_rf.iloc[:,11:].values

Y = main_data_rf.iloc[:,6].values



# BASIC TRAIN/TEST SPLIT

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=5)

dt=DecisionTreeRegressor()

dt.fit(X_train, Y_train) 

predictions= dt.predict(X_test)



errors = abs(predictions - Y_test)

np.mean(errors)

#test_score = dt.score(X_test, Y_test);test_score 
main_data_rf = main_data_moving_ave.copy()

dummy_columns = pd.get_dummies(main_data_moving_ave[['fare','train_type','train_class','origin','destination']])

main_data_rf = pd.concat([main_data_rf, dummy_columns], axis=1, ignore_index=False)



X = main_data_rf.iloc[:,11:].values

Y = main_data_rf.iloc[:,6].values



# BASIC TRAIN/TEST SPLIT

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=5)



# MODEL FITTING

rf= RandomForestRegressor(n_estimators = 1000, random_state = 42)

rf.fit(X_train, Y_train)

predictions = rf.predict(X_test)



# Finding mean absolute percentage erro

errors = abs(predictions - Y_test)

np.mean(errors)