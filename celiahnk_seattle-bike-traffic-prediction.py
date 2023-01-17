import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns
traffic_data = pd.read_csv('../input/Fremont_Bridge_Bicycle_Counter.csv', index_col='Date', parse_dates=True)

weather_data = pd.read_csv('../input/Seattle_weather_daily.csv', index_col='DATE', parse_dates=True)
traffic_data.head()
weather_data.head(2)
weather_data.shape
weather_data.describe()
traffic_data.shape
traffic_data.describe()
traffic_data.head()
daily = (

    traffic_data

    .resample('d')

    .sum()

    .loc[:, ['Fremont Bridge Total']]

    .rename(

        columns={

            'Fremont Bridge Total': 'Total'

        }

    )

)
daily.head()
monthly = (

    traffic_data

    .resample('m')

    .sum()

).plot(figsize = (15,5))
by_time = (

    traffic_data

    .groupby(traffic_data.index.time)

    .mean()

)



hourly_ticks = 4 * 60 * 60 * np.arange(6)



by_time.plot(xticks=hourly_ticks, style=[':', '--', '-']);
by_weekday = (

    traffic_data

    .groupby(traffic_data.index.dayofweek)

    .mean()

)

by_weekday.index = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']

by_weekday.plot(style=[':', '--', '-']);
### let's assign those days of the week labels in the dataset itself 



daily = (

    daily

    .assign(

        day_of_week=lambda _df: _df.index.dayofweek

    )

    .pipe(pd.get_dummies, columns=['day_of_week'])

    .rename(

        columns={

            'day_of_week_0': 'Mon',

            'day_of_week_1': 'Tue',

            'day_of_week_2': 'Wed',

            'day_of_week_3': 'Thu',

            'day_of_week_4': 'Fri',

            'day_of_week_5': 'Sat',

            'day_of_week_6': 'Sun'

        }

    )

)

daily.head()
from pandas.tseries.holiday import USFederalHolidayCalendar

cal = USFederalHolidayCalendar()

holidays = cal.holidays('2012', '2020')

daily = daily.join(pd.Series(1, index=holidays, name='holiday'))

daily['holiday'].fillna(0, inplace=True)
(

    daily

    .loc[daily.holiday == 1]

    .reset_index()

    .sort_values(by= "Date")

    .tail(10)

)
weather_data.describe()
weather_data.tail(1)
# Temperatures are in 1/10 deg C; convert to C

weather_data['Temp (C)'] = (weather_data['TAVG'] - 32)/(9/5)



# We can create a new binomial metric 'Dry day' as day with or without precipitation

weather_data['dry day'] = (weather_data['PRCP'] == 0).astype(int)



# Join the 2 datasets

daily = daily.join(weather_data[['PRCP', 'Temp (C)', 'dry day']])



# Drop any rows with null values

daily.dropna(axis=0, how='any', inplace=True)
daily.describe()
sns.boxplot(x='dry day', y='Total', data=daily, hue='dry day')
sns.jointplot(daily['Temp (C)'], daily['Total'], kind='kde')
daily.head()
column_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'holiday','dry day', 'Temp (C)']

X = daily[column_names] # define the independant varibles 

y = daily['Total'] # define the target value, the dependant variable



from sklearn.model_selection  import train_test_split   

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=1)  ## split the dataset in train and test sub-sets



from sklearn.linear_model import LinearRegression # 1. choose model class

model = LinearRegression(fit_intercept=False)     # 2. instantiate model

model.fit(Xtrain, ytrain)                         # 3. fit model to train data

y_model = model.predict(Xtest)                    # 4. predict on new test data



from sklearn.metrics import r2_score

r2_score(ytest, y_model)  ## check score of the model chosen
from sklearn.model_selection  import cross_validate

cv = cross_validate(model, X, y, cv=10, return_train_score=True)

cv_df = pd.DataFrame({"train_score": cv["train_score"], "test_score": cv["test_score"]})

cv_df
print(cv_df["train_score"].mean(),cv_df["test_score"].mean())
daily['predicted'] = model.predict(X) # add the predicted number of riders in the orginial data set

daily[['Total', 'predicted']].plot(figsize =(20,10), legend=True);
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])

coeff_df
daily["error"]= (daily.predicted - daily.Total).astype(int)

daily["error_abs"]= (daily.predicted - daily.Total).abs().astype(int)



monthly_error = daily.resample('m').mean().reset_index()

monthly_error.plot(x="Date", y=["Total","predicted"], figsize=(15,5))

year_error = daily.resample('y').mean().reset_index()

year_error.plot(x="Date", y=["error_abs"], figsize=(15,5))
monthly_error = (

    daily

    .resample('m')

    .mean()

    .reset_index()

)

monthly_error.plot(x='Date',y='error', figsize=(15,5))
(

    monthly_error

    .sort_values(by ='error')

    .tail(5)

)
daily =(

    daily

    .assign(

        month_num=lambda _df: _df.index.month, # get the month number from date

        Xmas_period=lambda _df: _df['month_num'] == 12,

        Summer_period=lambda _df: _df['month_num'].isin([7, 8])

    )

    .drop(columns=['month_num'])

)
column_names_fe = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'holiday','dry day', 'Temp (C)','Xmas_period','Summer_period']

X_fe = daily[column_names_fe] # define the independant varibles 

y_fe = daily['Total'] # define the target value, the dependant variable



Xtrain, Xtest, ytrain, ytest = train_test_split(X_fe, y_fe, random_state=1)  ## split the dataset in train and test sub-sets



model = LinearRegression(fit_intercept=False)     # 2. instantiate model

model.fit(Xtrain, ytrain)                         # 3. fit model to train data

y_model = model.predict(Xtest)



r2_score(ytest, y_model)  ## check score of the model chosen
cv = cross_validate(model, X_fe, y_fe, cv=10, return_train_score=True)

cv_df = pd.DataFrame({"train_score": cv["train_score"], "test_score": cv["test_score"]})

cv_df
print(cv_df["train_score"].mean(),cv_df["test_score"].mean())
coeff_df = pd.DataFrame(model.coef_, X_fe.columns, columns=['Coefficient'])

coeff_df
daily['predicted_fe'] = model.predict(X_fe) # incoporated the predicted number of riders in the orginial data set



daily["error_fe"]= (daily.predicted_fe - daily.Total).astype(int)

daily["error_fe_abs"]= (daily.predicted - daily.Total).abs().astype(int)



monthly_error = daily.resample('m').mean().reset_index()



monthly_error.plot(x="Date", y=["Total","predicted_fe","predicted"], figsize=(15,5))
year_error = daily.resample('y').mean().reset_index()

year_error.plot(x="Date", y=["error_abs","error_fe_abs"], figsize=(15,5))
#Dataset import

population_data = pd.read_csv("../input/Seattle_yearly_pop.csv", index_col='Date', parse_dates=True)



# Join the 2 datasets

daily = daily.join(population_data[['population']])



# Drop any rows with null values

daily.dropna(axis=0, how='any', inplace=True)
column_names_fe = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'holiday','dry day', 'Temp (C)','Xmas_period','Summer_period','population']

X_fe = daily[column_names_fe] # define the independant varibles 

y_fe = daily['Total'] # define the target value, the dependant variable



Xtrain, Xtest, ytrain, ytest = train_test_split(X_fe, y_fe, random_state=1)

model = LinearRegression(fit_intercept=False)     

model.fit(Xtrain, ytrain)                         

y_model = model.predict(Xtest)



r2_score(ytest, y_model)  ## check score of the model chosen



#daily['predicted_fe'] = model_fe.predict(X_fe) # incoporated the predicted number of riders in the orginial data set
coeff_df = pd.DataFrame(model.coef_, X_fe.columns, columns=['Coefficient'])

coeff_df
daily["error_fe"]= (daily.predicted_fe - daily.Total).astype(int)

daily["error_fe_abs"]= (daily.predicted - daily.Total).abs().astype(int)



monthly_error = daily.resample('m').mean().reset_index()



monthly_error.plot(x="Date", y=["Total","predicted_fe","predicted"], figsize=(15,5))
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection  import cross_val_score



daily_X = daily[['Mon','Tue','Wed','Thu','Fri','Sat','Sun','holiday','Temp (C)','dry day','Xmas_period','Summer_period']]

daily_y = daily['Total']



tree_reg = DecisionTreeRegressor(max_depth=6)

tree_reg.fit(daily_X, daily_y)
cross_val_score(tree_reg, X, y, cv=10)
scores_tree = cross_val_score(tree_reg, X, y, cv=10).mean()

scores_tree
# Visualize the trained Decision Tree by export_graphviz() method



from sklearn.tree import export_graphviz

from sklearn import tree

from IPython.display import SVG

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from graphviz import Source

from IPython.display import display
labels = daily_X.columns



graph = Source(tree.export_graphviz(tree_reg ,feature_names = labels,max_depth=5, filled = True))

display(SVG(graph.pipe(format='svg')))