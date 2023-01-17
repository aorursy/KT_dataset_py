# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import calendar

import seaborn as sb

import xgboost as xgb

import plotly.express as px

import matplotlib.pyplot as plt

from plotly.subplots import make_subplots

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.metrics import mean_squared_log_error,make_scorer

from sklearn.model_selection import train_test_split,GridSearchCV
#Reading the file

file = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")
file.describe()
file.isnull().sum(axis=0)
file.columns
corr = file[['temp','atemp','humidity', 'windspeed','casual', 'registered','count']].corr()

f,axes = plt.subplots(1,1,figsize = (8,8))

sb.heatmap(corr,square=True,annot = True,linewidth = .5,center = 1.4,ax = axes)
file = file

file['Date'] = pd.DatetimeIndex(file['datetime']).date

file['Hour'] = pd.DatetimeIndex(file['datetime']).hour

file['Day'] = pd.DatetimeIndex(file['datetime']).day

file['Month'] = pd.DatetimeIndex(file['datetime']).month

file['Year'] = pd.DatetimeIndex(file['datetime']).year

file['Weekday'] = pd.DatetimeIndex(file['datetime']).weekday_name
a = []

for i in file.index:

    a.append('Total Count : '+str(file['count'][i]))

file['count_vis'] = a
fig = px.line(x = 'Date', y = "count", data_frame = file,color = 'Hour',range_y = (0,1150),

              title = 'Interactive LinePlot of the whole dataset(Hover for more details)',

              hover_data = ['Hour','Date','casual','registered'],

              hover_name = 'count_vis', text = None,

              height = 670,width = 980)

fig.show()
f,axes = plt.subplots(1,3,figsize = (17,7))

sb.despine(left = True)

x = 'season'



sb.barplot(x = x , y = 'casual' , data = file, saturation = 1, ax =  axes[0])

sb.barplot(x = x , y = 'registered' , data = file, saturation = 1, ax = axes[1])

sb.barplot(x = x , y = 'count' , data = file, saturation = 1, ax = axes[2])
f,axes = plt.subplots(1,3,figsize = (17,7))

sb.despine(left = True)

x = 'holiday'



sb.barplot(x = x , y = 'casual' , data = file, saturation = 1, ax =  axes[0] ,)

sb.barplot(x = x , y = 'registered' , data = file, saturation = 1, ax = axes[1])

sb.barplot(x = x , y = 'count' , data = file, saturation = 1, ax = axes[2])
f,axes = plt.subplots(1,3,figsize = (17,7))

sb.despine(left = True)

x = 'workingday'



sb.barplot(x = x , y = 'casual' , data = file, saturation = 1, ax =  axes[0] ,)

sb.barplot(x = x , y = 'registered' , data = file, saturation = 1, ax = axes[1])

sb.barplot(x = x , y = 'count' , data = file, saturation = 1, ax = axes[2])
f,axes = plt.subplots(1,3,figsize = (17,7))

sb.despine(left = True)

x = 'weather'



sb.barplot(x = x , y = 'casual' , data = file, saturation = 1, ax =  axes[0] ,)

sb.barplot(x = x , y = 'registered' , data = file, saturation = 1, ax = axes[1])

sb.barplot(x = x , y = 'count' , data = file, saturation = 1, ax = axes[2])
f,axes = plt.subplots(1,3,figsize = (19,7))

sb.despine(left = True)

x = 'Hour'



sb.barplot(x = x , y = 'casual' , data = file, saturation = 1, ax =  axes[0] ,)

sb.barplot(x = x , y = 'registered' , data = file, saturation = 1, ax = axes[1])

sb.barplot(x = x , y = 'count' , data = file, saturation = 1, ax = axes[2])
file.groupby('Weekday').count().index
file1 = file.groupby(['Hour','Weekday']).mean().reset_index()

dic = {'Weekday':['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']}

dic1 = {'registered':'Average count of registered poeple commuting.','count': 'Average people commuting','Hour':'Hour of the day',

        'Weekday':'Day of the week'}

fig = px.line(x = 'Hour', y = "registered", data_frame = file1.reset_index(),color = 'Weekday',

              title = 'Interactive LinePlot of the registered separated by weekday(Hover for more details)',labels = dic1,

              hover_data = ['count'],category_orders = dic,range_y = [0,550],height = 670,width = 980)

fig.show()
f,axes = plt.subplots(1,3,figsize = (19,7))

sb.despine(left = True)

x = 'Day'



sb.barplot(x = x , y = 'casual' , data = file, saturation = 1, ax =  axes[0] ,)

sb.barplot(x = x , y = 'registered' , data = file, saturation = 1, ax = axes[1])

sb.barplot(x = x , y = 'count' , data = file, saturation = 1, ax = axes[2])
f,axes = plt.subplots(1,3,figsize = (19,7))

sb.despine(left = True)

x = 'Month'

#order = ['January','February','March','April','May','June','July','August','September','October','November','December']

plot = sb.barplot(x = x , y = 'casual' , data = file, saturation = 1, ax =  axes[0])

sb.barplot(x = x , y = 'registered' , data = file, saturation = 1, ax = axes[1])

sb.barplot(x = x , y = 'count' , data = file, saturation = 1, ax = axes[2])
f,axes = plt.subplots(1,3,figsize = (19,7))

sb.despine(left = True)

x = 'Year'



sb.barplot(x = x , y = 'casual' , data = file, saturation = 1, ax =  axes[0] ,)

sb.barplot(x = x , y = 'registered' , data = file, saturation = 1, ax = axes[1])

sb.barplot(x = x , y = 'count' , data = file, saturation = 1, ax = axes[2])
file.describe()
file.columns
for i in file.groupby('season').count().index:

    s = 's'+str(i)

    a=[]

    for j in file.season:

        if j==i:

            a.append(1)

        else:

            a.append(0)

    file[s]=a

file.sample(5)
for i in file.groupby('weather').count().index:

    s = 'w'+str(i)

    a=[]

    for j in file.weather:

        if j==i:

            a.append(1)

        else:

            a.append(0)

    file[s]=a

file.sample(5)
for i in file.groupby('Hour').count().index:

    s = 'Hour'+str(i)

    a=[]

    for j in file.Hour:

        if j==i:

            a.append(1)

        else:

            a.append(0)

    file[s]=a

file.sample(5)
for i in file.groupby("Month").count().index:

    s = 'Month' + str(i)

    a = []

    for j in file.Month:

        if j==i:

            a.append(1)

        else:

            a.append(0)

    file[s] = a

file.sample(5)
file.columns
feed = file[['Hour0', 'Hour1', 'Hour2', 'Hour3', 'Hour4', 'Hour5',

       'Hour6', 'Hour7', 'Hour8', 'Hour9', 'Hour10', 'Hour11', 'Hour12',

       'Hour13', 'Hour14', 'Hour15', 'Hour16', 'Hour17', 'Hour18', 'Hour19',

       'Hour20', 'Hour21', 'Hour22', 'Hour23','Month1', 'Month2', 'Month3',

       'Month4', 'Month5', 'Month6', 'Month7', 'Month8', 'Month9', 'Month10',

       'Month11', 'Month12','Year','s1','s2','s3','s4','holiday','workingday',

        'w1','w2','w3','w4','temp','humidity','casual','registered']]
feed.describe()
feed.columns
df_train_x = feed.drop('casual',axis = 1).drop('registered',axis=1)

df_train_x.describe()
df_reg_train_y = feed['registered']

df_reg_train_y.describe
df_cas_train_y = feed['casual']

df_cas_train_y.describe
x1_train, x1_test, y1_train, y1_test = train_test_split(df_train_x, df_cas_train_y, test_size=0.15, random_state=42)

x2_train, x2_test, y2_train, y2_test = train_test_split(df_train_x, df_reg_train_y, test_size=0.15, random_state=42)
poly = PolynomialFeatures(degree=2)

poly_x1_train = poly.fit_transform(x1_train)

poly_x1_test = poly.fit_transform(x1_test)

poly_x2_train = poly.fit_transform(x2_train)

poly_x2_test = poly.fit_transform(x2_test)
rf = RandomForestRegressor()



parameters = {'n_estimators':[50,100,150,200,250],

              'min_impurity_decrease':[0.0,0.001,0.01],

              'max_depth':[20,40,60,80,100]}



models = ['Normal Linear Regression: ','Linear Regression over polynomial: ',

          'Decision Tree Regressor: ','XG Boosting: ']
def custom_scorer(y_true,y_pred):

    for i in range(len(y_pred)):

        if y_pred[i]<0:

            y_pred[i] = 1

    return np.sqrt(mean_squared_log_error(y_true, y_pred ))

scorer = make_scorer(custom_scorer,greater_is_better = False)
predict = []

reg = LinearRegression().fit(x1_train, y1_train)

pre_reg = reg.predict(x1_test)



reg_poly = LinearRegression().fit(poly_x1_train, y1_train)

pre_reg_poly = reg_poly.predict(poly_x1_test)



rf_reg = GridSearchCV(rf, parameters, cv=5, verbose=2,scoring = scorer,n_jobs = -1)

rf_reg.fit(x1_train, y1_train)

pre_rf_reg = rf_reg.predict(x1_test)



predict.append(pre_reg)

predict.append(pre_reg_poly)

predict.append(pre_rf_reg)
for prediction in range(len(predict)):

    pre = []

    for p in predict[prediction]:

        if p < 1:

            pre.append(1)

        else:

            pre.append(p)

    print(models[prediction]+str(np.sqrt(mean_squared_log_error(y1_test, pre ))))
predict = []

cas = LinearRegression().fit(x2_train, y2_train)

pre_cas = cas.predict(x2_test)



cas_poly = LinearRegression().fit(poly_x2_train, y2_train)

pre_cas_poly = cas_poly.predict(poly_x2_test)



rf_cas = GridSearchCV(rf, parameters, cv=5, verbose=2,scoring = scorer,n_jobs = -1)

rf_cas.fit(x2_train, y2_train)

pre_rf_cas = rf_cas.predict(x2_test)



predict.append(pre_cas)

predict.append(pre_cas_poly)

predict.append(pre_rf_cas)
for prediction in range(len(predict)):

    pre = []

    for p in predict[prediction]:

        if p < 1:

            pre.append(1)

        else:

            pre.append(p)

    print(models[prediction]+str(np.sqrt(mean_squared_log_error(y2_test, pre ))))
print("For Random Forest Model: ")

print("\t Best Parametres for registered are: ",end='')

print(rf_reg.best_params_)

print("\t Best Parametres for casual are: ",end = '')

print(rf_cas.best_params_)
predict1 = []



reg1 = LinearRegression().fit(x1_train, y1_train)

pre_reg1 = reg1.predict(x1_test)



reg1_poly = LinearRegression().fit(poly_x1_train, y1_train)

pre_reg1_poly = reg1_poly.predict(poly_x1_test)



rf1 = RandomForestRegressor(n_estimators = 200,max_depth=80,min_impurity_decrease = 0.001).fit(x1_train, y1_train)

pre_rf1 = rf1.predict(x1_test)



for i in range(pre_reg1.size):

    if pre_reg1[i]<1:

        pre_reg1[i] = 1 

    if pre_reg1_poly[i]<1:

        pre_reg1_poly[i] = 1

    if pre_rf1[i]<1:

        pre_rf1[i] = 1



predict1.append(pre_reg1)

predict1.append(pre_reg1_poly)

predict1.append(pre_rf1)



x1_final = x1_test.copy()

x1_final['Output'] = y1_test

x1_final['Lin_reg'] = pre_reg1

x1_final['Lin_reg_poly'] = pre_reg1_poly

x1_final['RF_reg'] = pre_rf1

x1_final['Resid'] = y1_test-pre_reg1

x1_final['Resid_poly'] = y1_test-pre_reg1_poly



for prediction in predict1:

    print(np.sqrt(mean_squared_log_error( y1_test, prediction )))
predict2 = []



reg2 = LinearRegression().fit(x2_train, y2_train)

pre_reg2 = reg2.predict(x2_test)



reg2_poly = LinearRegression().fit(poly_x2_train, y2_train)

pre_reg2_poly = reg2_poly.predict(poly_x2_test)



rf2 = RandomForestRegressor(n_estimators = 150,max_depth=60,min_impurity_decrease = 0.0).fit(x2_train, y2_train)

pre_rf2 = rf2.predict(x2_test)



for i in range(pre_reg2.size):

    if pre_reg2[i]<1:

        pre_reg2[i] = 1 

    if pre_reg2_poly[i]<1:

        pre_reg2_poly[i] = 1

    if pre_rf2[i]<1:

        pre_rf2[i] = 1



predict2.append(pre_reg2)

predict2.append(pre_reg2_poly)

predict2.append(pre_rf2)



x2_final = x2_test.copy()

x2_final['Output'] = y2_test

x2_final['Lin_reg'] = pre_reg2

x2_final['Lin_reg_poly'] = pre_reg2_poly

x2_final['RF_reg'] = pre_rf2

x2_final['Resid'] = y2_test-pre_reg2

x2_final['Resid_poly'] = y2_test-pre_reg2_poly



for prediction in predict2:

    print(np.sqrt(mean_squared_log_error( y2_test, prediction )))
from plotly.subplots import make_subplots

name1  = ['Residual for casual without polynomial features'] *1633

name2  = ['Residual for casual with polynomial features'] *1633

name3  = ['Residual for registered without polynomial features'] *1633

name4  = ['Residual for registered with polynomial features'] *1633

dic = {'Lin_reg': 'Predicted Output','Resid':'Deviation from predicted','Output':'Expected Output','Lin_reg_poly': 'Predicted Output',

       'Resid_poly':'Deviation from predicted'}

fig1 = px.scatter(data_frame = x1_final,x = 'Lin_reg', y = 'Resid',hover_data = ['Output'],labels = dic,hover_name = name1,

                  color_discrete_sequence = ['red'])

fig2 = px.scatter(data_frame = x1_final,x = 'Lin_reg_poly', y = 'Resid_poly',hover_data = ['Output'],labels = dic,hover_name = name2,

                  color_discrete_sequence = ['blue'])

fig3 = px.scatter(data_frame = x2_final,x = 'Lin_reg', y = 'Resid',hover_data = ['Output'],labels = dic,hover_name = name3,

                  color_discrete_sequence = ['darkgreen'])

fig4 = px.scatter(data_frame = x2_final,x = 'Lin_reg_poly', y = 'Resid_poly',hover_data = ['Output'],labels = dic,hover_name = name4,

                  color_discrete_sequence = ['gold'])

trace1 = fig1['data'][0]

trace2 = fig2['data'][0]

trace3 = fig3['data'][0]

trace4 = fig4['data'][0]



fig = make_subplots(rows=2, cols=2,horizontal_spacing =0.1,vertical_spacing  = 0.2,

                    row_titles = ['Using Polynomial','Without Polynomial'],column_titles = ['Casual','Registered'],

                    x_title = 'Residual plots for Registered and Casual under different models (Hover for more details)')



fig.add_trace(trace1, row=1, col=1)

fig.add_trace(trace2, row=1, col=2)

fig.add_trace(trace3, row=2, col=1)

fig.add_trace(trace4, row=2, col=2)

fig.show()
rf1 = RandomForestRegressor(n_estimators = 200,max_depth=80,min_impurity_decrease = 0.001).fit(df_train_x,df_cas_train_y)

rf2 = RandomForestRegressor(n_estimators = 150,max_depth=60,min_impurity_decrease = 0.0).fit(df_train_x,df_reg_train_y)
test_file = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
test=test_file

test.describe()
test['mth'] = pd.DatetimeIndex(test['datetime']).month

test['yr'] = pd.DatetimeIndex(test['datetime']).year

test['dy'] = pd.DatetimeIndex(test['datetime']).day

test['hr'] = pd.DatetimeIndex(test['datetime']).hour



for i in test.groupby("season").count().index:

    s = 's' + str(i)

    a = []

    for j in test.season:

        if j==i:

            a.append(1)

        else:

            a.append(0)

    test[s] = a

for i in test.groupby("weather").count().index:

    s = 'w' + str(i)

    a = []

    for j in test.weather:

        if j==i:

            a.append(1)

        else:

            a.append(0)

    test[s] = a

for i in test.groupby('hr').count().index:

    s = 'hr'+str(i)

    a=[]

    for j in test.hr:

        if j==i:

            a.append(1)

        else:

            a.append(0)

    test[s]=a

for i in test.groupby("mth").count().index:

    s = 'm' + str(i)

    a = []

    for j in test.mth:

        if j==i:

            a.append(1)

        else:

            a.append(0)

    test[s] = a

test.sample(10)
test = test[['hr0','hr1','hr2','hr3','hr4','hr5','hr6','hr7','hr8','hr9','hr10','hr11','hr12','hr13','hr14','hr15','hr16','hr17','hr18',

                 'hr19','hr20','hr21','hr22','hr23','m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12','yr',

                 's1','s2','s3','s4','holiday','workingday','w1','w2','w3','w4','temp','humidity']]

test.describe
pre_cas = rf1.predict(test)

pre_reg = rf2.predict(test)

final_predictions = pd.DataFrame(pre_cas+pre_reg,columns = ['cout'])



final_predictions.describe
s=[]

for j in final_predictions.cout:

    if int(j)<1:

        s.append(1)

    else:

        s.append(j)

final_predictions['count'] = s 
final_predictions.describe
final_predictions['datetime']=test_file['datetime']

final_predictions = final_predictions[['datetime','count']]
final_predictions.describe()
final_predictions.to_csv('submission.csv',index=False)