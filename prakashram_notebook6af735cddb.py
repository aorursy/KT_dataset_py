import matplotlib.pyplot as pplt

import seaborn as sb

import numpy as np

import pandas as pa

import matplotlib as mlib

import statsmodels as stats

import statsmodels.formula.api as sm

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_log_error

from ggplot import ggplot
cyc_data = pa.read_csv('../input/train.csv')

cyc_data_tst = pa.read_csv('../input/test.csv')
to_appenddate = list(cyc_data_tst["datetime"])
cyc_data.describe()
cyc_data.info()
cyc_data.head()
data_Toplot=["datetime","season", "holiday","workingday", "weather", "temp", "atemp", "humidity", "windspeed"]
%matplotlib inline

sb.pairplot(data=cyc_data,x_vars=data_Toplot,y_vars="casual",size=3)

sb.pairplot(data=cyc_data,x_vars=data_Toplot,y_vars="registered",size=3)

sb.pairplot(data=cyc_data,x_vars=data_Toplot,y_vars="count",size=3)
cyc_data[data_Toplot].corr()
cyc_data.hist(figsize=(15,15))
cyc_data_comb = cyc_data.append(cyc_data_tst)

cyc_data_comb.reset_index(inplace=True)

cyc_data_comb.drop('index', axis=1,inplace=True)

cyc_data_comb.rename(columns={'count':'cnt'},inplace=True)
cyc_data_comb.head()
cyc_data_comb["datetime"] = cyc_data_comb["datetime"].map(lambda dt: dt.split(" ")[1].split(":")[0])
def datetime_fill(buf_time):

    buf_fill = int(buf_time)

    if( (buf_fill>=8) & (buf_fill<=12) ):

        return "P_AM"

    elif( (buf_fill>=13) & (buf_fill<=16) ):

        return "NP_PM"

    elif( (buf_fill>=17) & (buf_fill<=22) ):

        return "P_PM"

    else:

        return "NP_AM"

    

cyc_data_comb["datetime"] = cyc_data_comb["datetime"].map(lambda dt_time: datetime_fill(dt_time))
datetime_dummies = pa.get_dummies(cyc_data_comb["datetime"],prefix="Time")

cyc_data_comb = pa.concat([cyc_data_comb,datetime_dummies],axis=1)

cyc_data_comb.drop('datetime',axis=1,inplace=True)
season_map = {1:"Spring",2:"Summer",3:"Fall",4:"Winter"}

weather_map = {1:"Clear",2:"Misty",3:"Light_Rain",4:"Heavy_Rain"}



cyc_data_comb["season"] = cyc_data_comb.season.map(season_map)

cyc_data_comb["weather"] = cyc_data_comb.weather.map(weather_map)
seasonmap_dummies = pa.get_dummies(cyc_data_comb["season"],prefix="Season")

cyc_data_comb = pa.concat([cyc_data_comb,seasonmap_dummies],axis=1)

cyc_data_comb.drop('season',axis=1,inplace=True)



weathermap_dummies = pa.get_dummies(cyc_data_comb["weather"],prefix="Weather")

cyc_data_comb = pa.concat([cyc_data_comb,weathermap_dummies],axis=1)

cyc_data_comb.drop('weather',axis=1,inplace=True)
cyc_data_comb.corr()
cyc_data_comb.drop('temp',axis=1,inplace=True)
cyc_data_comb.hist(figsize=(15,15))
cyc_data = cyc_data_comb.loc[0:10885]

cyc_data_tst = cyc_data_comb.loc[10886:]

cyc_data.columns



cyc_data_train = cyc_data.loc[0:7000]

cyc_data_tstpart = cyc_data.loc[7001:]

cyc_data_train.reset_index

cyc_data_tstpart.reset_index
x_ipts = ['atemp','holiday', 'humidity','windspeed', 'workingday', 'Time_NP_AM',

          'Time_NP_PM', 'Time_P_AM','Time_P_PM', 'Season_Fall','Season_Spring', 'Season_Summer','Season_Winter', 

          'Weather_Clear', 'Weather_Heavy_Rain','Weather_Light_Rain', 'Weather_Misty']

opt_casl_train = cyc_data.casual

opt_regd_train = cyc_data.registered
casl_modl_lnr = LinearRegression()

casl_modl_lnr.fit(cyc_data[x_ipts],cyc_data.casual)



%matplotlib inline

sb.pointplot(x=cyc_data.casual,y=casl_modl_lnr.predict(cyc_data[x_ipts]))



casl_modl_dtree = DecisionTreeRegressor()

casl_modl_dtree.fit(cyc_data[x_ipts],cyc_data.casual)



sb.pointplot(x=cyc_data.casual,y=casl_modl_dtree.predict(cyc_data[x_ipts]),color='red')



print(casl_modl_lnr.score(cyc_data[x_ipts],cyc_data.casual))

print(casl_modl_dtree.score(cyc_data[x_ipts],cyc_data.casual))

print(casl_modl_dtree.feature_importances_)
regd_modl_lnr = LinearRegression()

regd_modl_lnr.fit(cyc_data[x_ipts],cyc_data.registered)



%matplotlib inline

sb.pointplot(x=cyc_data.registered,y=regd_modl_lnr.predict(cyc_data[x_ipts]))



regd_modl_dtree = DecisionTreeRegressor()

regd_modl_dtree.fit(cyc_data[x_ipts],cyc_data.registered)



sb.pointplot(x=cyc_data.registered,y=regd_modl_dtree.predict(cyc_data[x_ipts]),color='red')



print(regd_modl_lnr.score(cyc_data[x_ipts],cyc_data.registered))

print(regd_modl_dtree.score(cyc_data[x_ipts],cyc_data.registered))

print(regd_modl_dtree.feature_importances_)
#Predicting values using different methods



#Linear Regression Model

#casual

casl_modl_lnr = LinearRegression()

casl_modl_lnr.fit(cyc_data_train[x_ipts],cyc_data_train.casual)



#registered



regd_modl_lnr = LinearRegression()

regd_modl_lnr.fit(cyc_data_train[x_ipts],cyc_data_train.registered)



predicted = casl_modl_lnr.predict(cyc_data_tstpart[x_ipts]) + regd_modl_lnr.predict(cyc_data_tstpart[x_ipts])



%matplotlib inline

fig,ax = pplt.subplots(nrows=1,ncols=1)

fig.set_size_inches(20,5)

pplt.plot(cyc_data_tstpart.index,cyc_data_tstpart.cnt)

pplt.plot(cyc_data_tstpart.index,predicted,c='red',linewidth=3)

print(predicted)

#DecisionTree Regressor Model

#casual

casl_modl_lnr = DecisionTreeRegressor(max_depth=20,max_features='sqrt')

casl_modl_lnr.fit(cyc_data_train[x_ipts],cyc_data_train.casual)



#registered



regd_modl_lnr = DecisionTreeRegressor(max_depth=20,max_features='sqrt')

regd_modl_lnr.fit(cyc_data_train[x_ipts],cyc_data_train.registered)



predicted = casl_modl_lnr.predict(cyc_data_tstpart[x_ipts]) + regd_modl_lnr.predict(cyc_data_tstpart[x_ipts])



%matplotlib inline

fig,ax = pplt.subplots(nrows=1,ncols=1)

fig.set_size_inches(20,5)

pplt.plot(cyc_data_tstpart.index,cyc_data_tstpart.cnt)

pplt.plot(cyc_data_tstpart.index,predicted,c='red',linewidth=3)

print(predicted)
#RandomForest Regressor Model

#casual

casl_modl_lnr = RandomForestRegressor(n_estimators=2000,max_depth=10)

casl_modl_lnr.fit(cyc_data_train[x_ipts],cyc_data_train.casual)



#registered



regd_modl_lnr = RandomForestRegressor(n_estimators=2000,max_depth=10)

regd_modl_lnr.fit(cyc_data_train[x_ipts],cyc_data_train.registered)



predicted = casl_modl_lnr.predict(cyc_data_tstpart[x_ipts]) + regd_modl_lnr.predict(cyc_data_tstpart[x_ipts])



%matplotlib inline

#fig,ax = pplt.subplots(nrows=1,ncols=1)

#fig.set_size_inches(20,5)

#pplt.plot(cyc_data_tstpart.index,cyc_data_tstpart.cnt)

#pplt.plot(cyc_data_tstpart.index,predicted,c='red',linewidth=3)

print(predicted)
#Gradient boost Regressor Model



#casual

casl_modl_lnr = GradientBoostingRegressor(n_estimators=4000,max_depth=20,max_features='auto')

casl_modl_lnr.fit(cyc_data[x_ipts],cyc_data.casual)



#registered



regd_modl_lnr = GradientBoostingRegressor(n_estimators=4000,max_depth=20,max_features='auto')

regd_modl_lnr.fit(cyc_data[x_ipts],cyc_data.registered)



predicted = casl_modl_lnr.predict(cyc_data_tst[x_ipts]) + regd_modl_lnr.predict(cyc_data_tst[x_ipts])



app_str = str("datetime,count")

o_put = list(map(lambda a,b: str(a) + "," + str(b), to_appenddate,predicted))

o_put.insert(0,app_str)



%matplotlib inline

fig,ax = pplt.subplots(nrows=1,ncols=1)

fig.set_size_inches(30,10)

pplt.plot(cyc_data_tst.index,cyc_data_tst.cnt)

pplt.plot(cyc_data_tst.index,predicted,c='red',linewidth=3)



print(o_put)