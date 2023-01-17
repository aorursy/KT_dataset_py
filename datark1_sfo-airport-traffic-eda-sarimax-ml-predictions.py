import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualisation

import matplotlib.pyplot as plt # data visualisation

import datetime as dt # working with time data

import warnings  

warnings.filterwarnings('ignore')
PAX_raw = pd.read_csv("../input/air-traffic-passenger-statistics.csv")

PAX = PAX_raw.copy()

PAX.head()
print("Shape of passengers database: ",PAX.shape)

nulls = PAX.isnull().sum().to_frame().sort_values(by=0, ascending = False)

nulls.columns = ["Missing values"]

nulls[nulls.iloc[:,0] != 0]
PAX.loc[:,"Activity Period"] = pd.to_datetime(PAX.loc[:,"Activity Period"].astype(str), format="%Y%m")

PAX.loc[:,"Year"] = PAX["Activity Period"].dt.year

PAX.loc[:,"Month"] = PAX["Activity Period"].dt.month
time_begin = PAX.loc[:,"Activity Period"].min()

time_end = PAX.loc[:,"Activity Period"].max()

time_range = time_end-time_begin



print("First date: ", str(time_begin)[:11])

print("Last date: ", str(time_end)[:11])

print("Time range in days:", time_range.days)

print("Time range in months:", time_range/np.timedelta64(1,"M"))
PAX_yr = PAX.groupby(["Activity Period"])["Passenger Count"].sum().divide(1000)



fig, ax = plt.subplots(figsize=(15,7))



#Plotting the main PAX line

sns.lineplot(x=PAX_yr.index, y=PAX_yr.values, markers=True, ax=ax)



#Plotting vertical lines for beginning of each year 

years = PAX_yr.index.year.unique()

for year in years:

    plt.axvline(x=str(year) +"-01-01", ls = "--", c = "#3f5261", alpha=0.7)

    

#Looking for maximum PAX for each year

PAX_yr_maxs = PAX_yr.groupby(PAX_yr.index.year).max()

PAX_yr_max_complete = PAX_yr[PAX_yr.isin(PAX_yr_maxs.values)].to_frame()



#Marking points of interest

plt.scatter(PAX_yr_max_complete.index, PAX_yr_max_complete.values, color = "red")



#Annotating each marker

for t,v in PAX_yr_max_complete.reset_index().values:

    ax.text(t,v+100,int(v))

    

#Looking for minimum PAX for each year

PAX_yr_mins = PAX_yr.groupby(PAX_yr.index.year).min()

PAX_yr_min_complete = PAX_yr[PAX_yr.isin(PAX_yr_mins.values)].to_frame()

plt.scatter(PAX_yr_min_complete.index, PAX_yr_min_complete.values, color = "green")

for t,v in PAX_yr_min_complete.reset_index().values:

    ax.text(t,v-200,int(v))

    

plt.title("SFO passengers", size = 20)

plt.xlabel("Date", fontweight="bold", size = 12)

plt.ylabel("PAX (in tousands)", fontweight="bold", size = 12)

plt.show()
PAX_airline_yr = PAX.groupby(["Year","Operating Airline"])["Passenger Count"].sum().divide(1000)

PAX_airline_yr = PAX_airline_yr.reset_index()

pivot_1 = PAX_airline_yr.pivot_table(values="Passenger Count",index="Operating Airline",columns="Year", fill_value=0)

pivot_1.loc["United Airlines",:] = pivot_1.loc["United Airlines",:] + pivot_1.loc["United Airlines - Pre 07/01/2013",:]

pivot_1.drop("United Airlines - Pre 07/01/2013",axis=0, inplace=True)



#dropping the small airlines

dropped = pivot_1[pivot_1.sum(axis=1)<13]

pivot_1 = pivot_1.drop(dropped.index,axis=0)



sns.set(font_scale=0.7)

fig1 = plt.figure(figsize=(12,20))

p1 = sns.heatmap(pivot_1, annot=True, linewidths=.5, vmin=100, vmax=1000, fmt='.0f', cmap=sns.cm.rocket_r)

p1.set_title("Number of passengers carried (in thousands)", fontweight="bold")

p1.set_yticklabels(p1.get_yticklabels(), rotation=0)

plt.tight_layout()
avg_airline = pivot_1.mean(axis=1)

TOP5_airlines = avg_airline.nlargest(5).to_frame().mul(1000).astype("int64")

TOP5_airlines.columns = ["Mean no. of passengers per year"]

sum_of_all = TOP5_airlines.loc[:,"Mean no. of passengers per year"].sum()

TOP5_airlines.loc[:,"Share [in pct]"] = TOP5_airlines.loc[:,"Mean no. of passengers per year"].div(sum_of_all).mul(100).round(1)

TOP5_airlines
sns.set(font_scale=0.7)

fig2 = plt.figure(figsize=(12,5))

p2 = sns.heatmap(dropped, annot=True, linewidths=.5, cmap="YlGnBu", fmt='.1f')

p2.set_title("The smallest airlines operating at SFO", fontweight ="bold")

p2.set_yticklabels(p1.get_yticklabels(), rotation=0)

plt.tight_layout()
PAX_month_yr = PAX.groupby(["Year","Month"])["Passenger Count"].sum().divide(1000).round()

PAX_month_yr = PAX_month_yr.reset_index()



pivot_2 = PAX_month_yr.pivot_table(values="Passenger Count",index="Month",columns="Year", fill_value=0)

pivot_2.index=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]



sns.set(font_scale=0.8)

fig = plt.figure(figsize=(12,8))

g = sns.heatmap(pivot_2, annot=True, linewidths=.5, fmt="d", square =True, vmin=2000, cmap=sns.cm.rocket_r)

g.set_title("Number of passengers in each month (in thousands)", fontweight="bold")

g.set_yticklabels(g.get_yticklabels(), rotation=0)

plt.tight_layout()
bbb = PAX.groupby(["Year","GEO Region"])["Passenger Count"].sum()

bbb = bbb.reset_index()



pivot_5 = bbb.pivot_table(values="Passenger Count",index="Year",columns="GEO Region", fill_value=0)

pivot_5 = pivot_5.drop("US",axis=1)

pivot_5["Total"] = pivot_5.sum(axis=1)



for col in pivot_5.columns[:-1]:

        pivot_5["Share " + str(col)]=pivot_5[col]/pivot_5["Total"]



ratios_5 = pivot_5.iloc[:,-8:]

ratios_5.plot(figsize=(12,8), title="Share of GEO regions to total traffic generated")

DomIntPAX = PAX.groupby(["Activity Period","GEO Summary"])["Passenger Count"].sum()

DomIntPAX = DomIntPAX.reset_index()



pivot_4 = DomIntPAX.pivot_table(values="Passenger Count",index="Activity Period",columns="GEO Summary", fill_value=0)

pivot_4["Share"] = (pivot_4["International"]/(pivot_4["Domestic"]+pivot_4["International"])).mul(100)

mean_DomInt = pivot_4["Share"].mean()



fig4, ax4 = plt.subplots(figsize=(12,6))

plt.title("Domestic vs. International flights share in overall passenger traffic (in %)", fontdict={'fontsize':13,'fontweight' : "bold"})

plt.ylim(0,100)



pivot_4["Share"].plot(ax=ax4, c="white")

ax4.fill_between(pivot_4.index, 100, color='#f48342')

ax4.fill_between(pivot_4.index, pivot_4["Share"], color='#4189f4')

ax4.axhline(mean_DomInt, c="black", linestyle="--")

ax4.annotate("DOMESTIC",("2011-12",7), size=25, ha="center")

ax4.annotate("INTERNATIONAL",("2011-12",50), size=25, ha="center", va="center")

plt.show()
TS2 = PAX[PAX["GEO Summary"]=="Domestic"].groupby("Activity Period")["Passenger Count"].sum()

TS3 = PAX[PAX["GEO Summary"]=="International"].groupby("Activity Period")["Passenger Count"].sum()

TS = pd.concat([TS2,TS3],axis=1)

TS.columns=["Domestic","International"]

ax = TS.plot(figsize=(12,5))

ax.set_title("Absolute number of passengers by flight type", fontweight="bold")

plt.show()
rolled2 = TS2.pct_change(12).mul(100)

rolled3 = TS3.pct_change(12).mul(100)



mean_domestic = rolled2.mean()

mean_international = rolled3.mean()

rolled = pd.concat([rolled2,rolled3],axis=1)

rolled.columns = ["Domestic","International"]



rolled.plot(figsize=(10,7))

plt.axhline(mean_domestic, c="green", linestyle="--")

plt.axhline(mean_international, c="blue", linestyle="--")



arrowprops = dict(

    arrowstyle = "-|>",

    connectionstyle = "angle,angleA=0,angleB=90,rad=10",

    color = "red"

    )



text1 = "INTERNAT. MEAN\n {:0.2f}%".format(mean_international)

text2 = "DOMESTIC MEAN\n {:0.2f}%".format(mean_domestic)

plt.annotate(text1, xy=("2005-05",mean_international), xytext=(("2007-02",-10)), size=10, ha="center",arrowprops=arrowprops)

plt.annotate(text2, xy=("2005-05",mean_domestic), xytext=(("2007-10",16)), size=10, ha="center",arrowprops=arrowprops)

plt.xlim(["2004-12","2018-12"])

plt.ylim([-20,20])

plt.ylabel("Percentage change (12-month window)")

plt.show()
TS1 = PAX.groupby("Activity Period")["Passenger Count"].sum().to_frame()



f, ax1 = plt.subplots(1,1,figsize=(15,5))

TS1.plot(ax=ax1)

ax1.set_xlabel("Date")

ax1.set_ylabel("Passenger Count")

plt.grid(True)
from statsmodels.tsa.stattools import adfuller



results = adfuller(TS1["Passenger Count"])

print('ADF Statistic: %f' % results[0])

print('p-value: %f' % results[1])
from statsmodels.tsa.seasonal import seasonal_decompose

plt.rcParams['figure.figsize'] = 10, 5

# Additive decomposition

decomposed_add = seasonal_decompose(TS1, model="additive")

add = decomposed_add.plot()

plt.show()
TS1_diff = TS1.diff().dropna()

plt.figure(figsize=(12,5))

ax1 = TS1_diff["Passenger Count"].plot()

ax1.set_xlabel("Date")

ax1.set_ylabel("Passenger Count 12-months Difference")

plt.grid(True)

plt.show()
results = adfuller(TS1_diff["Passenger Count"])

print('ADF Statistic: %f' % results[0])

print("P-value of a test is: {}".format(results[1]))
results = adfuller(TS1.diff().diff().dropna()["Passenger Count"])

print('ADF Statistic: %f' % results[0])

print("P-value of a test is: {}".format(results[1]))
from pandas.plotting import autocorrelation_plot

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



fig,ax = plt.subplots(2,1,figsize=(20,10))

plot_acf(TS1_diff, lags=36, ax=ax[0])

plot_pacf(TS1_diff, lags=36, ax=ax[1])

plt.show()
from pandas.plotting import lag_plot



fig, ax = plt.subplots(figsize=(10, 10))

ax = lag_plot(TS1_diff, lag=1)

ax = lag_plot(TS1_diff, lag=2, c="orange")



plt.show()
from statsmodels.tsa.arima_model import ARIMA



arima_df = pd.DataFrame(columns=["p","q","aic","bic"])



i=0

# Loop over p values from 0-3

for p in range(4):

    # Loop over q values from 0-3

    for q in range(4):

        

        try:

            # creating and fitting ARIMA(p,1,q) model

            model = ARIMA(TS1.astype(float), order=(p,1,q))

            results = model.fit()

            

            # Printing order, AIC and BIC

            #print(p, q, results.aic, results.bic)

            arima_df.loc[i,"p"] = p

            arima_df.loc[i,"q"] = q

            arima_df.loc[i,"aic"] = results.aic

            arima_df.loc[i,"bic"] = results.bic

            i = i+1

        except:

            #print(p, q, None, None)

            i = i+1

    

arima_df["sum_aic_bic"] = arima_df["aic"]+arima_df["bic"]

arima_df.sort_values(by="sum_aic_bic", ascending=False, inplace=True)

arima_df
from statsmodels.tsa.statespace.sarimax import SARIMAX



model2 = SARIMAX(TS1, order=(2,1,1), seasonal_order=(0,1,0,12))

results = model2.fit()

results.summary()
plt.rcParams['figure.figsize'] = 12, 8

plot = results.plot_diagnostics()
# Create SARIMA mean forecast

forecast = results.get_forecast(steps=48)

lower = forecast.conf_int()["lower Passenger Count"]

upper = forecast.conf_int()["upper Passenger Count"]



# Plot mean SARIMA predictions

fig,ax = plt.subplots(1,1,figsize=(20,10))



plt.plot(TS1, label='original')

plt.plot(forecast.predicted_mean, label='SARIMAX', c="r")

plt.fill_between(forecast.conf_int().index, 

                 lower,upper,

                 color='pink')

plt.xlabel('Date')

plt.ylabel('No of passengers')

plt.legend()

plt.show()
TS1["Year"] = TS1.index.year

TS1["Month"] = TS1.index.month

TS1.head()

X = TS1[["Year","Month"]]

y = TS1[["Passenger Count"]]
plt.figure(figsize=(10,5))

plt.scatter(TS1["Year"],TS1["Passenger Count"],c=TS1["Month"])

plt.legend()

plt.show()
plt.figure(figsize=(10,5))

plt.scatter(TS1["Month"],TS1["Passenger Count"],c=TS1["Year"])

plt.legend()

plt.show()
from sklearn.svm import SVR, SVR

from sklearn.kernel_ridge import KernelRidge

from sklearn.model_selection import KFold, train_test_split, TimeSeriesSplit

from sklearn.metrics import mean_squared_error



def tsplit(X,y,model):

    tscv = TimeSeriesSplit(n_splits=3)

    fig,ax = plt.subplots(3, figsize=(15,8))

    axis = 0

    for train_index, test_index in tscv.split(X):

        #splitting data into training and test sets

        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]

        y_train, y_test = y.iloc[train_index,:], y.iloc[test_index,:]

        #fitting model

        model.fit(X_train,y_train.values.ravel())

        #predicting

        predictions = model.predict(X_test)

        #printing results

        print("MSE for split {0}:".format(axis+1))

        print(mean_squared_error(y_test,predictions))

        #ax[axis].plot(X_train.index, y_train) # needs fixing

        ax[axis].plot(list(X_test.index), predictions)

        ax[axis].plot(list(X_test.index), y_test)

        axis += 1

    return(None)
X2 = TS1[["Year","Month"]][1:]

y2 = np.log(TS1[["Passenger Count"]]).diff().dropna()

X2.head()
from sklearn.preprocessing import StandardScaler



sc_x = StandardScaler()

sc_y = StandardScaler()

# Scale x and y (two scale objects)

X2_scaled = pd.DataFrame(sc_x.fit_transform(X2))

y2_scaled = pd.DataFrame(sc_y.fit_transform(y2))
from sklearn.model_selection import GridSearchCV
rdg = KernelRidge(kernel='rbf')

parameters = {'alpha':np.arange(0.005,0.02,0.005), 'gamma':np.arange(0.001,0.01,0.001)}



tscv = TimeSeriesSplit(n_splits=3)

rdg_gs = GridSearchCV(rdg, parameters, cv=tscv, verbose=0, scoring='neg_mean_squared_error')

rdg_gs.fit(X2_scaled, y2_scaled)



rdg_gs.best_score_

best_rdg = rdg_gs.best_estimator_

print(best_rdg)
tsplit(X2_scaled,y2_scaled,best_rdg)
svr = SVR()

parameters = {'kernel':['rbf','poly'],

              'C':np.arange(0.2,0.8,0.1),

              'gamma':np.arange(0.2,1.2,0.02),

              'degree':[3,4,5]}



tscv = TimeSeriesSplit(n_splits=3)

reg = GridSearchCV(svr, parameters, cv=tscv, verbose=0, scoring='neg_mean_squared_error')

reg.fit(X2_scaled, y2_scaled.values.ravel())



reg.best_score_

best_svr = reg.best_estimator_

print(best_svr)
tsplit(X2_scaled,y2_scaled,best_svr)
from sklearn.neural_network import MLPRegressor



mlp = MLPRegressor(max_iter=600)

parameters = {'hidden_layer_sizes':np.arange(800,1400,50),'alpha':[0.0001,0.0002], 'momentum':[0.85,0.9,0.95]}



tscv = TimeSeriesSplit(n_splits=3)

reg = GridSearchCV(mlp, parameters, cv=tscv, verbose=0, scoring='neg_mean_squared_error')

reg.fit(X2_scaled, y2_scaled.values.ravel())



reg.best_score_

best_mlp = reg.best_estimator_

print(best_mlp)
tsplit(X2_scaled,y2_scaled,best_mlp)
LANDINGS_raw = pd.read_csv("../input/air-traffic-landings-statistics.csv")

LANDINGS = LANDINGS_raw.copy()

LANDINGS.head()
LANDINGS.loc[:,"Activity Period"] = pd.to_datetime(LANDINGS.loc[:,"Activity Period"].astype(str), format="%Y%m")

LANDINGS.loc[:,"Year"] = LANDINGS["Activity Period"].dt.year

LANDINGS.loc[:,"Month"] = LANDINGS["Activity Period"].dt.month
print(LANDINGS.loc[:,"Aircraft Body Type"].unique())
types = LANDINGS.groupby(["Year","Aircraft Body Type"])["Landing Count"].sum()

types = types.reset_index()



pivot_5 = types.pivot_table(values="Landing Count",index="Year",columns="Aircraft Body Type", fill_value=0)

pivot_5.plot(figsize=(12,5))

plt.show()