from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso
import matplotlib.pyplot as plt # plotting
%matplotlib inline
import seaborn as sns
import numpy as np # linear algebra
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,silhouette_samples
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.plotting.register_matplotlib_converters()
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from datetime import timedelta
from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from fbprophet import Prophet
import plotly.express as px
std=StandardScaler()
import geopandas as gpd
import folium
from folium import plugins
from folium.plugins import HeatMap
print("Setup Complete")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

covid = pd.read_csv('/kaggle/input/covid19-cases-in-so-paulo-daily-updated/dados_covid_sp.csv', delimiter=';')
covid.dataframeName = 'dados_covid_sp.csv'
covid.head(5)
covid = covid[pd.notnull(covid['codigo_ibge'])]
print("Size/Shape of the dataset: ",covid.shape)
print("Checking for null values:\n",covid.isnull().sum())
print("Checking Data-type of each column:\n",covid.dtypes)
covid['dateInt']="2020" + covid['mes'].astype(str).str.zfill(2)+ covid['dia'].astype(str).str.zfill(2)
covid['Date'] = pd.to_datetime(covid['dateInt'], format='%Y%m%d')
covid.codigo_ibge = covid.codigo_ibge.astype(str)
covid['codigo_ibge'] = covid['codigo_ibge'].str[:7]
print (covid.Date)
print(covid.codigo_ibge)
covid.head(5)
leitos_sus = pd.read_csv('/kaggle/input/leitos-sus-abril-2020/leitos_abril_2020_2.csv', delimiter=';')
covid.dataframeName = 'leitos_abril_2020_2.csv'
leitos_sus.head(5)
leitos_sus['Município']=leitos_sus['Município'].str.slice(start=6)
leitos_sus.head(5)

covid_geo = gpd.read_file("../input/so-paulo-shapefile/35MUE250GC_SIR.shp")
covid_geo.head(200)
covid = pd.merge (covid, covid_geo, how = 'left', left_on = ['codigo_ibge'], right_on = ['CD_GEOCMU'])
covid.shape
covid.head(5)
covid.rename(columns={'NM_MUNICIP':'City','casos':'Confirmed','obitos':'Deaths','latitude':'Lat','longitude':'Long','geometry':'Geometry','Quantidade_existente':'Beds'}, inplace=True)
covid.head(5)
covid.isna().sum()
covid.info
type(covid)
temp = covid.groupby('Date')['Confirmed', 'Deaths'].sum().reset_index()
temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)
temp.style.background_gradient(cmap='Pastel1')
map = folium.Map(width=800,height=500, location=[-22.164767, -48.605910], tiles='openstreetmap', zoom_start=7)
map.choropleth(geo_data=covid_geo, data=covid,
             columns=['Cases', 'Deaths'],
             key_on='feature.id',
             fill_color='YlGn', fill_opacity=0.7, line_opacity=0.2,
             legend_name='Participation Rate (%)')
map
# Create a base map
covid_heatmap = folium.Map(width=800,height=500, location=[-22.164767, -48.605910], tiles='openstreetmap', zoom_start=7)

# Add a heatmap to the base map
HeatMap(data=covid[['Lat', 'Long']], radius=15).add_to(covid_heatmap)

minimap = plugins.MiniMap()
covid_heatmap.add_child(minimap)

# Display the map
covid_heatmap
datewise=covid.groupby(["Date"]).agg({"casos":'sum',"obitos":'sum'})
datewise["Days Since"]=datewise.index-datewise.index.min()
print(datewise)
print("Basic Information")
print("Total number of cities in São Paulo with Disease Spread: ",len(covid["munic"].unique()))
print("Total number of Confirmed Cases in São Paulo: ",datewise["casos"].iloc[-1])
print("Total number of Deaths Cases in São Paulo: ",datewise["obitos"].iloc[-1])
datewise["WeekOfYear"]=datewise.index.weekofyear

week_num=[]
weekwise_confirmed=[]
weekwise_deaths=[]
w=1
for i in list(datewise["WeekOfYear"].unique()):
    weekwise_confirmed.append(datewise[datewise["WeekOfYear"]==i]["casos"].iloc[-1])
    weekwise_deaths.append(datewise[datewise["WeekOfYear"]==i]["obitos"].iloc[-1])
    week_num.append(w)
    w=w+1

plt.figure(figsize=(8,5))
plt.plot(week_num,weekwise_confirmed,linewidth=3)
plt.plot(week_num,weekwise_deaths,linewidth=3)
plt.ylabel("Number of Cases")
plt.xlabel("Week Number")
plt.title("Weekly progress of Different Types of Cases")
plt.xlabel
fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(15,5))
sns.barplot(x=week_num,y=pd.Series(weekwise_confirmed).diff().fillna(0),ax=ax1)
sns.barplot(x=week_num,y=pd.Series(weekwise_deaths).diff().fillna(0),ax=ax2)
ax1.set_xlabel("Week Number")
ax2.set_xlabel("Week Number")
ax1.set_ylabel("Number of Confirmed Cases")
ax2.set_ylabel("Number of Death Cases")
ax1.set_title("Weekly increase in Number of Confirmed Cases")
ax2.set_title("Weekly increase in Number of Death Cases")
print("Average increase in number of Confirmed Cases every day: ",np.round(datewise["casos"].diff().fillna(0).mean()))
print("Average increase in number of Deaths Cases every day: ",np.round(datewise["obitos"].diff().fillna(0).mean()))

plt.figure(figsize=(15,6))
plt.plot(datewise["casos"].diff().fillna(0),label="Daily increase in Confirmed Cases",linewidth=3)
plt.plot(datewise["obitos"].diff().fillna(0),label="Daily increase in Death Cases",linewidth=3)
plt.xlabel("Timestamp")
plt.ylabel("Daily Increment")
plt.title("Daily increase in different Types of Cases in São Paulo state")
plt.xticks(rotation=90)
plt.legend()
citywise=covid[covid["Date"]==covid["Date"].max()].groupby(["munic"]).agg({"casos":'sum',"obitos":'sum'}).sort_values(["casos"],ascending=False)
citywise["Mortality"]=(citywise["obitos"]/citywise["casos"])*100
fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(10,12))
top_15_confirmed=citywise.sort_values(["casos"],ascending=False).head(15)
top_15_deaths=citywise.sort_values(["obitos"],ascending=False).head(15)
sns.barplot(x=top_15_confirmed["casos"],y=top_15_confirmed.index,ax=ax1)
ax1.set_title("Top 15 cities as per Number of Confirmed Cases")
ax1.set_xlabel("Cases")
ax1.set_ylabel("Cities")
sns.barplot(x=top_15_deaths["obitos"],y=top_15_deaths.index,ax=ax2)
ax2.set_title("Top 15 cities as per Number of Death Cases")
ax2.set_xlabel("Cases")
ax2.set_ylabel("Cities")
# Set the width and height of the figure
plt.figure(figsize=(14,7))

# Add title
plt.title("Average Arrival Delay for Each Airline, by Month")

# Heatmap showing average arrival delay for each airline by month
temp=citywise.drop('obitos',1)
temp=temp.drop('Mortality',1)
sns.heatmap(temp.head(20), annot=True)

# Add label for horizontal axis
plt.xlabel("Airline")
citywise=covid[covid["Date"]==covid["Date"].max()].groupby(["munic"]).agg({"casos":'sum',"obitos":'sum'}).sort_values(["casos"],ascending=False)
citywise["Mortality"]=(citywise["obitos"]/citywise["casos"])*100
datewise["Mortality Rate"]=(datewise["obitos"]/datewise["casos"])*100

#Plotting Mortality and Recovery Rate 
fig, (ax1) = plt.subplots(1,figsize=(12,6))
ax1.plot(datewise["Mortality Rate"],label='Mortality Rate',linewidth=3)
ax1.axhline(datewise["Mortality Rate"].mean(),linestyle='--',color='black',label="Mean Mortality Rate")
ax1.set_ylabel("Mortality Rate")
ax1.set_xlabel("Timestamp")
ax1.set_title("Overall Datewise Mortality Rate")
ax1.legend()
for tick in ax1.get_xticklabels():
    tick.set_rotation(90)  
print("Average Mortality Rate",datewise["Mortality Rate"].mean())
print("Median Mortality Rate",datewise["Mortality Rate"].median())
###Top 25 Cities as per Mortatlity Rate with more than 500 Confirmed Cases
fig, (ax1) = plt.subplots(1,figsize=(10,8))
citywise_plot_mortal=citywise[citywise["casos"]>500].sort_values(["Mortality"],ascending=False).head(15)
sns.barplot(x=citywise_plot_mortal["Mortality"],y=citywise_plot_mortal.index,ax=ax1)
ax1.set_title("Top 15 Cities according High Mortatlity Rate")
ax1.set_xlabel("Mortality (in Percentage)")
ax1.set_ylabel("Cities")
X=citywise[["casos","obitos"]]
X=std.fit_transform(X)
wcss=[]
sil=[]
for i in range(2,11):
    clf=KMeans(n_clusters=i,init='k-means++',random_state=42)
    clf.fit(X)
    labels=clf.labels_
    centroids=clf.cluster_centers_
    sil.append(silhouette_score(X, labels, metric='euclidean'))
    wcss.append(clf.inertia_)
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(16,5))
x=np.arange(2,11)
ax1.plot(x,wcss,marker='o')
ax1.set_xlabel("Number of Clusters")
ax1.set_ylabel("Within Cluster Sum of Squares (WCSS)")
ax1.set_title("Elbow Method")
x=np.arange(2,11)
ax2.plot(x,sil,marker='o')
ax2.set_xlabel("Number of Clusters")
ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Score Method")
clf_final=KMeans(n_clusters=4,init='k-means++',random_state=42)
clf_final.fit(X)
citywise["Clusters"]=clf_final.predict(X)
cluster_summary=pd.concat([citywise[citywise["Clusters"]==1].head(6),citywise[citywise["Clusters"]==2].head(6),citywise[citywise["Clusters"]==3].head(6),citywise[citywise["Clusters"]==4].head(6),citywise[citywise["Clusters"]==0].head(6)])
cluster_summary.style.background_gradient(cmap='Reds')
datewise["Days Since"]=datewise.index-datewise.index[0]
datewise["Days Since"]=datewise["Days Since"].dt.days
train_ml=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid_ml=datewise.iloc[int(datewise.shape[0]*0.95):]
model_scores=[]
lin_reg=LinearRegression(normalize=True)
lin_reg.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["casos"]).reshape(-1,1))
prediction_valid_linreg=lin_reg.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))
model_scores.append(np.sqrt(mean_squared_error(valid_ml["casos"],prediction_valid_linreg)))
print("Root Mean Square Error for Linear Regression: ",np.sqrt(mean_squared_error(valid_ml["casos"],prediction_valid_linreg)))
plt.figure(figsize=(11,6))
prediction_linreg=lin_reg.predict(np.array(datewise["Days Since"]).reshape(-1,1))
plt.plot(datewise["casos"],label="Actual Confirmed Cases")
plt.plot(datewise.index,prediction_linreg, linestyle='--',label="Predicted Confirmed Cases using Linear Regression",color='black')
plt.xlabel('Time')
plt.ylabel('Confirmed Cases')
plt.title("Confirmed Cases Linear Regression Prediction")
plt.xticks(rotation=90)
plt.legend()
train_ml=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid_ml=datewise.iloc[int(datewise.shape[0]*0.95):]
poly = PolynomialFeatures(degree = 4) 
train_poly=poly.fit_transform(np.array(train_ml["Days Since"]).reshape(-1,1))
valid_poly=poly.fit_transform(np.array(valid_ml["Days Since"]).reshape(-1,1))
y=train_ml["casos"]
linreg=LinearRegression(normalize=True)
linreg.fit(train_poly,y)
prediction_poly=linreg.predict(valid_poly)
rmse_poly=np.sqrt(mean_squared_error(valid_ml["casos"],prediction_poly))
model_scores.append(rmse_poly)
print("Root Mean Squared Error for Polynomial Regression: ",rmse_poly)
comp_data=poly.fit_transform(np.array(datewise["Days Since"]).reshape(-1,1))
plt.figure(figsize=(11,6))
predictions_poly=linreg.predict(comp_data)
plt.plot(datewise["casos"],label="Train Confirmed Cases",linewidth=3)
plt.plot(datewise.index,predictions_poly, linestyle='--',label="Best Fit for Polynomial Regression",color='black')
plt.xlabel('Time')
plt.ylabel('Confirmed Cases')
plt.title("Confirmed Cases Polynomial Regression Prediction")
plt.xticks(rotation=90)
plt.legend()
new_prediction_poly=[]
for i in range(1,18):
    new_date_poly=poly.fit_transform(np.array(datewise["Days Since"].max()+i).reshape(-1,1))
    new_prediction_poly.append(linreg.predict(new_date_poly)[0])
train_ml=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid_ml=datewise.iloc[int(datewise.shape[0]*0.95):]
svm=SVR(C=1,degree=5,kernel='poly',epsilon=0.01)
svm.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["casos"]).reshape(-1,1))
prediction_valid_svm=svm.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))
plt.figure(figsize=(11,6))
prediction_svm=svm.predict(np.array(datewise["Days Since"]).reshape(-1,1))
plt.plot(datewise["casos"],label="Train Confirmed Cases",linewidth=3)
plt.plot(datewise.index,prediction_svm, linestyle='--',label="Best Fit for SVR",color='black')
plt.xlabel('Time')
plt.ylabel('Confirmed Cases')
plt.title("Confirmed Cases Support Vector Machine Regressor Prediction")
plt.xticks(rotation=90)
plt.legend()
model_scores.append(np.sqrt(mean_squared_error(valid_ml["casos"],prediction_valid_svm)))
print("Root Mean Square Error for Support Vectore Machine: ",np.sqrt(mean_squared_error(valid_ml["casos"],prediction_valid_svm)))
plt.figure(figsize=(11,6))
prediction_svm=svm.predict(np.array(datewise["Days Since"]).reshape(-1,1))
plt.plot(datewise["casos"],label="Train Confirmed Cases",linewidth=3)
plt.plot(datewise.index,prediction_svm, linestyle='--',label="Best Fit for SVR",color='black')
plt.xlabel('Time')
plt.ylabel('casos')
plt.title("Confirmed Cases Support Vector Machine Regressor Prediction")
plt.xticks(rotation=90)
plt.legend()
new_date=[]
new_prediction_lr=[]
new_prediction_svm=[]
for i in range(1,18):
    new_date.append(datewise.index[-1]+timedelta(days=i))
    new_prediction_lr.append(lin_reg.predict(np.array(datewise["Days Since"].max()+i).reshape(-1,1))[0][0])
    new_prediction_svm.append(svm.predict(np.array(datewise["Days Since"].max()+i).reshape(-1,1))[0])
pd.set_option('display.float_format', lambda x: '%.6f' % x)
model_predictions=pd.DataFrame(zip(new_date,new_prediction_lr,new_prediction_poly,new_prediction_svm),
                               columns=["Dates","Linear Regression Prediction","Polynonmial Regression Prediction","SVM Prediction"])
model_predictions.head()
model_train=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid=datewise.iloc[int(datewise.shape[0]*0.95):]
holt=Holt(np.asarray(model_train["casos"])).fit(smoothing_level=0.2, smoothing_slope=0.1,optimized=False)
y_pred=valid.copy()
y_pred["Holt"]=holt.forecast(len(valid))
model_scores.append(np.sqrt(mean_squared_error(y_pred["casos"],y_pred["Holt"])))
print("Root Mean Square Error Holt's Linear Model: ",np.sqrt(mean_squared_error(y_pred["casos"],y_pred["Holt"])))
plt.figure(figsize=(10,5))
plt.plot(model_train.casos,label="Train Set",marker='o')
valid.casos.plot(label="Validation Set",marker='*')
y_pred.Holt.plot(label="Holt's Linear Model Predicted Set",marker='^')
plt.ylabel("Confirmed Cases")
plt.xlabel("Date Time")
plt.title("Confirmed Holt's Linear Model Prediction")
plt.xticks(rotation=90)
plt.legend()
holt_new_date=[]
holt_new_prediction=[]
for i in range(1,18):
    holt_new_date.append(datewise.index[-1]+timedelta(days=i))
    holt_new_prediction.append(holt.forecast((len(valid)+i))[-1])

model_predictions["Holt's Linear Model Prediction"]=holt_new_prediction
model_predictions.head()
model_train=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid=datewise.iloc[int(datewise.shape[0]*0.95):]
y_pred=valid.copy()
es=ExponentialSmoothing(np.asarray(model_train['casos']),seasonal_periods=5,trend='mul', seasonal='mul').fit()
y_pred["Holt's Winter Model"]=es.forecast(len(valid))
model_scores.append(np.sqrt(mean_squared_error(y_pred["casos"],y_pred["Holt's Winter Model"])))
print("Root Mean Square Error for Holt's Winter Model: ",np.sqrt(mean_squared_error(y_pred["casos"],y_pred["Holt's Winter Model"])))
plt.figure(figsize=(10,5))
plt.plot(model_train.casos,label="Train Set",marker='o')
valid.casos.plot(label="Validation Set",marker='*')
y_pred["Holt\'s Winter Model"].plot(label="Holt's Winter Model Predicted Set",marker='^')
plt.ylabel("Confirmed Cases")
plt.xlabel("Date Time")
plt.title("Confiremd Cases Holt's Winter Model Prediction")
plt.xticks(rotation=90)
plt.legend()
holt_winter_new_prediction=[]
for i in range(1,18):
    holt_winter_new_prediction.append(es.forecast((len(valid)+i))[-1])
model_predictions["Holt's Winter Model Prediction"]=holt_winter_new_prediction
model_predictions.head()
model_train=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid=datewise.iloc[int(datewise.shape[0]*0.95):]
y_pred=valid.copy()
prophet_c=Prophet(interval_width=0.95,weekly_seasonality=True,)
prophet_confirmed=pd.DataFrame(zip(list(datewise.index),list(datewise["casos"])),columns=['ds','y'])
prophet_c.fit(prophet_confirmed)
forecast_c=prophet_c.make_future_dataframe(periods=30)
forecast_confirmed=forecast_c.copy()
confirmed_forecast=prophet_c.predict(forecast_c)
model_scores.append(np.sqrt(mean_squared_error(datewise["casos"],confirmed_forecast['yhat'].head(datewise.shape[0]))))
print("Root Mean Squared Error for Prophet Model: ",np.sqrt(mean_squared_error(datewise["casos"],confirmed_forecast['yhat'].head(datewise.shape[0]))))
print(prophet_c.plot(confirmed_forecast))
print(prophet_c.plot_components(confirmed_forecast))
prophet_c=Prophet(interval_width=0.95,weekly_seasonality=True,)
prophet_deaths=pd.DataFrame(zip(list(datewise.index),list(datewise["obitos"])),columns=['ds','y'])
prophet_c.fit(prophet_deaths)
forecast_c=prophet_c.make_future_dataframe(periods=30)
forecast_deaths=forecast_c.copy()
deaths_forecast=prophet_c.predict(forecast_c)
model_scores.append(np.sqrt(mean_squared_error(datewise["obitos"],deaths_forecast['yhat'].head(datewise.shape[0]))))
print("Root Mean Squared Error for Prophet Model: ",np.sqrt(mean_squared_error(datewise["obitos"],deaths_forecast['yhat'].head(datewise.shape[0]))))
print(prophet_c.plot(deaths_forecast))
print(prophet_c.plot_components(deaths_forecast))
model_names=["Linear Regression","Polynomial Regression","Support Vector Machine Regressor","Holt's Linear","Holt's Winter Model","Facebook's Prophet Model"]
model_summary=pd.DataFrame(zip(model_names,model_scores),columns=["Model Name","Root Mean Squared Error"]).sort_values(["Root Mean Squared Error"])
model_summary
model_predictions["Prophet's Prediction"]=list(confirmed_forecast["yhat"].tail(17))
model_predictions["Prophet's Upper Bound"]=list(confirmed_forecast["yhat_upper"].tail(17))
model_predictions.head()