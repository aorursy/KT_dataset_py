import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.ticker as ticker
import plotly.graph_objs as go
import missingno as msno
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
import descartes
import geopandas as gpd
from shapely.geometry import Point, Polygon
%matplotlib inline
cases_in_india = pd.read_csv("../input/coronavirus-cases-in-india/Covid cases in India.csv")
coordinates = pd.read_csv("../input/coronavirus-cases-in-india/Indian Coordinates.csv")
per_day = pd.read_excel("../input/coronavirus-cases-in-india/per_day_cases.xlsx")

age_group = pd.read_csv("../input/covid19-in-india/AgeGroupDetails.csv")
bedsIndia = pd.read_csv("../input/covid19-in-india/HospitalBedsIndia.csv")
testLabs = pd.read_csv("../input/covid19-in-india/ICMRTestingLabs.csv")
ind_detail = pd.read_csv("../input/covid19-in-india/IndividualDetails.csv")
statewise = pd.read_csv("../input/covid19-in-india/StatewiseTestingDetails.csv")
cov_india = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")
census2011 = pd.read_csv("../input/covid19-in-india/population_india_census2011.csv")
district = pd.read_csv("../input/district-dataset/district_level_latest.csv")
pnb_latlng = pd.read_csv("../input/punjab-latlng/punjab_lat_lng - Sheet1.csv")
census = census2011[["State / Union Territory", "Density"]]
# statewis = statewise.groupby("State").tail(1).reset_index()
statewis = statewise[statewise.groupby(['State'])['Positive'].transform(max) == statewise['Positive']]

den_pop = pd.merge(census, statewis, how = "inner", left_on = "State / Union Territory", right_on = "State")
den_pop["Density"] = den_pop["Density"].apply(lambda x : x.replace(".", ""))
den_pop["Density"] = den_pop["Density"].apply(lambda x : int(x.split("(")[0].split("/")[0].replace(",", "")))
den_pop
plt.figure(figsize = (6, 6))

sns.scatterplot("Density", "Positive", data = den_pop, s = 80, color = "red")
plt.xlim([0, 2000])
plt.ylim([0, 30000])
sns.set(font_scale = 1.5)
plt.title("Variation of Covid-19 cases with Population density", pad = 50)
sns.set(font_scale = 1)
plt.xlabel("Density")
plt.ylabel("Positive")
sns.despine()
ind_detail.head(1)
plt.figure(figsize = (5, 5))
sns.countplot(x = "gender", data = ind_detail)
ind_detail = ind_detail[(ind_detail["age"] != '28-35')]
ind_detail = ind_detail[~ind_detail['age'].isnull()]
ind_detail["age"] = ind_detail["age"].apply(float)
age_category = pd.cut(ind_detail.age,bins=[0,10,22,35,50,70,100],labels=['0-10','10-22','22-35','35-50', '50-70','70-100'])
ind_detail["age"] = age_category
data = ind_detail.copy()
index = data.groupby("age")["age"].count().index
# values = data.groupby("age")["age"].count().values
percent_values = []
for age_gp in data["age"].unique():
    total = len(data[data["current_status"] == "Recovered"])
    recovered = len(data[(data["age"] == age_gp) & (data["current_status"] == "Recovered")])
    percent_values.append((recovered/total) * 100)
    
    
plt.figure(figsize = (10, 6))
plt.xlabel("Age Groups")
plt.ylabel("Percentage of Recovered")
sns.barplot(x = index, y = percent_values)
plt.title("Recovered vs Age Group")
data["current_status"].unique()
percent_values = []
for age_gp in data["age"].unique():
    total = len(data[data["current_status"] == "Hospitalized"])
    recovered = len(data[(data["age"] == age_gp) & (data["current_status"] == "Hospitalized")])
    percent_values.append((recovered/total) * 100)
    
    
plt.figure(figsize = (10, 6))
plt.xlabel("Age Groups")
plt.ylabel("Percentage of hospitalized")
sns.barplot(x = index, y = percent_values)
plt.title("Hospitalized vs Age Group")
percent_values = []
for age_gp in data["age"].unique():
    total = len(data[data["current_status"] == "Deceased"])
    recovered = len(data[(data["age"] == age_gp) & (data["current_status"] == "Deceased")])
    percent_values.append((recovered/total) * 100)
    
    
plt.figure(figsize = (10, 6))
plt.xlabel("Age Groups")
plt.ylabel("Percentage of Deceased")
sns.barplot(x = index, y = percent_values)
plt.title("Deceased vs Age Group")
plt.figure(figsize = (15, 8))
sns.countplot(x = "age", data = ind_detail)
plt.xticks(rotation = 90)
punjab = district[district["state name"] == "Punjab"].reset_index(drop = True)
punjab.head()
plt.figure(figsize =(12, 7))
plt.title("Confirmed cases in Punjab districts")
ax = sns.barplot(x = "confirmed", y = "district", data = punjab.sort_values(by = "confirmed", ascending = False))
ax.set_xlabel("Confirmed cases")
plt.figure(figsize =(15, 8))
plt.title("Recovered cases in Punjab districts")
ax = sns.barplot(x = "recovered", y = "district", data = punjab.sort_values(by = "recovered", ascending = False))
ax.set_xlabel("recovered cases")
pnb_latlng["Lat"] = pnb_latlng["Lat"].apply(lambda x : x[:2])
pnb_latlng["Lng"] = pnb_latlng["Lng"].apply(lambda x : x[:2])
punjab = pd.merge(punjab, pnb_latlng, left_on = "district", right_on = "State", how = "inner")
punjab
fig = go.Figure()
fig.add_trace(go.Scatter(x=per_day['Date'], y=per_day['Total Cases'],
                    mode='lines+markers',name='Total Cases'))

fig.add_trace(go.Scatter(x=per_day['Date'], y=per_day['New Cases'], 
                mode='lines',name='New Cases'))

fig.update_layout(title_text='Trend of Coronavirus Cases in India(Cumulative cases)',plot_bgcolor='rgb(250, 242, 242)')

fig.show()
train = per_day.iloc[:85, 1:2].values
test = per_day.iloc[85:, 1:2].values

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
sc = MinMaxScaler(feature_range = (0, 1))
train_scaled = sc.fit_transform(train)

X_train = []
y_train = []
for i in range(20, len(train)):
    X_train.append(train_scaled[i - 20:i])
    y_train.append(train_scaled[i])

X_train = np.array(X_train)
y_train = np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


reg = Sequential()
reg.add(LSTM(units = 30, return_sequences = True, input_shape = (X_train.shape[1], 1)))
reg.add(Dropout(0.2))

reg.add(LSTM(units = 30, return_sequences = True))
reg.add(Dropout(0.2))

reg.add(LSTM(units = 30, return_sequences = True))
reg.add(Dropout(0.2))

reg.add(LSTM(units = 30, return_sequences = False))
reg.add(Dropout(0.2))

reg.add(Dense(units = 1))


reg.compile(optimizer = "adam", loss = "mean_squared_error")
history = reg.fit(X_train, y_train, epochs = 25, batch_size = 8)



total_data = per_day.iloc[:, 1:2]
inputs = total_data[len(total_data) - len(test) - 20:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(20, 20 + len(test)):
    X_test.append(inputs[i - 20:i])
    

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))



predictions = reg.predict(X_test)
predictions = sc.inverse_transform(predictions)

fig, ax = plt.subplots(1,1,figsize=(10,8))
ax.plot(np.arange(len(test)), test, label = "actual")
ax.plot(np.arange(len(test)), predictions, label = "predictions")
ax.set_xlabel("Days")
ax.set_ylabel("Actual vs Predicted")
plt.legend(loc = 2)
def mean_sq_error(pred, actual):
    mse = mean_squared_error(actual, pred)
    return np.sqrt(mse)
print(f"Root mean square error is equal to {mean_sq_error(predictions, test)}")
X_train = np.arange(1, len(per_day) + 1).reshape(-1, 1)
y_train = per_day.iloc[:, 1].values.reshape(-1, 1)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
polynomial_features= PolynomialFeatures(degree=3)
x_poly = polynomial_features.fit_transform(X_train)
model = LinearRegression()
model.fit(x_poly, y_train)
y_poly_pred = model.predict(x_poly)
plt.figure(figsize = (15, 8))
plt.plot(X_train, y_train, label = "actual")
plt.plot(X_train, y_poly_pred, label = "predictions")
plt.xlabel("Days")
plt.ylabel("Actual vs Predicted")
plt.legend(loc = 2)
rmse = np.sqrt(mean_squared_error(y_train, y_poly_pred))
print(f"Root Mean Square Error value is {rmse}")
X_test = np.arange(96, 96 + 60).reshape(-1, 1)
x_test_poly = polynomial_features.transform(X_test)
predictions = model.predict(x_test_poly)

dates = pd.date_range(start='5/4/2020', periods = 60)
df = pd.DataFrame()
df["Date"] = dates
df["predictions"] = predictions

df["predictions"] = df["predictions"].apply(lambda x : int(x))

print(f"Predictions for June 15: {df[df['Date'] == '2020-06-15'].values[0][1]}")
print(f"Predictions for June 30: {df[df['Date'] == '2020-06-30'].values[0][1]}")
data = per_day["Total Cases"].values
forecast_t = []
A_prev = 0
F_prev = 0
alpha = 0.75
for i in range(len(data)):
    F_t = alpha * A_prev + (1 - alpha) * F_prev
    forecast_t.append(int(round(F_t)))
    F_prev = F_t
    A_prev = data[i]
plt.figure(figsize = (15, 8))
plt.plot(np.arange(len(data)), data, label = "actual")
plt.plot(np.arange(len(data)), np.array(forecast_t), label = "predictions")
plt.xlabel("Days")
plt.ylabel("Actual vs Predicted")
plt.legend(loc = 2)
rmse = np.sqrt(mean_squared_error(data, np.array(forecast_t)))
print(f"Root Mean Square Error value is {rmse}")
per_day["shifted"] = per_day["Total Cases"].shift(-1)
per_day["growth_ratio"] = per_day["shifted"]/per_day["Total Cases"]
growth_ratio = per_day["growth_ratio"].median()
predictions = []
initial = per_day.iloc[-1, 1]
for i in range(60):
    new_pred = initial * growth_ratio
    predictions.append(int(round(new_pred)))
    initial = new_pred
dates = pd.date_range(start='5/4/2020', periods = 60)
df = pd.DataFrame()
df["Date"] = dates
df["predictions"] = predictions


print(f"Predictions for June 15: {df[df['Date'] == '2020-06-15'].values[0][1]}")
print(f"Predictions for June 30: {df[df['Date'] == '2020-06-30'].values[0][1]}")
plt.figure(figsize = (15, 8))
plt.xlabel("Next 60 days")
plt.ylabel("Total cases predicted")

plt.plot(predictions)
