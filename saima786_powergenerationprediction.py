import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
%matplotlib inline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import make_scorer, r2_score
%matplotlib inline
weather_actuals=pd.read_csv("/kaggle/input/climateconnect/weather_actuals.csv")
wf=pd.read_csv("/kaggle/input/climateconnect/weather_forecast.csv")
weather_actuals=weather_actuals.iloc[:,3:32]
weather_actuals.head()
weather_actuals.rename(columns={'datetime_local':'datetime'},inplace=True)
weather_actuals.precip_type.unique()
weather_actuals["precip_type"].fillna("no rain",inplace=True)
weather_actuals.precip_type.unique()
power_actuals=pd.read_csv("/kaggle/input/climateconnect/power_actual.csv")
power_actuals
import seaborn as sns
sns.scatterplot(x=power_actuals.index[4],y='power',data=power_actuals);
weather_actuals
merged = pd.merge(weather_actuals,power_actuals,on='datetime',how='right',sort=True) #merge target(power_actuals) and features(weather_actuals) dataset
merged
merged.fillna(method="ffill",inplace=True)
merged=merged.replace([-9999,-9999.0,-9999.00,'-9999'],np.nan) #Removing all the negative values with null values
merged.shape
merged.dropna(how='all',axis=1,inplace=True)
merged.shape
merged.dropna(inplace=True)
merged.shape
merged.columns
numericalFeatures = merged.select_dtypes(include = [np.number])
print("The number of numerical features is: {}".format(numericalFeatures.shape[1]))
numericalFeatures.columns
categoricalFeatures = merged.select_dtypes(exclude = [np.number])
print("The number of categorical features is: {}".format(categoricalFeatures.shape[1]))
categoricalFeatures.columns
merged['precip_type_en'] = LabelEncoder().fit_transform(merged['precip_type'])
merged[['precip_type', 'precip_type_en']]
merged['icon_encoded'] = LabelEncoder().fit_transform(merged['icon'])
merged[['icon', 'icon_encoded']] # special syntax to get just these two columns
merged['summary_encoded'] = LabelEncoder().fit_transform(merged['summary'])
merged[['summary', 'summary_encoded']]
merged['sunset']=pd.to_datetime(merged['sunset'])
merged['sunrise']=pd.to_datetime(merged['sunrise'])
merged['datetime']=pd.to_datetime(merged['datetime'])
merged["daylight"]=(merged["sunset"]-merged['sunrise']).dt.total_seconds() #to get the daylight time in seconds
merged
merged.drop(['precip_type','icon', 'summary','updated_at','sunrise','sunset','Unnamed: 0'],axis=1,inplace=True)
print(merged.isnull().sum())
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors=50, contamination='auto')
merged.describe()
merged.drop(merged[merged['power']>40].index,inplace=True)
merged.info()
merged.describe()
merged.shape
merged.drop(['dew_point','wind_bearing','apparent_temperature','wind_gust','precip_probability','ghi','gti','pressure','uv_index'],axis=1,inplace=True)
merged.plot(subplots=True,figsize = (10,15) )
merged.columns
corrmat = merged.corr(method='spearman')
f, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corrmat, ax=ax, cmap="YlGnBu", linewidths=0.1)
merged.info()
data=merged[['datetime','cloud_cover', 'temperature', 'humidity',
       'wind_speed', 'ozone',
       'precip_intensity', 'visibility', 'power',
       'precip_type_en', 'icon_encoded', 'summary_encoded', 'daylight']]
X=data.drop(['datetime','power'],axis=1)
y=data['power']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
sc = StandardScaler() # standardizing a feature by subtracting the mean and then scaling to unit variance.
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)
def test_model(model, x_train, y_train):
    cv = KFold(n_splits = 3, shuffle=True, random_state = 45)
    r2 = make_scorer(r2_score)
    r2_val_score = cross_val_score(model, x_train, y_train, cv=cv, scoring = r2)
    score = [r2_val_score.mean()]
    return score
def rsme(model, x, y):
    cv_scores = -cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv=10)
    return np.sqrt(cv_scores)
LR = LinearRegression()
acc_LR = test_model(LR, X_train, y_train)

LR_rsme = rsme(LR, X_train, y_train)


print('Score: {:.5f}'.format((acc_LR[0])))
print('RSME: {:.5f}'.format(LR_rsme.mean()))
LR.fit(X_train,y_train)
y_pred1=LR.predict(X_test)
for i in range(len(y_pred1)):
    if y_pred1[i]<1:
        y_pred1[i]=0
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred1})
df.head()
svr_reg = SVR(kernel='rbf')
acc_SVR = test_model(svr_reg, X_train, y_train)

svr_rsme = rsme(svr_reg, X_train, y_train)
print('Score: {:.5f}'.format((acc_SVR[0])))
print('RSME: {:.5f}'.format(svr_rsme.mean()))
svr_reg.fit(X_train,y_train)
y_pred2=svr_reg.predict(X_test)
for i in range(len(y_pred2)):
    if y_pred2[i]<1:
        y_pred2[i]=0
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred2})
df.head()
dt_reg = DecisionTreeRegressor(random_state=21)
acc_tree = test_model(dt_reg, X_train, y_train)

dt_rsme = rsme(dt_reg, X_train, y_train)
print('Score: {:.5f}'.format((acc_tree[0])))
print('RSME: {:.5f}'.format(dt_rsme.mean()))
dt_reg.fit(X_train,y_train)
y_pred3=dt_reg.predict(X_test)
for i in range(len(y_pred3)):
    if y_pred3[i]<1:
        y_pred3[i]=0
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred3})
df.tail(20)
from matplotlib.ticker import FuncFormatter,MaxNLocator
fig,ax=plt.subplots()
ax=fig.add_axes([0,0,1,1])
ax.grid(True)
ax.xaxis.set_major_locator(plt.MaxNLocator(30))
ax.set_xlabel('Date_Time')
ax.set_ylabel('Power_Predicted')
ax.plot(merged.index[0:45],y_pred3[0:45])
plt.xticks(rotation='vertical')
rf_reg = RandomForestRegressor(n_estimators =10, random_state=51)
acc_rf = test_model(rf_reg, X_train, y_train)

rf_rsme = rsme(rf_reg, X_train, y_train)
print('Score: {:.5f}'.format((acc_rf[0])))
print('RSME: {:.5f}'.format(rf_rsme.mean()))
rf_reg.fit(X_train,y_train)
y_pred4=rf_reg.predict(X_test)
for i in range(len(y_pred4)):
    if y_pred4[i]<1:
        y_pred4[i]=0
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred4})
df.head()
from matplotlib.ticker import FuncFormatter,MaxNLocator
fig,ax=plt.subplots()
ax=fig.add_axes([0,0,1,1])
ax.grid(True)
ax.xaxis.set_major_locator(plt.MaxNLocator(30))
ax.set_xlabel('Date_Time')
ax.set_ylabel('Power_Predicted')
ax.plot(merged.index[0:45],y_pred4[0:45])
plt.xticks(rotation='vertical')
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Support Vector Regressor', 
              'Decision Tree', 'Random Forest'],
    'Score': [acc_LR[0], acc_SVR[0], acc_tree[0], acc_rf[0]],
    'RSME': [LR_rsme[0], svr_rsme[0], dt_rsme[0], rf_rsme[0]]
             })

result = results.sort_values(by='RSME', ascending=True)
result = result.set_index('Model')
display(result.head(8))
wf=pd.read_csv("/kaggle/input/climateconnect/weather_forecast.csv")
wf=wf.iloc[:,3:]
wf.shape
wf.info()
wf.describe()
wf.precip_type.unique()
wf["precip_type"].fillna("no rain",inplace=True)
wf.precip_type.unique()
wf['icon_encoded'] = LabelEncoder().fit_transform(wf['icon'])
wf[['icon', 'icon_encoded']] # special syntax to get just these two columns
wf['summary_encoded'] = LabelEncoder().fit_transform(wf['summary'])
wf[['summary', 'summary_encoded']] # special syntax to get just these two columns
wf['precip_type_en'] = LabelEncoder().fit_transform(wf['precip_type'])
wf[['precip_type', 'precip_type_en']]
wf['sunset']=pd.to_datetime(wf['sunset'])
wf['sunrise']=pd.to_datetime(wf['sunrise'])

wf["daylight"]=(wf["sunset"]-wf['sunrise']).dt.total_seconds()
wf.columns
wf.rename(columns={'datetime_local':'datetime'},inplace=True)
#wf.set_index('datetime', inplace=True) 
wf.drop(['dew_point','wind_chill','heat_index','pressure','uv_index','wind_bearing','qpf','snow','pop','fctcode','precip_accumulation','precip_type','sunrise','sunset','icon','summary','updated_at'],axis=1,inplace=True)
wf.head()
wf.drop(['apparent_temperature','wind_gust','precip_probability'],axis=1,inplace=True)
print(wf.isnull().sum())
wf=wf.dropna()
wf.set_index('datetime', inplace=True)
wf['output']=dt_reg.predict(wf)
pred=wf['output']
for i in range(len(pred)):
    if pred[i]<1:
        pred[i]=0
prediction=wf[["output" ]]
prediction.rename(columns={'output':'power_predicted'},inplace=True)
prediction
submission = pd.DataFrame(prediction)
submission.to_csv("submission.csv", header=True)
submission
