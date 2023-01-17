import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns



import plotly.graph_objects as go
import plotly.express as px
data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
data.head()
data.info()
data['ObservationDate'] = pd.to_datetime(data['ObservationDate'])
#Türkiye verilerinin alınması
data_turkey = data[data['Country/Region'] == 'Turkey'] 
data_turkey = data_turkey.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
#Hafta sütununun eklenmesi
data_turkey["WeekofYear"]=data_turkey.index.weekofyear 
data_turkey["Days Since"]=(data_turkey.index-data_turkey.index[0])
data_turkey["Days Since"]=data_turkey["Days Since"].dt.days

data_turkey = data_turkey.reset_index()
data_turkey.head()
fig = px.bar(data_turkey, x='ObservationDate', y='Confirmed',color='Confirmed', height=500)
fig.update_layout(title='Türkiyedeki vaka sayısı',
                 xaxis_title="Tarih",
                 yaxis_title="Vaka Sayısı")
print("Tarihinde",data_turkey["ObservationDate"].iloc[-1], " toplam vaka sayısı ",data_turkey["Confirmed"].iloc[-1])
fig = px.bar(data_turkey, x='ObservationDate', y='Deaths',color='Deaths',template='ggplot2', height=500)
fig.update_layout(title='Türkiyedeki ölümle sonuçlanan vaka sayısı',
                 xaxis_title="Tarih",
                 yaxis_title="Ölümler")
print("Tarihinde ",data_turkey["ObservationDate"].iloc[-1], " ölümle sonuçlanan vaka sayısı ",data_turkey["Deaths"].iloc[-1])
fig = px.bar(data_turkey, x='ObservationDate', y='Recovered',color='Recovered',template='plotly_white', height=500)
fig.update_layout(title='Türkiyede iyileşen vaka sayısı',
                 xaxis_title="Tarih",
                 yaxis_title="İyileşen vaka")
print("Tarihiyle ",data_turkey["ObservationDate"].iloc[-1], " iyileşen vaka sayısı ",data_turkey["Recovered"].iloc[-1])
fig=go.Figure()
fig.add_trace(go.Scatter(x=data_turkey['ObservationDate'], y=data_turkey["Confirmed"],
                    mode='lines+markers',
                    name='Vaka'))
fig.add_trace(go.Scatter(x=data_turkey['ObservationDate'], y=data_turkey["Recovered"],
                    mode='lines+markers',
                    name='İyileşen vaka'))
fig.add_trace(go.Scatter(x=data_turkey['ObservationDate'], y=data_turkey["Deaths"],
                    mode='lines+markers',
                    name='Ölümle sonuçlanan vaka'))
fig.update_layout(title="Toplam vs İyileşen vs Ölümle sonuçlanan Türkiye'deki vaka sayıları",
                 xaxis_title="Date",yaxis_title="Vaka sayısı",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()
cases = 1
double_days=[]
C=[]
while(1):
    double_days.append(int(data_turkey[data_turkey["Confirmed"]<=cases].iloc[[-1]]["Days Since"]))
    C.append(cases)
    cases=cases*2
    if(cases<data_turkey["Confirmed"].max()):
        continue
    else:
        break
        
turkey_doubling=pd.DataFrame(list(zip(C,double_days)),columns=["Vaka Sayısı","İlk vakadan bu yana gün"])
turkey_doubling["Vakaları ikiye katlamak için gereken gün sayısı"]=turkey_doubling["İlk vakadan bu yana gün"].diff().fillna(turkey_doubling["İlk vakadan bu yana gün"].iloc[0])
turkey_doubling.loc[turkey_doubling['Vaka Sayısı']==1, 'Vakaları ikiye katlamak için gereken gün sayısı'] = 0
turkey_doubling.style.background_gradient(cmap='Reds')
data_turkey['Active'] = data_turkey['Confirmed'] - data_turkey['Recovered'] - data_turkey['Deaths']
fig=go.Figure(data=go.Pie(labels=['Aktif','İyileşen','Ölen'],
                values=[data_turkey.iloc[data_turkey['ObservationDate'].idxmax(axis=1)]['Active'],
                        data_turkey.iloc[data_turkey['ObservationDate'].idxmax(axis=1)]['Recovered'],
                        data_turkey.iloc[data_turkey['ObservationDate'].idxmax(axis=1)]['Deaths']
                       ]),layout={'template':'presentation'})
fig.update_layout(title_text="COVID-19 Türkiye Hasta Sayıları "+data_turkey['ObservationDate'].max().strftime("%d-%b'%y"))
fig.show()
columns=['Active','Recovered','Deaths']
meltedDF=pd.melt(data_turkey[columns[::-1]+['ObservationDate']],id_vars=['ObservationDate'], var_name='Value Type', value_name='Vaka oranları')
fig = px.bar(meltedDF, 
       x = "Vaka oranları",
       animation_frame = meltedDF['ObservationDate'].astype(str), 
       color = 'Value Type', 
       barmode = 'stack', height=400,
       template='seaborn',
       title='Zamana göre vaka oranları',
       orientation='h')
fig.show()
data_turkey.head()
train_ml=data_turkey.iloc[:int(data_turkey.shape[0]*0.95)]
valid_ml=data_turkey.iloc[int(data_turkey.shape[0]*0.95):]
model_scores=[]
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression(normalize=True)

lin_reg.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Confirmed"]).reshape(-1,1))
prediction_valid_linreg=lin_reg.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))
from sklearn.metrics import mean_squared_error,r2_score
model_scores.append(np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_linreg)))
print("Doğrusal Regresyon için Kök Ortalama Kare Hatası: ",np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_linreg)))
plt.figure(figsize=(11,6))
prediction_linreg=lin_reg.predict(np.array(data_turkey["Days Since"]).reshape(-1,1))
linreg_output=[]
for i in range(prediction_linreg.shape[0]):
    linreg_output.append(prediction_linreg[i][0])

fig=go.Figure()
fig.add_trace(go.Scatter(x=data_turkey.index, y=data_turkey["Confirmed"],
                    mode='lines+markers',name="Eğitim verisine göre vaka sayısı"))
fig.add_trace(go.Scatter(x=data_turkey.index, y=linreg_output,
                    mode='lines',name="Doğrusal regresyon doğrusu",
                    line=dict(color='black', dash='dot')))
fig.update_layout(title="Doğrusal regresyon tahmini",
                 xaxis_title="Tarih",yaxis_title="Vaka sayısı",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 10) 
train_poly=poly.fit_transform(np.array(train_ml["Days Since"]).reshape(-1,1))
valid_poly=poly.fit_transform(np.array(valid_ml["Days Since"]).reshape(-1,1))
y=train_ml["Confirmed"]
linreg=LinearRegression(normalize=True)
linreg.fit(train_poly,y)
prediction_poly=linreg.predict(valid_poly)
rmse_poly=np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_poly))
model_scores.append(rmse_poly)
print("Polinom Regresyonunda Kök Ortalama Kare Hatası: ",rmse_poly)
comp_data=poly.fit_transform(np.array(data_turkey["Days Since"]).reshape(-1,1))
plt.figure(figsize=(11,6))
predictions_poly=linreg.predict(comp_data)


fig=go.Figure()
fig.add_trace(go.Scatter(x=data_turkey.index, y=data_turkey["Confirmed"],
                    mode='lines+markers',name="Eğitim verisine göre vaka sayısı"))
fig.add_trace(go.Scatter(x=data_turkey.index, y=predictions_poly,
                    mode='lines',name="Doğrusal regresyon doğrusu",
                    line=dict(color='black', dash='dot')))
fig.update_layout(title="Polinom regresyon tahmini",
                 xaxis_title="Tarih",yaxis_title="Vaka sayısı",
                 legend=dict(x=0,y=1,traceorder="normal"))
fig.show()
from sklearn.svm import SVR
train_ml=data_turkey.iloc[:int(data_turkey.shape[0]*0.95)]
valid_ml=data_turkey.iloc[int(data_turkey.shape[0]*0.95):]
#SVR Modelin oluşturulması
svm=SVR(C=1,degree=5,kernel='poly',epsilon=0.01)

#Eğitim verisine model uydurma
svm.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Confirmed"]).reshape(-1,1))

prediction_valid_svm=svm.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))
model_scores.append(np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_svm)))
print("SVR için Kök Ortalama Kare Hatası: ",np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_svm)))
plt.figure(figsize=(11,6))
prediction_svm=svm.predict(np.array(data_turkey["Days Since"]).reshape(-1,1))
fig=go.Figure()

fig.add_trace(go.Scatter(x=data_turkey.index, y=data_turkey["Confirmed"],
                    mode='lines+markers',name="Eğitim verisine göre vaka sayısı"))
fig.add_trace(go.Scatter(x=data_turkey.index, y=prediction_svm,
                    mode='lines',name="SVR doğrusu",
                    line=dict(color='black', dash='dot')))
fig.update_layout(title="SVR'ye göre vaka tahmini",
                 xaxis_title="Tarih",yaxis_title="Vaka sayısı",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()