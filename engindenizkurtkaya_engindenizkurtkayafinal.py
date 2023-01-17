import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
covid19=pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
# covid19 değişkenine covid_19_data.csv dataframe i atadım.
# Tarihleri datetime'a dönüştürme
covid19['ObservationDate']=pd.to_datetime(covid19['ObservationDate'])
covid19.head()
covid19.info()
covid19.describe()
df=covid19.groupby(['ObservationDate'])[['Confirmed','Recovered','Deaths']].sum()
plt.figure(figsize=(20,10))
plt.title('22 Ocak 2020 den itibaren Novel Durumları',fontsize=30)
plt.xlabel('Date',fontsize=20)
plt.ylabel('Number of cases',fontsize=20)
plt.plot(df.index,df['Confirmed'],label='Enfekte',linewidth=3)
plt.plot(df.index,df['Recovered'],label='İyileşen',linewidth=3,color='green')
plt.plot(df.index,df['Deaths'],label='Ölen',linewidth=3,color='red')
plt.bar(df.index,df['Confirmed'],alpha=0.2,color='c')
plt.xticks(fontsize=15,rotation=90)
plt.yticks(fontsize=15)
plt.style.use('ggplot')
plt.legend()
df1=covid19.groupby(['Country/Region'])[['ObservationDate','Confirmed','Recovered','Deaths']]
india_cases=df1.get_group('India')
plt.figure(figsize=(20,8))
plt.title('Hindistandaki Vaka Sayılari',fontsize=20)

plt.ylabel('Vaka Sayıları ',fontsize=20)
plt.xlabel('Tarihler',fontsize=20)
plt.plot(india_cases['ObservationDate'],india_cases['Confirmed'],'-o',linewidth=2,markersize=10,mfc='red',mew=2.9,mec='black')
plt.xticks(rotation=90)
plt.grid()
country=covid19.groupby(['Country/Region'])[['Confirmed','Recovered','Deaths']].sum()
top_5=country.nlargest(5,['Confirmed'])
plt.figure(figsize=(20,16))
plt.subplot(311)
plt.title('Doğrulanmış, iyileşmiş ve ölüm vakası bulunan ilk 5 ülke',fontsize=20)
plt.barh(top_5.index,top_5['Confirmed'],color='blue')
plt.yticks(fontsize=20)
plt.xlabel('Doğrulanmış',fontsize=20)
plt.subplot(312)
plt.barh(top_5.index,top_5['Deaths'],color='red')
plt.yticks(fontsize=20)
plt.xlabel('Ölümler',fontsize=20)
plt.subplot(313)
plt.barh(top_5.index,top_5['Recovered'],color='green')
plt.yticks(fontsize=20)
plt.xlabel('İyileşen',fontsize=20)
covid19['day']=covid19['ObservationDate'].dt.day
import matplotlib.ticker as ticker
cv1=covid19[covid19['ObservationDate']>'2020-03']
fig, ax=plt.subplots(figsize=(15,8))
def draw_barchart(date):
    df=cv1[cv1['day'].eq(date)].sort_values(by='Confirmed',ascending=True).tail(10)
    ax.clear()
    ax.text(0,1.12,'18 Mart 2020 de en çok enfekte olan ülkeler',size=24,weight=600,transform=ax.transAxes,ha='left')
    ax.barh(df['Country/Region'],df['Confirmed'],color='orange')
    for i, (country,value) in enumerate(zip(df['Country/Region'],df['Confirmed'])):
        ax.text(value,i, country, size=14, ha='right',va='bottom')
        ax.text(value,i,f'{value:.0f}', size=14, ha='left',va='center')
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}')) 
    ax.set_yticks([])
    ax.set_axisbelow(True)
    ax.margins(0,0.1)
    ax.tick_params(axis='x',labelsize=15,colors='blue')
    ax.grid(which='major',axis='x',linestyle='--')
    plt.box(False) 
draw_barchart(18)

rank=country.nlargest(179,['Confirmed']).head(10)
confirmed=[]
recovered=[]
death=[]
for i in rank.index:
    df1=covid19[covid19['Country/Region']==i]
    confirmed.append(df1['Confirmed'].mean())
    recovered.append(df1['Recovered'].mean())
    death.append(df1['Deaths'].mean())
plt.figure(figsize=(20,20))

plt.subplot(311)
plt.title('Ortalamaları doğrulanmış, iyileşmiş ve ölüm vakaları olan ilk 10 ülke',fontsize=20,color='green')
plt.plot(rank.index,confirmed,'-o',mfc='black')
plt.ylabel('Onaylanmış',fontsize=20)
plt.grid()
plt.subplot(312)
plt.plot(rank.index,recovered,'-o',color='green',mfc='black')
plt.ylabel('İyileşen',fontsize=20)
plt.grid()
plt.subplot(313)
plt.plot(rank.index,death,'-o',color='red',mfc='black')
plt.ylabel('Ölümler',fontsize=20)
plt.grid()    
rank1=country.nlargest(179,['Confirmed']).head(20)
confirmed_perc=[]
for i in rank1['Confirmed']:
    confirmed_perc.append(i/rank1['Confirmed'].sum())
plt.figure(figsize=(20,20))    
plt.title('Onaylanan Vakaların Dağılımı',fontsize=20)    
plt.pie(confirmed_perc,autopct='%1.1f%%')
plt.legend(rank1.index,loc='best')
plt.show()    
symptoms={'Belirtiler':['Ateş',
        'Kuru Öksürük',
        'Yorgunluk',
        'Balgam',
        'Nefes Darlığı',
        'Kas Ağrısı',
        'Boğaz Ağrısı',
        'Bağ Ağrısı',
        'Titreme',
        'Mide Bulantısı',
        'Burun Tıkanıklığı',
        'İshal',
        'Hemoptizi',
        'Konjonktival tıkanıklık'],'Yüzdelik Oran':[87.9,67.7,38.1,33.4,18.6,14.8,13.9,13.6,11.4,5.0,4.8,3.7,0.9,0.8]}

symptoms=pd.DataFrame(data=symptoms,index=range(14))
symptoms
plt.figure(figsize=(20,15))
plt.title('Belirtilerin Dağılımı',fontsize=20)    
plt.pie(symptoms['Yüzdelik Oran'],autopct='%1.1f%%')
plt.legend(symptoms['Belirtiler'],loc='best')
plt.show() 
covid19['Country/Region']=covid19['Country/Region'].astype('str')
covid19['Province/State']=covid19['Province/State'].astype('str')
covid19['day']=covid19['ObservationDate'].dt.day
covid19['month']=covid19['ObservationDate'].dt.month
lbl=preprocessing.LabelEncoder()
for c in ['Province/State','Country/Region']:
    lbl.fit(covid19[c].unique())
    covid19[c]=lbl.transform(covid19[c])
x=covid19.drop(['Confirmed','SNo','Last Update','ObservationDate'],1)
y=covid19['Confirmed']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import r2_score

print(' R2 Score   : ',r2_score(y_test, y_pred))
y_pred = model.predict(x_test)

df_predictions = pd.DataFrame()
df_predictions['y_pred_log'] = y_pred
df_predictions['y_true_log'] = y_test.values
df_predictions['y_pred'] = 10 ** y_pred
df_predictions['y_true'] = 10 ** y_test.values

df_predictions['absolute_pct_error'] = abs((df_predictions['y_pred'] - df_predictions['y_true']) / df_predictions['y_true']) * 100