import pandas as pd
import numpy as np 
import seaborn as sns
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
avocado=pd.read_csv('../input/avocado.csv')
avocado.head(5)
del avocado['Unnamed: 0']
avocado['Date']=pd.to_datetime(avocado['Date'])
avocado.isnull().sum()
avocado_date=avocado.groupby('Date').mean()
avocado_date.head(5)
avocado_date.AveragePrice.plot(figsize=(15,4))
plt.ylabel('Average Price')
ax=sns.factorplot(x='Date', y='AveragePrice', data=avocado, hue='type',aspect=3)
avocado_date_t=avocado.pivot_table(index='Date', columns='type', aggfunc='mean')['AveragePrice']
avocado_date_t.plot(figsize=(15,4))
plt.text(x='2017-1-15', y=2.04, s='March 2017', color='green', fontsize=12)
plt.vlines(x='2017-3-1', ymin=0.8, ymax=2, color='green', linestyles=':', linewidth=3, label='March 2017')
plt.ylabel('Average Price')


sns.boxplot(data=avocado_date_t, palette='bone')
x=[]
for i in range(len(avocado)):
    m=avocado['Date'].loc[i].strftime("%B")
    x.append(m)
avocado['month']=x
avocado_year=avocado.pivot_table(index='month', columns='year', aggfunc='mean')['AveragePrice']
month_order={
    'January':1,
    'February':2,
    'March':3,
    'April':4,
    'May':5,
    'June':6,
    'July':7,
    'August':8,
    'September':9,
    'October':10,
    'November':11,
    'December':12
}
fig=avocado_year.loc[month_order].plot(figsize=(15,4), xticks=range(0,13), cmap='jet')
fig.set_facecolor('#7F7F7F')
plt.text(x=0.7, y=1.9, s='Winter', color='#98F5FF', fontsize=15)
plt.vlines(x=0, ymin=0.8, ymax=2, color='lightblue', linestyles=':', linewidth=3)
plt.text(x=3.8, y=1.9, s='Spring', color='#7FFF00', fontsize=15)
plt.vlines(x=2.5, ymin=0.8, ymax=2, color='#7FFF00', linestyles=':', linewidth=3)
plt.text(x=6.8, y=1.9, s='Summer', color='#FFB6C1', fontsize=15)
plt.vlines(x=5.5, ymin=0.8, ymax=2, color='pink', linestyles=':', linewidth=3)
plt.text(x=9.5, y=1.9, s='Fall', color='orange', fontsize=15)
plt.vlines(x=8.5, ymin=0.8, ymax=2, color='orange', linestyles=':', linewidth=3)
plt.ylabel('Average Price')
year_15=avocado.loc[avocado['year']==2015]
year_16=avocado.loc[avocado['year']==2016]
year_17=avocado.loc[avocado['year']==2017]
year_18=avocado.loc[avocado['year']==2018]
fig1=year_15.groupby('Date').mean().plot(y='AveragePrice', figsize=(10,3), kind='line', sharex=True, color='#98F5FF')
plt.text(x='2015-1-1', y=1.45, s='2015', color='#98F5FF', fontsize=15)
fig1.set_facecolor('#7F7F7F')

fig2=year_16.groupby('Date').mean().plot(y='AveragePrice', figsize=(10,3), kind='line', sharex=True, color='#FFE4C4')
plt.text(x='2016-1-1', y=1.55, s='2016', color='#FFE4C4', fontsize=15)
fig2.set_facecolor('#7F7F7F')

fig3=year_17.groupby('Date').mean().plot(y='AveragePrice', figsize=(10,3), kind='line', sharex=True, color='#CAFF70')
plt.text(x='2017-1-1', y=1.8, s='2017', color='#CAFF70', fontsize=15)
fig3.set_facecolor('#7F7F7F')
y=[]
a=avocado_date.index
for i in range(len(avocado_date)):
    m=a[i].strftime("%B")
    y.append(m)
avocado_date['month']=y

avocado_volume=avocado.pivot_table(index='Date', aggfunc='sum')
b=avocado_volume.index
z=[]
for i in range(len(avocado_volume)):
    m=b[i].strftime("%B")
    z.append(m)
ab=[]
for i in range(len(avocado_volume)):
    m=b[i].strftime("%Y")
    ab.append(m)
avocado_volume['month']=z
avocado_volume['year']=ab
plt.figure(figsize=(15,4))
sns.pointplot(x='month', y='AveragePrice', data=avocado_date, hue='year', palette='Set1')
plt.title('Average Price')
plt.ylabel('USD')
plt.figure(figsize=(15,4))
sns.pointplot(x='month', y='Total Volume', data=avocado_volume, hue='year', palette='Set1')
plt.title('Total Volume')
avocado_type_volume=avocado.pivot_table(index='Date', columns='type', aggfunc='sum')
avocado_type_volume.plot(y='Total Volume', figsize=(15,4), title='Total Volume')
e=avocado_type_volume['Total Volume'].sum(axis=1)
f=avocado_type_volume['Total Volume']['conventional']/e*100
g=100-avocado_type_volume['Total Volume']['conventional']/e*100
plt.figure(figsize=(10,4))
plt.text(x='2017-10-15', y=10, s='Organic', color='#00CD00', fontsize=18)
plt.bar(x=a, height=f, width=6, color='#FFE4B5')
plt.bar(x=a, height=g, width=7, color='#7CFC00')
plt.ylabel('Market share%')
avocado.region.unique()
west=avocado.loc[avocado['region']=='West']
southeast=avocado.loc[avocado['region']=='Southeast']
southcentral=avocado.loc[avocado['region']=='SouthCentral']
plains=avocado.loc[avocado['region']=='Plains']
northeast=avocado.loc[avocado['region']=='Northeast']
midsouth=avocado.loc[avocado['region']=='Midsouth']
totalus=avocado.loc[avocado['region']=='TotalUS']
df=pd.merge(west, southeast, on='Date', suffixes=('_west', '_southeast'))
df=pd.concat([west, southeast])
df1=pd.concat([df,southcentral])
df2=pd.concat([df1,plains])
df3=pd.concat([df2,northeast])
df4=pd.concat([df3,midsouth])
df5=pd.concat([df4,totalus])
df5_price=df5.pivot_table(index='Date', columns='region', aggfunc='mean')['AveragePrice']
ma_days=[10,20,50]
for ma in ma_days:
    column_name='MA for {} days'.format(ma)
    df5_price[column_name]=df5_price['TotalUS'].rolling(ma).mean()
test=df5_price.groupby('Date')[['TotalUS', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].mean()
test.drop('TotalUS', axis=1, inplace=True)
test.columns.name='Moving Average'
fig2=test.plot( figsize=(15,4), kind='line', cmap='Blues')
fig2.set_facecolor('#7A8B8B')
df5_price.plot(kind='line', figsize=(12,4))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('Average Price')
r=df5_price.index
c=[]
for i in range(len(df5_price)):
    m=r[i].strftime("%B")
    c.append(m)
df5_price['month']=c
#------------------------------------
f=[]
for i in range(len(df5_price)):
    m=r[i].strftime("%Y")
    f.append(m)
df5_price['year']=f    
med=df5_price.median()
med.sort_values()
median=['SouthCentral', 'West', 'Midsouth', 'Southeast', 'Plains', 'Northeast']
plt.figure(figsize=(15,4))
sns.boxplot( data=df5_price[['West','Midsouth','Northeast', 'Plains', 'SouthCentral', 'Southeast']], palette='summer', saturation=1, order=median)
plt.ylabel('Average Price')
plt.figure(figsize=(15,4))
sns.boxplot(x='month',y='TotalUS', data=df5_price, palette='cool', saturation=1)
plt.ylabel('Average Price')
plt.title('Average price fluctuations in US (years= 2015, 2016, 2017, 2018)')
df5_volume=df5.pivot_table(index='Date', columns='region', aggfunc='sum')['Total Volume']
df5_volume[['SouthCentral', 'West', 'Midsouth', 'Southeast', 'Plains', 'Northeast']].plot(kind='line', figsize=(12,4), cmap='Set1')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('Total Volume')
medi=df5_volume.median()
medi.sort_values()
order=['Plains','Midsouth', 'Southeast', 'Northeast', 'SouthCentral', 'West']
plt.figure(figsize=(15,4))
sns.violinplot( data=df5_volume[['West','Midsouth','Northeast', 'Plains', 'SouthCentral', 'Southeast']], palette='winter', saturation=1, order=order, orient='v')
plt.ylabel('Total Volume')
v=[]
for i in range(len(df5_volume)):
    m=r[i].strftime("%B")
    v.append(m)
df5_volume['month']=v    
    
plt.figure(figsize=(15,4))
sns.boxplot(x='month',y='TotalUS', data=df5_volume, palette='ocean', saturation=1)
plt.ylabel('Total Volume')
plt.title('Total volume fluctuations in US (years= 2015, 2016, 2017, 2018)')
avocado['Total indiv']=avocado['Total Volume']-avocado['Total Bags']
avocado['Revenue indiv']=avocado['Total indiv']*avocado['AveragePrice']
avocado['Revenue Bagged']=avocado['Total Bags']*avocado['AveragePrice']
avocado['Revenue Total']=avocado['Revenue Bagged']+avocado['Revenue indiv']
avocado_r_t=avocado.pivot_table(index='Date', columns='type', aggfunc='sum')[['Revenue indiv', 'Revenue Bagged', 'Revenue Total']]
avocado_r_t.plot(cmap='Set1', figsize=(10,4))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
avocado_rev=avocado.pivot_table(index='Date', aggfunc='sum')[['Revenue indiv', 'Revenue Bagged', 'Revenue Total']]
avocado_rev.plot(cmap='Set1', figsize=(10,4))
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
revenue=avocado[['Total indiv','Revenue indiv','Total Bags', 'Revenue Bagged', 'Total Volume', 'Revenue Total']]
rename=['Total_indiv', 'Revenue_indiv', 'Total_Bags', 'Revenue_Bagged','Total_Volume', 'Revenue_Total' ]
revenue.columns=rename
sns.pairplot(data=revenue)
lm3=smf.ols(formula='Revenue_Total ~ Total_indiv + Total_Bags', data=revenue).fit()
lm3.summary()
revenue['lm2']=0.8766*revenue['Total_indiv']+1.5155*revenue['Total_Bags']+29190  
revenue.plot(x='Revenue_Total', y='lm2', kind='scatter')
x=[0,7*10**7]
y=[0,7*10**7]
plt.plot(x,y, '-.', color='red')
plt.text(x=6.5*10**7, y=6*10**7, s='y=x', color='red', fontsize=13)
test=df5_price.loc[df5_price['year']=='2018']
train=df5_price.loc[(df5_price['year']=='2017') ^ (df5_price['year']=='2016') ^ (df5_price['year']=='2015') ]
price_reg=DataFrame(avocado.groupby('region')['AveragePrice'].mean())
price_reg.sort_values(by='AveragePrice').plot(kind='bar', figsize=(15,4))
price_reg.reset_index(inplace=True)
Bin=(price_reg['AveragePrice'].max()-price_reg['AveragePrice'].min())/5
class1=[]
class2=[]
class3=[]
class4=[]
class5=[]

for i in price_reg.index: 
    if (price_reg['AveragePrice'].loc[i]<price_reg['AveragePrice'].min()+Bin):
        class1.append(price_reg['region'].loc[i])
for i in price_reg.index: 
    if (price_reg['AveragePrice'].loc[i]>=price_reg['AveragePrice'].min()+Bin) & (price_reg['AveragePrice'].loc[i]<price_reg['AveragePrice'].min()+2*Bin):
        
        class2.append(price_reg['region'].loc[i])
#---------------------------------------------------------
for i in price_reg.index: 
    if (price_reg['AveragePrice'].loc[i]>=price_reg['AveragePrice'].min()+2*Bin) & (price_reg['AveragePrice'].loc[i]<price_reg['AveragePrice'].min()+3*Bin):
        
        class3.append(price_reg['region'].loc[i])
#---------------------------------------------------------
for i in price_reg.index: 
    if (price_reg['AveragePrice'].loc[i]>=price_reg['AveragePrice'].min()+3*Bin) & (price_reg['AveragePrice'].loc[i]<price_reg['AveragePrice'].min()+4*Bin):
        
        class4.append(price_reg['region'].loc[i])
#---------------------------------------------------------
for i in price_reg.index: 
    if (price_reg['AveragePrice'].loc[i]>=price_reg['AveragePrice'].min()+4*Bin):
        class5.append(price_reg['region'].loc[i])

        
temp=[]
for i in avocado['region']:
    if i in class1: 
        temp.append('class1')
    if i in class2:
        temp.append('class2')
    if i in class3:
        temp.append('class3')
    if i in class4:
        temp.append('class4')
    if i in class5:
        temp.append('class5')
avocado['class']=temp
avocado.head(3)
week=[]
for i in avocado['Date']:
    week.append(i.isocalendar()[1])
avocado['week']=week
year2015=avocado.loc[avocado['year']==2015]
year2015=year2015.pivot_table(index='week', columns='type', aggfunc='mean')['AveragePrice']
for i in avocado.index:
    if (avocado['week'].loc[i]==53) & (avocado['month'].loc[i]=='January'):
        avocado['week'].loc[i]=1
for i in avocado.index:
    if (avocado['week'].loc[i]==52) & (avocado['month'].loc[i]=='January'):
        avocado['week'].loc[i]=1
d=[]
for i in avocado.index:
    a=avocado['week'].loc[i]
    if avocado['type'].loc[i]=='conventional':
        d.append(year2015['conventional'].loc[a])
    else:
        d.append(year2015['organic'].loc[a])
avocado['base_price']=d
avocado['delta']=avocado['AveragePrice']-avocado['base_price']
df_modeling=avocado.drop(['Date', 'AveragePrice', 'year','region', 'month', 'base_price', 'Revenue indiv', 'Revenue Bagged', 'Revenue Total','Total indiv', 'Total Bags'], axis=1)
type_dum=pd.get_dummies(df_modeling['type'])
class_dum=pd.get_dummies(df_modeling['class'])
df_modeling=pd.concat([df_modeling, type_dum], axis=1)
df_modeling=pd.concat([df_modeling, class_dum], axis=1)
df_modeling.drop(['organic', 'class1', 'type', 'class'], axis=1, inplace=True)
df_modeling.head()
from sklearn.model_selection import train_test_split
X_multi=df_modeling.drop('delta', axis=1)
Y=df_modeling['delta']
X_train, X_test, Y_train, Y_test = train_test_split(X_multi, Y)
lreg=LinearRegression()
lreg.fit(X_train, Y_train)
coef=DataFrame(lreg.coef_)
features=DataFrame(X_multi.columns)
line1=pd.concat([features, coef], axis=1)
line1.columns=['feature', 'coefficient']
line1.index=line1['feature']
line1.sort_values(by='coefficient').plot(kind='bar')
pred_train=lreg.predict(X_train)
pred_test=lreg.predict(X_test)
plt.scatter(x=Y_test, y=pred_test, marker='.')
plt.plot(Y_test, Y_test, color='red')
plt.title('test data')
resid=Y_train-pred_train
resid2=Y_test-pred_test
train=plt.scatter(x=Y_train, y=resid, alpha=0.5)
test=plt.scatter(x=Y_test, y=resid2, marker='x', alpha=0.5)
line=plt.hlines(y=0, xmin=-1.5, xmax=1.5, color='red')
plt.legend((train,test), ('Training', 'Test'), loc='lower right')
plt.ylabel('residual')
plt.xlabel('delta')

