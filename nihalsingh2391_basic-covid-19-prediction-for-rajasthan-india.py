import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context("talk")
import datetime
import requests
import warnings
warnings.filterwarnings('ignore')
test_all=pd.read_csv('../input/covid19-in-india/covid_19_india.csv')
test_all=test_all.drop('Time',axis=1)
test_all=test_all.drop('ConfirmedIndianNational',axis=1)
test_all=test_all.drop('ConfirmedForeignNational',axis=1)
test_all
# for i in range(1670):
#     print(test_all['State/UnionTerritory'][i])
# #     if(test_all[i]['State/UnionTerritory']!=test_all[i+1]['State/UnionTerritory']):
# #         print(test_all[i])
dict={}
for i in range(1670):
    key=test_all['State/UnionTerritory'][i];
#     print(dict)
    d={key:test_all['Confirmed'][i]}
#     if(key in dict):
    dict.update(d)
#     else:
#         dict.add(key:test_all['Confirmed'][i])
# data1=pd.DataFrame(data=dict.keys())
# for i in dict.keys():
#     print(dict.get(i))
data1=pd.DataFrame(columns=['State','Confirmed'])
for i in dict.keys():
    dt=pd.DataFrame([[i,dict.get(i)]],columns=['State','Confirmed'])
#     print(dict.get(i))
    data1=data1.append(dt)
import seaborn as sns
sns.barplot(data1['Confirmed'],data1['State'],orient='h')
df1=pd.DataFrame(columns=['Confirmed','Cured','Deceased'])
for i in range(1671):
#     print(test_all['State/UnionTerritory'][i])
    if(test_all['State/UnionTerritory'][i]=='Rajasthan'):
        dt=pd.DataFrame([[(test_all['Confirmed'][i]),(test_all['Cured'][i]),(test_all['Deaths'][i])]],columns=['Confirmed','Cured','Deceased'])
        df1=df1.append(dt)
# dd=pd.DataFrame(columns=['Time'])
# for i in range(105):
#     dt=pd.DataFrame([[i+1]],columns=['Time'])
#     dd=dd.append(dt)
df1['date']=np.arange(len(df1))+1
df2=pd.read_csv('../input/rajasthan-corona/raj_covid.csv')
df2
plt.plot(df2['date'],df2['Confirmed'],label='Confirmed')
plt.plot(df2['date'],df2['Cured'],label='Cured')
plt.plot(df2['date'],df2['Deceased'],label='Deceased')
plt.plot(df2['date'],df2['Active'],label='Active')
plt.legend(loc='upper left', shadow=True, fontsize='small')
plt.xlabel("Days since first case")
plt.ylabel("No. of patients")
plt.plot(df2['date'],np.log(df2['Confirmed']),label='Confirmed')
plt.plot(df2['date'],np.log(df2['Cured']),label='Cured')
plt.plot(df2['date'],np.log(df2['Deceased']),label='Deceased')
plt.plot(df2['date'],np.log(df2['Active']),label='Active')
plt.legend(loc='upper left', shadow=True, fontsize='x-small')
plt.xlabel("Days since first case")
plt.ylabel("No. of patients(Log)")
df1cn=df2.drop('Cured',axis=1)
df1cn=df1cn.drop('Deceased',axis=1)
df1cn=df1cn.drop('Active',axis=1)
t=df1cn['date'].values
y=df1cn['Confirmed'].values
t1=np.array([i for i in t])
y1=np.array([i for i in y])
theta=0.4
n=len(t)
s=np.empty(n)
def yp(t,tau,theta,gam,b):
#     z=tau*(np.sin(theta*t))+b
    z=tau/(1+np.exp(-gam*(t-theta)))+b
#     z=tau*(t**3)+theta*(t**2)+theta*(t)
#     z=tau*np.exp(-((theta-t)**2)/(2*(gam**2)))+b
#     z=5.0*(1-np.exp(-(t-theta)/tau))+b
#     for i in range(n):
#         if(t[i]<theta):
#             s[i]=0
#         else:
#             s[i]=1
    return(z)
p0=[max(y1), np.median(t1),1,min(y1)]
print(yp(t1,max(y1), np.median(t1),1,min(y1)))
from scipy.optimize import curve_fit
c,cov=curve_fit(yp,t1,y1,p0,method='dogbox')
yo=yp(t1,c[0],c[1],c[2],c[3])
from sklearn.metrics import r2_score
print(r2_score(yo,y1))
plt.plot(t1,y1,'r.')
plt.plot(t1,yo,'b-')
plt.xlabel("Days since first case")
plt.ylabel("No. of patients")

df2cn=df2.drop('Cured',axis=1)
df2cn=df2cn.drop('Deceased',axis=1)
df2cn=df2cn.drop('Confirmed',axis=1)
t2=df2cn['date'].values
y2=df2cn['Active'].values
t21=np.array([i for i in t2])
y21=np.array([i for i in y2])
theta=0.4
n=len(t2)
s2=np.empty(n)
def yp2(t,tau,theta,gam,b):
    #     z=tau*(np.sin(theta*t))+b
    z=tau/(1+np.exp(-gam*(t-theta)))+b
#     z=tau*(t**3)+theta*(t**2)+theta*(t)
#     z=tau*np.exp(-((theta-t)**2)/(2*(gam**2)))+b
#     z=5.0*(1-np.exp(-(t-theta)/tau))+b
#     for i in range(n):
#         if(t[i]<theta):
#             s[i]=0
#         else:
#             s[i]=1
    return(z)
p0=[max(y21), np.median(t21),1,min(y21)]
print(yp2(t21,max(y21), np.median(t21),1,min(y21)))
from scipy.optimize import curve_fit
c2,cov2=curve_fit(yp2,t21,y21,p0,method='dogbox')
yo2=yp2(t21,c2[0],c2[1],c2[2],c[3])
from sklearn.metrics import r2_score
print(r2_score(yo2,y21))
plt.plot(t21,y21,'r.')
plt.plot(t21,yo2,'b-')
plt.xlabel("Days since first case")
plt.ylabel("No. of patients")
fut=np.array([36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101])
yo2_pred=yp(fut,c[0],c[1],c[2],c2[3])
# t21
plt.plot(fut,yo2_pred,'b-')
plt.xlabel("Days since first case")
plt.ylabel("No. of Confirmed Cases")