# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

def sigmoid(x, a, b, c):
    return c/(1+np.exp(-b*(x-a)))
def error(x,a,b,c,e):
    return c*np.sqrt(e)/(b*(c*x-x**2))    

def general_plot(dates,areas,areas_status,bound=((0,0,200),(185,1,np.inf))):
    for area in areas:
        if areas_status[areas.index(area)][-1]>200 and area!='': #and not(state=='Gujarat' or state=='Madhya Pradesh' or state=='Rajasthan'):
            area_status=areas_status[areas.index(area)].copy()
            area_status=area_status[area_status>0]
            popt, pcov = curve_fit(sigmoid, len(dates)-len(area_status)+np.arange(len(area_status)), area_status, bounds=bound)
            y=int(popt[2]*.99)#int((popt[2]+np.sqrt(popt[2]**2-4*s*popt[2]/popt[1]))/2)
            stop_date=int(popt[0]-(1/popt[1])*np.log((popt[2]/y-1)))
            er_stop_date=int(popt[2]*np.sqrt(pcov[2,2])/(popt[1]*(popt[2]*y-y**2)))    
            er_stop_date_line=min(er_stop_date,len(dates))
            
            print(area)
            #print(state_status)
            #print(popt,'\n',pcov)
            print('current:',area_status[-1],', Next 10 days:',[str(int(sigmoid(i,popt[0],popt[1],popt[2])))+'±'+str(int(error(i,popt[0],popt[1],popt[2],pcov[2,2]))) for i in range(len(dates),len(dates)+10)],', Final Infection Count:',int(popt[2]),'±',int(np.sqrt(pcov[2,2])))
            
            info=''
            if np.sqrt(pcov[2,2])/popt[2]>10 or 1.0*er_stop_date/stop_date>10:
                info='(unpredictable due to initial stage)'
            plt.title(area,loc='center')
            plt.scatter(len(dates)-len(area_status)+np.arange(len(area_status)),area_status,s=10)
            plt.plot(np.arange(200),sigmoid(np.arange(200),popt[0],popt[1],popt[2]),color='red')
            plt.scatter(stop_date,y,color='green')
            plt.errorbar(stop_date,y,xerr=er_stop_date_line)
            plt.text(x=0,y=y*0.9,s='Final Infected(curr_infection='+str(area_status[-1])+') : '+str(int(popt[2]))+'±'+str(int(np.sqrt(pcov[2,2])))+'\nStop Date(curr_day='+str(len(dates))+',assumed_final_infection(99%):'+str(y)+')='+str(stop_date)+'±'+str(er_stop_date),fontsize=8)
            plt.savefig(area+'.png')
            plt.show()
            plt.close()
            if area=='India':
                return [int(sigmoid(i,popt[0],popt[1],popt[2])) for i in range(55,55+37)]
# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')
#print(df.columns)

data=np.array(df)
#print(data)

dates=np.unique(data[:,1])
dates=dates.tolist()
dates.sort(key = lambda x: x.split('/')[1])
dates=np.array(dates)
#dates=dates[-21:]
print(dates[-1])

states=np.unique(data[:,3])
states=states.tolist()
#print(states)

status=np.zeros((len(states),len(dates)))
i=0
for date in dates:
	updated=data[data[:,1]==date]
	j=0
	for state in states:
		up_date=updated[updated[:,3]==state,8]
		if len(up_date)!=0:
			status[j,i]=status[j,i]+up_date[0]
		j=j+1
	i=i+1
    
general_plot(dates,states,status)
general_plot(dates,['India'],np.array([np.sum(status,axis=0)]))

dates=np.char.replace(dates,'/20','/2020')
result=np.array(general_plot(dates,['India'],np.array([np.sum(status, axis=0)])))
result_affected=result[1:].copy()
for i in range(len(result)):
    result[i+1:]=result[i+1:]-result[i]
result[0]=0
result_newcases=result[result>0].copy()
d=np.arange('2020-03-26','2020-05-01',dtype='datetime64[D]').astype(str)
df1=pd.DataFrame({'Date':d,'affected patients':result_affected,'new patients':result_newcases,'new fatality(death)':(result_newcases*425/12372).astype(int)})
df1.to_csv('with_lockdown.csv',index=False)

result=np.array(general_plot(dates[:56],['India'],np.array([np.sum(status, axis=0)[:56]]),((0,0,0),(92,1,np.inf))))
result_affected=result[1:].copy()
for i in range(len(result)):
    result[i+1:]=result[i+1:]-result[i]
result[0]=0
result_newcases=result[result>0].copy()
d=np.arange('2020-03-26','2020-05-01',dtype='datetime64[D]').astype(str)
df2=pd.DataFrame({'Date':d,'affected patients':result_affected,'new patients':result_newcases,'new fatality(death)':(result_newcases*16/730).astype(int)})
df2.to_csv('without_lockdown.csv',index=False)
df=pd.read_csv('/kaggle/input/covid19-in-india/IndividualDetails.csv',na_filter=False)
#print(df.columns)

data=np.array(df)
#print(data)

print(dates[-1])

districts=np.unique(data[:,6])
districts=districts.tolist()
#print(states)

status=np.zeros((len(districts),len(dates)))
i=0
for date in dates:
    updated=data[data[:,2]==date]
    j=0
    for district in districts:
        up_date=updated[updated[:,6]==district]
        status[j,i:]=status[j,i:]+len(up_date)
        j=j+1
    i=i+1

general_plot(dates,districts,status)