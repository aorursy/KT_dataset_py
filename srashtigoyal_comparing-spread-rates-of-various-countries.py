import pandas as pd

import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

#import plotly.graph_objects as go

#from fbprophet import Prophet

import pycountry

#import plotly.express as px
df_confirmed = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

df_recovered = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

df_deaths = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

df_confirmed.rename(columns={'Country/Region':'Country'}, inplace=True)

df_recovered.rename(columns={'Country/Region':'Country'}, inplace=True)

df_deaths.rename(columns={'Country/Region':'Country'}, inplace=True)



df_confirmed.head()
df_confirmed=df_confirmed.groupby(["Country"]).sum()

df_confirmed=df_confirmed.drop(columns=['Lat','Long'])

df_recovered=df_recovered.groupby(["Country"]).sum()

df_recovered=df_recovered.drop(columns=['Lat','Long'])

df_deaths=df_deaths.groupby(["Country"]).sum()

df_deaths=df_deaths.drop(columns=['Lat','Long'])

df_confirmed.head()
top_affected_countries=df_confirmed.sort_values(['3/20/20'],ascending=False).index[:63].values

top_affected_countries
np.savetxt('top_affected_countries.txt',top_affected_countries, delimiter=" ", fmt="%s")
df_confirmed_top=df_confirmed.loc[top_affected_countries,:]

df_recovered_top=df_recovered.loc[top_affected_countries,:]

df_deaths_top=df_deaths.loc[top_affected_countries,:]

#df_confirmed_top.to_csv('confirmed-data-top50-countries.csv')

df_confirmed_top.head()
df_active_top=df_confirmed_top-df_recovered_top - df_deaths_top
plt.figure(figsize=(13,13))

plt.title('total confirmed cases')

plt.barh(np.flip(top_affected_countries),np.log(np.flip(df_confirmed_top['3/21/20'].values)))

plt.xlabel('log cases')

#plt.xlim([9e4,0])

plt.grid()

#pd.plotting.table(data=df_confirmed_top['3/21/20'])
plt.figure(figsize=(13,13))

plt.title('total active cases')

plt.barh(np.flip(top_affected_countries),np.log(np.flip(df_active_top['3/21/20'].values)))

plt.xlabel('no. of cases')

plt.grid()
confirmed_mat=df_confirmed_top.as_matrix()

deaths_mat=df_deaths_top.as_matrix()

recovered_mat=df_recovered_top.as_matrix()

active_mat = df_active_top.as_matrix()
plt.figure(figsize=(10,7))

[plt.plot(np.log(confirmed_mat[i,:]),'*-',label=top_affected_countries[i]) for i in range(2,25)]

plt.plot(np.log(confirmed_mat[42,:]),'k*-',label=top_affected_countries[42])

plt.legend()
def plot_cases(confirmed_mat,thr=5,top_affected_countries=top_affected_countries):

    l=len(top_affected_countries)

    india_idx=np.where(top_affected_countries== 'India')[0][0]

    plt.figure(figsize=(10,10))

    idx=np.where(confirmed_mat[india_idx,:]>thr)[0]

    plt.plot(np.log(confirmed_mat[india_idx,idx.T]),'k*-',label=top_affected_countries[india_idx])

    for i in range(0,15):

        idx=np.where(confirmed_mat[i,:]>thr)[0]

        plt.plot(np.log(confirmed_mat[i,idx.T]),'*-',label=top_affected_countries[i])

    plt.xlabel('days(start from cases =' + str(thr) + ')')

    plt.ylabel('log # cases')

    plt.legend()

    plt.figure(figsize=(10,10))

    idx=np.where(confirmed_mat[india_idx,:]>thr)[0]

    plt.plot(np.log(confirmed_mat[i,idx.T]),'k*-',label=top_affected_countries[india_idx])

    for i in range(15,28):

        idx=np.where(confirmed_mat[i,:]>thr)[0]

        plt.plot(np.log(confirmed_mat[i,idx.T]),'*-',label=top_affected_countries[i])

    plt.xlabel('days(start from cases =' + str(thr) + ')')

    plt.ylabel('log # cases')

    plt.legend()

    plt.figure(figsize=(10,10))

    idx=np.where(confirmed_mat[india_idx,:]>thr)[0]

    plt.plot(np.log(confirmed_mat[india_idx,idx.T]),'k*-',label=top_affected_countries[india_idx])

    for i in range(28,40):

        idx=np.where(confirmed_mat[i,:]>thr)[0]

        plt.plot(np.log(confirmed_mat[i,idx.T]),'*-',label=top_affected_countries[i])

    plt.xlabel('days(start from cases =' + str(thr) + ')')

    plt.ylabel('log # cases')

    plt.legend()

    plt.figure(figsize=(10,10))

    for i in range(40,l):

        idx=np.where(confirmed_mat[i,:]>thr)[0]

        plt.plot(np.log(confirmed_mat[i,idx.T]),'*-',label=top_affected_countries[i])

    idx=np.where(confirmed_mat[india_idx,:]>thr)[0]

    plt.plot(np.log(confirmed_mat[india_idx,idx.T]),'k*-',label=top_affected_countries[india_idx])

    plt.xlabel('days(start from cases =' + str(thr) + ')')

    plt.ylabel('log # cases')

    plt.legend()
%matplotlib inline

print('confirmed_cases')

plot_cases(confirmed_mat,thr=50)
#print('active_cases')

#plot_cases(active_mat,thr=50)
#print('death_cases')

#plot_cases(deaths_mat,thr=1)
#print('recovered_cases')

#plot_cases(recovered_mat,thr=5)
def calc_slopes(confirmed_mat,thr=3,guess=1,extra=False):

    #slopes=np.zeros(50)

    l=len(top_affected_countries)

    fit_params = np.zeros([l,5])

    for i in range(l):

        idx=np.where(confirmed_mat[i,:]>=thr)[0]

        y = np.log(confirmed_mat[i,idx.T])

        if len(y)>0:

            x = np.arange(len(y))

            z, res, _, _, _ = np.polyfit(x, y, guess,full=True)

            #slopes[i]=z[0]

            fit_params[i,0]=z[0]

            fit_params[i,1]=z[1]

            fit_params[i,2]=idx[0]

            fit_params[i,3]= np.log(confirmed_mat[i,idx[0]])

            fit_params[i,4]=res                        

    return fit_params

    # p = np.poly1d(z)

    # fit = p(x)

  #  plt.figure()

  #  plt.hist(slopes,20,histtype='step')

  #  plt.xlabel('slope')

  #  print('mean,median = ',slopes.mean(),np.median(slopes))

  #  plt.figure(figsize=(30,10))

  #  plt.plot(slopes,'*-')

  #  plt.xticks(np.arange(0,50))

  #  plt.xlabel("Country")

  #  plt.ylabel("slope of log cases")

  #  plt.grid()

    
fit_params_confirmed=calc_slopes(confirmed_mat,thr=50)

fit_params_active=calc_slopes(active_mat,thr=50)
#confirmed cases 

def make_df_cases(fit_params_confirmed):

    confirmed_fit_df= pd.DataFrame()

    confirmed_fit_df['Country']=top_affected_countries

    confirmed_fit_df['slope(log cases)'] = fit_params_confirmed[:,0]

    confirmed_fit_df['start day'] = fit_params_confirmed[:,2].astype(int)

    confirmed_fit_df['start date']=df_confirmed_top.columns[confirmed_fit_df['start day']]

    confirmed_fit_df['intersecpt(log cases)'] = fit_params_confirmed[:,1]

    confirmed_fit_df['MSE fit'] = fit_params_confirmed[:,3]

    confirmed_fit_df=confirmed_fit_df.sort_values('slope(log cases)')

    confirmed_fit_df=confirmed_fit_df[confirmed_fit_df.Country != 'China']

    confirmed_fit_df=confirmed_fit_df[confirmed_fit_df.Country != 'Korea, South']

    confirmed_fit_df=confirmed_fit_df[confirmed_fit_df.Country != 'Cruise Ship']

    

    return confirmed_fit_df
confirmed_fit_df=make_df_cases(fit_params_confirmed)

#confirmed_fit_df.to_csv(r'47_countries_confirmed_fit.csv', index = False)

plt.figure(figsize=(13,13))

plt.title('slope(log # cases)')

plt.barh(confirmed_fit_df['Country'].values,confirmed_fit_df['slope(log cases)'].values)

#plt.xlim([9e4,0])

plt.grid()
# active cases

#active_fit_df=make_df_cases(fit_params_active)

#active_fit_df.to_csv(r'47_countries_active_fit.csv', index = False)

#active_fit_df
def y_pred(active_fit_df, t_len=65):

    for day in range(t_len):

        active_fit_df['exp MSE'] = np.exp(active_fit_df['MSE fit'])

        active_fit_df['pred_day ' + str(day)] = np.exp(active_fit_df['slope(log cases)']*day + active_fit_df['intersecpt(log cases)'])

    return active_fit_df
#active_fit_df_y=y_pred(active_fit_df)

#active_fit_df_y.to_csv(r'47_countries_active_fit_y.csv', index = False)

confirmed_fit_df_y=y_pred(confirmed_fit_df)

confirmed_fit_df_y.to_csv(r'top_countries_confirmed_fit_y.csv', index = False)

#active_fit_df_y[active_fit_df_y.columns[7:]].as_matrix()

confirmed_fit_df_y
#confirmed_mat_y=confirmed_fit_df_y[confirmed_fit_df_y.columns[7:]].as_matrix()

#plot_cases(confirmed_mat_y,thr=0,top_affected_countries=confirmed_fit_df_y['Country'].values)