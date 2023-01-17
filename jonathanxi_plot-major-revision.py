# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = '../input/bridata'

df_nonlinear_fval=pd.read_csv(f'{path}/nonlinear_fval.csv',header=None)

df_linear10_fval=pd.read_csv(f'{path}/linear_fval_10.csv',header=None)

df_linear11_fval=pd.read_csv(f'{path}/linear_fval_11.csv',header=None)

df_linear01_fval=pd.read_csv(f'{path}/linear_fval_01.csv',header=None)

df_sequential_fval=pd.read_csv(f'{path}/fval_linear_sequential.csv',header=None)
#df_plot_fval=pd.concat([df_nonlinear_fval,df_linear10_fval,df_linear11_fval,df_linear01_fval],axis=1)

df_plot_fval=pd.concat([df_sequential_fval,df_linear10_fval,df_linear11_fval,df_linear01_fval],axis=1)

df_plot_fval.columns=['Sequential Fval','Linear Fval 10','Linear Fval 11','Linear Fval 01']

df_plot_fval.index=range(2005,2017)
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={"figure.figsize": (8, 6)}); 

sns.set(style="white")

#sns.set(style="darkgrid")

sns.lineplot(data=df_plot_fval)

sns.despine()
df_country=pd.read_csv(f'{path}/df_country_plot.csv')

df_fdi=pd.read_csv(f'{path}/FDI.csv')

df_country=df_country.iloc[:,1:3]
lists=['South_Asia','Eastern_Europe','CIS','Western_Asia','Eastern_Asia','Middle_Asia']

def mean_calculate(df,lists):

    sum=[]

    for i in lists:

        temp=np.mean(df[i],axis=1)

        sum.append(temp)

    df_mean_plot=pd.concat(sum,axis=1)

    return df_mean_plot
df_fdi_new=df_fdi.drop('Indicator Name',axis=1)

df_fdi_new=pd.merge(df_country,df_fdi_new,on='Country Name',how='left')

df_fdi_new=df_fdi_new.T

df_fdi_new.columns=df_country['Country Name']

df_fdi_final=df_fdi_new.drop(['Country Name','Region'],axis=0)

df_fdi_final
lists=['South Asia','Eastern Europe','CIS','Western Asia','Eastern Asia','Central Asia']

df_fdi_final.columns=df_country['Region']

df_fdi_final_new=mean_calculate(df_fdi_final,lists)

df_fdi_final_new.columns=lists

df_fdi_final_new
fig = plt.figure(figsize=(16,12))

df_fdi_final_new.plot(style=['+-', 'd--', 'o-.', '.-', 'v:','*-'])

sns.despine()
import seaborn as sns

fig=plt.figure(figsize=(10,8))

sns.set(style="white",palette='muted',color_codes=False)

sns.kdeplot(df_fdi_final.iloc[0,:],label="2005" ,shade=True)

#sns.kdeplot(data=sp_nonlinear.iloc[1,:],label="2006" ,shade=True)

#sns.kdeplot(df_fdi_final.iloc[2,:],label="2007" ,shade=True)

#sns.kdeplot(df_fdi_final.iloc[3,:],label="2008" ,shade=True)

#sns.kdeplot(data=sp_nonlinear.iloc[4,:],label="2009" ,shade=True)

#sns.kdeplot(data=sp_nonlinear.iloc[5,:],label="2010" ,shade=True)

sns.kdeplot(df_fdi_final.iloc[6,:],label="2011" ,shade=True)

#sns.kdeplot(data=sp_nonlinear.iloc[7,:],label="2012" ,shade=True)

#sns.kdeplot(df_fdi_final.iloc[8,:],label="2013" ,shade=True)

#sns.kdeplot(data=sp_nonlinear.iloc[9,:],label="2014" ,shade=True)

#sns.kdeplot(df_fdi_final.iloc[10,:],label="2015" ,shade=True)

sns.kdeplot(df_fdi_final.iloc[11,:],label="2016" ,shade=True)

sns.despine()
tech_ineffi_nonlinear=pd.read_csv(f'{path}/tech_inefficiency_nonlinear.csv',header=None)

tech_ineffi_linear_11=pd.read_csv(f'{path}/tech_inefficiency_linear_11.csv',header=None)

tech_ineffi_linear_10=pd.read_csv(f'{path}/tech_inefficiency_linear_10.csv',header=None)

tech_ineffi_linear_01=pd.read_csv(f'{path}/tech_inefficiency_linear_01.csv',header=None)

tech_ineffi_sequential=pd.read_csv(f'{path}/tech_inefficiency_sequential.csv',header=None)

tech_ineffi_whole=pd.read_csv(f'{path}/tech_inefficiency_whole.csv',header=None)





df_country=pd.read_csv(f'{path}/df_country_plot.csv')

df_country=df_country.iloc[:,1:3]
df_country_list=df_country.T

df_country_list.columns=df_country_list.iloc[1,:]

df_country_list['Eastern Europe']
tech_ineffi_linear=['tech_ineffi_linear_11','tech_ineffi_linear_10','tech_ineffi_linear_01']
tech_ineffi_nonlinear.index=range(2005,2017)

tech_ineffi_linear_11.index=range(2005,2017)

tech_ineffi_linear_10.index=range(2005,2017)

tech_ineffi_linear_01.index=range(2005,2017)

tech_ineffi_sequential.index=range(2005,2017)

tech_ineffi_whole.index=range(2005,2017)





tech_ineffi_nonlinear.columns=df_country['Region']

tech_ineffi_linear_11.columns=df_country['Region']

tech_ineffi_linear_10.columns=df_country['Region']

tech_ineffi_linear_01.columns=df_country['Region']

tech_ineffi_sequential.columns=df_country['Region']

tech_ineffi_whole.columns=df_country['Region']
#tech_ineffi_nonlinear.index=range(2005,2017)

#tech_ineffi_linear_11.index=range(2005,2017)

#tech_ineffi_linear_10.index=range(2005,2017)

#tech_ineffi_linear_01.index=range(2005,2017)

#tech_ineffi_nonlinear.columns=df_country['Country Name']

#tech_ineffi_linear_11.columns=df_country['Country Name']

#tech_ineffi_linear_10.columns=df_country['Country Name']

#tech_ineffi_linear_01.columns=df_country['Country Name']
df_country.tail()
import seaborn as sns

fig = plt.figure(figsize=(10,8))

sns.set(style="white",palette='deep',color_codes=False)

#plt.xlim(0,8000)

sns.kdeplot(data=tech_ineffi_sequential.iloc[0,:],label="2005" )

#sns.kdeplot(data=sp_nonlinear.iloc[1,:],label="2006" ,shade=True)

#sns.kdeplot(data=tech_ineffi_sequential.iloc[2,:],label="2007")

sns.kdeplot(data=tech_ineffi_sequential.iloc[3,:],label="2008")

#sns.kdeplot(data=sp_nonlinear.iloc[4,:],label="2009" ,shade=True)

#sns.kdeplot(data=sp_nonlinear.iloc[5,:],label="2010" ,shade=True)

sns.kdeplot(data=tech_ineffi_sequential.iloc[6,:],label="2011" )

#sns.kdeplot(data=sp_nonlinear.iloc[7,:],label="2012" ,shade=True)

sns.kdeplot(data=tech_ineffi_sequential.iloc[8,:],label="2013")

#sns.kdeplot(data=sp_nonlinear.iloc[9,:],label="2014" ,shade=True)

sns.kdeplot(data=tech_ineffi_sequential.iloc[10,:],label="2015" )

sns.kdeplot(data=tech_ineffi_sequential.iloc[11,:],label="2016")

sns.despine()
lists=['South Asia','Eastern Europe','CIS','Western Asia','Eastern Asia','Central Asia']

def mean_calculate(df,lists):

    sum=[]

    for i in lists:

        temp=np.mean(df[i],axis=1)

        sum.append(temp)

    df_mean_plot=pd.concat(sum,axis=1)

    return df_mean_plot
df_linear_plot_11=mean_calculate(tech_ineffi_linear_11,lists)

df_linear_plot_11.columns=lists

df_linear_plot_10=mean_calculate(tech_ineffi_linear_10,lists)

df_linear_plot_10.columns=lists

df_linear_plot_01=mean_calculate(tech_ineffi_linear_01,lists)

df_linear_plot_01.columns=lists
df_nonlinear_plot=mean_calculate(tech_ineffi_sequential,lists)

df_nonlinear_plot.columns=lists

df_sequential_plot=mean_calculate(tech_ineffi_sequential,lists)

df_sequential_plot.columns=lists

df_whole_plot=mean_calculate(tech_ineffi_whole,lists)

df_whole_plot.columns=lists

df_nonlinear_plot.tail()
ax=df_whole_plot.plot(style=['+-', 'd--', 'o-.', '.-', 'v:','*-'])

ax.set_ylabel('Average Technical Efficiency')

ax.set_xlabel('Year')

sns.despine()
fig = plt.figure(figsize=(16,12))

ax1= fig.add_subplot(2,2,1)

ax2= fig.add_subplot(2,2,2)

ax3= fig.add_subplot(2,2,3)

ax4= fig.add_subplot(2,2,4)

df_nonlinear_plot.plot(ax=ax1,title='Nonlinear',style=['+-', 'd--', 'o-.', '.-', 'v:','*-'])

df_linear_plot_11.plot(ax=ax2,title='$(g_y,g_b)=(\\frac{\\sqrt{2}}{2},-\\frac{\\sqrt{2}}{2})$',style=['+-', 'd--', 'o-.', '.-', 'v:','*-'])

df_linear_plot_10.plot(ax=ax3,title='$(g_y,g_b)=(1,0)$',style=['+-', 'd--', 'o-.', '.-', 'v:','*-'])

df_linear_plot_01.plot(ax=ax4,title='$(g_y,g_b)=(0,-1)$',style=['+-', 'd--', 'o-.', '.-', 'v:','*-'])

#df_sequential_plot.plot

sns.despine()
reduction_potential_nonlinear=pd.read_csv(f'{path}/reduction_potential_nonlinear.csv',header=None)

reduction_potential_linear_11=pd.read_csv(f'{path}/reduction_potential_linear_11.csv',header=None)

reduction_potential_linear_10=pd.read_csv(f'{path}/reduction_potential_linear_10.csv',header=None)

reduction_potential_linear_01=pd.read_csv(f'{path}/reduction_potential_linear_01.csv',header=None)

reduction_potential_sequential=pd.read_csv(f'{path}/reduction_potential_sequential.csv',header=None)

reduction_potential_whole=pd.read_csv(f'{path}/reduction_potential_whole.csv',header=None)
df_co2=pd.read_csv(f'{path}/CO2.csv')

df_co2_new=df_co2.drop(['1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','API'],axis=1)
np.mean(reduction_potential_nonlinear,axis=1)
df_country.tail()
df_co2_final=pd.merge(df_country,df_co2_new,on='Country Name',how='left')

df_co2_final.iloc[:,2:14]=df_co2_final.iloc[:,2:14].astype(float)

for i in range(2,14):

    df_co2_final.iloc[:,i]=df_co2_final.iloc[:,i]/np.mean(df_co2_final.iloc[:,i])
df_co2_real=df_co2_final.drop(['Country Name','Region'],axis=1).T

df_co2_real.columns=df_country['Region']

df_co2_real.tail()
temp_1=np.mean(df_co2_real['South Asia'],axis=1)

temp_2=np.mean(df_co2_real['Eastern Europe'],axis=1)

temp_3=np.mean(df_co2_real['CIS'],axis=1)

temp_4=np.mean(df_co2_real['Western Asia'],axis=1)

temp_5=np.mean(df_co2_real['Eastern Asia'],axis=1)

temp_6=np.mean(df_co2_real['Central Asia'],axis=1)

temp=[temp_1,temp_2,temp_3,temp_4,temp_5,temp_6]
df_real_co2=pd.DataFrame(temp).T

lists=['South Asia','Eastern Europe','CIS','Western Asia','Eastern Asia','Central Asia']

df_real_co2.columns=lists

df_real_co2.tail()
reduction_potential_nonlinear.index=range(2005,2017)

reduction_potential_linear_11.index=range(2005,2017)

reduction_potential_linear_10.index=range(2005,2017)

reduction_potential_linear_01.index=range(2005,2017)

reduction_potential_sequential.index=range(2005,2017)

reduction_potential_whole.index=range(2005,2017)



reduction_potential_nonlinear.columns=df_country['Region']

reduction_potential_linear_11.columns=df_country['Region']

reduction_potential_linear_10.columns=df_country['Region']

reduction_potential_linear_01.columns=df_country['Region']

reduction_potential_sequential.columns=df_country['Region']

reduction_potential_whole.columns=df_country['Region']
import seaborn as sns

fig = plt.figure(figsize=(10,8))

sns.set(style="white",palette='deep',color_codes=False)

#plt.xlim(0,8000)

sns.kdeplot(data=reduction_potential_whole.iloc[0,:],label="2005" )

#sns.kdeplot(data=sp_nonlinear.iloc[1,:],label="2006" ,shade=True)

#sns.kdeplot(data=tech_ineffi_sequential.iloc[2,:],label="2007")

sns.kdeplot(data=reduction_potential_whole.iloc[3,:],label="2008")

#sns.kdeplot(data=sp_nonlinear.iloc[4,:],label="2009" ,shade=True)

#sns.kdeplot(data=sp_nonlinear.iloc[5,:],label="2010" ,shade=True)

sns.kdeplot(data=reduction_potential_whole.iloc[6,:],label="2011" )

#sns.kdeplot(data=sp_nonlinear.iloc[7,:],label="2012" ,shade=True)

sns.kdeplot(data=reduction_potential_whole.iloc[8,:],label="2013")

#sns.kdeplot(data=sp_nonlinear.iloc[9,:],label="2014" ,shade=True)

sns.kdeplot(data=reduction_potential_whole.iloc[10,:],label="2015" )

sns.kdeplot(data=reduction_potential_whole.iloc[11,:],label="2016")

sns.despine()
df_linear_plot_11=mean_calculate(reduction_potential_linear_11,lists)

df_linear_plot_11.columns=lists

df_linear_plot_10=mean_calculate(reduction_potential_linear_10,lists)

df_linear_plot_10.columns=lists

df_linear_plot_01=mean_calculate(reduction_potential_linear_01,lists)

df_linear_plot_01.columns=lists

df_nonlinear_plot=mean_calculate(reduction_potential_nonlinear,lists)

df_nonlinear_plot.columns=lists

df_sequential_plot=mean_calculate(reduction_potential_sequential,lists)

df_sequential_plot.columns=lists

df_whole_plot=mean_calculate(reduction_potential_whole,lists)

df_whole_plot.columns=lists

df_nonlinear_plot
df_whole_plot_new=df_whole_plot.copy()

df1=df_real_co2.copy()

m,n=np.shape(df1)

for i in range(n):

    for j in range(m):

        df_whole_plot_new.iloc[j,i]=df_whole_plot_new.iloc[j,i]/df1.iloc[j,i]
df_sequential_plot_new=df_sequential_plot.copy()

df1=df_real_co2.copy()

m,n=np.shape(df1)

for i in range(n):

    for j in range(m):

        df_sequential_plot_new.iloc[j,i]=df_sequential_plot_new.iloc[j,i]/df1.iloc[j,i]
df_nonlinear_plot_new=df_nonlinear_plot.copy()

df1=df_real_co2.copy()

m,n=np.shape(df1)

for i in range(n):

    for j in range(m):

        df_nonlinear_plot_new.iloc[j,i]=df_nonlinear_plot_new.iloc[j,i]/df1.iloc[j,i]
df_linear_plot_01_new=df_linear_plot_01.copy()

df1=df_real_co2.copy()

m,n=np.shape(df1)

for i in range(n):

    for j in range(m):

        df_linear_plot_01_new.iloc[j,i]=df_linear_plot_01_new.iloc[j,i]/df1.iloc[j,i]
df_linear_plot_10_new=df_linear_plot_10.copy()

df1=df_real_co2.copy()

m,n=np.shape(df1)

for i in range(n):

    for j in range(m):

        df_linear_plot_10_new.iloc[j,i]=df_linear_plot_10_new.iloc[j,i]/df1.iloc[j,i]
df_linear_plot_11_new=df_linear_plot_11.copy()

df1=df_real_co2.copy()

m,n=np.shape(df1)

for i in range(n):

    for j in range(m):

        df_linear_plot_11_new.iloc[j,i]=df_linear_plot_11_new.iloc[j,i]/df1.iloc[j,i]
df_real_co2
np.mean(df_whole_plot_new,0)
df_whole_plot_new
fig = plt.figure(figsize=(16,12))

ax=df_whole_plot_new.plot(style=['+-', 'd--', 'o-.', '.-', 'v:','*-'])

ax.set_ylabel('Average Reduction Potential')

ax.set_xlabel('Year')

sns.despine()
fig = plt.figure(figsize=(16,12))

ax1= fig.add_subplot(2,2,1)

ax2= fig.add_subplot(2,2,2)

ax3= fig.add_subplot(2,2,3)

ax4= fig.add_subplot(2,2,4)

df_nonlinear_plot_new.plot(ax=ax1,title='Nonlinear',style=['+-', 'd--', 'o-.', '.-', 'v:','*-'])

#df_sequential_plot_new.plot(ax=ax1,title='Sequential')

#df_nonlinear_plot_new.plot(ax=ax2,title='Nonlinear')

df_linear_plot_11_new.plot(ax=ax2,title='$(g_y,g_b)=(\\frac{\\sqrt{2}}{2},-\\frac{\\sqrt{2}}{2})$',style=['+-', 'd--', 'o-.', '.-', 'v:','*-'])

df_linear_plot_10_new.plot(ax=ax3,title='$(g_y,g_b)=(1,0)$',style=['+-', 'd--', 'o-.', '.-', 'v:','*-'])

df_linear_plot_01_new.plot(ax=ax4,title='$(g_y,g_b)=(0,-1)$',style=['+-', 'd--', 'o-.', '.-', 'v:','*-'])

sns.despine()
sp_nonlinear=pd.read_csv(f'{path}/shadowprice_nonlinear.csv',header=None)

sp_linear_11=pd.read_csv(f'{path}/shadowprice_linear_11.csv',header=None)

sp_linear_10=pd.read_csv(f'{path}/shadowprice_linear_10.csv',header=None)

sp_linear_01=pd.read_csv(f'{path}/shadowprice_linear_01.csv',header=None)

sp_sequential=pd.read_csv(f'{path}/shadow_price_sequential.csv',header=None)

sp_whole=pd.read_csv(f'{path}/shadow_price_whole.csv',header=None)



average_data=pd.read_csv(f'{path}/average_data_byyear.csv')
def scale_sp(df,average_data,df_country):

    m,n=np.shape(df)

    for i in range(m):

        df.iloc[i,:]=df.iloc[i,:]*average_data.iloc[3,i+1]/(average_data.iloc[4,i+1]*1000000)

    df.index=range(2005,2017)

    df.columns=df_country['Region']

    temp_1=np.mean(df['South Asia'],axis=1)

    temp_2=np.mean(df['Eastern Europe'],axis=1)

    temp_3=np.mean(df['CIS'],axis=1)

    temp_4=np.mean(df['Western Asia'],axis=1)

    temp_5=np.mean(df['Eastern Asia'],axis=1)

    temp_6=np.mean(df['Central Asia'],axis=1)

    temp=[temp_1,temp_2,temp_3,temp_4,temp_5,temp_6]

    df_new=pd.DataFrame(temp).T

    df_new.columns=lists

    df_new

    return df_new
fig = plt.figure(figsize=(16,12))

df_nonlinear_plot.plot(ax=ax1,title='Nonlinear',style=['+-', 'd--', 'o-.', '.-', 'v:','*-'])
sp_linear_10_new=scale_sp(sp_linear_10,average_data,df_country)

sp_linear_01_new=scale_sp(sp_linear_01,average_data,df_country)

sp_linear_11_new=scale_sp(sp_linear_11,average_data,df_country)

sp_nonlinear_new=scale_sp(sp_nonlinear,average_data,df_country)

sp_sequential_new=scale_sp(sp_sequential,average_data,df_country)

sp_whole_new=scale_sp(sp_whole,average_data,df_country)
fig = plt.figure(figsize=(16,12))

ax=sp_sequential_new.plot(style=['+-', 'd--', 'o-.', '.-', 'v:','*-'])

ax.set_ylabel('Average Shadow price')

ax.set_xlabel('Year')

sns.despine()
np.mean(sp_sequential_new,axis=1)
np.mean(sp_linear_11,axis=1)
fig = plt.figure(figsize=(20,8))

ax1= fig.add_subplot(1,2,1)

ax2= fig.add_subplot(1,2,2)

#ax3= fig.add_subplot(2,2,3)

#ax4= fig.add_subplot(2,2,4)

import seaborn as sns

sns.set(style="white",palette='deep',color_codes=False)

ax1.set_xlim(0,20000)

sns.kdeplot(data=sp_nonlinear.iloc[0,:],label="2005" ,shade=True,ax=ax1)

#sns.kdeplot(data=sp_nonlinear.iloc[1,:],label="2006" ,shade=True)

sns.kdeplot(data=sp_nonlinear.iloc[2,:],label="2007" ,shade=True,ax=ax1)

sns.kdeplot(data=sp_nonlinear.iloc[3,:],label="2008" ,shade=True,ax=ax1)

#sns.kdeplot(data=sp_nonlinear.iloc[4,:],label="2009" ,shade=True)

#sns.kdeplot(data=sp_nonlinear.iloc[5,:],label="2010" ,shade=True)

sns.kdeplot(data=sp_nonlinear.iloc[6,:],label="2011" ,shade=True,ax=ax1)

#sns.kdeplot(data=sp_nonlinear.iloc[7,:],label="2012" ,shade=True)

sns.kdeplot(data=sp_nonlinear.iloc[8,:],label="2013" ,shade=True,ax=ax1)

#sns.kdeplot(data=sp_nonlinear.iloc[9,:],label="2014" ,shade=True)

sns.kdeplot(data=sp_nonlinear.iloc[10,:],label="2015" ,shade=True,ax=ax1)

sns.kdeplot(data=sp_nonlinear.iloc[11,:],label="2016" ,shade=True,ax=ax1)

#sp_nonlinear_new.plot(ax=ax1,title='Nonlinear')

#sp_linear_11_new.plot(ax=ax2,title='$(g_y,g_b)=(1,-1)$')

#sp_linear_10_new.plot(ax=ax3,title='$(g_y,g_b)=(1,0)$')

#sp_linear_01_new.plot(ax=ax4,title='$(g_y,g_b)=(0,-1)$')

sns.despine()



ax2.set_xlim(0,4000)

sns.set(style="white",palette='deep',color_codes=False)

sns.kdeplot(data=sp_linear_11.iloc[0,:],label="2005" ,shade=True,ax=ax2)

#sns.kdeplot(data=sp_nonlinear.iloc[1,:],label="2006" ,shade=True)

sns.kdeplot(data=sp_linear_11.iloc[2,:],label="2007" ,shade=True,ax=ax2)

sns.kdeplot(data=sp_linear_11.iloc[3,:],label="2008" ,shade=True,ax=ax2)

#sns.kdeplot(data=sp_nonlinear.iloc[4,:],label="2009" ,shade=True)

#sns.kdeplot(data=sp_nonlinear.iloc[5,:],label="2010" ,shade=True)

sns.kdeplot(data=sp_linear_11.iloc[6,:],label="2011" ,shade=True,ax=ax2)

#sns.kdeplot(data=sp_nonlinear.iloc[7,:],label="2012" ,shade=True)

sns.kdeplot(data=sp_linear_11.iloc[8,:],label="2013" ,shade=True,ax=ax2)

#sns.kdeplot(data=sp_nonlinear.iloc[9,:],label="2014" ,shade=True)

sns.kdeplot(data=sp_linear_11.iloc[10,:],label="2015" ,shade=True,ax=ax2)

sns.kdeplot(data=sp_linear_11.iloc[11,:],label="2016" ,shade=True,ax=ax2)

sns.despine()
import seaborn as sns

fig = plt.figure(figsize=(10,8))

sns.set(style="white",palette='deep',color_codes=False)

plt.xlim(0,8000)

sns.kdeplot(data=sp_whole.iloc[0,:],label="2005" ,shade=True)

#sns.kdeplot(data=sp_nonlinear.iloc[1,:],label="2006" ,shade=True)

sns.kdeplot(data=sp_whole.iloc[2,:],label="2007" ,shade=True)

sns.kdeplot(data=sp_whole.iloc[3,:],label="2008" ,shade=True)

#sns.kdeplot(data=sp_nonlinear.iloc[4,:],label="2009" ,shade=True)

#sns.kdeplot(data=sp_nonlinear.iloc[5,:],label="2010" ,shade=True)

sns.kdeplot(data=sp_whole.iloc[6,:],label="2011" ,shade=True)

#sns.kdeplot(data=sp_nonlinear.iloc[7,:],label="2012" ,shade=True)

sns.kdeplot(data=sp_whole.iloc[8,:],label="2013" ,shade=True)

#sns.kdeplot(data=sp_nonlinear.iloc[9,:],label="2014" ,shade=True)

sns.kdeplot(data=sp_whole.iloc[10,:],label="2015" ,shade=True)

sns.kdeplot(data=sp_whole.iloc[11,:],label="2016" ,shade=True)

sns.despine()
import plotly_express as px	

gapminder = px.data.gapminder()	

gapminder2007 = gapminder.query('year == 2007')

gapminder2007.tail()
px.scatter(gapminder2007, x='gdpPercap', y='lifeExp',color='continent',size='pop',size_max=60)
df_country.index=df_country['Country Name']

sp_nonlinear_radar=sp_sequential.iloc[0,:]

#sp_nonlinear_radar=sp_nonlinear_radar*average_data['2005'][3]/(1000000*average_data['2005'][4])

sp_nonlinear_radar=pd.DataFrame(sp_nonlinear_radar).T

sp_nonlinear_radar.columns=df_country['Country Name']

sp_nonlinear_radar.index=['Shadow Price in 2005']

sp_nonlinear_radar_plot=sp_nonlinear_radar.append(df_country['Region'],ignore_index=True)

sp_nonlinear_radar_plot=sp_nonlinear_radar_plot.append(df_country['Country Name'],ignore_index=True)

sp_nonlinear_radar_plot.index=['shadow price in 2005','Region','Country Name']

sp_nonlinear_radar_plot_2005=sp_nonlinear_radar_plot.T

sp_nonlinear_radar_plot_2005.tail()

import plotly.express as px

wind = px.data.wind()

fig = px.scatter_polar(sp_nonlinear_radar_plot_2005, r="shadow price in 2005", theta="Country Name",

                   color="Region")

fig.update_layout(

    font_size=6,

)



fig.show()
df_country.index=df_country['Country Name']

sp_nonlinear_radar=sp_sequential.iloc[11,:]

#sp_nonlinear_radar=sp_nonlinear_radar*average_data['2005'][3]/(1000000*average_data['2005'][4])

sp_nonlinear_radar=pd.DataFrame(sp_nonlinear_radar).T

sp_nonlinear_radar.columns=df_country['Country Name']

sp_nonlinear_radar.index=['Shadow Price in 2005']

sp_nonlinear_radar_plot=sp_nonlinear_radar.append(df_country['Region'],ignore_index=True)

sp_nonlinear_radar_plot=sp_nonlinear_radar_plot.append(df_country['Country Name'],ignore_index=True)

sp_nonlinear_radar_plot.index=['shadow price in 2005','Region','Country Name']

sp_nonlinear_radar_plot_2005=sp_nonlinear_radar_plot.T

sp_nonlinear_radar_plot_2005.tail()

import plotly.express as px

wind = px.data.wind()

fig = px.scatter_polar(sp_nonlinear_radar_plot_2005, r="shadow price in 2005", theta="Country Name",

                   color="Region")

fig.update_layout(

    font_size=6,

)



fig.show()
df_country.index=df_country['Country Name']

sp_nonlinear_radar=sp_nonlinear.iloc[0,:]

#sp_nonlinear_radar=sp_nonlinear_radar*average_data['2005'][3]/(1000000*average_data['2005'][4])

sp_nonlinear_radar=pd.DataFrame(sp_nonlinear_radar).T

sp_nonlinear_radar.columns=df_country['Country Name']

sp_nonlinear_radar.index=['Shadow Price in 2005']

sp_nonlinear_radar_plot=sp_nonlinear_radar.append(df_country['Region'],ignore_index=True)

sp_nonlinear_radar_plot=sp_nonlinear_radar_plot.append(df_country['Country Name'],ignore_index=True)

sp_nonlinear_radar_plot.index=['shadow price in 2005','Region','Country Name']

sp_nonlinear_radar_plot_2005=sp_nonlinear_radar_plot.T

sp_nonlinear_radar_plot_2005.tail()

import plotly.express as px

wind = px.data.wind()

fig = px.scatter_polar(sp_nonlinear_radar_plot_2005, r="shadow price in 2005", theta="Country Name",

                   color="Region")

fig.update_layout(

    font_size=6,

)



fig.show()
sp_nonlinear_radar=sp_nonlinear.iloc[11,:]

#sp_nonlinear_radar=sp_nonlinear_radar*average_data['2005'][3]/(1000000*average_data['2005'][4])

sp_nonlinear_radar=pd.DataFrame(sp_nonlinear_radar).T

sp_nonlinear_radar.columns=df_country['Country Name']

sp_nonlinear_radar.index=['Shadow Price in 2005']

sp_nonlinear_radar_plot=sp_nonlinear_radar.append(df_country['Region'],ignore_index=True)

sp_nonlinear_radar_plot=sp_nonlinear_radar_plot.append(df_country['Country Name'],ignore_index=True)

sp_nonlinear_radar_plot.index=['shadow price in 2005','Region','Country Name']

sp_nonlinear_radar_plot_2005=sp_nonlinear_radar_plot.T

sp_nonlinear_radar_plot_2005.tail()

import plotly.express as px

wind = px.data.wind()

fig = px.scatter_polar(sp_nonlinear_radar_plot_2005, r="shadow price in 2005", theta="Country Name",

                   color="Region")

fig.update_layout(

    font_size=6,

)



fig.show()
df_country.index=df_country['Country Name']



sp_nonlinear_radar=sp_linear_10.iloc[0,:]

#sp_nonlinear_radar=sp_nonlinear_radar*average_data['2005'][3]/(1000000*average_data['2005'][4])

sp_nonlinear_radar=pd.DataFrame(sp_nonlinear_radar).T

sp_nonlinear_radar.columns=df_country['Country Name']

sp_nonlinear_radar.index=['Shadow Price in 2005']

sp_nonlinear_radar_plot=sp_nonlinear_radar.append(df_country['Region'],ignore_index=True)

sp_nonlinear_radar_plot=sp_nonlinear_radar_plot.append(df_country['Country Name'],ignore_index=True)

sp_nonlinear_radar_plot.index=['shadow price in 2005','Region','Country Name']

sp_nonlinear_radar_plot_2005=sp_nonlinear_radar_plot.T

sp_nonlinear_radar_plot_2005.tail()

import plotly.express as px

wind = px.data.wind()

fig = px.scatter_polar(sp_nonlinear_radar_plot_2005, r="shadow price in 2005", theta="Country Name",

                   color="Region")

fig.update_layout(

    font_size=6,

)



fig.show()
df_country.index=df_country['Country Name']



sp_nonlinear_radar=sp_linear_10.iloc[11,:]

#sp_nonlinear_radar=sp_nonlinear_radar*average_data['2016'][3]/(1000000*average_data['2016'][4])

sp_nonlinear_radar=pd.DataFrame(sp_nonlinear_radar).T

sp_nonlinear_radar.columns=df_country['Country Name']

sp_nonlinear_radar.index=['Shadow Price in 2016']

sp_nonlinear_radar_plot=sp_nonlinear_radar.append(df_country['Region'],ignore_index=True)

sp_nonlinear_radar_plot=sp_nonlinear_radar_plot.append(df_country['Country Name'],ignore_index=True)

sp_nonlinear_radar_plot.index=['shadow price in 2016','Region','Country Name']

sp_nonlinear_radar_plot_2016=sp_nonlinear_radar_plot.T

sp_nonlinear_radar_plot_2016.tail()

import plotly.express as px

wind = px.data.wind()

fig = px.scatter_polar(sp_nonlinear_radar_plot_2016, r="shadow price in 2016", theta="Country Name",

                   color="Region")

fig.update_layout(

    font_size=6,

)

fig.show()
df_country.index=df_country['Country Name']



sp_nonlinear_radar=sp_linear_11.iloc[0,:]

#sp_nonlinear_radar=sp_nonlinear_radar*average_data['2005'][3]/(1000000*average_data['2005'][4])

sp_nonlinear_radar=pd.DataFrame(sp_nonlinear_radar).T

sp_nonlinear_radar.columns=df_country['Country Name']

sp_nonlinear_radar.index=['Shadow Price in 2005']

sp_nonlinear_radar_plot=sp_nonlinear_radar.append(df_country['Region'],ignore_index=True)

sp_nonlinear_radar_plot=sp_nonlinear_radar_plot.append(df_country['Country Name'],ignore_index=True)

sp_nonlinear_radar_plot.index=['shadow price in 2005','Region','Country Name']

sp_nonlinear_radar_plot_2005=sp_nonlinear_radar_plot.T

sp_nonlinear_radar_plot_2005.tail()



import plotly.express as px

wind = px.data.wind()

fig = px.scatter_polar(sp_nonlinear_radar_plot_2005, r="shadow price in 2005", theta="Country Name",

                   color="Region")

fig.update_layout(

    font_size=6,

)



fig.show()
df_country.index=df_country['Country Name']



sp_nonlinear_radar=sp_linear_11.iloc[11,:]

#sp_nonlinear_radar=sp_nonlinear_radar*average_data['2016'][3]/(1000000*average_data['2016'][4])

sp_nonlinear_radar=pd.DataFrame(sp_nonlinear_radar).T

sp_nonlinear_radar.columns=df_country['Country Name']

sp_nonlinear_radar.index=['Shadow Price in 2016']

sp_nonlinear_radar_plot=sp_nonlinear_radar.append(df_country['Region'],ignore_index=True)

sp_nonlinear_radar_plot=sp_nonlinear_radar_plot.append(df_country['Country Name'],ignore_index=True)

sp_nonlinear_radar_plot.index=['shadow price in 2016','Region','Country Name']

sp_nonlinear_radar_plot_2016=sp_nonlinear_radar_plot.T

sp_nonlinear_radar_plot_2016.tail()

import plotly.express as px

wind = px.data.wind()

fig = px.scatter_polar(sp_nonlinear_radar_plot_2016, r="shadow price in 2016", theta="Country Name",

                   color="Region")

fig.update_layout(

    font_size=6,

)

fig.show()
sp_nonlinear_radar_plot_2016
tech_change_nonlinear=pd.read_csv(f'{path}/tech_change_nonlinear.csv',header=None)

tech_change_linear_11=pd.read_csv(f'{path}/tech_change_linear_11.csv',header=None)

tech_change_linear_10=pd.read_csv(f'{path}/tech_change_linear_10.csv',header=None)

tech_change_linear_01=pd.read_csv(f'{path}/tech_change_linear_01.csv',header=None)

tech_change_nonlinear_conventional=pd.read_csv(f'{path}/tech_change_nonlinear_conventional.csv',header=None)

tech_linear_sequential=pd.read_csv(f'{path}/tech_linear_sequential.csv',header=None)



effi_change_nonlinear=pd.read_csv(f'{path}/effi_change_nonlinear.csv',header=None)

effi_change_linear_11=pd.read_csv(f'{path}/effi_change_linear_11.csv',header=None)

effi_change_linear_10=pd.read_csv(f'{path}/effi_change_linear_10.csv',header=None)

effi_change_linear_01=pd.read_csv(f'{path}/effi_change_linear_01.csv',header=None)

tech_linear_sequential=pd.read_csv(f'{path}/tech_linear_sequential.csv',header=None)

effi_linear_sequential=pd.read_csv(f'{path}/effi_linear_sequential.csv',header=None)
tech_change=tech_change_nonlinear.copy()

tech_change.columns=df_country['Country Name']

tech_change.max().tail()
tech_linear_sequential.columns=df_country['Region']

tech_linear_sequential.index=['2005-2006','2006-2007','2007-2008','2008-2009','2009-2010',

                             '2010-2011','2011-2012','2012-2013',

                             '2013-2014','2014-2015','2015-2016']
effi_linear_sequential.columns=df_country['Region']

effi_linear_sequential.index=['2005-2006','2006-2007','2007-2008','2008-2009','2009-2010',

                             '2010-2011','2011-2012','2012-2013',

                             '2013-2014','2014-2015','2015-2016']
tech_change_nonlinear_conventional.columns=df_country['Region']

tech_change_nonlinear_conventional.index=['2005-2006','2006-2007','2007-2008','2008-2009','2009-2010',

                             '2010-2011','2011-2012','2012-2013',

                             '2013-2014','2014-2015','2015-2016']
tech_change_nonlinear.columns=df_country['Region']

tech_change_nonlinear.index=['2005-2006','2006-2007','2007-2008','2008-2009','2009-2010',

                             '2010-2011','2011-2012','2012-2013',

                             '2013-2014','2014-2015','2015-2016']
effi_change_nonlinear.columns=df_country['Region']

effi_change_nonlinear.index=['2005-2006','2006-2007','2007-2008','2008-2009','2009-2010',

                             '2010-2011','2011-2012','2012-2013',

                             '2013-2014','2014-2015','2015-2016']
tech_linear_sequential_plot=mean_calculate(tech_linear_sequential,lists)

tech_linear_sequential_plot.columns=lists

effi_linear_sequential_plot=mean_calculate(effi_linear_sequential,lists)

effi_linear_sequential_plot.columns=lists

fig = plt.figure(figsize=(8,6))

#ax1= fig.add_subplot(1,2,1)

#ax2= fig.add_subplot(1,2,2)

ax=tech_linear_sequential_plot.plot(style=['+-', 'd--', 'o-.', '.-', 'v:','*-'])

ax.set_ylabel('Technical Change')

sns.despine()
ax=effi_linear_sequential_plot.plot(style=['+-', 'd--', 'o-.', '.-', 'v:','*-'])

ax.set_ylabel('Efficiency Change')

sns.despine()
effi_linear_sequential_plot
tech_linear_sequential_plot
SLPI=tech_linear_sequential_plot+effi_linear_sequential_plot

ax=SLPI.plot(style=['+-', 'd--', 'o-.', '.-', 'v:','*-'])

ax.set_ylabel('SLPI')

sns.despine()
SLPI
tech_change_nonlinear_plot=mean_calculate(tech_change_nonlinear_conventional,lists)

tech_change_nonlinear_plot.columns=lists

effi_change_nonlinear_plot=mean_calculate(effi_change_nonlinear,lists)

effi_change_nonlinear_plot.columns=lists

fig = plt.figure(figsize=(16,6))

ax1= fig.add_subplot(1,2,1)

ax2= fig.add_subplot(1,2,2)

tech_change_nonlinear_plot.plot(ax=ax1,title='Technological Change',style=['+-', 'd--', 'o-.', '.-', 'v:','*-'])

effi_change_nonlinear_plot.plot(ax=ax2,title='Efficiency Change',style=['+-', 'd--', 'o-.', '.-', 'v:','*-'])



sns.despine()
tech_change_nonlinear_conventional['CIS']
tech_change_nonlinear_plot=mean_calculate(tech_change_nonlinear,lists)

tech_change_nonlinear_plot.columns=lists

effi_change_nonlinear_plot=mean_calculate(effi_change_nonlinear,lists)

effi_change_nonlinear_plot.columns=lists

fig = plt.figure(figsize=(16,6))

ax1= fig.add_subplot(1,2,1)

ax2= fig.add_subplot(1,2,2)

tech_change_nonlinear_plot.plot(ax=ax1,title='Technological Change')

effi_change_nonlinear_plot.plot(ax=ax2,title='Efficiency Change')



sns.despine()
np.mean(effi_linear_sequential_plot,axis=1)
np.mean(tech_linear_sequential_plot,axis=1)
np.mean(tech_linear_sequential_plot,axis=1)+np.mean(effi_linear_sequential_plot,axis=1)
np.mean(tech_change_linear_11,axis=1)
def effi_radar_plot(df):

    tech_change_nonlinear_radar=df

    tech_change_nonlinear_radar=pd.DataFrame(tech_change_nonlinear_radar).T

    tech_change_nonlinear_radar.columns=df_country['Country Name']

    tech_change_nonlinear_radar_new=tech_change_nonlinear_radar.append(df_country['Region'],ignore_index=True)

    tech_change_nonlinear_radar_new=tech_change_nonlinear_radar_new.append(df_country['Country Name'],ignore_index=True)

    tech_change_nonlinear_radar_new.index=['2005','Region','Country Name']

    tech_change_nonlinear_radar_new=tech_change_nonlinear_radar_new.T

    tech_change_nonlinear_radar_new.tail()

    import plotly.express as px

    wind = px.data.wind()

    fig = px.bar_polar(tech_change_nonlinear_radar_new, r="2005", theta="Country Name",

                   color="Region")

    fig.update_layout(

    font_size=6,

        )

    fig.show()
fig = plt.figure(figsize=(20,8))

effi_radar_plot(effi_change_linear_01.iloc[0,:])
effi_radar_plot(tech_change_linear_11.iloc[0,:])