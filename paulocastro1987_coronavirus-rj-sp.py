# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import math

import seaborn as sns

from scipy.integrate import odeint



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import os

df = pd.read_csv('/kaggle/input/corona-virus-brazil/brazil_covid19.csv')

# Any results you write to the current directory are saved as output.
df.head()
rj = df[df['state'] == 'Rio de Janeiro']
rjc = rj[rj['cases']>0]

data = rjc['date'].values

casos = rjc['cases'].values

mortes = rjc['deaths'].values

plt.figure(figsize=(10,6))

plt.title('Casos do novo coronavírus no RJ')

k=sns.barplot(x='date',y='cases',data=rjc,palette="Blues_d")

k.set_xticklabels(k.get_xticklabels(), rotation=45)

plt.show()
def exponencial(td,z0):

    s1 = np.sum(td*np.log(z0))

    s2 = (np.sum(np.log(z0)))*(np.sum(td))/len(td)

    s3 = np.sum(np.dot(td,td))

    s4 = ((np.sum(td))**2)/len(td)

    A = (s1 - s2)/(s3 - s4)

    s5 = np.sum(np.log(z0))/len(td)

    s6 = np.sum(td)/len(td)

    B = math.exp(s5 - A*s6)



    r1 = len(td)*np.sum(td*np.log(z0))

    r2 = (np.sum(td)*np.sum(np.log(z0)))

    r3 = ((len(td))*(np.sum(np.dot(td,td)))) - (np.sum(td))**2

    r4 = ((len(td))*(np.sum(np.dot(np.log(z0),np.log(z0))))) - (np.sum(np.log(z0)))**2



    R2 = ((r1 - r2)**2)/(r3*r4)

    return A,B,R2
dias = np.array([0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16])

g = exponencial(dias,casos)

a = g[0]

b = g[1]
r2 = g[2]

r2
sp = df[df['state'] == 'São Paulo']

spc = sp[sp['cases']>0]

data_sp = spc['date'].values

casos_sp = spc['cases'].values

plt.figure(figsize=(10,6))

plt.title('Casos do novo coronavírus em SP')

k=sns.barplot(x='date',y='cases',data=spc,palette="Blues_d")

k.set_xticklabels(k.get_xticklabels(), rotation=45)

plt.show()
dias_sp = np.arange(len(spc))

casos_sp = spc['cases'].values

g_sp = exponencial(dias_sp,casos_sp)

a_sp = g_sp[0]

b_sp = g_sp[1]
r2s = g_sp[2]

r2s
caso_proj_rj = b*np.exp(a*dias)

days_proj_rj10 = np.arange(len(dias)+10)

caso_proj_rj10 = b*np.exp(a*days_proj_rj10)

caso_proj_sp = b_sp*np.exp(a_sp*dias_sp)

days_proj_sp10 = np.arange(len(dias_sp)+10)

caso_proj_sp10 = b_sp*np.exp(a_sp*days_proj_sp10)
fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2,figsize=(22,17))

ax1.grid(True)

ax1.set_title('Casos de Coronavírus no RJ em 16 dias',fontsize=18)

ax1.set_xlabel('Dias',fontsize=15)

ax1.set_ylabel('Casos',fontsize=15)

ax1.scatter(dias,casos,label='Casos reais')

ax1.plot(dias,caso_proj_rj,label='Ajuste exponencial',color='red')

handles, labels = ax1.get_legend_handles_labels()

ax1.legend(loc='best')



ax2.grid(True)

ax2.set_yscale('log')

ax2.set_xlabel('Dias',fontsize=15)

ax2.set_ylabel('Casos (escala logaritmica)',fontsize=15)

ax2.set_title('Casos projetados para RJ nos próximos 10 dias',fontsize=18)

ax2.tick_params(labelsize='large', width=3)

ax2.legend('Ajuste exponencial',loc='upper left')

ax2.plot(days_proj_rj10,caso_proj_rj10,color='green',label='Ajuste exponencial em escala log')

handles, labels = ax2.get_legend_handles_labels()

ax2.legend(loc='best')



ax3.grid(True)

ax3.set_title('Casos de Coronavírus no SP em 24 dias',fontsize=18)

ax3.set_xlabel('Dias',fontsize=15)

ax3.set_ylabel('Casos',fontsize=15)

ax3.scatter(dias_sp,casos_sp,label='Casos reais')

ax3.plot(dias_sp,caso_proj_sp,label='Ajuste exponencial',color='red')

handles, labels = ax3.get_legend_handles_labels()

ax3.legend(loc='best')



ax4.grid(True)

ax4.set_yscale('log')

ax4.set_xlabel('Dias',fontsize=15)

ax4.set_ylabel('Casos (escala logaritmica)',fontsize=15)

ax4.set_title('Casos projetados para SP nos próximos 10 dias',fontsize=18)

ax4.tick_params(labelsize='large', width=3)

ax4.legend('Ajuste exponencial',loc='upper left')

ax4.plot(days_proj_sp10,caso_proj_sp10,color='green',label='Ajuste exponencial em escala log')

handles, labels = ax4.get_legend_handles_labels()

ax4.legend(loc='best')

plt.show()
def modelo(x,t,N,beta,g):

    S = x[0]

    I = x[1]

    R = x[2]

    dSdt = -beta*I*S/N 

    dIdt = beta*I*S/N - g*I

    dRdt = g*I 

    return [dSdt,dIdt,dRdt]
beta_rj = g[0] 

beta_sp = g_sp[0]

gamma = 1.0/14.0 #1/DIAS DE RECUPERAÇÃO

N_rj = 1.646e7 #pop do RJ

N_sp = 4.404e7 #pop de SP

dias_sim = 300 #número de dias

I0 = 1 #condição inicial de infecção

R0 = 0

S0_rj = N_rj - I0 - R0

S0_sp = N_sp - I0 - R0

t = np.linspace(0,dias_sim,5*dias_sim)

x0_rj = [S0_rj,I0,R0] #vetor com condições iniciais no RJ

x0_sp = [S0_sp,I0,R0] #vetor com condições iniciais em SP

y_rj = odeint(modelo,x0_rj,t,args=(N_rj,beta_rj,gamma))

y_rj_130 = odeint(modelo,x0_rj,t,args=(N_rj,1.3*beta_rj,gamma))

y_rj_70 = odeint(modelo,x0_rj,t,args=(N_rj,0.5*beta_rj,gamma))

y_sp = odeint(modelo,x0_sp,t,args=(N_sp,beta_sp,gamma))

y_sp_130 = odeint(modelo,x0_sp,t,args=(N_sp,1.3*beta_sp,gamma))

y_sp_70 = odeint(modelo,x0_sp,t,args=(N_sp,0.5*beta_sp,gamma))
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(22,8))

ax1.grid(True)

ax1.set_xlim([0,dias_sim])

ax1.set_ylabel('Número de pessoas',fontsize=14)

ax1.set_xlabel('Dias',fontsize=14)

handles, labels = ax1.get_legend_handles_labels()

ax1.set_title('Perfil de contaminação por COVID-19 no RJ pelo modelo SIR',fontsize=13)

ax1.tick_params(labelsize='large', width=3)

ax1.plot(t,y_rj[:,0],'b',label = 'Suscetíveis')

ax1.plot(t,y_rj[:,1],'r',label = 'Infectados')

ax1.plot(t,y_rj[:,2],'g',label = 'Recuperados')

ax1.plot(t,y_rj_130[:,1],'r:',label = 'Infectados com 30% a mais de contaminação')

ax1.plot(t,y_rj_70[:,1],'r--',label = 'Infectados com 50% a menos de contaminação')

ax1.legend(loc='upper right')#,handles=[caso, caso_proj])



ax2.grid(True)

ax2.set_xlim([0,dias_sim])

ax2.set_ylabel('Número de pessoas',fontsize=14)

ax2.set_xlabel('Dias',fontsize=14)

handles, labels = ax2.get_legend_handles_labels()

ax2.set_title('Perfil de contaminação por COVID-19 em SP pelo modelo SIR',fontsize=13)

ax2.tick_params(labelsize='large', width=3)

ax2.plot(t,y_sp[:,0],'c',label = 'Suscetíveis')

ax2.plot(t,y_sp[:,1],'k',label = 'Infectados')

ax2.plot(t,y_sp[:,2],'m',label = 'Recuperados')

ax2.plot(t,y_sp_130[:,1],'k:',label = 'Infectados sem 30% a mais de contaminação')

ax2.plot(t,y_sp_70[:,1],'k--',label = 'Infectados com 50% a menos de contaminação')

ax2.legend(loc='upper right')

plt.show()