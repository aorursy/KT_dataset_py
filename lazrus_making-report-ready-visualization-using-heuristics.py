# Libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import missingno as ms

from scipy import stats

from scipy.stats import norm, skew

%matplotlib inline



#Data

df = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')



#Looking at the Data

df.head(10)
#Simple And Default Histogram

plt.figure(figsize=(12,10))

plt.hist(df['platelets'],bins=30)

plt.ylabel('Frequency')

plt.xlabel('Platelets');
#Improved Visualization

plt.figure(figsize=(15,10))



#Using a low alpha and orange color

sns.distplot(df['platelets'],ax=plt.gca(),color='orange',fit=norm,kde_kws={'linewidth':2})

plt.tick_params(axis='both', which='major', labelsize=13) #adjusting ticks 



mu,sigma = norm.fit(df['platelets']) # adding additional information to increase Data Ink Ratio

plt.legend(['Normal dist. $\mu=$ {:.2f} and $\sigma=$ {:.2f}'.format(mu, sigma)],

            loc='best',frameon=False,fontsize=13)

    

sp = plt.gca().spines

sp['top'].set_visible(False)

sp['right'].set_visible(False)



plt.ylabel('Frequency',fontsize=14,labelpad=10)

plt.xlabel('Platelets (Counts)',fontsize=14,labelpad=10)

plt.title('Platelets Distribution',fontsize=17,pad=7)



plt.grid( alpha=0.5,color='lightslategrey');

# Data ink is reduced by adding a grid but in a plot like 

#this we cannot directly label the bar and hence grid
#Making Canvas

canv, axs = plt.subplots(1,2)

canv.set_size_inches(20,10)

canv.tight_layout(pad=4)



#First 'Before' Plot

plt.sca(axs[0])

plt.hist(df['platelets'],bins=30)

plt.ylabel('Frequency')

plt.xlabel('Platelets')

plt.title('Platelets Distribution W/O Using Heuristics')



#Second 'After' Plot 



plt.sca(axs[1])

sns.distplot(df['platelets'],ax=plt.gca(),color='orange',fit=norm,kde_kws={'linewidth':2.5}) 

plt.tick_params(axis='both', which='major', labelsize=13)



mu,sigma = norm.fit(df['platelets']) 

plt.legend(['Normal dist. $\mu=$ {:.2f} and $\sigma=$ {:.2f}'.format(mu, sigma)],

            loc='best',frameon=False,fontsize=13)

    

sp = plt.gca().spines

sp['top'].set_visible(False)

sp['right'].set_visible(False)



plt.ylabel('Frequency',fontsize=14,labelpad=10)

plt.xlabel('Platelets (Counts)',fontsize=14,labelpad=10)

plt.title('Platelets Distribution Using Heuristics',fontsize=17,pad=7)



plt.grid( alpha=0.5,color='lightslategrey');
df.head(2)
from sklearn.preprocessing import MinMaxScaler



cols = ['creatinine_phosphokinase','ejection_fraction','platelets','serum_sodium'] #Continous Features

df_cont = df[cols]



scale = MinMaxScaler(feature_range=(0,12))#Scaling to range of [0,12]

scaled = scale.fit_transform(df_cont)



df_sc = pd.DataFrame(data=scaled,columns=cols)

df_sc.head(5)
#Making Canvas

canv, axs = plt.subplots(2,2)

canv.set_size_inches(20,18)

canv.tight_layout(pad=10)



#Plotting



cnt = 0

for rw in axs:   # Little Bit of Automation is not bad right!!!

    for ax in rw:

        plt.sca(ax)

        sns.distplot(df_sc[cols[cnt]],ax=plt.gca(),color='orange',

                     fit=norm,kde_kws={'linewidth':2.5}) 

        plt.tick_params(axis='both', which='major', labelsize=13)



        mu,sigma = norm.fit(df_sc[cols[cnt]])  

        plt.legend(['Normal dist. $\mu=$ {:.2f} and $\sigma=$ {:.2f}'.format(mu, sigma)],

            loc=1,frameon=False,fontsize=13)

    

        sp = plt.gca().spines

        sp['top'].set_visible(False)

        sp['right'].set_visible(False)



        plt.ylabel('Frequency',fontsize=14,labelpad=10)

        plt.xlabel('{}'.format(cols[cnt]),fontsize=14,labelpad=10)

        plt.title('{} Distribution Using Heuristics'.format(cols[cnt]),fontsize=17,pad=10)



        plt.grid( alpha=0.5,color='lightslategrey');

        cnt += 1
df['time'] = pd.cut(df['time'],bins=5)

from sklearn.preprocessing import LabelEncoder

lbl = LabelEncoder().fit(df['time'])

df['time'] = lbl.transform(df['time'])
ccol = ['anaemia','diabetes','high_blood_pressure','sex','smoking','time']
pt = pd.pivot_table(df,index='DEATH_EVENT',columns=ccol[2],values='smoking',

                    aggfunc ='count').fillna(0)



#Making Canvas

canv, axs = plt.subplots(1,2)

canv.set_size_inches(20,10)

canv.tight_layout(pad=4)



#First 'Don't' plot

plt.sca(axs[0])

plt.bar(df[ccol[2]].value_counts().index-0.4,np.array(pt.query('DEATH_EVENT==["0"]'))[0],

        width=0.4,align='center',label='Not Dead')

plt.bar(df[ccol[2]].value_counts().index,np.array(pt.query('DEATH_EVENT==["1"]'))[0],

        width=0.4,align='center',label='Dead')

plt.xticks(df[ccol[2]].value_counts().index-0.2,df[ccol[2]].value_counts().index)

plt.ylabel('Number of Patients',fontsize=14)

plt.title(ccol[2],fontsize=14)



#Second Do's Plot

plt.sca(axs[1])

plt.title(ccol[2],fontsize=14)

        

bars = plt.bar(df[ccol[2]].value_counts().index-0.4,

        np.array(pt.query('DEATH_EVENT==["0"]'))[0],

      width=0.4,align='center',label='Not Dead',

        color='lightslategrey',alpha=0.9)

        

for bar,value in zip(bars,np.array(pt.query('DEATH_EVENT==["0"]'))[0]):

    plt.text((bar.get_x()+0.158),(bar.get_height()-5),'{}'.format(value),

             color='white',fontsize=18)

    

bars = plt.bar(df[ccol[2]].value_counts().index,

        np.array(pt.query('DEATH_EVENT==["1"]'))[0],

        width=0.4,align='center',label='Dead',

        color='orange',alpha=0.8)

        

for bar,value in zip(bars,np.array(pt.query('DEATH_EVENT==["0"]'))[0]):

    plt.text((bar.get_x()+0.158),(bar.get_height()-5),'{}'.format(value),

             color='white',fontsize=18)

        

plt.legend(fontsize=12,frameon=False)

plt.xticks(df[ccol[2]].value_counts().index-0.2,df[ccol[2]].value_counts().index)

plt.ylabel('Number of Patients',fontsize=14)

        

for key,spine in plt.gca().spines.items():

    spine.set_visible(False)

        

plt.tick_params(axis='x', which='both',length=0,labelsize=12)

plt.tick_params(axis='y', which='both',length=0,labelsize=0);
cnt = 0

canv , axs = plt.subplots(2,3,sharey=False)

canv.set_size_inches(20,18)

canv.tight_layout(pad=5)



for row in axs:   # Automation is Awesome!!!!!

    for axis in row:

        try:

            if ccol[cnt] != 'smoking':

                pt = pd.pivot_table(df,index='DEATH_EVENT',columns=ccol[cnt],values='smoking'

                                    ,aggfunc ='count').fillna(0)

            else:

                pt = pd.pivot_table(df,index ='DEATH_EVENT',columns=ccol[cnt],

                                    values ='diabetes',

                                    aggfunc ='count').fillna(0)

        except:

            continue

        

        plt.sca(axis)

        plt.title(ccol[cnt],fontsize=14)

        

        bars = plt.bar(df[ccol[cnt]].value_counts().index-0.4,

                np.array(pt.query('DEATH_EVENT==["0"]'))[0],

                width=0.4,align='center',label='Not Dead',

                color='lightslategrey',alpha=0.9)

        

        for bar,value in zip(bars,np.array(pt.query('DEATH_EVENT==["0"]'))[0]):

            plt.text((bar.get_x()+0.11),(bar.get_height()+1.1),'{}'.format(value),

                     color='k',fontsize=14)

        

        bars = plt.bar(df[ccol[cnt]].value_counts().index,

                np.array(pt.query('DEATH_EVENT==["1"]'))[0],

                width=0.4,align='center',label='Dead',

                color='orange',alpha=0.7)

        

        for bar,value in zip(bars,np.array(pt.query('DEATH_EVENT==["1"]'))[0]):

            plt.text((bar.get_x()+0.11),(bar.get_height()+1.1),'{}'.format(value),

                     color='k',fontsize=14)

        

        plt.legend(fontsize=12,frameon=False)

        plt.xticks(df[ccol[cnt]].value_counts().index-0.2,df[ccol[cnt]].value_counts().index)

        if cnt == 0 or cnt == 3:

            plt.ylabel('Number of Patients',fontsize=14)

        

        for key,spine in plt.gca().spines.items():

            spine.set_visible(False)

        

        plt.tick_params(axis='x', which='both',length=0,labelsize=12)

        plt.tick_params(axis='y', which='both',length=0,labelsize=0)

        cnt+=1