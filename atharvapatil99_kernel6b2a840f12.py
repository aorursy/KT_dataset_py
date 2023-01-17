import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
file_path='../input/covid19-in-india/AgeGroupDetails.csv'

df=pd.read_csv(file_path, index_col='Sno')

df
df['AgeGroup'].replace('Missing', '0', inplace=True)

df
l1=list()

for item in df['Percentage']:

    item=item.replace('%','')

    l1.append(float(item))

    

df['Percentage']=l1
df.dtypes


fig, ax =plt.subplots(1,2,figsize=(14,5))









sns.barplot(x=df['AgeGroup'],y=df['Percentage'], ax=ax[0])





sns.lineplot(x=df['Percentage'],y=df['TotalCases'], ax=ax[1])

covid_19_path="../input/covid19-in-india/covid_19_india.csv"

covid_19_df=pd.read_csv(covid_19_path, index_col='Sno')
covid_19_df.head()
covid_19_df.tail(20)
state_list=covid_19_df['State/UnionTerritory'].tolist()

unique_state=covid_19_df['State/UnionTerritory'].unique().tolist()
sno_conf=list()

for item in unique_state:

    res = len(state_list) - 1 - state_list[::-1].index(item)

    sno_conf.append(res+1)


final_covid19_df=covid_19_df.loc[sno_conf,['State/UnionTerritory','Confirmed']]

    
final_covid19_df.reset_index(inplace=True)

del final_covid19_df['Sno']
final_covid19_df=final_covid19_df.sort_values(by='State/UnionTerritory')

final_covid19_df.set_index('State/UnionTerritory',inplace=True)
final_covid19_df=final_covid19_df.drop(['Unassigned','Nagaland#','Ladakh','Jharkhand#'])

final_covid19_df
plt.figure(figsize=(20,10))

sns.barplot(x=final_covid19_df['Confirmed'], y=final_covid19_df.index)
file_path_hospital="../input/covid19-in-india/HospitalBedsIndia.csv"

hospital_df=pd.read_csv(file_path_hospital, index_col='State/UT')

hospital_df.head(10)
import numpy as np

hospital_df['NumSubDistrictHospitals_HMIS'].replace(np.nan,0,inplace=True)
hospital_df.head()
hospital_df.tail()
hospital_df['Total Beds in State']=hospital_df['NumPublicBeds_HMIS']+hospital_df['NumUrbanBeds_NHP18']

del hospital_df['Sno']


hospital_df.head()

hospital_df=hospital_df.drop(['All India'])
hospital_df.drop(['Sikkim','Lakshadweep','Daman & Diu'],inplace=True)
plt.figure(figsize=(20,10))

sns.barplot(x=hospital_df['Total Beds in State'],y=hospital_df.index)
import numpy as np

ypos=np.arange(len(hospital_df.index))



plt.figure(figsize=(16,8))



plt.bar(ypos-0.2,hospital_df['Total Beds in State'],width=0.4,label='Total Beds in State',color='red')

plt.bar(ypos+0.2,final_covid19_df['Confirmed'],width=0.4,label='Confirmed',color='green')

plt.xticks(ypos,hospital_df.index,rotation=90)







plt.legend()

#show to show the graph

plt.show()

#savefig to save the figure in png with name 'graph'

plt.savefig('graph')

ICMR_Details_path="../input/covid19-in-india/ICMRTestingDetails.csv"

ICMR_Details_df=pd.read_csv(ICMR_Details_path, index_col='DateTime', parse_dates=True)
ICMR_Details_df.head()
plt.figure(figsize=(16,8))

sns.lineplot(data=ICMR_Details_df['TotalSamplesTested'],label='Samples Tested')

sns.lineplot(data=ICMR_Details_df['TotalIndividualsTested'],label='Individuals Tested')

sns.lineplot(data=ICMR_Details_df['TotalPositiveCases'],label='Positive Cases')

ICMR_Details_df.index
ICMR_labs_path="../input/covid19-in-india/ICMRTestingLabs.csv"

ICMR_labs_df=pd.read_csv(ICMR_labs_path)
ICMR_labs_df.head()
l1=ICMR_labs_df['state'].value_counts()
d1=pd.DataFrame(l1.index, columns=['State'])

d2=pd.DataFrame(l1.values, columns=['No. of labs'])

state_labs_df=pd.merge(d1,d2, left_index=True, right_index=True)
state_labs_df.sort_values('State', axis=0, ascending=True, inplace=True, na_position ='last')
state_labs_df.reset_index(inplace=True)

del state_labs_df['index']

state_labs_df
plt.figure(figsize=(10,8))

sns.barplot(x=state_labs_df['No. of labs'], y=state_labs_df['State'])
statewise_testing_path='../input/covid19-in-india/StatewiseTestingDetails.csv'

statewise_testing_df=pd.read_csv(statewise_testing_path, index_col='State',parse_dates=True)

statewise_testing_df['Date']=pd.to_datetime(statewise_testing_df['Date'])

statewise_testing_df.dtypes
df_mah=statewise_testing_df.loc['Maharashtra']

df_mah.set_index(['Date'],inplace=True)

df_mah
plt.figure(figsize=(15,7))

plt.xticks(rotation=90)

sns.lineplot(data=df_mah)

top10_states_df=final_covid19_df.sort_values(by='Confirmed', ascending=False).head(10)

top10_states_df.index[0]



count=0

fig, ax =plt.subplots(4,3,figsize=(25,18))



plt.figure(0)





g0=sns.lineplot(data=statewise_testing_df.loc[top10_states_df.index[0]].set_index(['Date']), ax=ax[0,0])

ax[0,0].set_title('MAHARASHTRA')



g1=sns.lineplot(data=statewise_testing_df.loc[top10_states_df.index[1]].set_index(['Date']), ax=ax[0,1])

g2=sns.lineplot(data=statewise_testing_df.loc[top10_states_df.index[2]].set_index(['Date']), ax=ax[0,2])

g3=sns.lineplot(data=statewise_testing_df.loc[top10_states_df.index[3]].set_index(['Date']), ax=ax[1,0])

g4=sns.lineplot(data=statewise_testing_df.loc[top10_states_df.index[4]].set_index(['Date']), ax=ax[1,1])

g5=sns.lineplot(data=statewise_testing_df.loc[top10_states_df.index[5]].set_index(['Date']), ax=ax[1,2])

g6=sns.lineplot(data=statewise_testing_df.loc[top10_states_df.index[6]].set_index(['Date']), ax=ax[2,0])

g7=sns.lineplot(data=statewise_testing_df.loc[top10_states_df.index[7]].set_index(['Date']), ax=ax[2,1])

g8=sns.lineplot(data=statewise_testing_df.loc[top10_states_df.index[8]].set_index(['Date']), ax=ax[2,2])

g9=sns.lineplot(data=statewise_testing_df.loc[top10_states_df.index[9]].set_index(['Date']), ax=ax[3,1])



for i in range(4):

    for j in range(3):

        if i==3 and j==0 or i==3 and j==2:

            break

        

        ax[i,j].set_title(top10_states_df.index[count])

        count=count+1



fig.delaxes(ax[3,0])

fig.delaxes(ax[3,2])



fig.autofmt_xdate(rotation=45)







        