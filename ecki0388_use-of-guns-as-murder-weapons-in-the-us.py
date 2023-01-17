import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import calendar

#We are only intereted in murder and manslaughter, so we discard the rest.

mm=pd.read_csv('../input/database.csv',low_memory=False).groupby('Crime Type').get_group('Murder or Manslaughter')
threshold=0.05

weaponCountM= mm[mm['Perpetrator Sex']=='Male']['Weapon'].value_counts(normalize=True).sort_values()

weaponCountF= mm[mm['Perpetrator Sex']=='Female']['Weapon'].value_counts(normalize=True).sort_values()

weaponCountM['Other']=weaponCountM[weaponCountM<threshold].sum()

weaponCountF['Other']=weaponCountF[weaponCountF<threshold].sum()

plt.figure()

plt.pie(weaponCountM[weaponCountM>threshold].values,labels=weaponCountM[weaponCountM>threshold].index)

plt.title('Weapons used by male Perpetrators')

plt.show()

plt.figure()

plt.pie(weaponCountF[weaponCountF>threshold].values,labels=weaponCountF[weaponCountF>threshold].index)

plt.title('Weapons used by female Perpetrators')

plt.show()
weaponsByYear=mm.groupby('Year')['Weapon'].value_counts(normalize=True)

#To get all firearms we have to include the tags Rifle, Shotgun, Gun and Firearm

GunsByYear=weaponsByYear.loc[:,'Handgun',:]+weaponsByYear.loc[:,'Rifle',:]+weaponsByYear.loc[:,'Shotgun',:]+weaponsByYear.loc[:,'Firearm',:]+weaponsByYear.loc[:,'Gun',:]



f,ax=plt.subplots(1)

weaponsByYear.loc[:,'Handgun',:].plot(ax=ax,label='Handguns')

GunsByYear.plot(ax=ax,label='All guns')

ax2=ax.twinx()

mm.groupby('Year').size().plot(ax=ax2,color='orange',label='Number of cases')

ax2.grid(False)

ax.legend(loc=0)

ax2.legend(loc=1)

plt.show()
#group the data

weaponCountStates= mm.groupby('State')['Weapon'].value_counts(normalize=True).sort_index()

wS=weaponCountStates.loc[:,'Handgun',:]+weaponCountStates.loc[:,'Rifle',:]+weaponCountStates.loc[:,'Shotgun',:]+weaponCountStates.loc[:,'Firearm',:]+weaponCountStates.loc[:,'Gun',:]

#in some states the term 'gun' is not used, this results in missing values if we add them up

wSnoGun=weaponCountStates.loc[:,'Handgun',:]+weaponCountStates.loc[:,'Rifle',:]+weaponCountStates.loc[:,'Shotgun',:]+weaponCountStates.loc[:,'Firearm',:]

for state in ['Maine','North Dakota','South Dakota','Vermont']:

    wS.loc[state]=wSnoGun.loc[state]



restrictive=['California','New Jersey','Massachusetts','New York','Connecticut','Hawaii','Maryland','Rhodes Island','Illnoise','Pennsylvania']

loose=['Louisiana','Mississippi','Arizona','Kentucky','Wyoming','Missouri','Alaska','South Dakota','Vermont','Kansas']

plt.figure(figsize=(12,12))

ax=sns.barplot(y=wS.sort_values(ascending=False).index,x=wS.sort_values(ascending=False).values)

for a in ax.get_yticklabels():

    if a.get_text() in restrictive:

        a.set_color('green')

    if a.get_text() in loose:

        a.set_color('orange')

plt.show()