import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/us-accidents/US_Accidents_Dec19.csv')
print(df["Severity"].value_counts())
df['time'] = pd.to_datetime(df.Start_Time, format='%Y-%m-%d %H:%M:%S')



plt.subplots(2,2,figsize=(15,10))

for s in np.arange(1,5):

    plt.subplot(2,2,s)

    plt.hist(pd.DatetimeIndex(df.loc[df["Severity"] == s]['time']).month, bins=[1,2,3,4,5,6,7,8,9,10,11,12,13], align='left', rwidth=0.8)

    plt.title("Accident Count by Month with Severity " + str(s), fontsize=14)

    plt.xlabel("Month", fontsize=16)

    plt.ylabel("Accident Count", fontsize=16)

    plt.xticks(fontsize=16)

    plt.yticks(fontsize=16)

plt.tight_layout()

plt.show()
df['DayOfWeek'] = df['time'].dt.dayofweek

plt.subplots(2,2,figsize=(15,10))

for s in np.arange(1,5):

    plt.subplot(2,2,s)

    plt.hist(df.loc[df["Severity"] == s]['DayOfWeek'], bins=[0,1,2,3,4,5,6,7], align='left', rwidth=0.8)

    plt.title("Accident Count by Day with Severity " + str(s), fontsize=16)

    plt.xlabel("Day", fontsize=16)

    plt.ylabel("Accident Count", fontsize=16)

    plt.xticks(fontsize=16)

    plt.yticks(fontsize=16)

plt.tight_layout()

plt.show()
for s in np.arange(1,5):

    plt.subplots(figsize=(12,5))

    df.loc[df["Severity"] == s]['Weather_Condition'].value_counts().sort_values(ascending=False).head(20).plot.bar(width=0.5,color='y',edgecolor='k',align='center',linewidth=1)

    plt.xlabel('Weather Condition',fontsize=16)

    plt.ylabel('Accident Count',fontsize=16)

    plt.title('20 of The Main Weather Conditions for Accidents of Severity ' + str(s),fontsize=16)

    plt.xticks(fontsize=16)

    plt.yticks(fontsize=16)
for s in ["Fog","Light Rain","Rain","Heavy Rain","Snow"]:

    plt.subplots(1,2,figsize=(12,5))

    plt.suptitle('Accident Severity Under ' + s,fontsize=16)

    plt.subplot(1,2,1)

    df.loc[df["Weather_Condition"] == s]['Severity'].value_counts().plot.bar(width=0.5,color='y',edgecolor='k',align='center',linewidth=1)

    plt.xlabel('Severity',fontsize=16)

    plt.ylabel('Accident Count',fontsize=16)

    plt.xticks(fontsize=16)

    plt.yticks(fontsize=16)

    plt.subplot(1,2,2)

    df.loc[df["Weather_Condition"] == s]['Severity'].value_counts().plot.pie(autopct='%1.0f%%',fontsize=16)
factors = ['Temperature(F)','Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']



for factor in factors:

    # remove some of the extreme values

    factorMin = df[factor].quantile(q=0.0001)

    factorMax = df[factor].quantile(q=0.9999)

    # print df["Severity"].groupby(pd.cut(df[factor], np.linspace(factorMin,factorMax,num=20))).count()

    plt.subplots(figsize=(15,5))

    for s in np.arange(1,5):

        df["Severity"].groupby(pd.cut(df[factor], np.linspace(factorMin,factorMax,num=20))).mean().plot()

        plt.title("Mean Severity as a Function of " + factor, fontsize=16)

        plt.xlabel(factor + " Range", fontsize=16)

        plt.ylabel("Mean Severity", fontsize=16)

        plt.xticks(fontsize=11)

        plt.yticks(fontsize=16)
for s in ['Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']:

    # check if infrastructure type is found in any record 

    if (df[s] == True).sum() > 0:

        plt.subplots(1,2,figsize=(12,5))

        plt.xticks(fontsize=14)

        plt.suptitle('Accident Severity Near ' + s,fontsize=16)

        plt.subplot(1,2,1)

        df.loc[df[s] == True]['Severity'].value_counts().plot.bar(width=0.5,color='y',edgecolor='k',align='center',linewidth=1)

        plt.xlabel('Severity',fontsize=16)

        plt.ylabel('Accident Count',fontsize=16)

        plt.xticks(fontsize=16)

        plt.yticks(fontsize=16)

        plt.subplot(1,2,2)

        df.loc[df[s] == True]['Severity'].value_counts().plot.pie(autopct='%1.0f%%',fontsize=16)