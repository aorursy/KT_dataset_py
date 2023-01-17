import numpy as np 

import pandas as pd

import seaborn as sns



from matplotlib import pyplot as plt
df = pd.read_csv('/kaggle/input/us-border-crossing-data/Border_Crossing_Entry_Data.csv')

df['Date'] = pd.to_datetime(df['Date'])
d1 = df.groupby('Date').sum()

d1 = d1.sort_index()



plt.figure(figsize=(12,12))

plt.plot(d1.index,d1['Value'])

plt.tick_params(axis='both',labelsize=14)



plt.ylabel('Value (10 million)',fontsize=12)

plt.title('US-Canada/US-Mexico monthly border crossings',fontsize=12)





plt.show()
plt.figure(figsize=(12,12))

plt.plot(d1.index,d1['Value'])

plt.ylabel('Number of people (10 million)',fontsize=12)

plt.tick_params(axis='both',labelsize=14)

plt.title('US-Canada/US-Mexico monthly border crossings',fontsize=20)

year_loc = pd.Timestamp('1996-07-13') + pd.timedelta_range(start='0 day', periods=26, freq='365D')

year_lab = [x.year for x in year_loc]

plt.xticks(year_loc, year_lab,rotation=90)

plt.xlabel('Year',fontsize=20)





year_start = pd.Timestamp('1996-01-02') + pd.timedelta_range(start='0 day', periods=26, freq='365D')

for y in year_start:

    plt.axvline(y,color='red',alpha=0.4)





plt.figure(figsize=(12,12))

plt.plot(d1.index,d1['Value'])

plt.ylabel('Value (10 million)',fontsize=12)

plt.tick_params(axis='both',labelsize=14)

plt.title('US-Canada/US-Mexico monthly border crossings',fontsize=20)

year_loc = pd.Timestamp('1996-07-13') + pd.timedelta_range(start='0 day', periods=26, freq='365D')

year_lab = [x.year for x in year_loc]

plt.xticks(year_loc, year_lab,rotation=90)

plt.xlabel('Year',fontsize=20)





year_start = pd.Timestamp('1996-01-02') + pd.timedelta_range(start='0 day', periods=26, freq='365D')

for y in year_start:

    plt.axvline(y,color='red',alpha=0.4)

    



df1 = df[['Value','Date']].groupby('Date').sum()

df1['Date'] = df1.index

df1.index = np.arange(df1.count()[0])

df1['Year'] = df1['Date'].apply(lambda x: x.year)

max_vals = df1.sort_values('Value', ascending=False).drop_duplicates(['Year']).sort_values('Year')

plt.scatter(max_vals['Date'].values, max_vals['Value'],c='red')





plt.figure(figsize=(8,8))

months = max_vals['Date'].apply(lambda x: x.month)

plt.bar(months.groupby(months).count().index,months.groupby(months).count())

plt.xticks(np.arange(1,13),'Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec'.split(','))

plt.ylabel('Count')

plt.title('Months of maximum number of crossings')
plt.figure(figsize=(12,12))

plt.plot(d1.index,d1['Value'])

plt.ylabel('Number of people (10 million)',fontsize=12)

plt.tick_params(axis='both',labelsize=14)

plt.title('US-Canada/US-Mexico monthly border crossings',fontsize=20)

year_loc = pd.Timestamp('1996-07-13') + pd.timedelta_range(start='0 day', periods=26, freq='365D')

year_lab = [x.year for x in year_loc]

plt.xticks(year_loc, year_lab,rotation=90)

plt.xlabel('Year',fontsize=20)





year_start = pd.Timestamp('1996-01-02') + pd.timedelta_range(start='0 day', periods=26, freq='365D')

for y in year_start:

    plt.axvline(y,color='red',alpha=0.4)

    



df1 = df[['Value','Date']].groupby('Date').sum()

df1['Date'] = df1.index

df1.index = np.arange(df1.count()[0])

df1['Year'] = df1['Date'].apply(lambda x: x.year)

min_vals = df1.sort_values('Value', ascending=True).drop_duplicates(['Year']).sort_values('Year')

plt.scatter(min_vals['Date'].values, min_vals['Value'],c='red')
plt.figure(figsize=(8,8))

months = min_vals['Date'].apply(lambda x: x.month)

plt.bar(months.groupby(months).count().index,months.groupby(months).count())

plt.xticks(np.arange(1,13),'Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec'.split(','))

plt.ylabel('Count')

plt.title('Months of minimum number of crossings')
plt.figure(figsize=(8,8))



#Number of people crossing both borders

US_Canada = df[df['Border'] == 'US-Canada Border']['Value'].sum()

US_Mexico = df[df['Border'] == 'US-Mexico Border']['Value'].sum()

plt.bar(['Canada','Mexico'],height=[US_Canada,US_Mexico])

plt.title("Total number of crossings")

plt.ylabel("Value (1 billion)")
means_of_transport = df.groupby('Measure').sum()['Value']

means_of_transport = means_of_transport.sort_values(ascending=False)



truncated = means_of_transport[means_of_transport > 67036035]

truncated = truncated.append(pd.Series({'Others': means_of_transport[means_of_transport < 67036035].sum()}))



plt.figure(figsize=(10,10))

plt.pie(truncated,labels=truncated.index)

plt.title('Means of crossing (Both borders)')
means_of_tr_both_cntr = df.groupby(['Border','Measure']).sum()['Value']

plt.figure(figsize=(5,10))





tr_can = means_of_tr_both_cntr['US-Canada Border'].sort_values(ascending=False)[:5]



pd.Series({'Others': means_of_tr_both_cntr['US-Canada Border'].sort_values(ascending=False)[5:].sum()})

tr_can = tr_can.append(pd.Series({'Others': means_of_tr_both_cntr['US-Canada Border'].sort_values(ascending=False)[5:].sum()}))

plt.subplot(2,1, 1)

plt.pie(tr_can,

        labels=tr_can.index)

plt.title('US-Canada border')





plt.subplot(2, 1, 2)

tr_mex = means_of_tr_both_cntr['US-Mexico Border'].sort_values(ascending=False)[:5]

tr_mex = tr_mex.append(pd.Series({'Others': means_of_tr_both_cntr['US-Mexico Border'].sort_values(ascending=False)[5:].sum()}))

plt.pie(tr_mex,

        labels=tr_mex.index)

plt.title('US-Mexico border')
crossing_states = df.groupby(['Border','State']).sum()['Value']

cross_can = crossing_states['US-Canada Border']

cross_can = cross_can.sort_values(ascending=False)



cross_can_tr = cross_can[:7]

cross_can_tr = cross_can_tr.append(pd.Series({'Others': cross_can[7:].sum()}))



plt.figure(figsize=(10,10))

plt.pie(cross_can_tr.values,labels=cross_can_tr.index)





plt.title('Which state is most frequently entered into? US-Canada border')
#For mexico, we get

cross_mex = crossing_states['US-Mexico Border']

cross_mex = cross_mex.sort_values(ascending=False)



plt.figure(figsize=(10,10))

plt.pie(cross_mex.values,labels=cross_mex.index)



plt.title('Which state is most frequently entered into? US-Mexico border')