import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

import datetime, time

%matplotlib inline

plt.rcParams["figure.figsize"] = [10,10]
################## USING THE PROVIDED CODE TO PREPARE THE DATA #########################



## 1. Reading data into a pandas DataFrame, and inspecting the columns a bit



df = pd.read_csv("../input/train_electricity.csv")

dfTest = pd.read_csv("../input/test_electricity.csv")

#insert column to test data to match formats

dfTest.insert(1, 'Consumption_MW', '')



print("Dataset has", len(df), "entries.")



print(f"\n\t{'Column':20s} | {'Type':8s} | {'Min':12s} | {'Max':12s}\n")

for col_name in df.columns:

    col = df[col_name]

    print(f"\t{col_name:20s} | {str(col.dtype):8s} | {col.min():12.1f} | {col.max():12.1f}")



## 2. Adding some datetime related features



def add_datetime_features(df):

    features = ["Year", "Week", "Day", "Dayofyear", "Month", "Dayofweek",

                "Is_year_end", "Is_year_start", "Is_month_end", "Is_month_start",

                "Hour", "Minute",]

    one_hot_features = ["Month", "Dayofweek"]



    datetime = pd.to_datetime(df.Date * (10 ** 9))



    df['Datetime'] = datetime  # We won't use this for training, but we'll remove it later



    for feature in features:

        new_column = getattr(datetime.dt, feature.lower())

        if feature in one_hot_features:

            df = pd.concat([df, pd.get_dummies(new_column, prefix=feature)], axis=1)

        else:

            df[feature] = new_column

    return df



df = add_datetime_features(df)

dfTest = add_datetime_features(dfTest)

df.columns



########################################## END ##############################################
print(df.isnull().sum().sum())

print(dfTest.isnull().sum().sum())
df.plot(x = 'Datetime', y = 'Consumption_MW', figsize=(20,15), linewidth = 0.5,  c = 'g', alpha = 0.5, xlim=(df['Datetime'].iloc[[0,-1]]), title = 'Electricity consumption over time'); 
f, ax = plt.subplots(1, 1)

df2 = df[(df['Consumption_MW'] > 1000) & (df['Consumption_MW'] < 10000)]

df2.plot(x = 'Datetime', y = 'Consumption_MW', figsize=(20,5), ax = ax, linewidth = 0.5, c = 'g', alpha = 0.5) 



dfSmoothed = df2.copy()

dfSmoothed['Consumption_MW'] = dfSmoothed['Consumption_MW'].rolling(window=20000,center=True).mean()

dfSmoothed.plot(x = 'Datetime', y = 'Consumption_MW', c = 'w', figsize=(20,5), ax = ax, xlim=(df['Datetime'].iloc[[0,-1]]))

ax.legend(('Real', 'Smoothed')); ax.set_title('Cleaner consumption data');
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))

color=iter(plt.cm.YlGn(np.linspace(0,1,9)))





for i in range(2010,2018):

    data = dfSmoothed[dfSmoothed['Year'] == i]

    c=next(color)

    ax1.plot(data.Dayofyear, data.Consumption_MW, c = c)

    ax2.bar(i, df2.loc[df2['Year'] == i,'Consumption_MW'].sum(), align='center', color = c, width = 0.4)

    

ax1.legend(('2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018'))

ax1.set_xlabel('Day of the year'); ax1.set_ylabel('Consumption (smoothed), MW');

ax1.set_title('Seasonal variation during the year (2010-2017)')

    

ax2.set_ylim(3.1e8, 3.75e8)

ax2.set_xlabel('Year'); ax2.set_ylabel('Total consumption, MW')

ax2.text(2012.6, 3.33e8, 'Price \nincreases'); ax2.text(2016.4, 3.69e8, 'Elon Musk \ndrops hot \nTesla Model 3');

ax2.set_title('Change of the annual electricity consumption');
color=iter(plt.cm.YlGn(np.linspace(0,1,9)))



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))

ax1 = plt.subplot(121, projection='polar')

for i in range(2010,2019):

    data = dfSmoothed[dfSmoothed['Year'] == i].dropna()

    c=next(color)

    ax1.plot(data.Dayofyear/365*2*np.pi, data.Consumption_MW, c = c)



ax1.set_rmin(6000)

ax1.set_title('365 days of each Year')

ax1.legend(('2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018'))

    

ax2 = plt.subplot(122,projection='polar')

for i in range(1,13):

    data = df2[df2.iloc[:,14+i]==1].dropna()

    ax2.bar(i/12*2*np.pi, data.Consumption_MW.sum(), width = 2*np.pi/14, color = 'g', alpha = 0.3)

    ax2.annotate('Month  '+str(i), xy=(i/12*2*np.pi, 2.4e8))



ax2.set_rmin(1.7e8)

ax2.set_title('Month-wise data summed over all the years');
f, ax1 = plt.subplots(1, 1, figsize=(20,10))

color=iter(plt.cm.YlGn(np.linspace(0,1,52)))

for i in range(0,52):

    data = df2[(df2['Year'] == 2017) &  (df2['Week'] == i)]

    c=next(color)

    ax1.plot(np.linspace(0,1,len(data)), data.Consumption_MW, c = c)



ax1.set_ylim(0); ax1.set_xlim(0,1)

ax1.text(0.005, 7500, 'Night'); ax1.text(0.035, 9500, 'Morning'); 

ax1.text(0.09, 9700, 'Evening'); ax1.text(0.15, 7500, 'Night'); 

ax1.annotate('EEML \nparticipant \npreparing \nKaggle \nsubmission', xy=( 0.875, 7300), xytext=(0.85, 8500), arrowprops=dict(arrowstyle='->'))

ax1.set_xlabel('Days of the week, Monday trough Sunday'); ax1.set_ylabel('Consumption, MW'); ax1.set_title('Weekly electricity consumption, 2017')

ax1.plot([0,1],[5000, 5000]);
#pd.plotting.scatter_matrix(df.loc[:,'Date':'Production_MW'], alpha=0.2, figsize=(20, 20), diagonal='kde')

#sns.pairplot(df2.loc[:,'Date':'Production_MW'], diag_kind="kde")



removed = df[(df['Consumption_MW'] < 1000) | (df['Consumption_MW'] > 10000)]

# data which has been removed



f, axs = plt.subplots(10,10, figsize=(20,20))

for m in range(10):

    for n in range(10):

        axs[m][n].set_axis_off()

        if m == 0:

            axs[m][n].set_title(df2.columns[n])

        if n == 0:

            axs[m][n].set_title(df2.columns[m], x = -0.1, y = 0.5, rotation = 'vertical')

        if m==n:

            axs[m][n].hist(df2.iloc[:,m], bins=100, color = 'r', alpha = 0.6)

            continue

        axs[m][n].scatter(df2.iloc[:,m], df2.iloc[:,n], s = 1, color = 'g', alpha = 0.2)

        axs[m][n].scatter(removed.iloc[:,m], removed.iloc[:,n], s = 10, color = 'r', alpha = 0.7)



corr = df2.loc[:,'Date':'Production_MW'].corr()

f, ax = plt.subplots(1,1, figsize=(20,10))

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(130, 10, as_cmap=True))



for i in range(np.size(corr, 0)):

    for j in range(np.size(corr, 1)):

        text = ax.text(j+0.5, i+0.5, round(corr.iloc[i, j],2), ha="center", va="center", color="w", size=15)
delta = np.mean(np.abs(df2['Consumption_MW']-df2['Production_MW']))

rmse = np.sqrt(np.mean((df2['Consumption_MW']-df2['Production_MW'])**2))

print('Mean absolute difference =', delta, 'MW')

print('RMSE = ', rmse, 'MW')



f, (ax1, ax2, ax3) = plt.subplots(3, 1)

ax1.fill_between(df2.iloc[50000:52000,0], df2.iloc[50000:52000,1]-rmse, df2.iloc[50000:52000,1]+rmse, color='g', alpha=0.1)

df2.iloc[50000:52000].plot(x = 'Date', y = 'Production_MW', figsize=(20,5), ax = ax1, color = 'r', alpha=0.7) 

df2.iloc[50000:52000].plot(x = 'Date', y = 'Consumption_MW', figsize=(20,5), ax = ax1, color = 'g', alpha=0.7)  



ax2.fill_between(df2.iloc[100000:102000,0], df2.iloc[100000:102000,1]-rmse, df2.iloc[100000:102000,1]+rmse, color='g', alpha=0.1)

df2.iloc[100000:102000].plot(x = 'Date', y = 'Production_MW', figsize=(20,5), ax = ax2, color = 'r', alpha=0.7) 

df2.iloc[100000:102000].plot(x = 'Date', y = 'Consumption_MW', figsize=(20,5), ax = ax2, color = 'g', alpha=0.7)  

ax2.legend(loc='center left');



ax3.fill_between(df2.iloc[150000:152000,0], df2.iloc[150000:152000,1]-rmse, df2.iloc[150000:152000,1]+rmse, color='g', alpha=0.1)

df2.iloc[150000:152000].plot(x = 'Date', y = 'Production_MW', figsize=(20,5), ax = ax3, color = 'r', alpha=0.7) 

df2.iloc[150000:152000].plot(x = 'Date', y = 'Consumption_MW', figsize=(20,5), ax = ax3, color = 'g', alpha=0.7)  



ax1.set_xlim(df2.Date.iloc[[50000,52000]]);

ax2.set_xlim(df2.Date.iloc[[100000,102000]]);

ax3.set_xlim(df2.Date.iloc[[150000,152000]]);





j = 0

out = np.empty([8,2])

for i in range(2010,2018):

    data = df2[df2['Year'] == i]

    out[j,0] = sum(data.Consumption_MW)

    out[j,1] = sum(data.Production_MW)

    j = j + 1

    

plt.xlabel('Year'); plt.ylabel('Total consumption, MW'); plt.legend(loc='upper left');

fig, ax = plt.subplots(figsize=(20,5)); width = 0.22;

ax.barh(np.arange(2010,2018) - width/2, out[:,0], width, label='Consumption_MW', alpha = 0.5, color = 'g')

ax.barh(np.arange(2010,2018) + width/2, out[:,1], width, label='Production_MW', alpha = 0.5, color = 'r')

plt.xlabel('Year'); plt.ylabel('Total consumption, MW'); plt.legend(loc='upper left');

out = np.empty([9,7]) 

j = 0

for i in range(2010,2019):

    out[j,:] = np.asarray(df2.loc[df2['Year'] == i,'Coal_MW':'Biomass_MW'].sum())

    out[j,:] /= sum(out[j,:])

    j += 1

    

dfRatio = pd.DataFrame(out, index= range(2010,2019), columns=df2.columns[2:9])

dfRatio.plot.barh(stacked=True, figsize = (17,10), colormap='YlGn', title='Ratio of eneregy sources in total production');

for i in range(np.size(out, 1)):

    for j in range(np.size(out, 0)):

        text = plt.text(i/20+0.7, j+0.35, round(out[j,i],2), ha="center", va="center", color="k")

plt.ylabel('Year');
fig, axs = plt.subplots(4, 2, figsize=(20, 20), sharex=True, sharey=False)



# data which has been removed

removed = df[(df['Consumption_MW'] < 1000) | (df['Consumption_MW'] > 10000)]



#concatenate with data for 2018 to have more information

dfC = pd.concat([df2, dfTest])





for i in range(4):

    for j in range(2):

        if i*2+j == 7: 

            break

        df2.plot(x = 'Datetime', y = df2.columns[i*2+j+2], ax = axs[i][j], linewidth = 0.2, c = 'g', alpha = 0.6)

        dfTest.plot(x = 'Datetime', y = dfTest.columns[i*2+j+2], ax = axs[i][j], linewidth = 0.2, c = 'r', alpha = 0.6)

        axs[i][j].plot_date(removed['Datetime'], removed.iloc[:,i*2+j+2], marker='o', ls = '', color='r')



dfCSmoothed = dfC.copy()

dfCSmoothed.loc[:,'Coal_MW':'Biomass_MW'] = dfCSmoothed.loc[:,'Coal_MW':'Biomass_MW'].rolling(window=20000,center=True).mean()

axs[3][1].plot_date(dfCSmoothed['Datetime'], dfCSmoothed.loc[:,'Coal_MW':'Biomass_MW'], marker='', ls='-')

axs[3][1].legend(dfCSmoothed.columns[2:9])

plt.tight_layout()
def recursive(nYears, resampled, xp, yp, pred, color, i):



    if nYears > 0:

        nYears -= 1

        resampled = resampled - pd.Timedelta('365 days')

        

        i, pred = recursive(nYears, resampled, xp, yp, pred, color, i)

        

        x = [time.mktime(t.timetuple()) for t in resampled]

        pred[:,i] = np.interp(x, xp, yp)

        c = next(color)

        ax1.plot(np.linspace(0,1,3000), pred[:3000,i], c = c)

        ax2.bar(i, pred[:,i].sum(), align='center', color = c, width = 0.3)

        i += 1  

        return i, pred      

    else:

        return i, pred



for nYears in range(2,6):

    fig = plt.figure(figsize = (20,4))

    gs = plt.GridSpec(1, 5, figure=fig)

    ax1 = fig.add_subplot(gs[0, :-1])

    ax2 = fig.add_subplot(gs[0, -1])



    color=iter(plt.cm.YlGn(np.linspace(0,1,nYears+4)))

    c = next(color);c = next(color);

    

    # select needed time interval

    resampled = dfTest.Datetime

    

    xp = np.asarray(df2.Date)

    yp = np.asarray(df2.Consumption_MW)

    pred = np.empty([len(resampled),nYears])

    i = 0

    

    i, pred = recursive(nYears, resampled, xp, yp, pred, color, i)



    predFinal = pred.mean(axis=1) * 1.06 



    ax1.plot(np.linspace(0,1,3000), predFinal[:3000], c = 'k')

    ax1.set_xlim(0,1); ax1.set(xlabel='time, a.u.', ylabel='Consumption, MW')

    

    ax2.bar(nYears, predFinal.sum(), align='center', color ='k', width = 0.3)

    ax2.set(xlabel='Year', ylabel='Total Consumption, MW')

    ax2.set_xticklabels(list(range(2017-nYears, 2019)))

    ax2.set_yticklabels([])

    

    # to save csv

    # pd.DataFrame(data={'Date': dfTest.Date, 'Consumption_MW': predFinal}).to_csv('res_'+str(nYears)+'.csv', index=False)