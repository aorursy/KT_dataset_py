import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

print('Libraries ready')
mydata = pd.read_csv(r'../input/the-human-freedom-index/hfi_cc_2019.csv')

mydata.head()
rawdata = mydata[['year', 'ISO_code', 'countries', 'region', 'pf_ss_women_fgm', 

                  'pf_ss_women_inheritance', 'pf_ss_women', 'pf_movement_women', 'ef_money_growth', 

                  'ef_government', 'ef_legal', 'ef_money', 'ef_trade', 'ef_regulation', 'ef_score']].copy()

rawdata.head()
rawdata.info()

rawdata.describe()
rawdata['pf_ss_women_fgm'] = pd.to_numeric(rawdata['pf_ss_women_fgm'], errors = 'coerce')

rawdata['pf_ss_women_inheritance'] = pd.to_numeric(rawdata['pf_ss_women_inheritance'], errors = 'coerce')

rawdata['pf_ss_women'] = pd.to_numeric(rawdata['pf_ss_women'], errors = 'coerce')

rawdata['pf_movement_women'] = pd.to_numeric(rawdata['pf_movement_women'], errors = 'coerce')

rawdata['ef_money_growth'] = pd.to_numeric(rawdata['ef_money_growth'], errors = 'coerce')

rawdata['ef_government'] = pd.to_numeric(rawdata['ef_government'], errors = 'coerce')

rawdata['ef_legal'] = pd.to_numeric(rawdata['ef_legal'], errors = 'coerce')

rawdata['ef_money'] = pd.to_numeric(rawdata['ef_money'], errors = 'coerce')

rawdata['ef_trade'] = pd.to_numeric(rawdata['ef_trade'], errors = 'coerce')

rawdata['ef_regulation'] = pd.to_numeric(rawdata['ef_regulation'], errors = 'coerce')

rawdata['ef_score'] = pd.to_numeric(rawdata['ef_score'], errors = 'coerce')
# Let's look at the data one more time

rawdata.info()
rawdata = rawdata.dropna()

rawdata.info()
rawdata.hist(column = 'pf_ss_women')

plt.title('Frequency Distribution for Safety & Security')



rawdata.hist(column = 'pf_movement_women')

plt.title('Frequency Distribution for Freedom of Movement of Women')



rawdata.hist(column = 'ef_score')

plt.title('Frequency Distribution for Economic Freedom')
fig, ax = plt.subplots(figsize=(5, 5))

plt.title('Data in 2017')

rawdata[rawdata['year'] == 2017]['pf_ss_women'].plot.hist(color = 'red')

rawdata[rawdata['year'] == 2017]['ef_score'].plot.hist(color = 'darkorange')

rawdata[rawdata['year'] == 2017]['pf_movement_women'].plot.hist(color = 'gold')
fig, ax = plt.subplots(figsize=(5, 5))

plt.title('Data in 2016')

rawdata[rawdata['year'] == 2016]['pf_ss_women'].plot.hist(color = 'red')

rawdata[rawdata['year'] == 2016]['ef_score'].plot.hist(color = 'darkorange')

rawdata[rawdata['year'] == 2016]['pf_movement_women'].plot.hist(color = 'gold')



fig, ax = plt.subplots(figsize=(5, 5))

plt.title('Data in 2015')

rawdata[rawdata['year'] == 2015]['pf_ss_women'].plot.hist(color = 'red')

rawdata[rawdata['year'] == 2015]['ef_score'].plot.hist(color = 'darkorange')

rawdata[rawdata['year'] == 2015]['pf_movement_women'].plot.hist(color = 'gold')



fig, ax = plt.subplots(figsize=(5, 5))

plt.title('Data in 2014')

rawdata[rawdata['year'] == 2014]['pf_ss_women'].plot.hist(color = 'red')

rawdata[rawdata['year'] == 2014]['ef_score'].plot.hist(color = 'darkorange')

rawdata[rawdata['year'] == 2014]['pf_movement_women'].plot.hist(color = 'gold')



fig, ax = plt.subplots(figsize=(5, 5))

plt.title('Data in 2013')

rawdata[rawdata['year'] == 2013]['pf_ss_women'].plot.hist(color = 'red')

rawdata[rawdata['year'] == 2013]['ef_score'].plot.hist(color = 'darkorange')

rawdata[rawdata['year'] == 2013]['pf_movement_women'].plot.hist(color = 'gold')



fig, ax = plt.subplots(figsize=(5, 5))

plt.title('Data in 2012')

rawdata[rawdata['year'] == 2012]['pf_ss_women'].plot.hist(color = 'red')

rawdata[rawdata['year'] == 2012]['ef_score'].plot.hist(color = 'darkorange')

rawdata[rawdata['year'] == 2012]['pf_movement_women'].plot.hist(color = 'gold')



fig, ax = plt.subplots(figsize=(5, 5))

plt.title('Data in 2011')

rawdata[rawdata['year'] == 2011]['pf_ss_women'].plot.hist(color = 'red')

rawdata[rawdata['year'] == 2011]['ef_score'].plot.hist(color = 'darkorange')

rawdata[rawdata['year'] == 2011]['pf_movement_women'].plot.hist(color = 'gold')



fig, ax = plt.subplots(figsize=(5, 5))

plt.title('Data in 2010')

rawdata[rawdata['year'] == 2010]['pf_ss_women'].plot.hist(color = 'red')

rawdata[rawdata['year'] == 2010]['ef_score'].plot.hist(color = 'darkorange')

rawdata[rawdata['year'] == 2010]['pf_movement_women'].plot.hist(color = 'gold')



fig, ax = plt.subplots(figsize=(5, 5))

plt.title('Data in 2009')

rawdata[rawdata['year'] == 2009]['pf_ss_women'].plot.hist(color = 'red')

rawdata[rawdata['year'] == 2009]['ef_score'].plot.hist(color = 'darkorange')

rawdata[rawdata['year'] == 2009]['pf_movement_women'].plot.hist(color = 'gold')



fig, ax = plt.subplots(figsize=(5, 5))

plt.title('Data in 2008')

rawdata[rawdata['year'] == 2008]['pf_ss_women'].plot.hist(color = 'red')

rawdata[rawdata['year'] == 2008]['ef_score'].plot.hist(color = 'darkorange')

rawdata[rawdata['year'] == 2008]['pf_movement_women'].plot.hist(color = 'gold')
fig, ax = plt.subplots(figsize=(20,10))

plt.title('Safety & Security of Women by Region and Year')

plt.xlabel('Region')

plt.ylabel('Score')

sns.lineplot(x = 'region', y = 'pf_ss_women', data = rawdata[rawdata['year'] == 2017], label = '2017', ci = None)

sns.lineplot(x = 'region', y = 'pf_ss_women', data = rawdata[rawdata['year'] == 2016], label = '2016', ci = None)

sns.lineplot(x = 'region', y = 'pf_ss_women', data = rawdata[rawdata['year'] == 2015], label = '2015', ci = None)

sns.lineplot(x = 'region', y = 'pf_ss_women', data = rawdata[rawdata['year'] == 2014], label = '2014', ci = None)

sns.lineplot(x = 'region', y = 'pf_ss_women', data = rawdata[rawdata['year'] == 2013], label = '2013', ci = None)

sns.lineplot(x = 'region', y = 'pf_ss_women', data = rawdata[rawdata['year'] == 2012], label = '2012', ci = None)

sns.lineplot(x = 'region', y = 'pf_ss_women', data = rawdata[rawdata['year'] == 2011], label = '2011', ci = None)

sns.lineplot(x = 'region', y = 'pf_ss_women', data = rawdata[rawdata['year'] == 2010], label = '2010', ci = None)

sns.lineplot(x = 'region', y = 'pf_ss_women', data = rawdata[rawdata['year'] == 2009], label = '2009', ci = None)

sns.lineplot(x = 'region', y = 'pf_ss_women', data = rawdata[rawdata['year'] == 2008], label = '2008', ci = None)
fig, ax = plt.subplots(figsize=(20,10))

plt.title('Freedom of Movement of Women by Region and Year')

plt.xlabel('Region')

plt.ylabel('Score')

sns.lineplot(x = 'region', y = 'pf_movement_women', data = rawdata[rawdata['year'] == 2017], label = '2017', ci = None)

sns.lineplot(x = 'region', y = 'pf_movement_women', data = rawdata[rawdata['year'] == 2016], label = '2016', ci = None)

sns.lineplot(x = 'region', y = 'pf_movement_women', data = rawdata[rawdata['year'] == 2015], label = '2015', ci = None)

sns.lineplot(x = 'region', y = 'pf_movement_women', data = rawdata[rawdata['year'] == 2014], label = '2014', ci = None)

sns.lineplot(x = 'region', y = 'pf_movement_women', data = rawdata[rawdata['year'] == 2013], label = '2013', ci = None)

sns.lineplot(x = 'region', y = 'pf_movement_women', data = rawdata[rawdata['year'] == 2012], label = '2012', ci = None)

sns.lineplot(x = 'region', y = 'pf_movement_women', data = rawdata[rawdata['year'] == 2011], label = '2011', ci = None)

sns.lineplot(x = 'region', y = 'pf_movement_women', data = rawdata[rawdata['year'] == 2010], label = '2010', ci = None)

sns.lineplot(x = 'region', y = 'pf_movement_women', data = rawdata[rawdata['year'] == 2009], label = '2009', ci = None)

sns.lineplot(x = 'region', y = 'pf_movement_women', data = rawdata[rawdata['year'] == 2008], label = '2008', ci = None)
fig, ax = plt.subplots(figsize=(20,10))

plt.title('Economic Freedom by Region and Year')

plt.xlabel('Region')

plt.ylabel('Score')

sns.lineplot(x = 'region', y = 'ef_score', data = rawdata[rawdata['year'] == 2017], label = '2017', ci = None)

sns.lineplot(x = 'region', y = 'ef_score', data = rawdata[rawdata['year'] == 2016], label = '2016', ci = None)

sns.lineplot(x = 'region', y = 'ef_score', data = rawdata[rawdata['year'] == 2015], label = '2015', ci = None)

sns.lineplot(x = 'region', y = 'ef_score', data = rawdata[rawdata['year'] == 2014], label = '2014', ci = None)

sns.lineplot(x = 'region', y = 'ef_score', data = rawdata[rawdata['year'] == 2013], label = '2013', ci = None)

sns.lineplot(x = 'region', y = 'ef_score', data = rawdata[rawdata['year'] == 2012], label = '2012', ci = None)

sns.lineplot(x = 'region', y = 'ef_score', data = rawdata[rawdata['year'] == 2011], label = '2011', ci = None)

sns.lineplot(x = 'region', y = 'ef_score', data = rawdata[rawdata['year'] == 2010], label = '2010', ci = None)

sns.lineplot(x = 'region', y = 'ef_score', data = rawdata[rawdata['year'] == 2009], label = '2009', ci = None)

sns.lineplot(x = 'region', y = 'ef_score', data = rawdata[rawdata['year'] == 2008], label = '2008', ci = None)
chartdata = rawdata[rawdata['year'] == 2017][['pf_ss_women', 'pf_movement_women', 'ef_score']].copy()

corr = chartdata.corr()

fig, ax = plt.subplots(figsize=(10,10))

plt.title('2017 Matrix')

sns.heatmap(

    corr, 

    square=True,

    linewidths=.5,

    annot = True,

    ax = ax

)
chartdata = rawdata[rawdata['year'] == 2016][['pf_ss_women', 'pf_movement_women', 'ef_score']].copy()

corr = chartdata.corr()

fig, ax = plt.subplots(figsize=(10,10))

plt.title('2016 Matrix')

sns.heatmap(

    corr, 

    square=True,

    linewidths=.5,

    annot = True,

    ax = ax

)



chartdata = rawdata[rawdata['year'] == 2015][['pf_ss_women', 'pf_movement_women', 'ef_score']].copy()

corr = chartdata.corr()

fig, ax = plt.subplots(figsize=(10,10))

plt.title('2015 Matrix')

sns.heatmap(

    corr, 

    square=True,

    linewidths=.5,

    annot = True,

    ax = ax

)



chartdata = rawdata[rawdata['year'] == 2014][['pf_ss_women', 'pf_movement_women', 'ef_score']].copy()

corr = chartdata.corr()

fig, ax = plt.subplots(figsize=(10,10))

plt.title('2014 Matrix')

sns.heatmap(

    corr, 

    square=True,

    linewidths=.5,

    annot = True,

    ax = ax

)



chartdata = rawdata[rawdata['year'] == 2013][['pf_ss_women', 'pf_movement_women', 'ef_score']].copy()

corr = chartdata.corr()

fig, ax = plt.subplots(figsize=(10,10))

plt.title('2013 Matrix')

sns.heatmap(

    corr, 

    square=True,

    linewidths=.5,

    annot = True,

    ax = ax

)



chartdata = rawdata[rawdata['year'] == 2012][['pf_ss_women', 'pf_movement_women', 'ef_score']].copy()

corr = chartdata.corr()

fig, ax = plt.subplots(figsize=(10,10))

plt.title('2012 Matrix')

sns.heatmap(

    corr, 

    square=True,

    linewidths=.5,

    annot = True,

    ax = ax

)



chartdata = rawdata[rawdata['year'] == 2011][['pf_ss_women', 'pf_movement_women', 'ef_score']].copy()

corr = chartdata.corr()

fig, ax = plt.subplots(figsize=(10,10))

plt.title('2011 Matrix')

sns.heatmap(

    corr, 

    square=True,

    linewidths=.5,

    annot = True,

    ax = ax

)



chartdata = rawdata[rawdata['year'] == 2010][['pf_ss_women', 'pf_movement_women', 'ef_score']].copy()

corr = chartdata.corr()

fig, ax = plt.subplots(figsize=(10,10))

plt.title('2010 Matrix')

sns.heatmap(

    corr, 

    square=True,

    linewidths=.5,

    annot = True,

    ax = ax

)



chartdata = rawdata[rawdata['year'] == 2009][['pf_ss_women', 'pf_movement_women', 'ef_score']].copy()

corr = chartdata.corr()

fig, ax = plt.subplots(figsize=(10,10))

plt.title('2009 Matrix')

sns.heatmap(

    corr, 

    square=True,

    linewidths=.5,

    annot = True,

    ax = ax

)



chartdata = rawdata[rawdata['year'] == 2008][['pf_ss_women', 'pf_movement_women', 'ef_score']].copy()

corr = chartdata.corr()

fig, ax = plt.subplots(figsize=(10,10))

plt.title('2008 Matrix')

sns.heatmap(

    corr, 

    square=True,

    linewidths=.5,

    annot = True,

    ax = ax

)
datatrend = {'Year': [2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008],

            'Safety & Security | Economic Freedom': [0.44, 0.5, 0.49, 0.45, 0.45, 0.46, 0.46, 0.4, 0.44, 0.45],

            'Movement | Economic Freedom': [0.26, 0.41, 0.39, 0.36, 0.37, 0.5, 0.48, 0.18, 0.24, 0.21]

            }

datatrend = pd.DataFrame(datatrend, columns = ['Year', 'Safety & Security | Economic Freedom', 

                                               'Movement | Economic Freedom'])

datatrend.head(10)
plt.title('Degree of Correlation by Year')

plt.xlabel('Year')

sns.regplot(x = 'Year', y = 'Safety & Security | Economic Freedom', data = datatrend, color = 'red')
plt.title('Degree of Correlation by Year')

plt.xlabel('Year')

sns.regplot(x = 'Year', y = 'Movement | Economic Freedom', data = datatrend, color = 'blue')