import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv('../input/train.csv')

df.head()
enumSeries = []

numSeries = []

for colname in df.columns:

    datatype = str(df[colname].dtype)

    if colname == 'Id' or colname == 'SalePrice':

        pass

    elif datatype == 'int64' or datatype == 'float64':

        numSeries.append(colname)

    else:

        enumSeries.append(colname)

df[numSeries].head()
import matplotlib.pyplot as plt

plt.hist(df['SalePrice'], bins=50)
plt.hist(np.log10(df['SalePrice']), bins=50)
numSeriesForVq = ['OverallQual', 'GrLivArea', 'GarageCars', '1stFlrSF']

df[numSeries].corrwith(np.log10(df['SalePrice'])).sort_values(ascending=False)
pd.concat([np.log10(df['GrLivArea']), np.log10(df['1stFlrSF'])], axis=1).corrwith(np.log10(df['SalePrice']))
ndf = pd.concat([df, pd.Series(np.log10(df['GrLivArea']),name='GrLivArea10'), pd.Series(np.log10(df['1stFlrSF']),name='1stFlrSF10'), pd.Series(np.log10(df['SalePrice']),name='SalePrice10')], axis=1,)

numSeriesForVq = ['OverallQual', 'GrLivArea10', 'GarageCars', '1stFlrSF10']

ndf[numSeriesForVq + ['SalePrice10'] + ['SalePrice']].head()
from sklearn.linear_model import LinearRegression

x_train = ndf[numSeriesForVq].values

y_train = ndf['SalePrice10'].values

lin_reg = LinearRegression()

lin_reg.fit(x_train, y_train)

colors = ['blue', 'red', 'green', 'orange']

choice = 0

for xval in numSeriesForVq:

    plt.scatter(ndf[xval], y_train, color=colors[choice])

    plt.xlabel(numSeriesForVq[choice])

    plt.ylabel('SalePrice10')

    plt.show()

    choice += 1
from matplotlib.ticker import FormatStrFormatter



choices = np.sort(np.random.choice(x_train.shape[0], 10))

predictions = np.power(10, lin_reg.predict(x_train))

targets = np.power(10, y_train)

preds = predictions[choices] / 1e3

acts = targets[choices] / 1e3

ind = np.arange(len(preds))

width = 0.35

fig, ax = plt.subplots()

rects1 = ax.bar(ind - width / 2, preds, width, label='Prediction')

rects2 = ax.bar(ind + width / 2, acts, width, label='Actual')

ax.set_ylabel('Price')

ax.set_xticks(ind)

ax.set_xticklabels(tuple(choices))

ax.legend()

ax.yaxis.set_major_formatter(FormatStrFormatter('%.0fk'))

fig.tight_layout()

plt.show()
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(predictions, targets)

mae
print(np.corrcoef(ndf['OverallQual'].values, np.log10(targets))[0][1])

print(np.corrcoef(np.log10(predictions), np.log10(targets))[0][1])
err = targets - predictions

df[numSeries].corrwith(pd.Series(err)).sort_values(ascending=False).head()
allEnums = []

for enumSeriesName in enumSeries:

    valueCounts = ndf[enumSeriesName].value_counts()

    for key in valueCounts.index:

        if valueCounts[key] > df.shape[0] / 10 and valueCounts[key] < 9 * df.shape[0] / 10:

            allEnums.append({'series': enumSeriesName, 'key': key})

allEnums
enumDetails = allEnums[0]

print(enumDetails['series'] + ' == "' + enumDetails['key'] + '"')

rdf = ndf.query(enumDetails['series'] + ' == "' + enumDetails['key'] + '"')

rdf.head()
allMean = ndf['SalePrice10'].mean()

allStd = ndf['SalePrice10'].std()

enumInds = []

enumScores = []

enumCount = []

for enumDetails in allEnums:

    rdf = ndf.query(enumDetails["series"] + ' == "' + enumDetails["key"] + '"')

    enumInds.append(enumDetails["series"] + '-' + enumDetails["key"])

    enumScores.append((rdf['SalePrice10'].mean() - allMean) / allStd)

    enumCount.append(rdf.shape[0])

series = pd.Series(enumScores, index = enumInds)

countSeries = pd.Series(enumCount, index=enumInds)

res = pd.DataFrame([series, series.abs(), countSeries]).transpose().rename(columns={0: 'NormalizedDiff', 1: 'Absolute', 2: 'Count'}).sort_values(by=['Absolute'], ascending=False).head()

res
def buildQuery(vals):

    query = ''

    for ind in range(len(vals)):

        if ind > 0:

            query += ' and '

        if vals[ind]:

            query += res.iloc[ind].name.replace('-', ' == "') + '"'

        else:

            query += res.iloc[ind].name.replace('-', ' != "') + '"'

    return  query



import itertools

lst = list(itertools.product([0, 1], repeat=3))

divisions = []

for val in lst:

    query = buildQuery(val)

    divisions.append({ "query": query, "df": ndf.query(query) })

    print(divisions[-1]['query'] + ': ' + str(divisions[-1]['df'].shape[0]))
import itertools

lst = list(itertools.product([0, 1], repeat=3))

divisions = []

for val in lst[0:4]:

    query = buildQuery(val)

    divisions.append({ "query": query, "df": ndf.query(query) })

    print(divisions[-1]['query'] + ': ' + str(divisions[-1]['df'].shape[0]))

query = 'MSZoning == "RM"'

divisions.append({ "query": query, "df": ndf.query(query) })

print(divisions[-1]['query'] + ': ' + str(divisions[-1]['df'].shape[0]))
for division in divisions:

    x_train = division['df'][numSeriesForVq].values

    y_train = division['df']['SalePrice10'].values

    lin_reg = LinearRegression()

    lin_reg.fit(x_train, y_train)

    division['regression'] = lin_reg

    division['x_train'] = x_train

    division['y_train'] = y_train
from matplotlib.ticker import FormatStrFormatter



predictions = np.array([])

targets = np.array([])

for division in divisions:

    predictions = np.concatenate((predictions, np.power(10, division['regression'].predict(division['x_train']))))

    targets = np.concatenate((targets, np.power(10, division['y_train'])))

choices = np.sort(np.random.choice(len(predictions), 10))

preds = predictions[choices] / 1e3

acts = targets[choices] / 1e3

ind = np.arange(len(preds))

width = 0.35

fig, ax = plt.subplots()

rects1 = ax.bar(ind - width / 2, preds, width, label='Prediction')

rects2 = ax.bar(ind + width / 2, acts, width, label='Actual')

ax.set_ylabel('Price')

ax.set_xticks(ind)

ax.set_xticklabels(tuple(choices))

ax.legend()

ax.yaxis.set_major_formatter(FormatStrFormatter('%.0fk'))

fig.tight_layout()

plt.show()
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(predictions, targets)

mae
print(np.corrcoef(np.log10(predictions), np.log10(targets))[0][1])
tdf = pd.read_csv('../input/test.csv')

ntdf = pd.concat([tdf, pd.Series(np.log10(tdf['GrLivArea']),name='GrLivArea10'),pd.Series(np.log10(tdf['1stFlrSF']),name='1stFlrSF10')], axis=1,)

ntdf = ntdf.fillna(pd.Series([5, 3, 3, 0], index=['OverallQual', 'GrLivArea10', 'GarageCars', '1stFlrSF10']))

for division in divisions:

    divisionTdf = ntdf.query(division['query'])

    division['tdf'] = divisionTdf

    division['x_test'] = divisionTdf[numSeriesForVq].values

    division['x_ids'] = divisionTdf['Id'].values

    division['predictions'] = np.power(10, division['regression'].predict(division['x_test']))

xf = pd.DataFrame(ntdf.query(divisions[4]['query'])[numSeriesForVq+['Id']])



predictions = np.array([])

ids = np.array([])

for division in divisions:

    predictions = np.concatenate((predictions, division['predictions']))

    ids = np.concatenate((ids, division['x_ids']))

res = pd.DataFrame([ids, predictions]).transpose().rename(columns={0: 'Id', 1: 'SalePrice'}).sort_values(by=['Id'])

res.Id = res.Id.astype(int)

res.to_csv('submission.csv', index=False)

res