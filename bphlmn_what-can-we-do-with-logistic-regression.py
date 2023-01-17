# import some packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# let's load in the data and look around

df_train = pd.read_json('../input/train.json')

df_train.info()
df_test = pd.read_json('../input/test.json')

df_test.info()
# coerce incidence angle to numeric

df_train['inc_angle'] = pd.to_numeric(df_train['inc_angle'], errors='coerce')



# combine training and test set for feature engineering

df_full = pd.concat([df_train, df_test], axis=0, ignore_index=True)
def get_stats(df, label=1):

    df['max'+str(label)] = [np.max(np.array(x)) for x in df['band_'+str(label)] ]

    df['maxpos'+str(label)] = [np.argmax(np.array(x)) for x in df['band_'+str(label)] ]

    df['min'+str(label)] = [np.min(np.array(x)) for x in df['band_'+str(label)] ]

    df['minpos'+str(label)] = [np.argmin(np.array(x)) for x in df['band_'+str(label)] ]

    df['med'+str(label)] = [np.median(np.array(x)) for x in df['band_'+str(label)] ]

    df['std'+str(label)] = [np.std(np.array(x)) for x in df['band_'+str(label)] ]

    df['mean'+str(label)] = [np.mean(np.array(x)) for x in df['band_'+str(label)] ]

    df['p25_'+str(label)] = [np.sort(np.array(x))[int(0.25*75*75)] for x in df['band_'+str(label)] ]

    df['p75_'+str(label)] = [np.sort(np.array(x))[int(0.75*75*75)] for x in df['band_'+str(label)] ]

    df['mid50_'+str(label)] = df['p75_'+str(label)]-df['p25_'+str(label)]



    return df



df_full = get_stats(df_full, 1)

df_full = get_stats(df_full, 2)
def plot_var(name, nbins=50):

    minval = df_full[name].min()

    maxval = df_full[name].max()

    plt.hist(df_full.loc[df_full.is_iceberg==1,name],range=[minval,maxval],

             bins=nbins,color='b',alpha=0.5,label='Boat')

    plt.hist(df_full.loc[df_full.is_iceberg==0,name],range=[minval,maxval],

             bins=nbins,color='r',alpha=0.5,label='Iceberg')

    plt.legend()

    plt.xlim([minval, maxval])

    plt.xlabel(name)

    plt.ylabel('Number')

    plt.show()

    

for col in ['inc_angle','min1','max1','std1','med1','mean1','mid50_1', 'p25_1', 'p75_1', 'minpos1', 'maxpos1']:

    plot_var(col)
for col in ['min2','max2','std2','med2','mean2','mid50_2','p25_2', 'p75_2']:

    plot_var(col)
df_full_stats = df_full.drop(['id','is_iceberg','band_1','band_2', 'inc_angle'], axis=1)

corr = df_full_stats.corr()

fig = plt.figure(1, figsize=(10,10))

plt.imshow(corr,cmap='inferno')

labels = np.arange(len(df_full_stats.columns))

plt.xticks(labels,df_full_stats.columns,rotation=90)

plt.yticks(labels,df_full_stats.columns)

plt.title('Correlation Matrix of Global Variables')

cbar = plt.colorbar(shrink=0.85,pad=0.02)

plt.show()
np_test = np.array(df_full.loc[1, 'band_1'])
len(np_test[np_test > -10])
def get_threshold_size(df, label=1, threshold=-15):

    df['gt_'+str(threshold)+'_'+str(label)] = [len(np.array(x)[np.array(x) > threshold]) for x in df_full['band_'+str(label)]]

    

    return df
for lbl in range(1, 3):

    for thr in range(-15, 35, 5):

        get_threshold_size(df_full, label=lbl, threshold=thr)
for col in ['gt_-15_1', 'gt_-10_1', 'gt_-5_1', 'gt_0_1', 'gt_5_1', 'gt_10_1', 'gt_15_1',

            'gt_20_1', 'gt_25_1', 'gt_30_1']:

    plot_var(col)
df_full.drop(['gt_15_1', 'gt_20_1', 'gt_25_1', 'gt_30_1'], axis=1, inplace=True)
for col in ['gt_-15_2', 'gt_-10_2', 'gt_-5_2', 'gt_0_2', 'gt_5_2', 'gt_10_2', 'gt_15_2',

            'gt_20_2', 'gt_25_2', 'gt_30_2']:

    plot_var(col)
df_full.drop(['gt_0_2', 'gt_5_2', 'gt_10_2', 'gt_15_2', 'gt_20_2', 'gt_25_2', 'gt_30_2'], 

             axis=1, inplace=True)
df_full.info()
df_full.drop(['inc_angle', 'maxpos1', 'minpos1', 'maxpos2', 'minpos2'], axis=1, inplace=True)
df_full.info()
X = df_full.drop(['band_1', 'band_2', 'id', 'is_iceberg'], axis=1)[:1604]

y = df_full['is_iceberg'][:1604]
# sklearn imports

from sklearn.cross_validation import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import log_loss

from sklearn.metrics import classification_report



skf = StratifiedKFold(y, n_folds=3)
for train_index, test_index in skf:

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    logmodel = LogisticRegression()

    logmodel.fit(X_train, y_train)

    predictions = logmodel.predict(X_test)

    predictions_prob = logmodel.predict_proba(X_test)[:,1]

    print(classification_report(y_test, predictions))

    print(log_loss(y_test, predictions_prob))
X_test = df_full.drop(['band_1', 'band_2', 'id', 'is_iceberg'], axis=1)[1604:]

logmodel = LogisticRegression()

logmodel.fit(X, y)

predictions_prob = logmodel.predict_proba(X_test)[:,1]
df_predictions = pd.DataFrame({'id' : df_full['id'][1604:], 'is_iceberg' : predictions_prob})

df_predictions.to_csv('logistic_regression_submission.csv', index=False)