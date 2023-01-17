import numpy as np

import pandas as pd



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold

#from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler, LabelEncoder

#from sklearn.neural_network import MLPClassifier



import matplotlib.pyplot as plt #plotting

import seaborn as sns #higher-lever plotting



import os 

print(os.listdir("../input")) # let's print available data

import warnings

warnings.filterwarnings('ignore') # ignore warnings
%%time

train_df = pd.read_csv('../input/X_train.csv')

test_df = pd.read_csv('../input/X_test.csv')

target_df = pd.read_csv('../input/y_train.csv')
train_df.head(n=5)
train_df.info()
len(train_df.measurement_number.value_counts())
target_df['surface'].value_counts().reset_index().rename(columns={'index': 'target'})
sns.set(style='whitegrid')

sns.countplot(y = 'surface',

              data = target_df,

              order = target_df['surface'].value_counts().index)

plt.show()
fig, ax = plt.subplots(1,1, figsize = (15,6));

corr = train_df.corr();

mask = np.zeros_like(corr);

mask[np.triu_indices_from(mask)] = True

hm = sns.heatmap(corr,

                ax = ax,

                mask = mask,

                cmap = 'Blues',

                annot = True,

                fmt = '.2f',

                linewidths = 0.05);

fig.subplots_adjust(top=0.93);

fig.suptitle('Features Correlation Heatmap', 

              fontsize=14, 

              fontweight='bold');
print('Are there NaNs in {}?: {}\n'.format('train_df',train_df.isnull().values.any())+

      'Are there NaNs in {}?: {}\n'.format('test_df',test_df.isnull().values.any())+

      'Are there NaNs in {}?: {}\n'.format('target_df',target_df.isnull().values.any()))
le = LabelEncoder()

le.fit(target_df['surface'])

target_df['surface'] = le.transform(target_df['surface'])
target_df['surface'].value_counts()
def get_features(df):

    result_df = pd.DataFrame()

    for col in df.columns:

        if col in ['row_id', 'series_id', 'measurement_number']:

            continue

        result_df['{}_mean'.format(col)] = df.groupby(['series_id'])[col].mean()

        result_df['{}_max'.format(col)] = df.groupby(['series_id'])[col].max()

        result_df['{}_min'.format(col)] = df.groupby(['series_id'])[col].min()

        result_df['{}_sum'.format(col)] = df.groupby(['series_id'])[col].sum()

        result_df['{}_mean_abs_change'.format(col)] = df.groupby(['series_id']\

        )[col].apply(lambda x: np.mean(np.abs(np.diff(x))))

    return result_df
%%time

train_df = get_features(train_df)

test_df = get_features(test_df)
# replace NAN to 0

train_df.fillna(0, inplace=True)

test_df.fillna(0, inplace=True)



# replace infinite value to zero

train_df.replace(-np.inf, 0, inplace=True)

train_df.replace(np.inf, 0, inplace=True)

test_df.replace(-np.inf, 0, inplace=True)

test_df.replace(np.inf, 0, inplace=True)
#Feature scaling

sc = StandardScaler()

train_df = sc.fit_transform(train_df)

test_df = sc.transform(test_df)
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=111222)

sub_preds_rf = np.zeros((test_df.shape[0],9))

oof_preds_rf = np.zeros((train_df.shape[0]))

score = 0

counter = 0



print('start training')



for train_index, test_index in folds.split(train_df, target_df['surface']):

    

    print('Fold {}'.format(counter+1))

    

    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)

    clf.fit(train_df[train_index], target_df['surface'][train_index])

    oof_preds_rf[test_index] = clf.predict(train_df[test_index])

    sub_preds_rf += clf.predict_proba(test_df) / folds.n_splits

    score += clf.score(train_df[test_index], target_df['surface'][test_index])

    counter += 1

    

    print('score : {}'.format(clf.score(train_df[test_index], target_df['surface'][test_index])))



print('avg accuracy : {}'.format(score / folds.n_splits))

submit = pd.read_csv('../input/sample_submission.csv')

submit['surface'] = le.inverse_transform(sub_preds_rf.argmax(axis=1))

submit.to_csv('submit.csv', index=False)



print('Ready')