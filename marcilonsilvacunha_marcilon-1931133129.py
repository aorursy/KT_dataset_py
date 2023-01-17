# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/costa-rican-household-poverty-prediction/train.csv')

test = pd.read_csv('/kaggle/input/costa-rican-household-poverty-prediction/test.csv')



df.shape, test.shape
df_all = df.append(test)



df_all.shape
df['Target'].hist(grid = False, bins = 10)
df.Target.value_counts()/100
df_all.describe().T
df_all.select_dtypes('object').head()
df_all['edjefa'].value_counts()
mapeamento = {'yes': 1, 'no': 0}



df_all['edjefa'] = df_all['edjefa'].replace(mapeamento).astype(int)

df_all['edjefe'] = df_all['edjefe'].replace(mapeamento).astype(int)
df_all.select_dtypes('object').head()
df_all['dependency'].value_counts()
df_all['dependency'] = df_all['dependency'].replace(mapeamento).astype(float)
df_all.select_dtypes('object').head()
df_all.isnull().sum().sort_values()
data_na = df_all.isnull().sum().values / df_all.shape[0] *100

df_na = pd.DataFrame(data_na, index=df_all.columns, columns=['Count'])

df_na = df_na.sort_values(by=['Count'], ascending=False)



missing_value_count = df_na[df_na['Count']>0].shape[0]



print(f'We got {missing_value_count} rows which have missing value in train set ')

df_na.head(6)



# rez_esc represents "years behind in school", missing value could be filled as 0

# meaneduc represents "average years of education for adults (18+)", missing value could be filled as 0

# v18q1 really depends on v18q

# v2a1 depends on tipovivi3

# We do not really need SQBxxxx features for polynomial in our case, and i will use fillna as 0 after at the last step of feature engineering

df_all[df_all['parentesco1'] == 1]['v2a1'].isnull().sum()
df_all['v18q'].value_counts()
df_all['v2a1'].fillna(0, inplace=True)

df_all['v18q1'].fillna(0, inplace=True)

df_all['rez_esc'].fillna(0, inplace=True)

df_all['v2a1'].hist(grid = False, bins = 10)
df_all['v18q1'].hist(grid = False, bins = 10)
df_all.meaneduc.describe().T
df_all.SQBmeaned.describe().T
#df_all.loc[df_all.meaneduc.isnull(), "meaneduc"] = 0

#df_all.loc[df_all.SQBmeaned.isnull(), "SQBmeaned"] = 0



df_all['meaneduc'].fillna(df_all['meaneduc'].median(), inplace=True) 

df_all['SQBmeaned'].fillna(df_all['SQBmeaned'].median(), inplace=True)

df_all['meaneduc'].hist(grid = False, bins = 10)
df_all['SQBmeaned'].hist(grid = False, bins = 10)
df_all.isnull().sum().sort_values()
df_all.fillna(-1, inplace=True)

df_all['hsize-pc'] = df_all['hhsize'] / df_all['tamviv']

df_all['phone-pc'] = df_all['qmobilephone'] / df_all['tamviv']

df_all['tablets-pc'] = df_all['v18q1'] / df_all['tamviv']

df_all['rooms-pc'] = df_all['rooms'] / df_all['tamviv']

df_all['rent-pc'] = df_all['v2a1'] / df_all['tamviv']


import seaborn as sns



variables = ['Target', 'dependency', 'v2a1', 'v18q1', 'rez_esc', 'meaneduc' ,'SQBmeaned']



# Calculate the correlations

corr_mat = df_all[variables].corr().round(2)



# Draw a correlation heatmap

plt.rcParams['font.size'] = 12

plt.figure(figsize = (12, 12))

sns.heatmap(corr_mat, vmin = -0.5, vmax = 0.8, center = 0, 

            cmap = plt.cm.RdYlBu, annot = True);





feats = [c for c in df_all.columns if c not in ['Id', 'idhogar', 'Target']]
train, test = df_all[df_all['Target'] != -1], df_all[df_all['Target'] == -1]
heads = train[train['parentesco1'] == 1]
import lightgbm as lgb

from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold

#parameter value is copied from 

clf = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',

                             random_state=None, silent=True, metric='None', 

                             n_jobs=4, n_estimators=700, class_weight='balanced',

                             colsample_bytree =  0.93, min_child_samples = 95, num_leaves = 14, subsample = 0.96)





clf.fit(heads[feats], heads['Target'])



accuracy_score(heads['Target'], clf.predict(heads[feats]))
test['Target'] = clf.predict(test[feats]).astype(int)

test['Target'].value_counts(normalize=True)


#test[['Id', 'Target']].to_csv('submission.csv', index=False)
# Trabalhando com CatBoost

from catboost import CatBoostClassifier

cbc = CatBoostClassifier(random_state=42)

cbc.fit(heads[feats], heads['Target'])

accuracy_score(test['Target'], cbc.predict(test[feats]))
test['Target'] = cbc.predict(test[feats]).astype(int)

test['Target'].value_counts(normalize=True)

#test[['Id', 'Target']].to_csv('submission.csv', index=False)
fig=plt.figure(figsize=(15, 20))



pd.Series(cbc.feature_importances_, index=feats).sort_values().plot.barh()




rf = RandomForestClassifier(max_depth=None, random_state=42, n_jobs=4, n_estimators=700,

                            min_impurity_decrease=1e-3, min_samples_leaf=2,

                            verbose=0, class_weight='balanced')

rf.fit(heads[feats], heads['Target'])
test['Target'] = rf.predict(test[feats]).astype(int)
test['Target'].value_counts(normalize=True)

accuracy_score(heads['Target'], rf.predict(heads[feats]))
test[['Id', 'Target']].to_csv('submission.csv', index=False)

fig=plt.figure(figsize=(15, 20))



pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()
from sklearn.metrics import confusion_matrix

import itertools



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.figure(figsize = (10, 10))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title, size = 18)

    plt.colorbar(aspect=4)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45, size = 12)

    plt.yticks(tick_marks, classes, size = 12)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    

    # Labeling the plot

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt), fontsize = 16,

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")

        

    plt.grid(None)

    plt.tight_layout()

    plt.ylabel('True label', size = 12)

    plt.xlabel('Predicted label', size = 12)

    cm = confusion_matrix(heads['Target'], rf.predict(heads[feats]))



    plot_confusion_matrix(cm, classes = ['Extreme', 'Moderate', 'Vulnerable', 'Non-Vulnerable'],

                      title = 'Poverty Confusion Matrix')