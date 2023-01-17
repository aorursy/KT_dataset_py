import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
df = pd.read_csv('train.csv')
df.head()
# Compute the correlation matrix

corr = df.corr().abs()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.5, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
df.columns
X = df[['id', 'chem_0', 'chem_1', 'chem_2', 'chem_3', 'chem_4', 'chem_5',

       'chem_6', 'chem_7', 'attribute']]

y = df[['class']]
X_eng = ['chem_0', 'chem_1', 'chem_4', 'chem_5',

       'chem_6', 'attribute']

X_ = df[X_eng]
X_eng = ['chem_0', 'chem_1','chem_4', 'chem_5',

       'chem_6']

X_ = df[X_eng]
X_
df_test = pd.read_csv('test.csv')
from sklearn.metrics import precision_score



rfc = RandomForestClassifier(n_estimators=2000)

rfc.fit(X_, np.ravel(y))

predicted = rfc.predict(df_test[X_eng])

tmp=pd.concat([pd.DataFrame(df_test['id'], columns=['id']), pd.DataFrame(predicted, columns=['class'])], axis=1)

tmp['class'] = [int(x) for x in tmp['class']]

pd.DataFrame.to_csv(tmp, 'tmp.csv', index=False)

print(precision_score(y['class'], rfc.predict(X_), average='macro'))
X_new = X[['chem_1', 'chem_2', 'chem_4', 'chem_5','chem_6', 'attribute']]
from sklearn.metrics import precision_score



for i in range(30):

    rfc = RandomForestClassifier(n_estimators=2000, random_state=i)

    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3)

    rfc.fit(X_train, np.ravel(y_train))

    print("Random state : ", i, "Score : ", rfc.score(X_test, y_test))
rfc = RandomForestClassifier(n_estimators=2000)

# X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3)

rfc.fit(X_new, np.ravel(y))



predicted = rfc.predict(df_test[['chem_1', 'chem_2', 'chem_4', 'chem_5', 'chem_6', 'attribute']])

tmp=pd.concat([pd.DataFrame(df_test['id'], columns=['id']), pd.DataFrame(predicted, columns=['class'])], axis=1)

tmp['class'] = [int(x) for x in tmp['class']]

pd.DataFrame.to_csv(tmp, 'sub18.csv', index=False)

print("Score : ", precision_score(y['class'], rfc.predict(X_new), average='macro'))


from sklearn.ensemble import VotingClassifier

rfc_2000_5 = RandomForestClassifier(n_estimators=2000, random_state=5)

rfc_2000_9 = RandomForestClassifier(n_estimators=2000, random_state=9)

rfc_2000_11 = RandomForestClassifier(n_estimators=2000, random_state=11)

rfc_1800_5 = RandomForestClassifier(n_estimators=1800, random_state=5)

rfc_1800_9 = RandomForestClassifier(n_estimators=1800, random_state=9)

rfc_1800_11 = RandomForestClassifier(n_estimators=1800, random_state=11)

rfc_1500_5 = RandomForestClassifier(n_estimators=1500, random_state=5)

rfc_1500_9 = RandomForestClassifier(n_estimators=1500, random_state=9)

rfc_1500_11 = RandomForestClassifier(n_estimators=1500, random_state=11)
vc1 = VotingClassifier(estimators=[('rfc_2000_5', rfc_2000_5), ('rfc_2000_9', rfc_2000_9), ('rfc_2000_11', rfc_2000_11),

                            ('rfc_1800_5', rfc_1800_5), ('rfc_1800_9', rfc_2000_9), ('rfc_1800_11', rfc_1800_11),

                            ('rfc_1500_5', rfc_1500_5), ('rfc_1500_9', rfc_1500_9), ('rfc_1500_11', rfc_1500_11)],

                             voting='soft', n_jobs=-1)
vc1.fit(X_new, np.ravel(y))
predicted = vc1.predict(df_test[['chem_1', 'chem_4', 'chem_5', 'chem_6', 'attribute']])

tmp=pd.concat([pd.DataFrame(df_test['id'], columns=['id']), pd.DataFrame(predicted, columns=['class'])], axis=1)

tmp['class'] = [int(x) for x in tmp['class']]

pd.DataFrame.to_csv(tmp, 'tmp.csv', index=False)

print(precision_score(y['class'], rfc.predict(X_new), average='macro'))