import warnings

warnings.filterwarnings("ignore", category=FutureWarning)



import pandas as pd

from pandas.api.types import CategoricalDtype



import numpy as np

from scipy import sparse



import string



import matplotlib.pyplot as plt

import matplotlib.style as style

style.use('seaborn-bright')



import seaborn as sns



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go



init_notebook_mode(connected=True)





from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import average_precision_score

from sklearn.metrics import precision_recall_curve

from sklearn.utils import resample

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import OneHotEncoder
train = pd.read_csv('../input/cat-in-the-dat/train.csv')

test = pd.read_csv('../input/cat-in-the-dat/test.csv')
train.head()
Combined_data = pd.concat([train.drop(['target'], axis=1), test], axis=0, ignore_index=True)

print('Shape of training dataset: {}'.format(train.shape))

print('Shape of testing dataset: {}'.format(test.shape))

print('Shape of combined dataset: {}'.format(Combined_data.shape))
total = Combined_data.isnull().sum().sort_values(ascending=False)

percent = (Combined_data.isnull().sum())/Combined_data.isnull().count().sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'], sort=False).sort_values('Total', ascending=False)

missing_data.head(5)
Combined_data.columns
X_train = train.drop(['target'], axis=1)

y_train = train['target']
bin_cats = X_train[['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']]

bin_cats.bin_3 = 1 *(bin_cats.bin_3 == 'T')

bin_cats.bin_4 = 1 *(bin_cats.bin_3 == 'Y') 

bin_cats.head()
nom_cats = X_train[['nom_1','nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']]

nom_encoder=OneHotEncoder()

nom_encoder.fit(nom_cats)

nom_cats = nom_encoder.transform(nom_cats)

nom_cats
ord_cat_names = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']

ord_cats = X_train[ord_cat_names]

ord_cats.ord_1 = pd.factorize(ord_cats.ord_1.astype(CategoricalDtype(categories=["Novice", "Contibutor", "Expert", "Master", "Grandmaster"], ordered=True)))[0]

ord_cats.ord_2 = pd.factorize(ord_cats.ord_2.astype(CategoricalDtype(categories=["Freezing", "Cold", "Warm", "Hot", "Boiling Hot", "Lava Hot"], ordered=True)))[0]

ord_cats.ord_3 = ord_cats['ord_3'].apply(lambda x: ord(x)-ord('a'))

ord_cats.ord_4 = ord_cats['ord_4'].apply(lambda x: ord(x)-ord('A'))



d = {}

for i, s in enumerate(string.ascii_letters):

    d[s] = i



ord_cats.ord_5 = ord_cats['ord_5'].apply(lambda x: 10*d[x[0]]+ d[x[1]])

ord_cats.head()
cyclic_cats = X_train[['day', 'month']]
X = np.concatenate((bin_cats.values, ord_cats.values), axis=1)

X = sparse.csr_matrix(X)

X = sparse.hstack((X, nom_cats))



y = train['target']
print('The original input representation has shape {}'.format(train.shape))

print('The one-hot encoded input representation has shape {}'.format(X.shape))
data = [

    go.Bar(

        y=train['target'].value_counts().to_dense().keys(),

        x=train['target'].value_counts(),

        orientation='h',

        text="d",

    )]

layout = go.Layout(

    height=500,

    title='Target populations',

    hovermode='closest',

    xaxis=dict(title='Training set count', ticklen=5, zeroline=False, gridwidth=2, domain=[0.1, 1]),

    yaxis=dict(title='', ticklen=5, gridwidth=2),

    showlegend=False

)

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='Target Populations')
X_train, X_val, y_train, y_val = train_test_split(X, y, 

                                                    test_size=0.2, 

                                                    random_state=42)

print('The shape of the training input is {}'.format(X_train.shape))

print('The shape of the validation input is {}'.format(X_val.shape))

print('The shape of the training output is {}'.format(y_train.shape))

print('The shape of the validation output is {}'.format(y_val.shape))
def adaboost(X_train, X_val, y_train):

    model = AdaBoostClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    return y_pred
y_baseline = adaboost(X_train, X_val, y_train)
#SMOTE

sm = SMOTE(random_state=42)

X_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)

y_smote = adaboost(X_train_sm, X_val, y_train_sm)
#RUS

X_full = X_train.copy()

X_full['target'] = y_train

X_maj = X_full[X_full.target == 0]

X_min = X_full[X_full.target == 1]

X_maj_rus = resample(X_maj, replace=False, n_samples=len(X_min), random_state=44)

X_rus = pd.concat([X_maj_rus, X_min])

X_train_rus = X_rus.drop(['target'], axis=1)

y_train_rus = X_rus.target

y_rus = adaboost(X_train_rus, X_val, y_train_rus)

print('RUS Adaboost')

print(classification_report(y_rus, y_val))
print('Vanilla AdaBoost')

print(classification_report(y_baseline, y_val))



print('SMOTE AdaBoost')

print(classification_report(y_smote, y_val))



print('RUS Adaboost')

print(classification_report(y_rus, y_val))
#y_pred = adaboost(train.drop('target'), test, train.target)