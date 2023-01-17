import itertools



import pickle

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.tree import DecisionTreeClassifier



from sklearn.impute import SimpleImputer

from sklearn.dummy import DummyClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import LabelEncoder, LabelBinarizer, StandardScaler



from sklearn.pipeline import Pipeline



from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split, cross_validate
np.random.seed(42)
df = pd.read_csv('../input/online_shoppers_intention.csv')

df.head()
df.shape
label_encoder = LabelEncoder()

label_binarizer = LabelBinarizer()



df['VisitorType'] = label_encoder.fit_transform(df['VisitorType'].values)

df['Weekend'] = label_binarizer.fit_transform(df['Weekend'].values)

df['Revenue'] = label_binarizer.fit_transform(df['Revenue'].values)



df = pd.concat([

    df[df.columns[:-1]],

    pd.get_dummies(df['Month'], prefix='Month_'),

    df['Revenue']],

    axis=1)



df = df.drop('Month', 1)
df.head()
df.describe()
missing_val_count_by_column = (df.isnull().sum())

missing_vals_columns = missing_val_count_by_column[

    missing_val_count_by_column > 0].keys()



simple_imputer = SimpleImputer(strategy = 'mean')

df[missing_vals_columns] = simple_imputer.fit_transform(

    df[missing_vals_columns])
features = df.columns[:-1]



X = df[features]

y = df['Revenue']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
dummy_classifier = DummyClassifier()

dummy_classifier.fit(X_train, y_train)

dummy_classifier.score(X_test, y_test)
scaler = StandardScaler()

model = DecisionTreeClassifier(max_depth = 5)



pipeline = Pipeline(steps=[('preprocessor', scaler),('estimator', model)])
cv = StratifiedKFold(n_splits=10, shuffle = True)

scores = cross_validate(pipeline, X, y, cv=cv, n_jobs=-1, return_train_score=True, return_estimator=True)
scoresdf = pd.DataFrame(scores)

scoresdf = scoresdf.rename({

    'test_score': 'test score',

    'train_score': 'train score'

}, axis=1)



scoresdf.head()
mean = scoresdf['test score'].mean()

std = scoresdf['test score'].std()



(mean - 2 * std, mean + 2 * std)
fig, ax = plt.subplots(figsize=(15,5))



score_plot = sns.lineplot(data=scoresdf[['test score', 'train score']], marker='o', ax=ax)

score_plot.set_title('Scores')



ax.set(xticks=scoresdf.index, xlim=(scoresdf.index[0], scoresdf.index[-1]))

fig.show()
scoresdf['distance'] = abs(scoresdf['test score'] - scoresdf['train score'])

index = scoresdf[scoresdf['distance'].min() == scoresdf['distance']].index[0]



classifier = scores['estimator'][index]

pickle.dump(classifier, open('model.h5', 'wb'))