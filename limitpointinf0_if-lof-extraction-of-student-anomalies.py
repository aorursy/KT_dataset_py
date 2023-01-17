import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



NO_COMPONENTS = 2



scaler = StandardScaler()

pca = PCA(n_components=NO_COMPONENTS)



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')

df.head()
import seaborn as sns

sns.boxplot(x=df['math score'])

plt.show()
plt.hist(df['math score'])

plt.show()
def convert_to_categorical(df, *categorical_cols):

    _df = df.copy()

    for col in categorical_cols:

        _df[col] = pd.Categorical(_df[col])

    return _df



def one_hot_dataframe(df, *categorical_cols):

    _df = df.copy()

    for col in categorical_cols:

        df_dummies = pd.get_dummies(_df[col], prefix='cat_{}'.format(col))

        _df = pd.concat([df_dummies, _df], axis=1)

    return _df



categorical_columns = [

    'gender', 

    'race/ethnicity', 

    'parental level of education',

    'lunch',

    'test preparation course',

]



percentage_cols = [

    'math score',

]



df = convert_to_categorical(df, *categorical_columns)

df = one_hot_dataframe(df, *categorical_columns)

df = df.drop(categorical_columns, axis=1)

df = df.drop(['reading score', 'writing score'], axis=1)

for col in percentage_cols:

    df[col] = df[col].map(lambda x: x / 100) 
import seaborn as sn

import matplotlib.pyplot as plt



corrMatrix = df.corr()

corrMatrix = corrMatrix.where(np.tril(np.ones(corrMatrix.shape)).astype(np.bool))



plt.figure(figsize = (20,15))

sn.heatmap(corrMatrix, annot=True)

plt.show()
scaler.fit(df.values)

X_scaled = scaler.transform(df.values)

pca.fit(X_scaled)

X_pca = pca.transform(X_scaled)

X_pca
from sklearn.ensemble import IsolationForest

clf_if = IsolationForest(random_state=0, contamination=0.01, verbose=True).fit(X_pca)
df_prediction_if = df.copy()

df_prediction_if['prediction'] = pd.Series(clf_if.predict(X_pca)).map(lambda x: 'outlier' if x==-1 else 'inlier')

df_prediction_if['color'] = df_prediction_if.prediction.map(lambda x: 'red' if x=='outlier' else 'blue')
plt.figure(figsize = (10,10))

plt.scatter(x=X_pca[:, 0], y=X_pca[:, 1], c=df_prediction_if.color)

plt.grid(True)

plt.show()
df_prediction_if[df_prediction_if.prediction == 'outlier']
from sklearn.neighbors import LocalOutlierFactor



N_NEIGHBORS = 6



clf = LocalOutlierFactor(n_neighbors=N_NEIGHBORS, contamination=0.01)

df_prediction_lof = df.copy()

df_prediction_lof['prediction'] = clf.fit_predict(X_pca)

df_prediction_lof['prediction'] = df_prediction_lof.prediction.map(lambda x: 'outlier' if x==-1 else 'inlier')

df_prediction_lof['color'] = df_prediction_lof.prediction.map(lambda x: 'red' if x=='outlier' else 'blue')

df_prediction_lof[df_prediction_lof.prediction == 'outlier']
plt.figure(figsize = (20,10))

plt.subplot(1, 2, 1)

plt.title('LOF {}-NN'.format(N_NEIGHBORS))

plt.scatter(x=X_pca[:, 0], y=X_pca[:, 1], c=df_prediction_lof.color)



plt.subplot(1, 2, 2)

plt.title('Isolation Forest')

plt.scatter(x=X_pca[:, 0], y=X_pca[:, 1], c=df_prediction_if.color)



plt.show()