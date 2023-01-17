# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score, recall_score

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import FeatureUnion

from sklearn.preprocessing import LabelBinarizer

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/autism-screening/Autism_Data.arff')

data = data.dropna()

data.columns = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'age', 'gender', 'ethnicity', 'jaundice', 'autism', 'country_of_res', 'used_app_before', 'result', 'age_desc', 'relation', 'Class_ASD']

data.head()
attr = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'age', 'gender', 'ethnicity', 'jaundice', 'autism', 'country_of_res', 'used_app_before', 'result', 'age_desc', 'relation']

bin_attr = ['Class_ASD']
class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X , y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names].values



hot_pipeline = Pipeline([

    ('selector', DataFrameSelector(attr)),

    ('hot', OneHotEncoder(handle_unknown='ignore')),

])



full_pipeline = FeatureUnion(transformer_list=[

                             ('hot_pipe', hot_pipeline),

                             ])
x = full_pipeline.fit_transform(data)



classASD = data['Class_ASD'].values.reshape(-1,1)

labeler = LabelEncoder()

y = labeler.fit_transform(classASD)



shuffle_index = np.random.permutation(704)

x1, y1 = x[shuffle_index], y[shuffle_index]



ax_tr, ax_te, ay_tr, ay_te = x1[:500], x1[500:], y1[:500], y1[500:]
amodel = SGDClassifier(random_state=41)

amodel.fit(ax_tr.astype(float), ay_tr.astype(float))

cross_val_score(amodel, ax_tr, ay_tr, cv=3, scoring='accuracy')
y_train_pred = cross_val_predict(amodel, ax_tr, ay_tr, cv=3)

confusion_matrix(ay_tr, y_train_pred)
precision_score(ay_tr, y_train_pred)
recall_score(ay_tr, y_train_pred)
f1_score(ay_tr, y_train_pred)
pca = PCA(n_components=3)

X = pca.fit_transform(ax_tr.toarray())



result=pd.DataFrame(X, columns=['PCA%i' % i for i in range(3)])

color=pd.DataFrame(ay_tr, columns=['Class_ASD'])



color['Class_ASD']=pd.Categorical(color['Class_ASD'])

my_color=color['Class_ASD'].cat.codes

    

# Plot initialisation

fig = plt.figure(figsize=(40,30))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(result['PCA0'], result['PCA1'], result['PCA2'], c=my_color, cmap="Set2_r", s=60)

 

# make simple, bare axis lines through space:

xAxisLine = ((min(result['PCA0']), max(result['PCA0'])), (0, 0), (0,0))

ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')

yAxisLine = ((0, 0), (min(result['PCA1']), max(result['PCA1'])), (0,0))

ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')

zAxisLine = ((0, 0), (0,0), (min(result['PCA2']), max(result['PCA2'])))

ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')

 

# label the axes

ax.set_xlabel("PC1")

ax.set_ylabel("PC2")

ax.set_zlabel("PC3")

plt.show()

pca = PCA(n_components=2)

X = pca.fit_transform(ax_tr.toarray())



result=pd.DataFrame(X, columns=['PCA%i' % i for i in range(2)])

color=pd.DataFrame(ay_tr, columns=['Class_ASD'])



color['Class_ASD']=pd.Categorical(color['Class_ASD'])

my_color=color['Class_ASD'].cat.codes

    

# Plot initialisation

fig = plt.figure(figsize=(25,18))

plt.scatter(result['PCA0'], result['PCA1'], c=my_color, cmap="spring", s=60)



plt.show()



plt.figure(figsize=(15,15))

sns.heatmap(data.corr(),annot = True,fmt = ".2f",cbar = True)

plt.xticks(rotation=90)

plt.yticks(rotation = 0)