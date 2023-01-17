# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import seaborn as sns               # Provides a high level interface for drawing attractive and informative statistical graphics

%matplotlib inline

sns.set()

from subprocess import check_output



import warnings                                            # Ignore warning related to pandas_profiling

warnings.filterwarnings('ignore') 



def annot_plot(ax,w,h):                                    # function to add data to plot

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    for p in ax.patches:

        ax.annotate('{0:.1f}'.format(p.get_height()), (p.get_x()+w, p.get_height()+h))



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df.columns
df['target'].count()
df[df['target']==1].count()
percentage_of_target_0 = ((df[df['target']==0].count())/df['target'].count())*100

percentage_of_target_1 = ((df[df['target']==1].count())/df['target'].count())*100

print(percentage_of_target_0['target'],'%')

print(percentage_of_target_1['target'],'%')


ax = sns.countplot('target', data = df)

annot_plot(ax,0.08,1)
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



y = df['target']

X = df.drop(['id','target'], axis = 1)



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42 )



model = XGBClassifier()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print(accuracy_score(y_test,y_pred)*100)
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,y_pred))
model.fit(X_train[['ps_calc_01']],y_train)

y_pred = model.predict(X_test[['ps_calc_01']])



print(accuracy_score(y_test,y_pred)*100)
# Class count

count_class_0, count_class_1 = df.target.value_counts()



# Divide by class

df_class_0 = df[df['target'] == 0]

df_class_1 = df[df['target'] == 1]
df_class_0_under = df_class_0.sample(count_class_1)

df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)



print('Random under-sampling:')

print(df_test_under.target.value_counts())

df_test_under.target.value_counts().plot(kind='bar',title = 'count(target)')
df_class_1_over = df_class_1.sample(count_class_0, replace=True)

df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)



print('Random over-sampling:')

print(df_test_over.target.value_counts())



df_test_over.target.value_counts().plot(kind='bar', title='Count (target)');
import imblearn
from sklearn.datasets import make_classification



X, y = make_classification(

    n_classes=2, class_sep=1.5, weights=[0.9, 0.1],

    n_informative=3, n_redundant=1, flip_y=0,

    n_features=20, n_clusters_per_class=1,

    n_samples=100, random_state=10

)



df = pd.DataFrame(X)

df['target'] = y

df.target.value_counts().plot(kind='bar', title='Count (target)');
def plot_2d_space(X, y, label='Classes'):   

    colors = ['#1F77B4', '#FF7F0E']

    markers = ['o', 's']

    for l, c, m in zip(np.unique(y), colors, markers):

        plt.scatter(

            X[y==l, 0],

            X[y==l, 1],

            c=c, label=l, marker=m

        )

    plt.title(label)

    plt.legend(loc='upper right')

    plt.show()
from sklearn.decomposition import PCA



pca = PCA(n_components=2)

X = pca.fit_transform(X)



plot_2d_space(X, y, 'Imbalanced dataset (2 PCA components)')
from imblearn.under_sampling import RandomUnderSampler



rus = RandomUnderSampler(return_indices=True)

X_rus, y_rus, id_rus = rus.fit_sample(X, y)



print('Removed indexes:', id_rus)



plot_2d_space(X_rus, y_rus, 'Random under-sampling')
from imblearn.over_sampling import RandomOverSampler



ros = RandomOverSampler()

X_ros, y_ros = ros.fit_sample(X, y)



print(X_ros.shape[0] - X.shape[0], 'new random picked points')



plot_2d_space(X_ros, y_ros, 'Random over-sampling')
from imblearn.under_sampling import TomekLinks



tl = TomekLinks(return_indices=True, ratio='majority')

X_tl, y_tl, id_tl = tl.fit_sample(X, y)



print('Removed indexes:', id_tl)



plot_2d_space(X_tl, y_tl, 'Tomek links under-sampling')
from imblearn.under_sampling import ClusterCentroids



cc = ClusterCentroids(ratio={0: 10})

X_cc, y_cc = cc.fit_sample(X, y)



plot_2d_space(X_cc, y_cc, 'Cluster Centroids under-sampling')
from imblearn.over_sampling import SMOTE



smote = SMOTE(ratio='minority')

X_sm, y_sm = smote.fit_sample(X, y)



plot_2d_space(X_sm, y_sm, 'SMOTE over-sampling')
len(X_sm), len(y_sm)
from imblearn.combine import SMOTETomek



smt = SMOTETomek(ratio='auto')

X_smt, y_smt = smt.fit_sample(X, y)



plot_2d_space(X_smt, y_smt, 'SMOTE + Tomek links')
len(X_smt), len(y_smt)