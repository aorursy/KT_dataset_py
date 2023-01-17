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
df = pd.read_csv('../input/creditcard.csv')
df.head()
len(df[df['Class']==1]), len(df[df['Class']==0])
percentage_of_Class_0 = ((df[df['Class']==0].count())/df['Class'].count())*100

percentage_of_Class_1 = ((df[df['Class']==1].count())/df['Class'].count())*100

print(percentage_of_Class_0['Class'],'%')

print(percentage_of_Class_1['Class'],'%')
ax = sns.countplot('Class',data = df)

annot_plot(ax, 0.08, 1)
y = df['Class']

x = df.drop('Class', axis = 1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 42)
from sklearn.preprocessing import StandardScaler

Scaler_X = StandardScaler()

X_train = Scaler_X.fit_transform(X_train)

X_test = Scaler_X.transform(X_test)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)
lr.score(X_test,y_test)
from sklearn.metrics import accuracy_score, confusion_matrix

print(accuracy_score(y_pred,y_test))

print(confusion_matrix(y_pred,y_test))
from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_pred,y_test))

print(confusion_matrix(y_pred,y_test))
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
# Class count

count_class_0, count_class_1 = df.Class.value_counts()



# Divide by class

df_class_0 = df[df['Class'] == 0]

df_class_1 = df[df['Class'] == 1]
df_class_0_under = df_class_0.sample(count_class_1)

df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)



print('Random under-sampling:')

print(df_test_under.Class.value_counts())

df_test_under.Class.value_counts().plot(kind='bar',title = 'count(Class)')
y = df_test_under['Class']

x = df_test_under.drop('Class', axis = 1)
X_train_under, X_test_under, y_train_under, y_test_under = train_test_split(x,y, test_size = 0.20, random_state = 42)



model = XGBClassifier()

model.fit(X_train_under,y_train_under)

y_under_pred = model.predict(X_test_under)



print(accuracy_score(y_under_pred,y_test_under)) 

confusion_matrix(y_under_pred,y_test_under)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train_under,y_train_under)

ylr_under_pred = lr.predict(X_test_under)



acuuracy_score = accuracy_score(y_under_pred,y_test_under)

print(acuuracy_score) 

cm = confusion_matrix(ylr_under_pred,y_test_under)

cm
df_class_1_over = df_class_1.sample(count_class_0, replace=True)

df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)



print('Random over-sampling:')

print(df_test_over.Class.value_counts())



df_test_over.Class.value_counts().plot(kind='bar', title='Count (target)');
y_over = df_test_over['Class']

X_over = df_test_over.drop('Class', axis = 1)
X_train_over, X_test_over, y_train_over, y_test_over = train_test_split(X_over,y_over, test_size = 0.20, random_state = 42)

model.fit(X_train_over,y_train_over)

y_over_pred = model.predict(X_test_over)



print(accuracy_score(y_over_pred,y_test_over))

confusion_matrix(y_over_pred, y_test_over)
lr.fit(X_train_over,y_train_over)

ylr_over_pred = lr.predict(X_test_over)



print(accuracy_score(ylr_over_pred,y_test_over))

confusion_matrix(ylr_over_pred, y_test_over)

import imblearn
from sklearn.datasets import make_classification



X, y = make_classification(

    n_classes=2, class_sep=1.5, weights=[0.9, 0.1],

    n_informative=3, n_redundant=1, flip_y=0,

    n_features=20, n_clusters_per_class=1,

    n_samples=200, random_state=10

)



df = pd.DataFrame(X)

df['Class'] = y



df.Class.value_counts().plot(kind = 'bar', title = 'count(Class)')
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

#Random under-sampling and over-sampling with imbalanced-learn



from imblearn.under_sampling import RandomUnderSampler



rus = RandomUnderSampler(return_indices = True)

X_rus, y_rus, id_rus = rus.fit_sample(X,y)



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
from imblearn.combine import SMOTETomek



smt = SMOTETomek(ratio='auto')

X_smt, y_smt = smt.fit_sample(X, y)



plot_2d_space(X_smt, y_smt, 'SMOTE + Tomek links')
# ROC CURVE

lr = LogisticRegression(C = best_c, penalty = 'l1')

y_pred_undersample_score = lr.fit(X_train_under,y_train_under.values.ravel()).decision_function(X_test_under.values)



fpr, tpr, thresholds = roc_curve(y_test_under.values.ravel(),y_pred_under_score)

roc_auc = auc(fpr,tpr)



# Plot ROC

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.0])

plt.ylim([-0.1,1.01])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
