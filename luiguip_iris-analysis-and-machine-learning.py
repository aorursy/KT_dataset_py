# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import scale

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

from tensorflow.keras.layers import Input, Dense, Activation,Dropout

from tensorflow.keras.models import Model

from xgboost import XGBClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv(os.path.join(os.sep, 'kaggle', 'input', 'iris', 'Iris.csv'), index_col='Id')
df.info()
df.head()
df.isna().sum()
n_classes = 3

plot_colors = "ryb"

plot_markers = "o^s"

plot_step = 0.02



unique_species = list(df['Species'].unique())

features = list(df.select_dtypes(include=['float64']).columns)
fig = plt.figure(figsize=(20,10))

for i, specie in enumerate(unique_species):

    df_specie = df[df['Species'] == unique_species[i]]

    for j, feature in enumerate(features):

        plt.subplot(3,4, i*len(features)+j+1)

        plt.title(specie+ ' ' + feature)

        plt.hist(df_specie[feature], bins=20, facecolor=plot_colors[i])

        plt.grid(True)

plt.suptitle("Histograms")

_ = plt.legend(loc='lower right', borderpad=0, handletextpad=0)
def scatter_iris(df):

    fig = plt.figure(figsize=(15, 10))

    for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):



        feature1 = features[pair[0]]

        feature2 = features[pair[1]]



        plt.subplot(2, 3, pairidx + 1)

        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

        plt.xlabel(feature1)

        plt.ylabel(feature2)



        for i, color, marker in zip(range(n_classes), plot_colors, plot_markers):

            df_specie = df[df['Species'] == unique_species[i]]

            x_plot = df_specie[feature1]

            y_plot = df_specie[feature2]

            plt.scatter(x_plot, y_plot, c=color, label=unique_species[i],

                        marker=marker, cmap=plt.cm.RdYlBu, edgecolor='black', s=15)



    plt.suptitle("Scatter plots")

    plt.legend(loc='lower right', borderpad=0, handletextpad=0)

    _ = plt.axis("tight")
scatter_iris(df)
features_scalars = StandardScaler().fit_transform(df[features])

df_scalar = pd.DataFrame(features_scalars, columns=features, index=list(range(1,151)))

df_scalar = pd.concat([df_scalar, df['Species']], axis=1)

df_scalar.head()
scatter_iris(df_scalar)
_ = sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1)
X = df[features]

y = df['Species']



Xs = StandardScaler().fit(X).transform(X)



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

X_val, X_test, y_val, y_test = train_test_split(X_train, y_train, test_size=0.5)



#X_trains, X_vals, y_trains, y_vals = train_test_split(Xs, y, test_size=0.3)

#X_vals, X_tests, y_vals, y_tests = train_test_split(X_trains, y_train, test_size=0.5)
class MlModel:

    def __init__(self,SkModel, kwargs):

        self.SkModel = SkModel

        self.kwargs = kwargs
def sklearn_model(ml_model):

    ml_model.model = ml_model.SkModel(**ml_model.kwargs)

    ml_model.model.fit(X_train, y_train.values.ravel())

    predict = ml_model.model.predict(X_val)

    ml_model.accuracy = accuracy_score(y_val, predict)

    print(ml_model.accuracy)

    print(ml_model.model)

    return ml_model
def regression_results(accuracy, confusion_matrix, report):

    print('Accuracy: {:.2f}'.format(accuracy))

    print('Confusion matrix.\n{}'.format(confusion_matrix))

    print('Report.\n{}'.format(report))
def regression_metrics(predict, y_c):

    accuracy = accuracy_score(y_c, predict)

    model_confusion_matrix = confusion_matrix(y_c, predict)

    report = classification_report(y_c, predict)

    regression_results(accuracy, model_confusion_matrix, report)
def sklearn_model_fast(ml_model, X_train, X_val, y_train, y_val, **kwargs):

    model = ml_model(**kwargs)

    model.fit(X_train, y_train.values.ravel())

    predict = model.predict(X_val)

    regression_metrics(predict, y_val)

    return model
lr = sklearn_model_fast(LogisticRegression, X_train, X_val, y_train, y_val, solver='lbfgs')
nb = sklearn_model_fast(GaussianNB, X_train, X_val, y_train, y_val)
svm = sklearn_model_fast(SVC, X_train, X_val, y_train, y_val, gamma='auto')
dt = sklearn_model_fast(DecisionTreeClassifier, X_train, X_val, y_train, y_val)
rf = sklearn_model_fast(RandomForestClassifier, X_train, X_val, y_train, y_val)
model_xgb = sklearn_model_fast(XGBClassifier, X_train, X_val, y_train, y_val)



print(regression_metrics(lr.predict(X_test), y_test))
print(regression_metrics(svm.predict(X_test), y_test))
print(regression_metrics(nb.predict(X_test), y_test))
print(regression_metrics(dt.predict(X_test), y_test))
print(regression_metrics(rf.predict(X_test), y_test))
print(regression_metrics(model_xgb.predict(X_test), y_test))