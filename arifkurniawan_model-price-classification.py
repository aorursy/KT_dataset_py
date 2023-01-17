import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
data_train = pd.read_csv("/kaggle/input/mobile-price-classification/train.csv")

data_test = pd.read_csv("/kaggle/input/mobile-price-classification/test.csv")
fig,ax=plt.subplots(figsize=(5,3))

data_train.groupby("price_range").count()[['blue']].plot(kind='bar',ax=ax,legend=False,color='blue')

for i in ax.spines:

    if i!="left" and i!="bottom":

        ax.spines[i].set_visible(False)

ax.set_title("Price_Range",fontdict={'fontweight':'bold'})

ax.set_xlabel("")

ax.set_xticklabels(ax.get_xticklabels(),rotation=0)

plt.show()
data_train.info()
fig,ax=plt.subplots(figsize=(15,7))

sns.heatmap(data_train.corr(),annot=True,fmt='.1g',cmap="Blues")

plt.show()
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier
list_methods = [SVC,DecisionTreeClassifier,ExtraTreesClassifier,LogisticRegression,

                GaussianNB,GradientBoostingClassifier,RandomForestClassifier,AdaBoostClassifier]

methods_name = ["SVC","DecisionTreeClassifier","ExtraTreesClassifier","LogisticRegression",

                "GaussianNB","GradientBoostingClassifier","RandomForestClassifier","AdaBoostClassifier"]
X = data_train.drop(['price_range'],axis=1)

y = data_train['price_range']

X_train,X_test,y_train,y_test = train_test_split(X,y)
def highlight_max(data, color='yellow'):

    '''

    highlight the maximum in a Series or DataFrame

    '''

    attr = 'background-color: {}'.format(color)

    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1

        is_max = data == data.max()

        return [attr if v else '' for v in is_max]

    else:  # from .apply(axis=None)

        is_max = data == data.max().max()

        return pd.DataFrame(np.where(is_max, attr, ''),

                            index=data.index, columns=data.columns)
classification_result = []

for j in range(len(list_methods)):

    model_result = []

    for i in range(1,len(X.columns)+1):

        pca = PCA(n_components=i)

        pca.fit(X_train,y_train)

        X_t_train = pca.fit_transform(X_train)

        X_t_test = pca.fit_transform(X_test)

        clf = RandomForestClassifier()

        clf.fit(X_t_train,y_train)

        model_result.append({'n_feature':i,methods_name[j]:clf.score(X_t_test,y_test)})

    classification_result.append(pd.DataFrame(model_result))



df_result = classification_result[0]

for i in range(1,len(classification_result)):

    df_result = df_result.merge(classification_result[i],on='n_feature')

df_result = df_result[['n_feature']+methods_name]

df_result = df_result.set_index('n_feature')
df_result.style.apply(highlight_max)
result = SelectKBest(score_func=chi2,k=10)

hasil = result.fit(np.array(X),np.array(y))

feature_select = pd.DataFrame({'result':hasil.scores_,'field':X.columns}).sort_values('result',ascending=False)

feature_select = feature_select.reset_index(drop=True)
print(feature_select)
classification_result = []

for j in range(len(list_methods)):

    model_result = []

    for i in range(1,len(feature_select)+1):

        X_f_train = X_train[list(feature_select['field'])[:i]]

        X_f_test = X_test[list(feature_select['field'])[:i]]

        clf = list_methods[j]()

        clf.fit(X_f_train,y_train)

        model_result.append({'n_feature':i,methods_name[j]:clf.score(X_f_test,y_test)})

    classification_result.append(pd.DataFrame(model_result))

df_result = classification_result[0]

for i in range(1,len(classification_result)):

    df_result = df_result.merge(classification_result[i],on='n_feature')

df_result = df_result[['n_feature']+methods_name]

df_result = df_result.set_index("n_feature")
df_result.style.apply(highlight_max)
X_f_train = X_train[list(feature_select['field'])[:4]]

X_f_test = X_test[list(feature_select['field'])[:4]]

clf = ExtraTreesClassifier()

clf.fit(X_f_train,y_train)
clf.predict(data_test[list(feature_select['field'])[:4]])