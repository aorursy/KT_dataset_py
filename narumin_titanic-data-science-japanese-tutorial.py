# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# warnings
import warnings
warnings.filterwarnings('ignore')
# 変数の相関関係をヒートマップで表示
def plot_correlation_map(df, title=''):
    corr = df.corr()
    _, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.diverging_palette(220, 10, n=24, as_cmap=True)
    _ = sns.heatmap(
        corr,
        vmin=-1,
        vmax=1,
        center=0,
        cmap=cmap,
        square=True,
        cbar_kws={'shrink': .9},
        ax=ax,
        annot=True,
        annot_kws={'fontSize': 12}
    )
    ax.set_title(title)

# 定量変数とターゲット変数の関係を可視化
def plot_distribution(df, var, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, hue=target, aspect=4, row=row, col=col)
    facet.map(sns.kdeplot, var, shade=True)
    facet.set(xlim=(0, df[var].max()))
    facet.add_legend()

# カテゴリ変数とターゲット変数の関係を可視化
def plot_categories(df, cat, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, row=row, col=col)
    facet.map(sns.barplot, cat, target, order=df[cat].dropna().unique())
    facet.add_legend()
    
def plot_variable_importance(X, y):
    tree = DecisionTreeClassifier(random_state=99)
    tree.fit(X, y)
    plot_model_var_imp(tree, X, y)
    
def plot_model_var_imp(model, X, y):
    imp = pd.DataFrame(
        model.feature_importances_,
        columns=['importance'],
        index=X.columns
    )
    imp = imp.sort_values(['importance'], ascending=False)
    imp[:10].plot(kind='barh')
    print(model.score(X, y))
    
def plot_plotly_bar(x, y, title, y_title):
    trace = go.Bar(
        x=x,
        y=y,
        width=0.5,
        marker=dict(
            color=y,
            colorscale='Portland',
            showscale=True,
            reversescale=False
        ),
        opacity=0.6
    )
    layout= go.Layout(
        autosize=True,
        title=title,
        hovermode='closest',
        yaxis=dict(
            title=y_title,
            ticklen=5,
            gridwidth=2
        ),
        showlegend=False
    )
    fig = go.Figure(data=[trace], layout=layout)
    py.iplot(fig)
    
def plot_plotly_correlation_map(df, title):
    trace = go.Heatmap(
        z=df.astype(float).corr().values ,
        x=df.columns.values,
        y=df.columns.values,
        colorscale='Viridis',
        showscale=True,
        reversescale = True
    )
    layout= go.Layout(
        autosize=True,
        title=title,
        hovermode='closest',
        showlegend=False
    )
    fig = go.Figure(data=[trace], layout=layout)
    py.iplot(fig)
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
combine = [df_train, df_test]
df_train.isnull().sum() # 欠損値を確認
df_test.isnull().sum() # 欠損値を確認
df_train.head() # 最初の5件表示
df_train.describe() # 定量変数の統計量を表示
df_train.describe(include=['O']) # カテゴリ変数の統計量を表示
plot_correlation_map(df_train)
# df_train.loc[:, ['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
plot_distribution( df_train , var='Age' , target ='Survived' , row ='Sex')
grid = sns.FacetGrid(df_train, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
guess_ages = np.zeros((2,3)) # 年齢の推測値

# PclassとSexでのグループ別の年齢中央値で補う
sex_table = ['male', 'female']
for dataset in combine:
  for i in range(0, 2):
    for j in range(0, 3):
      df_guess = dataset[(dataset['Sex'] == sex_table[i]) & (dataset['Pclass'] == j+1)]['Age'].dropna()
      #age_mean = df_guess.mean()
      #age_std = df_guess.std()
      #age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)
      age_guess = df_guess.median()
      guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5
      
  for i in range(0, 2):
    for j in range(0, 3):
      dataset.loc[(dataset['Age'].isnull()) & (dataset['Sex'] == sex_table[i]) & (dataset['Pclass'] == j+1), 'Age'] = guess_ages[i, j]
      
  dataset['Age'] = dataset['Age'].astype(int)

df_train.head()
for dataset in combine:
    dataset['AgeBand'] = pd.cut(dataset['Age'], 5) # 年齢層（5分割）の変数を作成
    
df_train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
    dataset['AgeBand'] = dataset['AgeBand'].astype(np.str)
    le = LabelEncoder()
    dataset['AgeBand'] = le.fit_transform(dataset['AgeBand']) #  数値に変換
  
df_train.head()
#grid = sns.FacetGrid(df_train, col='Survived')
#grid.map(plt.hist, 'Age', bins=20)

plot_distribution( df_train , var='Fare' , target ='Survived')
guess_fares = np.zeros((2,3)) # 運賃の推測値

for dataset in combine:
    for j in range(0, 3):
        df_guess = dataset[(dataset['Pclass'] == j+1)]['Fare'].dropna()
        #age_mean = df_guess.mean()
        #age_std = df_guess.std()
        #age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)
        fare_guess = df_guess.median()
        guess_fares[i, j] = int(fare_guess / 0.5 + 0.5) * 0.5

    for j in range(0, 3):
        dataset.loc[(dataset['Fare'].isnull()) &  (dataset['Pclass'] == j+1), 'Fare'] = guess_fares[i, j]
for dataset in combine:
    dataset['FareBand'] = pd.cut(dataset['Fare'], 4) # 価格帯（4分割）の変数を作成

df_train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
    dataset['FareBand'] = dataset['FareBand'].astype(np.str)
    le = LabelEncoder()
    dataset['FareBand'] = le.fit_transform(dataset['FareBand']) #  数値に変換
  
df_train.head()
#df_train.loc[:, ['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
plot_categories( df_train , cat='Pclass' , target='Survived' )
plot_categories( df_train , cat='Embarked' , target='Survived' )
freq_port = df_train['Embarked'].dropna().mode()[0]
freq_port
for dataset in combine:
  dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
  
df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
    le = LabelEncoder()
    dataset['Embarked'] = le.fit_transform(dataset['Embarked']) #  数値に変換
  
df_train.head()
plot_categories( df_train , cat='Sex' , target='Survived' )
for dataset in combine:
    le = LabelEncoder()
    dataset['Sex'] = le.fit_transform(dataset['Sex']) #  数値に変換
    
df_train.head()
# df_train.loc[:, ['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
plot_categories( df_train , cat='SibSp' , target='Survived' )
# df_train.loc[:, ['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
plot_categories( df_train , cat='Parch' , target='Survived' )
for dataset in combine:
  dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1 # 親族数の変数を作成
  
df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
  dataset['IsAlone'] = 0
  dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1 # 単身フラグ変数を作成
  
df_train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
  dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
  
pd.crosstab(df_train['Title'], df_train['Sex'])
for dataset in combine:
  dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
  dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
  dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
  dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
  
df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
    le = LabelEncoder()
    dataset['Title'] = le.fit_transform(dataset['Title']) #  数値に変換

df_train.head()
for dataset in combine:
    dataset['Cabin'] = dataset['Cabin'].fillna('U').map(lambda c: c[0])

df_train[['Cabin', 'Survived']].groupby(['Cabin'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
    le = LabelEncoder()
    dataset['Cabin'] = le.fit_transform(dataset['Cabin']) # 数値に変換
    
df_train.head()
# チケットの種別のみ抽出する
def clean_ticket(ticket):
    ticket = ticket.replace('.' , '')
    ticket = ticket.replace('/' , '')
    ticket = ticket.split()
    ticket = map(lambda t: t.strip(), ticket)
    ticket = list(filter(lambda t: not t.isdigit(), ticket))
    if len(ticket) > 0:
        return ticket[0]
    else:
        return 'XXX'
    
for dataset in combine:
    dataset['Ticket'] = dataset['Ticket' ].map(clean_ticket)

df_train[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
    le = LabelEncoder()
    dataset['Ticket'] = le.fit_transform(dataset['Ticket']) # 数値に変換
    
df_train.head()
drop_table = ['Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Name']
df_train = df_train.drop(drop_table, axis=1)
df_test = df_test.drop(drop_table, axis=1)
df_train.describe()
df_test.describe()
df_train.head()
df_test.head()
X_train = df_train.drop(['PassengerId', 'Survived'], axis=1)
Y_train = df_train['Survived']
X_test = df_test.drop(['PassengerId'], axis=1)

X_train.shape, Y_train.shape, X_test.shape
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        self.clf = clf(**params)
        
    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)
        
    def predict(self , x):
        return  self.clf.predict(x)
    
    def has_feature_importances(self):
        return hasattr(self.clf, "feature_importances_");
    
    def get_feature_importances(self):
        return self.clf.feature_importances_
    
    def plot_feature_importances(self, title, columns):
        trace = go.Scatter(
            y=self.clf.feature_importances_, 
            x=columns, 
            mode='markers',
            marker=dict(
                sizemode = 'diameter',
                sizeref=1,
                size=25,
                color=self.clf.feature_importances_,
                colorscale='Portland',
                showscale=True
            ),
             text=columns
        )
        layout= go.Layout(
            autosize=True, 
            title=title,
            hovermode= 'closest',
            yaxis=dict(
                title='Feature Importance',
                ticklen=5,
                gridwidth=2
            ),
            showlegend= False
        )
        fig = go.Figure(data=[trace], layout=layout)
        py.iplot(fig)
SEED = 0 # 乱数シード
NFOLDS = 5 # 訓練データの分割数

kf = KFold(n_splits= NFOLDS, random_state=SEED)

# K-分割交差検証
# 標本群をK個に分離して、1つを検証に残りを訓練に使う
# 訓練と検証をK回行う
# K回の結果を平均して1つの推定する
def get_oof(clf, x_train, y_train, x_pred):
    oof_test = np.zeros((x_train.shape[0],))
    oof_pred = np.zeros((x_pred.shape[0],))
    oof_pred_skf  = np.empty((NFOLDS, x_pred.shape[0]))
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr, x_ts = x_train.loc[train_index], x_train.loc[test_index]
        y_tr, y_ts = y_train.loc[train_index], y_train.loc[test_index]
        clf.fit(x_tr, y_tr)
        oof_test[test_index] = clf.predict(x_ts)
        oof_pred_skf[i, :] = clf.predict(x_pred)
    oof_pred = oof_pred_skf.mean(axis=0)
    return oof_test, oof_pred
classifiers = {
    'SVC': {
        'model': SVC,
        'params': {
            'kernel' : 'linear',
            'C' : 0.025,
            'random_state': SEED
        },
    },
    "KNeighborsClassifier": {
        'model': KNeighborsClassifier,
        'params': {
            'n_neighbors' : 3,
            #'random_state': SEED
        }
    },
    'GaussianNB': {
        'model': GaussianNB,
        'params': {
            #'random_state': SEED
        }
    },
    'Perceptron': {
        'model': Perceptron,
        'params': {
             'random_state': SEED
        }
    },
    'LinearSVC': {
        'model': LinearSVC,
        'params': {
            'random_state': SEED
        }
    },
    'SGDClassifier': {
        'model': SGDClassifier,
        'params': {
            'random_state': SEED
        }
    },
    'DecisionTreeClassifier': {
        'model': DecisionTreeClassifier,
        'params': {
             'random_state': SEED
        }
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier,
        'params': {
            'n_jobs': -1,
            'n_estimators': 500,
            'warm_start': True, 
            #'max_features': 0.2,
            'max_depth': 6,
            'min_samples_leaf': 2,
            'max_features' : 'sqrt',
            'verbose': 0,
            'random_state': SEED
        }
    },
    'AdaBoostClassifier': {
        'model': AdaBoostClassifier,
        'params': {
            'n_estimators': 500,
            'learning_rate' : 0.75,
            'random_state': SEED
        }
    },
    'ExtraTreesClassifier': {
        'model': ExtraTreesClassifier,
        'params': {
            'n_jobs': -1,
            'n_estimators':500,
            #'max_features': 0.5,
            'max_depth': 8,
            'min_samples_leaf': 2,
            'verbose': 0,
            'random_state': SEED
        }
    },
    'GradientBoostingClassifier': {
        'model': GradientBoostingClassifier,
        'params': {
            'n_estimators': 500,
             #'max_features': 0.2,
            'max_depth': 5,
            'min_samples_leaf': 2,
            'verbose': 0,
            'random_state': SEED
        }
    }
}
df_score = pd.DataFrame(columns=['name', 'score'])
df_train_pred = pd.DataFrame()
df_test_pred = pd.DataFrame()
for i, name in enumerate(classifiers):
    clf = classifiers[name] 
    clf['helper'] = SklearnHelper(clf=clf['model'], seed=SEED, params=clf['params'])
    oof_train, oof_test = get_oof(clf['helper'].clf, X_train, Y_train, X_test)
    df_score = df_score.append(
        pd.Series([name,  accuracy_score(oof_train, Y_train)], index=df_score.columns), 
        ignore_index=True)
    df_train_pred[name] = oof_train
    df_test_pred[name] = oof_test
df_score.sort_values(by='score', ascending=False)
feature_columns = X_train.columns.values
df_feature = pd.DataFrame({
    'features': feature_columns
})
# 予測モデル別に特徴量の重要度を可視化
for i, name in enumerate(classifiers):
    clf = classifiers[name]
    helper = clf['helper']
    if helper.has_feature_importances():
        df_feature[name] = helper.get_feature_importances()
        helper.plot_feature_importances(title=name, columns=feature_columns)
# 全予測モデルの特徴量の重要度を平均を可視化
df_feature['mean'] = df_feature.mean(axis=1)
df_feature = df_feature.sort_values(by='mean', ascending=False)
plot_plotly_bar(x=feature_columns, y=df_feature['mean'], title='Barplots of Mean Feature Importance', y_title='Feature Importance')
# ランダムフォレストの結果を使用
submission = pd.DataFrame({
    "PassengerId": df_test["PassengerId"].astype(np.int),
    "Survived": df_test_pred['RandomForestClassifier']
})
submission.to_csv('submission.csv', index=False)
# 各予測モデルの予測結果の相関関係を表示
plot_plotly_correlation_map(df=df_train_pred, title='Classifier predict correlation　map')
practical_classifiers =  [
    'RandomForestClassifier',
    'GradientBoostingClassifier',
    'ExtraTreesClassifier',
    'DecisionTreeClassifier',
    'AdaBoostClassifier',
]
df_train_pred = df_train_pred.loc[:, practical_classifiers]
df_test_pred = df_test_pred.loc[:, practical_classifiers]

gbm = XGBClassifier(
    #learning_rate = 0.02,
    n_estimators= 2000,
    max_depth= 4,
    min_child_weight= 2,
    #gamma=1,
    gamma=0.9,                        
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread= -1,
    scale_pos_weight=1
)

gbm_train_pred, gbm_test_pred = get_oof(gbm, df_train_pred, Y_train, df_test_pred)
accuracy_score(gbm_train_pred, Y_train)
rfecv = RFECV(estimator=classifiers['RandomForestClassifier']['helper'].clf, step=1, cv=StratifiedKFold(2), scoring='accuracy')
rfecv.fit(X_train, Y_train)
print(rfecv.score(X_train, Y_train))
print("Optimal number of feature: ", rfecv.n_features_)

for index, selected in enumerate(rfecv.support_):
    if selected:
        print("Selected column:", X_train.columns[index])
        
plt.plot(range(1 , len( rfecv.grid_scores_ ) + 1), rfecv.grid_scores_)