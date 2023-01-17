%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from IPython.display import display



import warnings

warnings.filterwarnings('ignore')

# from warnings import simplefilter

# simplefilter(action='ignore', category=FutureWarning)



df = pd.read_csv('../input/survey.csv')



############################## 

# 欠測値の処理

############################## 

def replace_nan(df, c, replace):

    df.loc[df[c].isnull(), c] = replace

    return df



# 各変数について欠測値処理。

df = replace_nan(df, 'state', 'None')

df = replace_nan(df, 'self_employed', 'No')

df = replace_nan(df, 'work_interfere', 'Never')



# 変数commentsについての欠測値処理。コメントの内容に関係なく、欠測値は一律0（コメント無し）とした。

df = replace_nan(df, 'comments', 0)
############################## 

# 異常値の処理

############################## 

# Age:負の値、100を超過する値は異常値とし、データ行削除。

c = 'Age'

df.drop(df.index[(df[c] < 0) | (100 < df[c])], inplace=True)



# Gender:男性をM、女性をF、両性をHに名寄せ。解釈不明な入力値はNaNにしてデータ行削除。

c = 'Gender'

replace_map = {'Female':'F','M':'M','Male':'M','male':'M','female':'F','m':'M','Male-ish':'M','maile':'M','Trans-female':'F','Cis Female':'F','F':'F','something kinda male?':np.nan,'Cis Male':'M','Woman':'F','f':'F','Mal':'M','Male (CIS)':'M','queer/she/they':'H','non-binary':'H','Femake':'F','woman':'F','Make':'M','Nah':np.nan,'All':'H','Enby':'H','fluid':'H','Genderqueer':'H','Female ':'F','Androgyne':'H','Agender':'H','cis-female/femme':'F','Guy (-ish) ^_^':'M','male leaning androgynous':'H','Male ':'M','Man':'M','Trans woman':'F','msle':'M','Neuter':'H','Female (trans)':'F','queer':'H','Female (cis)':'F','Mail':'M','cis male':'M','A little about you':np.nan,'Malr':'M','p':np.nan,'femail':'F','Cis Man':'M','ostensibly male, unsure what that really means':np.nan}

df[c].replace(replace_map, inplace=True)

df.drop(df.index[df[c].isnull()], inplace=True)
############################## 

# 特徴量エンジニアリング

############################## 

# コメント欄：空でなければ1（コメントあり）、空ならば0に置換。

# なお、Notebookのセルを連続して実行してもエラーならないように２重実行を防止しておく。

# ※一度実行すると変数の型が自動的にbool型に変わることを利用して、元のobject型であるときだけ実行するように制御。

c = 'comments'

df.loc[(df[c]!=0)&(df[c].str.strip()!=''), c] = 1



# 01変換（対象は2値変数）

replace_map = {'Yes':1, 'No':0}

# なお、Notebookのセルを連続して実行してもエラーならないように２重実行を防止しておく。

# ※一度実行すると変数の型が自動的にint64型に変わることを利用して、元のobject型であるときだけ実行するように制御。

# print(df.dtypes['treatment'])

if (df.dtypes['treatment'] == 'object'):

    df['treatment'].replace(replace_map, inplace=True)

    df['family_history'].replace(replace_map, inplace=True)

    df['obs_consequence'].replace(replace_map, inplace=True)



# 説明変数としなかった変数は削除。

df.drop(['Timestamp','Age','Country','self_employed','no_employees','remote_work','tech_company','wellness_program','seek_help','mental_health_consequence','phys_health_consequence','coworkers','supervisor','mental_health_interview','phys_health_interview','mental_vs_physical'], axis=1, inplace=True)



# onehot-encodingの準備

# ただし、onehot-encodingの前に、新規変数名を"＜元の変数名＞_＜カテゴリ値＞"とするためにスペースやシングルコーテーションを含むカテゴリ値を適切に変換しておく。

# 変数benefits、anonymity

replace_map={"Don't know":'DontKnow'}

df['benefits'].replace(replace_map, inplace=True)

df['anonymity'].replace(replace_map, inplace=True)



# 変数care_options

replace_map={"Not sure":'NotSure'}

df['care_options'].replace(replace_map, inplace=True)



# 変数leave

replace_map={'Very easy':'VeryEasy', 'Somewhat easy':'SomewhatEasy', 'Somewhat difficult':'SomewhatDifficult', 'Very difficult':'VeryDifficult', "Don't know":'DontKnow'}

df['leave'].replace(replace_map, inplace=True)



# onehot-encoding

def create_dummy_var(orgdf, orgcol):

    tempcol = orgcol + '_str'

    # 一時列を作る。値は＜元の列名＞_＜カテゴリ値＞

    orgdf[tempcol] = orgdf[orgcol].astype(str).map(lambda x : orgcol + '_' + x)

    # 一時列の値をダミー変数として追加する。

    newdf = pd.concat([orgdf, pd.get_dummies(orgdf[tempcol])], axis=1)

    # 一時列と元の列は削除する。

    newdf.drop([tempcol, orgcol], axis=1, inplace=True)

    return newdf



# onehot-encoding実行。

df_fin = df # 最初は元のdfを引数としているので注意。

df_fin = create_dummy_var(df_fin, 'Gender')

df_fin = create_dummy_var(df_fin, 'state')

df_fin = create_dummy_var(df_fin, 'work_interfere')

df_fin = create_dummy_var(df_fin, 'benefits')

df_fin = create_dummy_var(df_fin, 'care_options')

df_fin = create_dummy_var(df_fin, 'anonymity')

df_fin = create_dummy_var(df_fin, 'leave')



# 後々の見易さのために目的変数treatmentを一番左に移動しておく。

cols = df_fin.columns.tolist()

cols.remove('treatment')

cols.insert(0, 'treatment')

df_fin = df_fin[cols]

#df_fin.head()
df_fin.corr().style.background_gradient().format('{:.2f}')
df_fin.drop(['Gender_F','work_interfere_Never','benefits_DontKnow','care_options_No','anonymity_DontKnow'], axis=1, inplace=True)

#display(df_fin)
# アルゴリズムを実行する関数を作成しておく。

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



##############################

# モデル生成・あてはめ実行メソッド

# 与えられたアルゴリズムとデータで学習（モデル作成）と予測（あてはめ）を行い、混同行列を表示するところまで行う。

##############################

def go_fit_pred(algo, df):

    # 説明変数をX、目的変数をyとしてロジスティック回帰を実行。

    X = df.drop(['treatment'], axis=1)

    y = df['treatment']

    algo.fit(X, y)

    # print(algo.coef_)

    # print(algo.intercept_)



    # 混同行列

    # 真の値の抽出

    y_true = df['treatment'].values

    # 予測値の算出。ただし、テストデータではなく元のデータセットを使った算出。

    y_pred = algo.predict(X)

    # なお、デフォルトの混同行列では行、列とも表示順が[0(陰性), 1(陽性)]となって理解しづらいので、行、列とも[1(陽性),0(陰性)]の順に変更。

    confmat = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[1, 0])

    print('行、列ともに[1(陽性),0(陰性)]の順に直した混同行列:\r\n{}'.format(confmat))



    # 分類レポート

    print('')

    print('分類レポート:')

    report = classification_report(y_true, y_pred, digits=3) # 小数点以下3桁まで表示

    print(report)

    

    # データが反映されたモデルを返す

    return algo
from sklearn.tree import DecisionTreeClassifier, export_graphviz

clf = DecisionTreeClassifier(criterion="gini", max_depth=None, min_samples_split=3, min_samples_leaf=3, random_state=1234)

clf = go_fit_pred(clf, df_fin)
# 重要度の表示

# scikit-learnで算出される重要度は、ある説明変数による不純度の減少量合計である。

print(clf.feature_importances_)

columns=df_fin.drop('treatment', axis=1).columns

pd.DataFrame(clf.feature_importances_, index=columns).plot.bar(figsize=(14,2))

plt.ylabel("Importance")

plt.xlabel("Features")

plt.show()
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10, max_depth=2, criterion="gini", min_samples_leaf=2, min_samples_split=2, random_state=1234)

go_fit_pred(rf, df_fin)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5) # n_neighborsのデフォルト値が5

go_fit_pred(knn, df_fin)
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LinearRegression, LogisticRegression



############################## 

# データ分割

############################## 

X, y = df_fin.drop(['treatment'], axis=1), df_fin['treatment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

#display(X_train)
############################## 

# グリッドサーチ実行メソッド

############################## 

from sklearn.metrics import classification_report

def go_grid(estimator, param_grid, cv, X_train, X_test, y_train, y_test):

    gs = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, n_jobs=4) # 処理時間短縮のため並列度も指定。

    gs.fit(X_train, y_train)

    print('Best Params  : {0}'.format(gs.best_params_)) # 最良モデルのパラメータ

    print('Best Score   : %.3f' % gs.best_score_) # 最良モデルの平均スコア

    print('Test Accuracy: %.3f' % gs.best_estimator_.score(X_test, y_test)) # 最良モデルによるテストデータの正解率=(TP+TN)/(TP+TN+FP+FN)

    print('Classification Report:') # 最良モデルによるテストデータの分類レポート

    y_pred = gs.best_estimator_.predict(X_test)

    report = classification_report(y_test, y_pred, digits=3) # 小数点以下3桁まで表示

    print(report)
%%time

# Kaggle上での処理時間短縮のため、ローカルでいろいろ試した結果で範囲を絞った。

param_grid={

    'penalty':['l1', 'l2'],

    'C':np.arange(1.0, 2.1, 0.1)

}

go_grid(LogisticRegression(), param_grid, 5, X_train, X_test, y_train, y_test)
%%time

# Kaggle上での処理時間短縮のため、ローカルでいろいろ試した結果で範囲を絞った。

param_grid={

    'criterion':['gini', 'entropy'], 

    'max_depth':np.arange(5,11), 

    'min_samples_split':np.arange(2,6), 

    'min_samples_leaf':np.arange(11,21),

    'random_state':[1234]

}

go_grid(DecisionTreeClassifier(), param_grid, 5, X_train, X_test, y_train, y_test)
%%time

# Kaggle上での処理時間短縮のため、ローカルでいろいろ試した結果で範囲を絞った。 

param_grid={

    'n_estimators':np.arange(16,21),

    'criterion':['entropy'],

    'max_depth':np.arange(16,21),

    'min_samples_split':np.arange(11,16),

    'min_samples_leaf':[1], 

    'random_state':[1234]

}

go_grid(RandomForestClassifier(), param_grid, 5, X_train, X_test, y_train, y_test)

%%time

param_grid={

    'n_neighbors':np.arange(1,11), 

}

go_grid(KNeighborsClassifier(), param_grid, 5, X_train, X_test, y_train, y_test)
def go_grid_with_precision(estimator, param_grid, cv, X_train, X_test, y_train, y_test):

    gs = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, scoring='precision', n_jobs=4) # 処理時間短縮のため並列度も指定。

    gs.fit(X_train, y_train)

    print('Best Params  : {0}'.format(gs.best_params_))

    print('Best Score   : %.3f' % gs.best_score_)

    print('Test Accuracy: %.3f' % gs.best_estimator_.score(X_test, y_test))

    print('Classification Report:')

    y_pred = gs.best_estimator_.predict(X_test)

    report = classification_report(y_test, y_pred, digits=3) # 小数点以下3桁まで表示

    print(report)
%%time

# Kaggle上での処理時間短縮のため、ローカルでいろいろ試した結果で範囲を絞った。

param_grid={

    'penalty':['l1', 'l2'],

    'C':np.arange(1.0, 2.1, 0.1)

}

go_grid_with_precision(LogisticRegression(), param_grid, 5, X_train, X_test, y_train, y_test)
%%time

# Kaggle上での処理時間短縮のため、ローカルでいろいろ試した結果で範囲を絞った。

param_grid={

    'criterion':['gini', 'entropy'], 

    'max_depth':np.arange(5,11), 

    'min_samples_split':np.arange(2,6), 

    'min_samples_leaf':np.arange(1,6),

    'random_state':[1234]

}

go_grid_with_precision(DecisionTreeClassifier(), param_grid, 5, X_train, X_test, y_train, y_test)
%%time

# Kaggle上での処理時間短縮のため、ローカルでいろいろ試した結果で範囲を絞った。 

param_grid={

    'n_estimators':np.arange(11,15), 

    'criterion':['entropy'], 

    'max_depth':np.arange(11,15),

    'min_samples_split':np.arange(11,15), # min_samples_splitは2以上。

    'min_samples_leaf':[1,6],

    'random_state':[1234]

}

go_grid_with_precision(RandomForestClassifier(), param_grid, 5, X_train, X_test, y_train, y_test)
%%time

param_grid={

    'n_neighbors':np.arange(1,21), 

}

go_grid_with_precision(KNeighborsClassifier(), param_grid, 5, X_train, X_test, y_train, y_test)
from sklearn.ensemble import AdaBoostClassifier

def go_adab(estimator, X_train, X_test, y_train, y_test):

    adab = AdaBoostClassifier(base_estimator=estimator, n_estimators=50, random_state=1234)

    adab.fit(X_train, y_train)

    print('Test Accuracy: %.3f' % adab.score(X_test, y_test))

    print('Classification Report:')

    y_pred = adab.predict(X_test)

    report = classification_report(y_test, y_pred, digits=3) # 小数点以下3桁まで表示

    print(report)
estimator = LogisticRegression()

go_adab(estimator, X_train, X_test, y_train, y_test)
estimator=DecisionTreeClassifier(criterion="gini", max_depth=None, min_samples_split=3, min_samples_leaf=3, random_state=1234)

go_adab(estimator, X_train, X_test, y_train, y_test)
estimator=RandomForestClassifier(n_estimators=10, max_depth=2, criterion="gini", min_samples_leaf=2, min_samples_split=2, random_state=1234)

go_adab(estimator, X_train, X_test, y_train, y_test)
# estimator = KNeighborsClassifier(n_neighbors=5)

# go_adab(estimator, X_train, X_test, y_train, y_test)

# ⇒エラー"ValueError: KNeighborsClassifier doesn't support sample_weight."が発生するのでコメントアウトしている。