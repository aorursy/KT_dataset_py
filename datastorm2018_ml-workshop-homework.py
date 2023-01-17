%matplotlib inline 

#グラフをnotebook内に描画させるための設定

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from IPython.display import display
df = pd.read_csv('../input/survey.csv')

display(df.columns)

display(df.head())

display(df.tail())
# 各変数ごとに、unique要素のリストおよびそれらのデータ個数を列挙。

for c in df.columns:

    if (c != 'Timestamp') & (c != 'comments'):

        print('##############################')

        print(c)

        print('##############################')

        display(df[c].unique())

        display(df[c].value_counts().sort_index())

        print('\r\n')
y = 'treatment'

# 各変数の集計値を棒グラフにする。※なお、各列をfor文でループしたが20個を超過すると警告が出るらしい。

for c in df.columns:

    if (c != 'treatment') &(c != 'Timestamp') & (c != 'comments'):

        pd.crosstab(df[c],df['treatment']).plot.bar(stacked=True)
# Gender:男性をM、女性をF、両性をHに名寄せ。ただし解釈不明な入力値はNaNにしてから後でデータ行を削除。

c = 'Gender'

replace_map = {'Female':'F','M':'M','Male':'M','male':'M','female':'F','m':'M','Male-ish':'M','maile':'M','Trans-female':'F','Cis Female':'F','F':'F','something kinda male?':np.nan,'Cis Male':'M','Woman':'F','f':'F','Mal':'M','Male (CIS)':'M','queer/she/they':'H','non-binary':'H','Femake':'F','woman':'F','Make':'M','Nah':np.nan,'All':'H','Enby':'H','fluid':'H','Genderqueer':'H','Female ':'F','Androgyne':'H','Agender':'H','cis-female/femme':'F','Guy (-ish) ^_^':'M','male leaning androgynous':'H','Male ':'M','Man':'M','Trans woman':'F','msle':'M','Neuter':'H','Female (trans)':'F','queer':'H','Female (cis)':'F','Mail':'M','cis male':'M','A little about you':np.nan,'Malr':'M','p':np.nan,'femail':'F','Cis Man':'M','ostensibly male, unsure what that really means':np.nan}



print('処理前')

display(df[c].unique())



# mapを使って置換し、NaNはデータ行削除。

df[c].replace(replace_map, inplace=True)

df.drop(df.index[df[c].isnull()], inplace=True)



print('')

print('処理後')

display(df[c].unique())

pd.crosstab(df['Gender'],df['treatment']).plot.bar(stacked=True)
# 欠測値NaNが1個以上ある変数を列挙。

df_null=pd.DataFrame(df.isnull().sum(), columns=['NullCount'])

display(df_null[df_null['NullCount'] > 0])
# state

c = 'state'

replace = 'None'



print('処理前')

display(df[c].head())

print('NaN件数：{0}'.format(df[c].isnull().sum()))



df.loc[df[c].isnull(), c] = replace



print('')

print('処理後')

display(df[c].head())

print('NaN件数：{0}'.format(df[c].isnull().sum()))
# self_employed

c = 'self_employed'

replace = 'No'



print('処理前')

display(df[c].head())

print('NaN件数：{0}'.format(df[c].isnull().sum()))



df.loc[df[c].isnull(), c] = replace



print('')

print('処理後')

display(df[c].head())

print('NaN件数：{0}'.format(df[c].isnull().sum()))
# work_interfere

c = 'work_interfere'

replace = 'Never'



print('処理前')

display(df[c].head())

print('NaN件数：{0}'.format(df[c].isnull().sum()))



df.loc[df[c].isnull(), c] = replace



print('')

print('処理後')

display(df[c].head())

print('NaN件数：{0}'.format(df[c].isnull().sum()))
# Age

c = 'Age'



print('処理前')

display(df[c].count())



df.drop(df.index[(df[c] < 0) | (100 < df[c])], inplace=True)



print('')

print('処理後')

display(df[c].count())

# 説明変数としなかった変数は削除する。

df.drop(['Timestamp','Age','Country','self_employed','no_employees','remote_work','tech_company','wellness_program','seek_help','mental_health_consequence','phys_health_consequence','coworkers','supervisor','mental_health_interview','phys_health_interview','mental_vs_physical','comments'], axis=1, inplace=True)

df.head()
# 2値のみ持つ変数は01変換する。

replace_map = {'Yes':1, 'No':0}

# なお、Notebookのセルを連続して実行してもエラーならないように２重実行を防止しておく。

# ※一度実行すると変数の型が自動的にint64型に変わることを利用して、元のobject型であるときだけ実行するように制御。

print(df.dtypes['treatment'])

if (df.dtypes['treatment'] == 'object'):

    df['treatment'].replace(replace_map, inplace=True)

    df['family_history'].replace(replace_map, inplace=True)

    df['obs_consequence'].replace(replace_map, inplace=True)
# 一旦確認。

display(df.head())

print(df.dtypes)
# onehot-encoding

# ただし、onehot-encodingの前に、新規変数名を"＜元の変数名＞_＜カテゴリ値＞"とするためにスペースやシングルコーテーションを含むカテゴリ値を適切に変換しておく。

replace_map={"Don't know":'DontKnow'}

df['benefits'].replace(replace_map, inplace=True)

df['anonymity'].replace(replace_map, inplace=True)



replace_map={"Not sure":'NotSure'}

df['care_options'].replace(replace_map, inplace=True)



replace_map={'Very easy':'VeryEasy', 'Somewhat easy':'SomewhatEasy', 'Somewhat difficult':'SomewhatDifficult', 'Very difficult':'VeryDifficult', "Don't know":'DontKnow'}

df['leave'].replace(replace_map, inplace=True)



# 一旦確認。

df.head()
# onehot-encodingのダミー変数追加関数を作っておく。

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
# 特徴量エンジニアリングの結果を確認。

df_fin.head()
from sklearn.linear_model import LinearRegression,LogisticRegression

from sklearn.metrics import classification_report



X = df_fin.drop(['treatment'], axis=1)

y = df_fin['treatment']



lr = LogisticRegression()

lr.fit(X, y)

print(lr.coef_)

print(lr.intercept_)
from sklearn.metrics import confusion_matrix



# 真の値と予測値

y_true = df_fin['treatment'].as_matrix()

y_pred = lr.predict(X)



# 混同行列の作成。

# ただし、デフォルトのままでは行、列とも[0(陰性), 1(陽性)]の順になってしまう。

confmat0 = confusion_matrix(y_true=y_true, y_pred=y_pred)

print('デフォルトのままの混同行列:\r\n{}'.format(confmat0))



# そこで、目的変数treatmentの値が1の場合が陽性（精神治療を受けた）であるので、

# 混同行列を理解しやすくするために、行、列ともに[1(陽性),0(陰性)]の順で作成しておく。

labels_posi_nega = [1, 0]

confmat = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels_posi_nega)

print('')

print('行、列ともに[1(陽性),0(陰性)]の順に直した混同行列:\r\n{}'.format(confmat))



# 分かりやすく図示。

print('')

print('以下、分かりやすく図示した混同行列。')

labels_cell = [

    ['TP', 'FN'],

    ['FP', 'TN']

]

fig = plt.figure()

ax = fig.add_subplot(111)

ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)

for i in range(confmat.shape[0]):

    for j in range(confmat.shape[1]):

        ax.text(x=j, y=i, s='{0} = {1}'.format(labels_cell[i][j], confmat[i,j]), va='center', ha='center')



ax.set_xlabel('predicted')

ax.xaxis.set_label_position('top') 

ax.set_ylabel('true')

# ティックラベルを[1(陽性),0(陰性)]の順にする。

ax.set_xticklabels([''] + labels_posi_nega)

ax.set_yticklabels([''] + labels_posi_nega)

plt.show()
TP=confmat[0,0]

FN=confmat[0,1]

FP=confmat[1,0]

TN=confmat[1,1]



# Accuracy(正解率)

Accuracy = (TP + TN)/(TP + TN + FP + FN)

print('Accuracy = {0}'.format(Accuracy))



# Precision(適合率)

Precision = TP/(TP + FP)

print('Precision = {0}'.format(Precision))



# Recall(再現率)

Recall = TP/(TP + FN)

print('Recall = {0}'.format(Recall))



# F1値

F1 = 2.0/((1.0/Precision)+(1.0/Recall))

print('F1 = {0}'.format(F1))
report = classification_report(y_true, y_pred)

print(report)