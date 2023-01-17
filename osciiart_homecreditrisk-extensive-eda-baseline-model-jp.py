from plotly.offline import init_notebook_mode, iplot
from wordcloud import WordCloud
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly import tools
from datetime import date
import pandas as pd
import numpy as np 
import seaborn as sns
import random 
import warnings
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)

path = "../input/"

def bar_hor(df, col, title, color, w=None, h=None, lm=0, limit=100, return_trace=False, rev=False, xlb = False):
    """
    横向き棒グラフ作成関数
    df:  
    col: 
    title: 
    color: 
    w=None: 
    h=None: 
    lm=0: 
    limit=100: 
    return_trace=False: 
    rev=False: 
    xlb = False:
    """
    cnt_srs = df[col].value_counts()
    yy = cnt_srs.head(limit).index[::-1] 
    xx = cnt_srs.head(limit).values[::-1] 
    if rev:
        yy = cnt_srs.tail(limit).index[::-1] 
        xx = cnt_srs.tail(limit).values[::-1] 
    if xlb:
        trace = go.Bar(y=xlb, x=xx, orientation = 'h', marker=dict(color=color))
    else:
        trace = go.Bar(y=yy, x=xx, orientation = 'h', marker=dict(color=color))
    if return_trace:
        return trace 
    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)

def bar_hor_noagg(x, y, title, color, w=None, h=None, lm=0, limit=100, rt=False):
    trace = go.Bar(y=x, x=y, orientation = 'h', marker=dict(color=color))
    if rt:
        return trace
    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)


def bar_ver_noagg(x, y, title, color, w=None, h=None, lm=0, rt = False):
    trace = go.Bar(y=y, x=x, marker=dict(color=color))
    if rt:
        return trace
    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
    
def gp(col, title):
    """
    グループ化棒グラフを表示
    col: 表示する列
    title: 図のタイトル
    """
    df1 = app_train[app_train["TARGET"] == 1]
    df0 = app_train[app_train["TARGET"] == 0]
    a1 = df1[col].value_counts()
    b1 = df0[col].value_counts()

    trace1 = go.Bar(x=a1.index, y=a1.values, name='Target : 1', marker=dict(color="#44ff54"))
    trace2 = go.Bar(x=b1.index, y=b1.values, name='Target : 0', marker=dict(color="#ff4444"))

    data = [trace1, trace2]
    layout = go.Layout(barmode='group', height=300, title = title)

    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename='grouped-bar')
app_train = pd.read_csv(path + "application_train.csv") # application train データの読み込み
app_train.head() # 最初の5行を表示
# 目的変数の分布
# 横向き棒グラフの表示
bar_hor(app_train, # 表示するデータ 
        "TARGET", # 表示する列名
        "Distribution of Target Variable" , # 図のタイトル
        ["#44ff54", '#ff4444'], # 棒グラフの色
        h=400, # 図の高さ
        w=800, # 図の幅
        lm=100, # ?
        xlb = ['Target : 1','Target : 0'] # 各バーの説明
       )
tr1 = bar_hor(app_train, "CODE_GENDER", "Distribution of CODE_GENDER Variable" ,"#639af2", w=700, lm=100, return_trace= True) # 性別の棒グラフ
tr2 = bar_hor(app_train, "NAME_CONTRACT_TYPE", "Distribution of NAME_CONTRACT_TYPE Variable" ,"#a4c5f9", w=700, lm=100, return_trace = True) # 契約タイプの棒グラフ

# 図の整形・表示
fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles = ['Gender' , 'Contract Type'])
fig.append_trace(tr1, 1, 1);
fig.append_trace(tr2, 1, 2);
fig['layout'].update(height=400, showlegend=False, margin=dict(b=100));
iplot(fig);
# 性別・目的変数のグループ化棒グラフ
gp('CODE_GENDER', # 表示する列名
   'Distribution of Target with Gender' # 図のタイトル
  ) 
gp('NAME_CONTRACT_TYPE', 'Distribution of Target with Contract Type') # 契約タイプ・目的変数のグループ化棒グラフ
tr1 = bar_hor(app_train, "FLAG_OWN_REALTY", "Distribution of FLAG_OWN_REALTY" ,"#639af2", w=700, lm=100, return_trace= True)
tr2 = bar_hor(app_train, "FLAG_OWN_CAR", "Distribution of FLAG_OWN_CAR" ,"#639af2", w=700, lm=100, return_trace = True)

fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles = ['Own Realty' , 'Own CAR'])
fig.append_trace(tr1, 1, 1);
fig.append_trace(tr2, 1, 2);
fig['layout'].update(height=400, showlegend=False, margin=dict(b=100));
iplot(fig);


gp('FLAG_OWN_REALTY', 'Distribution of Target with FLAG_OWN_REALTY')
gp('FLAG_OWN_CAR', 'Distribution of Target with FLAG_OWN_CAR Type')
tr1 = bar_hor(app_train, "NAME_TYPE_SUITE", "Distribution of NAME_TYPE_SUITE" ,"#639af2", w=700, lm=100, return_trace= True) # 世帯構成タイプの棒グラフ
tr2 = bar_hor(app_train, "NAME_INCOME_TYPE", "Distribution of NAME_INCOME_TYPE" ,"#a4c5f9", w=700, lm=100, return_trace = True) # 収入タイプの棒グラフ

fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles = ['Applicants Suite Type' , 'Applicants Income Type'])
fig.append_trace(tr1, 1, 1);
fig.append_trace(tr2, 1, 2);
fig['layout'].update(height=400, showlegend=False, margin=dict(l=100));
iplot(fig);
gp('NAME_TYPE_SUITE', 'Target with respect to Suite Type of Applicants')
gp('NAME_INCOME_TYPE', 'Target with respect to Income Type of Applicants')
tr1 = bar_hor(app_train, "NAME_FAMILY_STATUS", "Distribution of NAME_FAMILY_STATUS" ,"#639af2", w=700, lm=100, return_trace= True)
tr2 = bar_hor(app_train, "NAME_HOUSING_TYPE", "Distribution of NAME_HOUSING_TYPE" ,"#a4c5f9", w=700, lm=100, return_trace = True)

fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles = ['NAME_FAMILY_STATUS' , 'NAME_HOUSING_TYPE'])
fig.append_trace(tr1, 1, 1);
fig.append_trace(tr2, 1, 2);
fig['layout'].update(height=400, showlegend=False, margin=dict(l=100));
iplot(fig);
gp('NAME_FAMILY_STATUS', 'Target and Family Status')
gp('NAME_HOUSING_TYPE', 'Target and Housing Type')
tr1 = bar_hor(app_train, "NAME_EDUCATION_TYPE", "Distribution of NAME_EDUCATION_TYPE" ,"#639af2", w=700, lm=100, return_trace= True)
# tr2 = bar_hor(app_train, "NAME_INCOME_TYPE", "Distribution of NAME_FAMILY_STATUS" ,"#a4c5f9", w=700, lm=100, return_trace = True)

fig = tools.make_subplots(rows=1, cols=1, print_grid=False, subplot_titles = ['NAME_EDUCATION_TYPE' ])
fig.append_trace(tr1, 1, 1);
# fig.append_trace(tr2, 1, 2);
fig['layout'].update(height=400, showlegend=False, margin=dict(l=100));
iplot(fig);
gp('NAME_EDUCATION_TYPE', 'Education Type and Target')
tr1 = bar_hor(app_train, "ORGANIZATION_TYPE", "Distribution of ORGANIZATION_TYPE" ,"#639af2", w=700, lm=100, return_trace= True)
tr2 = bar_hor(app_train, "OCCUPATION_TYPE", "Distribution of OCCUPATION_TYPE" ,"#a4c5f9", w=700, lm=100, return_trace = True)

fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles = ['ORGANIZATION_TYPE' , 'OCCUPATION_TYPE'])
fig.append_trace(tr1, 1, 1);
fig.append_trace(tr2, 1, 2);
fig['layout'].update(height=600, showlegend=False, margin=dict(l=200));
iplot(fig);
gp('ORGANIZATION_TYPE', 'Organization Type and Target')
gp('OCCUPATION_TYPE', 'Occupation Type and Target')
tr1 = bar_hor(app_train, "WALLSMATERIAL_MODE", "Distribution of FLAG_OWN_CAR" ,"#a4c5f9", w=700, lm=100, return_trace = True)
tr2 = bar_hor(app_train, "FONDKAPREMONT_MODE", "Distribution of FLAG_OWN_REALTY" ,"#639af2", w=700, lm=100, return_trace= True)
tr3 = bar_hor(app_train, "HOUSETYPE_MODE", "Distribution of FLAG_OWN_CAR" ,"#a4c5f9", w=700, lm=100, return_trace = True)

fig = tools.make_subplots(rows=1, cols=3, print_grid=False, subplot_titles = ['WALLSMATERIAL_MODE' , 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE'])
fig.append_trace(tr1, 1, 1);
fig.append_trace(tr2, 1, 2);
fig.append_trace(tr3, 1, 3);

fig['layout'].update(height=400, showlegend=False, margin=dict(l=100));
iplot(fig);
gp('WALLSMATERIAL_MODE', 'Wall Material and Target')
gp('FONDKAPREMONT_MODE', 'Foundation and Target')
gp('HOUSETYPE_MODE', 'House Type and Target')
plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_CREDIT")
ax = sns.distplot(app_train["AMT_CREDIT"])
plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_ANNUITY")
ax = sns.distplot(app_train["AMT_ANNUITY"].dropna())
plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_GOODS_PRICE")
ax = sns.distplot(app_train["AMT_GOODS_PRICE"].dropna())
# ax = sns.distplot(app_train["CNT_CHILDREN"], kde = False)
plt.figure(figsize=(12,5))
plt.title("Distribution of REGION_POPULATION_RELATIVE")
ax = sns.distplot(app_train["REGION_POPULATION_RELATIVE"])
plt.figure(figsize=(12,5))
plt.title("Distribution of DAYS_BIRTH")
ax = sns.distplot(app_train["DAYS_BIRTH"])
plt.figure(figsize=(12,5))
plt.title("Distribution of DAYS_EMPLOYED")
ax = sns.distplot(app_train["DAYS_EMPLOYED"])
plt.figure(figsize=(12,5))
plt.title("Distribution of DAYS_REGISTRATION")
ax = sns.distplot(app_train["DAYS_REGISTRATION"])
plt.figure(figsize=(12,5))
plt.title("Distribution of CNT_FAM_MEMBERS")
ax = sns.distplot(app_train["CNT_FAM_MEMBERS"].dropna())
bureau = pd.read_csv(path + "bureau.csv")
bureau.head()
bureau_balance = pd.read_csv(path + "bureau_balance.csv")
bureau_balance.head()
credit_card_balance = pd.read_csv(path + "credit_card_balance.csv")
credit_card_balance.head()
pcb = pd.read_csv(path + "POS_CASH_balance.csv")
pcb.head()
previous_application = pd.read_csv(path + "previous_application.csv")
previous_application.head()
# 契約状況 (受理、拒否、申請キャンセル、申請なし (unused offer))の比率の円グラフを表示
t = previous_application['NAME_CONTRACT_STATUS'].value_counts()
labels = t.index
values = t.values

colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1']

trace = go.Pie(labels=labels, 
               values=values,
               hoverinfo='', 
               textinfo='',
               textfont=dict(size=12), # フォントサイズ
               marker=dict(colors=colors, # 色設定
                           line=dict(color='#fff', width=2))
              )

layout = go.Layout(title='Name Contract Status in Previous Applications', height=400)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)
t = previous_application['NAME_TYPE_SUITE'].value_counts()
labels = t.index
values = t.values

colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1']

trace = go.Pie(labels=labels, values=values,
               hoverinfo='', textinfo='',
               textfont=dict(size=12),
               marker=dict(colors=colors,
                           line=dict(color='#fff', width=2)))

layout = go.Layout(title='Suite Type in Previous Application Distribution', height=400)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)
t = previous_application['NAME_CLIENT_TYPE'].value_counts()
labels = t.index
values = t.values

colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1']

trace = go.Pie(labels=labels, values=values,
               hoverinfo='', textinfo='',
               textfont=dict(size=12),
               marker=dict(colors=colors,
                           line=dict(color='#fff', width=2)))

layout = go.Layout(title='Client Type in Previous Applications', height=400)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)
t = previous_application['CHANNEL_TYPE'].value_counts()
labels = t.index
values = t.values

colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1']

trace = go.Pie(labels=labels, values=values,
               hoverinfo='', textinfo='',
               textfont=dict(size=12),
               marker=dict(colors=colors,
                           line=dict(color='#fff', width=2)))

layout = go.Layout(title='Channel Type in Previous Applications', height=400)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)
installments_payments = pd.read_csv(path + "installments_payments.csv")
installments_payments.head()
from sklearn.model_selection import train_test_split 
import lightgbm as lgb

# テストファイル読み込み
app_test = pd.read_csv('../input/application_test.csv')

app_test['is_test'] = 1 
app_test['is_train'] = 0
app_train['is_test'] = 0
app_train['is_train'] = 1

# 目的変数
Y = app_train['TARGET']
train_X = app_train.drop(['TARGET'], axis = 1)

# テストID
test_id = app_test['SK_ID_CURR']
test_X = app_test

# 前処理のためにトレインとテストを連結
data = pd.concat([train_X, test_X], axis=0)
# カテゴリ変数を取得する関数
def _get_categorical_features(df):
    feats = [col for col in list(df.columns) if df[col].dtype == 'object']
    return feats

# カテゴリ変数をファクトライズ (整数に置換)する関数
def _factorize_categoricals(df, cats):
    for col in cats:
        df[col], _ = pd.factorize(df[col])
    return df 

# カテゴリ変数のダミー変数 (二値変数化)を作成する関数
def _get_dummies(df, cats):
    for col in cats:
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
    return df 

# カテゴリ変数を取得
data_cats = _get_categorical_features(data)
prev_app_cats = _get_categorical_features(previous_application)
bureau_cats = _get_categorical_features(bureau)

# ダミー変数を取得
previous_application = _get_dummies(previous_application, prev_app_cats)
bureau = _get_dummies(bureau, bureau_cats)

# カテゴリ変数をファクトライズ
data = _factorize_categoricals(data, data_cats)
## 参考: より良い特徴量 : https://www.kaggle.com/ogrellier/good-fun-with-ligthgbm 

### 過去の申請書情報を現在の申請書情報と結びつける

# IDごとに過去の申請回数をカウント
prev_apps_count = previous_application[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
previous_application['SK_ID_PREV'] = previous_application['SK_ID_CURR'].map(prev_apps_count['SK_ID_PREV'])

# IDごとに過去の申請書の特徴量の値の平均を取る
prev_apps_avg = previous_application.groupby('SK_ID_CURR').mean()
prev_apps_avg.columns = ['p_' + col for col in prev_apps_avg.columns]
data = data.merge(right=prev_apps_avg.reset_index(), how='left', on='SK_ID_CURR')

### 信用情報機関を現在の申請書情報と結びつける

# IDごとに信用情報機関の特徴量の値の平均を取る
bureau_avg = bureau.groupby('SK_ID_CURR').mean()
bureau_avg['buro_count'] = bureau[['SK_ID_BUREAU','SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']
bureau_avg.columns = ['b_' + f_ for f_ in bureau_avg.columns]
data = data.merge(right=bureau_avg.reset_index(), how='left', on='SK_ID_CURR')

# 最終的なtrainとtestデータを用意
ignore_features = ['SK_ID_CURR', 'is_train', 'is_test']
relevant_features = [col for col in data.columns if col not in ignore_features]
trainX = data[data['is_train'] == 1][relevant_features]
testX = data[data['is_test'] == 1][relevant_features]
x_train, x_val, y_train, y_val = train_test_split(trainX, Y, test_size=0.2, random_state=18)
lgb_train = lgb.Dataset(data=x_train, label=y_train)
lgb_eval = lgb.Dataset(data=x_val, label=y_val)
params = {'task': 'train', 'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc', 
          'learning_rate': 0.01, 'num_leaves': 48, 'num_iteration': 5000, 'verbose': 0 ,
          'colsample_bytree':.8, 'subsample':.9, 'max_depth':7, 'reg_alpha':.1, 'reg_lambda':.1, 
          'min_split_gain':.01, 'min_child_weight':1}
model = lgb.train(params, lgb_train, valid_sets=lgb_eval, early_stopping_rounds=150, verbose_eval=200)
lgb.plot_importance(model, figsize=(12, 50));
preds = model.predict(testX)
sub_lgb = pd.DataFrame()
sub_lgb['SK_ID_CURR'] = test_id
sub_lgb['TARGET'] = preds
sub_lgb.to_csv("lgb_baseline.csv", index=False)
sub_lgb.head()
