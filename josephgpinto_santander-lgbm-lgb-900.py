import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt#visualization
from PIL import  Image
%matplotlib inline
import seaborn as sns#visualization
import itertools
import warnings
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")
import io
import plotly.offline as py#visualization
py.init_notebook_mode(connected=True)#visualization
import plotly.graph_objs as go#visualization
import plotly.tools as tls#visualization
import plotly.figure_factory as ff#visualization

random_state = 42
np.random.seed(random_state)
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
data = pd.read_csv('../input/train.csv')
data.shape
data.describe()
#labels
lab = data["target"].value_counts().keys().tolist()
#values
val = data["target"].value_counts().values.tolist()

trace = go.Pie(labels = lab ,
               values = val ,
               marker = dict(colors =  [ 'royalblue' ,'lime'],
                             line = dict(color = "white",
                                         width =  1.3)
                            ),
               rotation = 90,
               hoverinfo = "label+value+text",
               hole = .5
              )
layout = go.Layout(dict(title = "class imbalance",
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                       )
                  )

data1 = [trace]
fig = go.Figure(data = data1,layout = layout)
py.iplot(fig)
features = [c for c in data.columns if c not in ['ID_code', 'target']]
X = data[features].head(n=5000)
Y = data['target'].head(n=5000)

def tsne_plot(x1, y1, name="graph.png"):
    tsne = TSNE(n_components=2, random_state=0)
    X_t = tsne.fit_transform(x1)

    plt.figure(figsize=(20, 8))
    plt.scatter(X_t[np.where(y1 == 0), 0], X_t[np.where(y1 == 0), 1], marker='o', color='g', linewidth='1', alpha=0.8, label='0')
    plt.scatter(X_t[np.where(y1 == 1), 0], X_t[np.where(y1 == 1), 1], marker='o', color='r', linewidth='1', alpha=0.8, label='1')

    plt.legend(loc='best');
    plt.show();
    
tsne_plot(X, Y, "tsne.png")
X = data.iloc[:,1:202]

pca = PCA(2)  
projected = pca.fit_transform(X)

plt.figure(figsize=(20, 8))
plt.scatter(projected[:, 0], projected[:, 1],
            c=X['target'], edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('YlGnBu', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();

def process_data(df_train, df_test):
    idx = [c for c in df_train.columns if c not in ['ID_code', 'target']]
    for df in [df_test, df_train]:
        for feat in idx:
            df['r2_'+feat] = np.round(df[feat], 2)
            df['r2_'+feat] = np.round(df[feat], 2)
        #df['sum'] = df[idx].sum(axis=1)
        #df['min'] = df[idx].min(axis=1)
        #df['max'] = df[idx].max(axis=1)
        #df['mean'] = df[idx].mean(axis=1)
        df['std'] = df[idx].std(axis=1)
        df['skew'] = df[idx].skew(axis=1)
        df['kurt'] = df[idx].kurtosis(axis=1)
        df['med'] = df[idx].median(axis=1)
    print('Train and test shape:',df_train.shape, df_test.shape)
    return df_train, df_test


#process_data(df_train,df_test)


def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y



lgb_params = {
    "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 13,
    "learning_rate" : 0.01,
    "bagging_freq": 5,
    "bagging_fraction" : 0.335,
    "feature_fraction" : 0.041,
    "min_data_in_leaf": 80,
    "min_sum_hessian_in_leaf": 10,
    "tree_learner": "serial",
    "boost_from_average": "false",
    "num_threads": 4,
    "scale_pos_weight":0.8882836,
    #"lambda_l1" : 5,
    #"lambda_l2" : 5,
    "bagging_seed" : random_state,
    "verbosity" : -1,
    "seed": random_state
}

target = df_train['target']
num_folds = 8
features = [c for c in df_train.columns if c not in ['ID_code', 'target']]

folds = KFold(n_splits=num_folds, random_state=2319)
oof = np.zeros(len(df_train))
getVal = np.zeros(len(df_train))
predictions = np.zeros(len(target))
feature_importance_df = pd.DataFrame()

print('Light GBM Model')
for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train.values, target.values)):
    X_train, y_train = df_train.iloc[trn_idx][features], target.iloc[trn_idx]
    X_valid, y_valid = df_train.iloc[val_idx][features], target.iloc[val_idx]

    #N = 2
    #for i in range(N):
        #print("fold split:{}".format(fold_ + 1))
        #print("fold N:{}".format(i))

    X_tr, y_tr = augment(X_train.values, y_train.values)
    X_tr = pd.DataFrame(X_tr)

    print("Fold idx:{}".format(fold_ + 1))
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_valid, label=y_valid)
    val_data = lgb.Dataset(df_train.iloc[val_idx][features], label=target.iloc[val_idx])

    clf = lgb.train(lgb_params, trn_data, 1000000, valid_sets=[trn_data, val_data], verbose_eval=4000,
                    early_stopping_rounds=4000)
    oof[val_idx] = clf.predict(df_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    getVal[val_idx] += clf.predict(df_train.iloc[val_idx][features], num_iteration=clf.best_iteration) / folds.n_splits

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += clf.predict(df_test[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))


num_sub = 10
print('Saving the Submission File')
sub = pd.DataFrame({"ID_code": df_test.ID_code.values})
sub["target"] = predictions
sub.to_csv('submission{}.csv'.format(num_sub), index=False)

cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)
best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
data=best_features.sort_values(by="importance",ascending=False)

data = [go.Bar(
            x=data['importance'],
            y=data['feature'],
            orientation = 'h'
)]

layout = go.Layout(
    title='Feature importance',
    font=dict(family='Courier New, monospace', size=12, color='#9467bd'),
    autosize=False,
    width=1200,
    height=2500,
    plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='automargin')


