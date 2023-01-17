import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
from scipy.special import boxcox
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#PLOTLY
import plotly.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import cufflinks as cf
from plotly.graph_objs import Scatter, Figure, Layout
cf.set_config_file(offline=True)
print(">> Loading Data...")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Train shape {}".format(train.shape))
print("Test shape {}".format(test.shape))
target = train['Target'].astype('int')
data = [go.Histogram(x=target)]
layout = go.Layout(title = "Target Histogram")
fig = go.Figure(data=data, layout=layout)
iplot(fig)
plt.figure(figsize=(20,8))
sns.countplot(train.Target)
plt.title("Value Counts of Target Variable")
print(f"Numer of Missing values in train: ", train.isnull().sum().sum())
print(f"Number of Missing values in train: ", test.isnull().sum().sum())
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

labels = []
values = []
for col in train.columns:
    if col not in ["Id", "Target"]:
        labels.append(col)
        values.append(spearmanr(train[col].values, train["Target"].values)[0])
corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})
corr_df = corr_df.sort_values(by='corr_values')
 
corr_df = corr_df[(corr_df['corr_values']>0.1) | (corr_df['corr_values']<-0.1)]
ind = np.arange(corr_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,30))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='red')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
plt.show()
plt.figure(figsize=(15,15))
sns.heatmap(train[corr_df.col_labels[:50]].corr())
plt.figure(figsize=(15,15))
sns.heatmap(train[corr_df.col_labels[:10]].corr(), annot=True)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
train.head()
train.drop(['Id','Target'], axis=1, inplace=True)
obj_columns = [f_ for f_ in train.columns if train[f_].dtype == 'object']
for col in tqdm(obj_columns):
    le = LabelEncoder()
    le.fit(train[col].astype(str))
    train[col] = le.transform(train[col].astype(str))
lgbm = LGBMClassifier()
xgbm = XGBClassifier()
train = train.astype('float32') # For faster computation
lgbm.fit(train, target , verbose=False)
xgbm.fit(train, target ,verbose=False)
LGBM_FEAT_IMP = pd.DataFrame({'Features':train.columns, "IMP": lgbm.feature_importances_}).sort_values(by='IMP', ascending=False)

XGBM_FEAT_IMP = pd.DataFrame({'Features':train.columns, "IMP": xgbm.feature_importances_}
                            ).sort_values(
                              by='IMP', ascending=False)
LGBM_FEAT_IMP.head(10).transpose()
XGBM_FEAT_IMP.head(10).transpose()
data = [go.Bar(
            x= LGBM_FEAT_IMP.head(50).Features,
            y= LGBM_FEAT_IMP.head(50).IMP, 
            marker=dict(color='green',))
       ]
layout = go.Layout(title = "LGBM Top 50 Feature Importances")
fig = go.Figure(data=data, layout=layout)
iplot(fig)
data = [go.Bar(
            x= XGBM_FEAT_IMP.head(50).Features,
            y= XGBM_FEAT_IMP.head(50).IMP, 
            marker=dict(color='blue',))
       ]
layout = go.Layout(title = "XGBM Top 50 Feature Importances")
fig = go.Figure(data=data, layout=layout)
iplot(fig)
cols_imp = list(set(LGBM_FEAT_IMP[LGBM_FEAT_IMP.IMP > 0 ].Features.values) & set(
    XGBM_FEAT_IMP[XGBM_FEAT_IMP.IMP > 0 ].Features.values))
MUTUAL_50 = cols_imp[:50]
DIFF_DESCRIBE = train[MUTUAL_50].describe().transpose() - test[MUTUAL_50].describe().transpose()
DIFF_DESCRIBE.style.format("{:.2f}").bar(align='mid', color=['#d65f5f', '#5fba7d'])
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist
X = train[cols_imp].dropna()
distortions = []
for k in tqdm(range(1,8)):
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
# Create a trace
trace = go.Line(
    x = [1,2,3,4,5,6,7,8],
    y = distortions,
    line = dict(
    color = 'red',
    width = 2),
    mode = 'lines+markers',
    name = 'lines+markers'
)
data = [trace]
layout = go.Layout(title = "Elbow Method Optimal Clusters - 3 (From Graph)")
fig = go.Figure(data=data, layout=layout)
iplot(fig)
from sklearn.metrics import silhouette_score
k_clusters = []
sil_coeffecients = []

for n_cluster in range(2,6):
    kmeans = KMeans(n_clusters = n_cluster).fit(X)
    label = kmeans.labels_
    sil_coeff = silhouette_score(X, label)
    print("For n_clusters={}, Silhouette Coefficient = {}".format(n_cluster, sil_coeff))
    sil_coeffecients.append(sil_coeff)
    k_clusters.append(n_cluster)
# Create a trace
trace = go.Line(
    x = [1,2,3,4,5,6],
    y = sil_coeffecients,
    line = dict(
    color = 'orange',
    width = 2),
    mode = 'lines+markers',
    name = 'lines+markers'
)
data = [trace]
layout = go.Layout(title = "Silhouette Optimal Clusters - 3 (From Graph)")
fig = go.Figure(data=data, layout=layout)
iplot(fig)