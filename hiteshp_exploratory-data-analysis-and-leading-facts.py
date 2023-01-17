import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
color = sns.color_palette()

%matplotlib inline

pd.options.mode.chained_assignment = None  # default='warn'


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Train shape : ", train.shape)
print("Test shape : ", test.shape)
train.head()
test.describe()
test.describe()
plt.figure(figsize=(20,8))
sns.countplot(train.Target)
plt.title("Value Counts of Target Variable")
missing_train = train.isnull().sum(axis=0).reset_index()
missing_train.columns = ['column_name', 'missing_count']
missing_train = missing_train[missing_train['missing_count']>0]
missing_train['d_type']='train'

missing_test = test.isnull().sum(axis=0).reset_index()
missing_test.columns = ['column_name', 'missing_count']
missing_test = missing_test[missing_test['missing_count']>0]
missing_test['d_type']='test'

ind = np.arange(missing_train.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,5))
rects = ax.barh(ind, missing_train.missing_count.values, color='y')
ax.set_yticks(ind)
ax.set_yticklabels(missing_train.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column for train data set")
plt.show()

ind = np.arange(missing_test.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,5))
rects = ax.barh(ind, missing_test.missing_count.values, color='y')
ax.set_yticks(ind)
ax.set_yticklabels(missing_test.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column for test data set")
plt.show()

dtype_df = train.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()
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
 
cols_to_use = corr_df[(corr_df['corr_values']>0.21) | (corr_df['corr_values']<-0.21)].col_labels.tolist()

temp_df = train[cols_to_use]
corrmat = temp_df.corr(method='spearman')
f, ax = plt.subplots(figsize=(20, 20))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=1., square=True, cmap="YlGnBu", annot=True)
plt.title("Important variables correlation map", fontsize=15)
plt.show()
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
target = train['Target'].astype('int')
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
plt.figure(figsize=(12,8))
sns.countplot(x="rooms", data=train)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Number of rooms', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()
plt.figure(figsize=(12,8))
sns.countplot(x="qmobilephone", data=train)
plt.ylabel('Count', fontsize=12)
plt.xlabel('No of mobile phones', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()
plt.figure(figsize=(12,8))
sns.countplot(x="overcrowding", data=train)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Number persons per room', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()
plt.figure(figsize=(12,8))
sns.countplot(x="meaneduc", data=train)
plt.ylabel('Count', fontsize=12)
plt.xlabel('average years of education for adults (18+)', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()
plt.figure(figsize=(12,8))
sns.countplot(x="edjefa", data=train)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Years of education of female head of household', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()
plt.figure(figsize=(12,8))
sns.countplot(x="r4t2", data=train)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Persons 12 years of age and older', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()
plt.figure(figsize=(12,8))
sns.countplot(x="r4h2", data=train)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Males 12 years of age and older', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()
plt.figure(figsize=(12,8))
sns.countplot(x="SQBedjefe", data=train)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Square of years of education of male head of household', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()
plt.figure(figsize=(12,8))
sns.countplot(x="dependency", data=train)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Dependency', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()
plt.figure(figsize=(12,8))
sns.countplot(x="SQBdependency", data=train)
plt.ylabel('Count', fontsize=12)
plt.xlabel('dependency squared', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()











