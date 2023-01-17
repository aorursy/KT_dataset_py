import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer, FunctionTransformer
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score, classification_report
from sklearn.decomposition import PCA, FactorAnalysis, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
pd.set_option('display.max_columns', None)
train = pd.read_csv("/kaggle/input/av-janatahack-crosssell-prediction/train.csv")
test = pd.read_csv("/kaggle/input/av-janatahack-crosssell-prediction/test.csv")
train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)
sample = pd.read_csv("/kaggle/input/av-janatahack-crosssell-prediction/sample.csv")
train.info()
train.head()
train.isna().sum()
test.isna().sum()
for col in train.columns:
    print(f"{col} : {train[col].nunique()}")
    print(train[col].unique())
#separating continuous and categorical variables
cat_var = ["Gender","Driving_License","Previously_Insured","Vehicle_Age","Vehicle_Damage"]
con_var = list(set(train.columns).difference(cat_var+["Response"]))
train.Response.value_counts(normalize=True)
sns.countplot(train.Response)
plt.title("Class count")
plt.show()
#sns.pairplot(train, hue='Response', diag_kind='hist')
#plt.show()
def map_val(data):
    data["Gender"] = data["Gender"].replace({"Male":1, "Female":0})
    data["Vehicle_Age"] = data["Vehicle_Age"].replace({'> 2 Years':2, '1-2 Year':1, '< 1 Year':0 })
    data["Vehicle_Damage"] = data["Vehicle_Damage"].replace({"Yes":1, "No":0})
    return data

train = map_val(train)
test = map_val(test)
fig, ax = plt.subplots(2,3 , figsize=(16,6))
ax = ax.flatten()
i = 0
for col in cat_var:
    sns.pointplot(col, 'Response', data=train, ax = ax[i])
    i+=1
plt.tight_layout()
plt.show()
sns.catplot('Gender', 'Response',hue='Vehicle_Age', row = 'Previously_Insured',col='Vehicle_Damage',data=train, kind='point', height=3, aspect=2)
plt.show()
fig, ax = plt.subplots(2,3 , figsize=(16,6))
ax = ax.flatten()
i = 0
for col in con_var:
    sns.boxplot( 'Response', col, data=train, ax = ax[i])
    i+=1
plt.tight_layout()
plt.show()
sns.catplot('Gender', 'Vintage',hue='Response', row = 'Previously_Insured',col='Vehicle_Damage',data=train, kind='box', height=3, aspect=2)
plt.show()
sns.catplot('Gender', 'Age',hue='Response', row = 'Previously_Insured',col='Vehicle_Damage',data=train, kind='box', height=3, aspect=2)
plt.show()
sns.catplot('Gender', 'Annual_Premium',hue='Response', row = 'Previously_Insured',col='Vehicle_Damage',data=train, kind='box', height=3, aspect=2)
plt.show()
plt.figure(figsize=(30,5))
sns.heatmap(pd.crosstab([train['Previously_Insured'], train['Vehicle_Damage']], train['Region_Code'],
                        values=train['Response'], aggfunc='mean', normalize='columns'), annot=True, cmap='inferno')
plt.show()
corr = train.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)]=True
plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='YlGnBu', mask=mask)
plt.title("Correlation Heatmap")
plt.show()
train.skew()
train['log_premium'] = np.log(train.Annual_Premium)
train['log_age'] = np.log(train.Age)
test['log_premium'] = np.log(test.Annual_Premium)
test['log_age'] = np.log(test.Age)
train.groupby(['Previously_Insured','Gender'])['log_premium'].plot(kind='kde')
plt.show()
train.groupby(['Previously_Insured','Gender'])['log_age'].plot(kind='kde')
plt.show()
def feature_engineering(data, col):
    mean_age_insured = data.groupby(['Previously_Insured','Vehicle_Damage'])[col].mean().reset_index()
    mean_age_insured.columns = ['Previously_Insured','Vehicle_Damage','mean_'+col+'_insured']
    mean_age_gender = data.groupby(['Previously_Insured','Gender'])[col].mean().reset_index()
    mean_age_gender.columns = ['Previously_Insured','Gender','mean_'+col+'_gender']
    mean_age_vehicle = data.groupby(['Previously_Insured','Vehicle_Age'])[col].mean().reset_index()
    mean_age_vehicle.columns = ['Previously_Insured','Vehicle_Age','mean_'+col+'_vehicle']
    data = data.merge(mean_age_insured, on=['Previously_Insured','Vehicle_Damage'], how='left')
    data = data.merge(mean_age_gender, on=['Previously_Insured','Gender'], how='left')
    data = data.merge(mean_age_vehicle, on=['Previously_Insured','Vehicle_Age'], how='left')
    data[col+'_mean_insured'] = data['log_age']/data['mean_'+col+'_insured']
    data[col+'_mean_gender'] = data['log_age']/data['mean_'+col+'_gender']
    data[col+'_mean_vehicle'] = data['log_age']/data['mean_'+col+'_vehicle']
    data.drop(['mean_'+col+'_insured','mean_'+col+'_gender','mean_'+col+'_vehicle'], axis=1, inplace=True)
    return data

train = feature_engineering(train, 'log_age')
test = feature_engineering(test, 'log_age')

train = feature_engineering(train, 'log_premium')
test = feature_engineering(test, 'log_premium')

train = feature_engineering(train, 'Vintage')
test = feature_engineering(test, 'Vintage')
X = train.drop(["Response"], axis=1)
Y = train["Response"]
dummy = ["Vehicle_Age"]
passthru = con_var = list(set(X.columns).difference(dummy))

onehot = OneHotEncoder(handle_unknown='ignore')
label = OrdinalEncoder()
scaler = StandardScaler()

feat_rf = RandomForestClassifier(n_jobs=4, random_state=1, class_weight='balanced_subsample')
feat_xgb = XGBClassifier(n_jobs=4, random_state=1, objective='binary:logistic')
selector_rf = SelectFromModel(feat_xgb, threshold=0.001)

transformers_onehot = [('pass','passthrough',passthru),
                       ('onehot', onehot, dummy) ]
ct_onehot = ColumnTransformer( transformers=transformers_onehot )

transformers_label = [('pass','passthrough',passthru),
                      ('onehot', label, dummy) ]
ct_label = ColumnTransformer( transformers=transformers_label )

pipe = Pipeline([('ct', ct_onehot),
                 ('scaler', scaler)])
poly = PolynomialFeatures(degree= 2, interaction_only=True)
pca = PCA(n_components=0.99)
kbest = SelectKBest(k=10)

pipe_pca = Pipeline([('ct', ct_onehot),
                      ('poly', poly),
                      ('scaler', scaler),
                      ('pca',pca)])

pipe_kbest = Pipeline([('ct', ct_onehot),
                       ('poly', poly),
                       ('scaler', scaler),
                       ('kbest',kbest)])

pipe_union = FeatureUnion([('pca',pipe_pca),
                           ('kbest',pipe_kbest)])
# merging the PCA components and KBest features from the data
pipe_union.fit(X, Y)
X_union = pipe_union.transform(X)
test_union = pipe_union.transform(test)
#np.cumsum(pipe_union.transformer_list[0][1].named_steps['pca'].explained_variance_ratio_)
ct_onehot.fit(X)
categories = ct_onehot.named_transformers_['onehot'].categories_
onehot_cols = [col+"_"+str(cat) for col,cats in zip(dummy, categories) for cat in cats]
all_columns = passthru + onehot_cols

X_transform = pd.DataFrame(pipe.fit_transform(X), columns = all_columns)
test_transform = pd.DataFrame(pipe.transform(test), columns = all_columns)

selector_rf.fit(X_transform, Y)
rf_cols = [col for col, flag in zip(X_transform.columns, selector_rf.get_support()) if flag]
print(rf_cols)
X_select = pd.DataFrame(selector_rf.transform(X_transform), columns = rf_cols)
test_select = pd.DataFrame(selector_rf.transform(test_transform), columns = rf_cols)
from tpot import TPOTClassifier
split = StratifiedKFold(n_splits=3, random_state=1)
model = TPOTClassifier(generations=5, population_size=50, scoring='roc_auc', cv=split, verbosity=2, random_state=1, n_jobs=-1)

model.fit(X_select, Y)
model.export('tpot_cross_sell.py')
