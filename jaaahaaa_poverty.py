%load_ext autoreload
%autoreload 2

%matplotlib inline
from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from IPython.display import display

from sklearn import metrics
df_raw = pd.read_csv('../input/train.csv', low_memory=False)
d={}
weird=[]
for row in df_raw.iterrows():
    idhogar=row[1]['idhogar']
    target=row[1]['Target']
    if idhogar in d:
        if d[idhogar]!=target:
            weird.append(idhogar)
    else:
        d[idhogar]=target

for i in set(weird):
    hhold=df_raw[df_raw['idhogar']==i][['idhogar', 'parentesco1', 'Target']]
    target=hhold[hhold['parentesco1']==1]['Target'].tolist()[0]
    for row in hhold.iterrows():
        idx=row[0]
        if row[1]['parentesco1']!=1:
            df_raw.at[idx, 'Target']=target
            
df_raw[df_raw['idhogar']==weird[1]][['idhogar','parentesco1', 'Target']]
print("Duplicates removed")
def data_cleaning(data):
    data['dependency']=np.sqrt(data['SQBdependency'])
    data['rez_esc']=data['rez_esc'].fillna(0)
    data['v18q1']=data['v18q1'].fillna(0)
    data['v2a1']=data['v2a1'].fillna(0)
    
    conditions = [
    (data['edjefe']=='no') & (data['edjefa']=='no'), #both no
    (data['edjefe']=='yes') & (data['edjefa']=='no'), # yes and no
    (data['edjefe']=='no') & (data['edjefa']=='yes'), #no and yes 
    (data['edjefe']!='no') & (data['edjefe']!='yes') & (data['edjefa']=='no'), # number and no
    (data['edjefe']=='no') & (data['edjefa']!='no') # no and number
    ]
    choices = [0, 1, 1, data['edjefe'], data['edjefa']]
    data['edjefx']=np.select(conditions, choices)
    data['edjefx']=data['edjefx'].astype(int)
    data.drop(['edjefe', 'edjefa'], axis=1, inplace=True)
    
    
    #Figure out if head of family is male or female
    conditions = [
    (data['male']==1)   & (data['parentesco1']==1),
    (data['female']==1) & (data['parentesco1']==1)
    ]
    choices = [0, 1]
    data['head_gender']=np.select(conditions, choices)
    data['head_gender']=data['head_gender'].astype(int)
    
    meaneduc_nan=data[data['meaneduc'].isnull()][['Id','idhogar','escolari']]
    me=meaneduc_nan.groupby('idhogar')['escolari'].mean().reset_index()
    for row in meaneduc_nan.iterrows():
        idx=row[0]
        idhogar=row[1]['idhogar']
        m=me[me['idhogar']==idhogar]['escolari'].tolist()[0]
        data.at[idx, 'meaneduc']=m
        data.at[idx, 'SQBmeaned']=m*m
       
    data.drop(['idhogar', 'Id'], axis=1, inplace=True)
    data.drop(['SQBmeaned', 'SQBdependency', 'SQBovercrowding', 'SQBhogar_nin', 'hogar_total', 'SQBage', 'agesq', 'SQBescolari', 'SQBhogar_total', 'tamhog', 'r4t3', 'tamviv', 'v18q'], axis=1, inplace=True)
    return data

data_cleaning(df_raw)
print("Data cleaned")
train_cats(df_raw)
df, y, nas = proc_df(df_raw, 'Target')
len (df)
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
display_all(df.tail().T)
display_all(df_raw.describe(include='all').T)
def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 1900
n_trn = len(df)-n_valid
raw_train, raw_valid = split_vals(df_raw, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape, n_trn
set_rf_samples(3000)
reset_rf_samples()
m = RandomForestClassifier(n_estimators=50, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
preds=m.predict(X_valid)
print(classification_report(y_valid, preds))
print(m.oob_score_)
#print(confusion_matrix(y_valid, preds))

#print(gs.best_params_)
#print(gs.best_estimator_)
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
fi = rf_feat_importance(m, df);
plot_fi(fi[:25]);
m = RandomForestClassifier(n_estimators=50, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
preds=m.predict(X_valid)
print(classification_report(y_valid, preds))
print(m.oob_score_)
to_keep = fi[fi.imp>0.010].cols; len(to_keep)
df_keep = df_raw[to_keep].copy()
X_train, X_valid = split_vals(df_keep, n_trn)
print(X_valid.shape)
m = RandomForestClassifier(n_estimators=50, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
preds=m.predict(X_valid)
print(classification_report(y_valid, preds))
print(m.oob_score_)
fi = rf_feat_importance(m, df_keep);
plot_fi(fi[:25]);
from scipy.cluster import hierarchy as hc
corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)
plt.show()
def get_oob(df):
    m = RandomForestRegressor(n_estimators=50, n_jobs=-1, oob_score=True)
    x, _ = split_vals(df, n_trn)
    m.fit(x, y_train)
    return m.oob_score_
get_oob(df_keep)
#for c in ('hhsize'): #'r4t3', 'tamviv'
#    print(c, get_oob(df_keep.drop(c, axis=1)))
#Final training with all data
m = RandomForestClassifier(n_estimators=1000, max_features=0.6, n_jobs=-1, oob_score=True)
m.fit(df_keep, y)
print(m.oob_score_)
df_test_raw = pd.read_csv('../input/test.csv', low_memory=False)
ids=df_test_raw['Id'] #Save for later
data_cleaning(df_test_raw)
print("Data cleaned")
train_cats(df_test_raw)
df_test, _, _ = proc_df(df_test_raw)

df_keep_test = df_test_raw[to_keep].copy()
predicted_target = m.predict(df_keep_test)

submit=pd.DataFrame({'Id': ids, 'Target': predicted_target})
submit.to_csv('submission.csv', index=False)