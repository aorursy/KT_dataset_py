%load_ext autoreload
%autoreload 2

%matplotlib inline

from fastai.imports import *
from fastai.structured import *
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
import os
print(os.listdir("../input"))
PATH = '../input'
train = pd.read_csv(f'{PATH}/train.csv', low_memory=False)
train
building_structure = pd.read_csv(f'{PATH}/Building_Structure.csv', low_memory=False)
building_ownership = pd.read_csv(f'{PATH}/Building_Ownership_Use.csv', low_memory=False)
test = pd.read_csv(f'{PATH}/test.csv', low_memory=False)
print(train.shape,'\n',building_ownership.shape,'\n', building_structure.shape)
train = train.merge(building_structure, on = 'building_id',how = 'left')
train = train.merge(building_ownership, on = 'building_id', how = 'left')
print(train.columns)
print(train.shape)
train.drop(['district_id_x', 'district_id_y', 'vdcmun_id_x', 'vdcmun_id_y', 'ward_id_y'], axis=1, inplace=True)
print(train.shape,train.columns)
test.shape
test = test.merge(building_structure, on = 'building_id',how = 'left')
test = test.merge(building_ownership, on = 'building_id', how = 'left')
test.drop(['district_id_x', 'district_id_y', 'vdcmun_id_x', 'vdcmun_id_y', 'ward_id_y'], axis=1, inplace=True)
test.shape
#function to display all rows and columns
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
display_all(train.tail().T)

display_all(train.describe(include='all').T)
display_all(train.describe(include='all').T)
# Create table for missing data analysis
def draw_missing_data_table(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
draw_missing_data_table(train)
#count familes have only 1 missing values we'll fill that
train['count_families'].fillna(train['count_families'].mode()[0],inplace=True)
print(train['has_repair_started'].value_counts())
print(test['has_repair_started'].value_counts())

train['has_repair_started'].fillna(False,inplace=True)
test['has_repair_started'].fillna(False,inplace=True)
print(train.columns.hasnans)
print(test.columns.hasnans)
Y = {'Grade 1': 1, 'Grade 2': 2, 'Grade 3': 3, 'Grade 4': 4, 'Grade 5': 5}
train['damage_grade'].replace(Y, inplace = True)
train['damage_grade'].unique()
train.dtypes
print(train.select_dtypes('object').nunique())
print(train.select_dtypes('object').nunique())
#Remove column 'building_id' as it is unique for every row & doesnt have any impact
train_building_id = train['building_id']
test_building_id = test['building_id']
train.drop(['building_id'], axis=1, inplace=True)
test.drop(['building_id'], axis=1, inplace=True)
display_all(train)
train['count_floors_change'] = (train['count_floors_post_eq']/train['count_floors_pre_eq'])
train['height_ft_change'] = (train['height_ft_post_eq']/train['height_ft_pre_eq'])
test['count_floors_change'] = (test['count_floors_post_eq']/test['count_floors_pre_eq'])
test['height_ft_change'] = (test['height_ft_post_eq']/test['height_ft_pre_eq'])

train.drop(['count_floors_post_eq', 'height_ft_post_eq'], axis=1, inplace=True)
test.drop(['count_floors_post_eq', 'height_ft_post_eq'], axis=1, inplace=True)
sns.barplot(train['condition_post_eq'],train['damage_grade']);
sns.barplot(train['plan_configuration'],train['damage_grade']);
train_cats(train)
apply_cats(test, train)
df, y, nas = proc_df(train, 'damage_grade', max_n_cat=6)
test_df, _, _ = proc_df(test, na_dict=nas, max_n_cat=6)

print(test_df.shape, df.shape)
from sklearn.metrics import f1_score

def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [m.score(x_train, y_train), m.score(x_test, y_test)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
from sklearn.model_selection import train_test_split
x = df
x_train, x_test, y_train, y_test = train_test_split(x, y, 
test_size=0.2, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
set_rf_samples(100000)
m = RandomForestClassifier(n_jobs=-1)
%time m.fit(x_train, y_train)
print(m.score(x_train, y_train))
print_score(m)
m = RandomForestRegressor(n_estimators=150, min_samples_leaf=1, max_features=0.6, n_jobs=-1, oob_score=True)
%time m.fit(x_train, y_train)
print_score(m)
fi = rf_feat_importance(m, df); fi[:10]
fi.plot('cols', 'imp', figsize=(10,6), legend=False);

def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plot_fi(fi[:30]);
to_keep = fi[fi.imp>0.001].cols; len(to_keep)
to_keep
def split_vals(a,n): return a[:n], a[n:]
df_keep = df[to_keep].copy()

from scipy.cluster import hierarchy as hc

corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)
plt.show()
correlations = df.corr()
print('Most Positive Correlations:\n', correlations.tail(10))
print('\nMost Negative Correlations:\n', correlations.head(10))
imp_features=['height_ft_change', 'condition_post_eq', 'count_floors_change' , 'ward_id_x','age_building', 'plinth_area_sq_ft']
scor = train[imp_features+['damage_grade']]
data_corrs = scor.corr()
data_corrs
plt.figure(figsize = (8, 6))

# Heatmap of correlations
sns.heatmap(data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');
reset_rf_samples()
m = RandomForestRegressor(n_estimators=150, min_samples_leaf=1, max_features=0.6, n_jobs=-1, oob_score=True)
%time m.fit(x_train, y_train)
print_score(m)
test_df.head(5)
ypreds = m.predict(test_df)
ypreds = ypreds.round()
prediction=pd.DataFrame({'building_id': test_building_id, 'damage_grade':ypreds})
target = {1: 'Grade 1', 2: 'Grade 2', 3: 'Grade 3', 4: 'Grade 4', 5: 'Grade 5'}
prediction.damage_grade.replace(target, inplace=True)
prediction.to_csv('submission.csv', index=False)
prediction.head()
from xgboost import XGBClassifier
xgbc = XGBClassifier(n_estimators=100, learning_rate=0.2, max_depth=6, random_state=42) #random state = 42 as for Feature Imp above 0.01 there were 42 cols
xgbc.fit(x_train, y_train)
print_score(xgbc)
pred_test_y = pd.Series(list(xgbc.predict(test_df)))
prediction=pd.DataFrame({'building_id': test_building_id, 'damage_grade':pred_test_y})
target = {1: 'Grade 1', 2: 'Grade 2', 3: 'Grade 3', 4: 'Grade 4', 5: 'Grade 5'}
prediction.damage_grade.replace(target, inplace=True)
prediction.to_csv('submission.csv', index=False)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(x_train, y_train)
print_score(knn)