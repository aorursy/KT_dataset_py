!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html -q
!pip install --upgrade kornia -q
!pip install allennlp==1.1.0.rc4 -q
!pip install --upgrade fastai -q
#importing the libraries
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


from fastai.tabular.all import *
# Setting the directory structure
path = Path('../input/house-prices-advanced-regression-techniques')
Path.BASE_PATH = path
path.ls()
#importing the data
train = pd.read_csv(path/'train.csv', low_memory=False)
test = pd.read_csv(path/'test.csv', low_memory=False)
train.shape
train.columns
train.head()
#Ordinal Variable
# train['OverallQual']
#dependent variable
dep_var = 'SalePrice'
train[dep_var] = np.log(train[dep_var])
proc = [Categorify, FillMissing]
cont, cat = cont_cat_split(train, 1, dep_var=dep_var)
splits = RandomSplitter(0.2)
splits = splits(train)
to = TabularPandas(train, procs=proc, cat_names=cat, cont_names=cont,splits = splits, y_names=dep_var)
len(to.train), len(to.valid)
to.show(5)
to.items.head(5)
xs, y = to.train.xs, to.train.y
valid_xs, valid_y = to.valid.xs, to.valid.y
m = DecisionTreeRegressor()
m.fit(xs, y);
def r_mse(pred,y): return round(math.sqrt(((pred-y)**2).mean()), 6)
def m_rmse(m, xs, y): return r_mse(m.predict(xs), y)
m_rmse(m, xs, y)
m_rmse(m, valid_xs, valid_y)
m.get_n_leaves(), len(train)
m = DecisionTreeRegressor(min_samples_leaf=10)
m.fit(to.train.xs, to.train.y)
m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y)
m.get_n_leaves()
def rf(xs, y, n_estimators=40, max_samples=800,
       max_features=0.5, min_samples_leaf=5, **kwargs):
    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators,
        max_samples=max_samples, max_features=max_features,
        min_samples_leaf=min_samples_leaf, oob_score=True).fit(xs, y)
m = rf(xs, y);
m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y)
preds = np.stack([t.predict(valid_xs) for t in m.estimators_])
r_mse(preds.mean(0), valid_y)
plt.plot([r_mse(preds[:i+1].mean(0), valid_y) for i in range(40)]);
r_mse(m.oob_prediction_, y)

preds = np.stack([t.predict(valid_xs) for t in m.estimators_])
preds.shape
preds_std = preds.std(0)
preds_std[:5]

def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)
fi = rf_feat_importance(m, xs)
fi[:10]
def plot_fi(fi):
    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)

plot_fi(fi[:30]);
to_keep = fi[fi.imp>0.005].cols
len(to_keep)

xs_imp = xs[to_keep]
valid_xs_imp = valid_xs[to_keep]

m = rf(xs_imp, y)
m_rmse(m, xs_imp, y), m_rmse(m, valid_xs_imp, valid_y)
len(xs.columns), len(xs_imp.columns)
plot_fi(rf_feat_importance(m, xs_imp));
from scipy.cluster import hierarchy as hc

def cluster_columns(df, figsize=(10,6), font_size=12):
    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
    corr_condensed = hc.distance.squareform(1-corr)
    z = hc.linkage(corr_condensed, method='average')
    fig = plt.figure(figsize=figsize)
    hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=font_size)
    plt.show()
cluster_columns(xs_imp)
p = valid_xs_imp['OverallQual'].value_counts(sort=False).plot.barh()
# c = to.classes['OverallQual']
# plt.yticks(range(len(c)), c);
# GrLivArea vs y
plt.scatter(xs_imp['GrLivArea'],y)
# GarageArea vs y
plt.scatter(xs_imp['GarageArea'],y)
from sklearn.inspection import plot_partial_dependence

fig,ax = plt.subplots(figsize=(12, 4))
plot_partial_dependence(m, valid_xs_imp, ['GrLivArea','GarageArea'],
                        grid_resolution=20, ax=ax);
# !pip install treeinterpreter -q
# !pip install waterfallcharts -q
#hide
import warnings
warnings.simplefilter('ignore', FutureWarning)

from treeinterpreter import treeinterpreter
from waterfall_chart import plot as waterfall
row = valid_xs_imp.iloc[:5]
prediction,bias,contributions = treeinterpreter.predict(m, row.values)

prediction[0], bias[0], contributions[0].sum()
waterfall(valid_xs_imp.columns, contributions[0], threshold=0.08, 
          rotation_value=45,formatting='{:,.3f}');
