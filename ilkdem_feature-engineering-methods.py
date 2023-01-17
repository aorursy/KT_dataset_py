import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV,StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, RFE,SelectPercentile
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
dfdata = pd.read_csv("../input/melbourne-housing-market/Melbourne_housing_FULL.csv")
dfdata.columns = dfdata.columns.str.lower()
dfdata.head()
dfdata.describe().T
print("Data : ",dfdata.shape)
print("Duplicate rows : ",dfdata[dfdata.duplicated()].shape)
dfdata.drop_duplicates(inplace = True)
print("After drop duplicates : ",dfdata.shape)

dfdata.info()
dfdata.isnull().sum()
msno.matrix(dfdata,figsize=(20,6))
plt.show()
msno.heatmap(dfdata,figsize=(8,8))
plt.show()
msno.dendrogram(dfdata,figsize=(12,12))
plt.show()
dfdata.isnull().sum()
dfdata[dfdata['distance'].isnull() == True]
# from dendrogram 9 columns are null, drop it
dfdata.drop(dfdata[dfdata['distance'].isnull() == True].index,axis=0,inplace=True)
dfdata.shape
dfdata[dfdata['bedroom2'].isnull() == True][['rooms','price','landsize','car']].describe().T
pd.crosstab(dfdata['bedroom2'],dfdata['rooms'])

sns.countplot(dfdata[dfdata['bedroom2'].isnull() == True]['rooms'])
plt.show()
dfdata['bedroom2'].fillna(dfdata['rooms'],inplace=True)
dfdata.isnull().sum()
pd.crosstab(dfdata['bathroom'],dfdata['rooms'])
dftemp = dfdata.groupby(['rooms'],as_index=False)['bathroom'].median()
indices = dfdata[dfdata['bathroom'].isnull() == True].index
dfdata.loc[indices,'bathroom'] = dfdata.loc[indices,'rooms'].apply(lambda x : dftemp[dftemp['rooms']==x]['bathroom'].values[0])
dfdata.isnull().sum()
dfdata[dfdata['bathroom'] == 'Nan']
def fill_via_suburb(colname):
    indices = dfdata[dfdata[colname].isnull() == True].index
    dfdata.loc[indices,colname] = dfdata.loc[indices,'suburb'].map(lambda x: dfdata[dfdata['suburb'] == x][colname].mode()[0])

fill_via_suburb('councilarea')
fill_via_suburb('regionname')
fill_via_suburb('propertycount')
dfdata.isnull().sum()
dfdata.drop(['car','landsize','buildingarea','yearbuilt','lattitude','longtitude'],axis=1,inplace=True)
dfdata.isnull().sum()
dfdata.dropna(subset = {'price'},inplace = True)
dfdata.describe()
fig,ax = plt.subplots(2,3,figsize = (24,8))
sns.distplot(dfdata['price'], ax = ax[0][0])
sns.boxplot(dfdata['price'],ax = ax[0][1])
stats.probplot(dfdata['price'],plot = ax[0][2])
sns.distplot(dfdata['distance'], ax = ax[1][0])
sns.boxplot(dfdata['distance'],ax = ax[1][1])
stats.probplot(dfdata['distance'],plot = ax[1][2])
plt.show()
fig,ax = plt.subplots(1,3,figsize = (24,5))
sns.distplot(dfdata['rooms'], ax = ax[0])
sns.distplot(dfdata['bedroom2'],ax = ax[1])
sns.distplot(dfdata['distance'],ax = ax[2])
plt.show()
dfskew = dfdata[['distance','price']]
dfskew_normal = Normalizer().fit_transform(dfskew)
dfskew_standard = StandardScaler().fit_transform(dfskew)
fig,ax = plt.subplots(2,2,figsize = (18,4))
sns.distplot(dfskew_normal[:,0], ax = ax[0][0])
sns.distplot(dfskew_normal[:,1], ax = ax[0][1])
sns.distplot(dfskew_standard[:,0], ax = ax[1][0])
sns.distplot(dfskew_standard[:,1], ax = ax[1][1])
plt.show()
dfskew_log = np.log1p(dfskew)
fig,ax = plt.subplots(1,2,figsize = (18,4))
sns.distplot(dfskew_log['distance'], ax = ax[0])
sns.distplot(dfskew_log['price'], ax = ax[1])
plt.show()
def sigmoid(x):
    e = np.exp(1)
    y = 1 /(1 + e** (-x))
    return y
dfskew_sigmoid = sigmoid(dfskew)
fig,ax = plt.subplots(1,2,figsize = (18,4))
sns.distplot(dfskew_sigmoid['distance'], ax = ax[0])
sns.distplot(dfskew_sigmoid['distance'], ax = ax[1])
plt.show()
dfskew_tanh = np.tanh(dfskew)
fig,ax = plt.subplots(1,2,figsize = (18,4))
sns.distplot(dfskew_tanh['distance'], ax = ax[0])
sns.distplot(dfskew_tanh['price'], ax = ax[1])
plt.show()
dfskew_root3 = dfskew**(1/3)
fig,ax = plt.subplots(1,2,figsize = (18,4))
sns.distplot(dfskew_root3['distance'], ax = ax[0])
sns.distplot(dfskew_root3['price'], ax = ax[1])
plt.show()
dfskew_cube = dfskew**(3)
fig,ax = plt.subplots(1,2,figsize = (18,4))
sns.distplot(dfskew_cube['distance'], ax = ax[0])
sns.distplot(dfskew_cube['price'], ax = ax[1])
plt.show()
dfskew_boxcox = pd.DataFrame()
dfskew_boxcox['price'],power_selected = boxcox(dfskew['price'])
print("lambda selected",round(power_selected,2))
fig,ax = plt.subplots(1,2,figsize = (18,4))
sns.distplot(dfskew['price'], ax = ax[0])
sns.distplot(dfskew_boxcox['price'], ax = ax[1])
plt.show()

dfskew_linear = dfskew.rank(method='min').apply(lambda x:(x-1)/(dfskew.shape[0] - 1))
fig,ax = plt.subplots(1,2,figsize = (18,4))
sns.distplot(dfskew_linear['distance'], ax = ax[0])
sns.distplot(dfskew_linear['price'], ax = ax[1])
plt.show()
def get_lr_result(X,y):
    lr = LinearRegression().fit(X,y)
    return round(100 * lr.score(X, y),1),lr
dfdatanew = dfdata[['rooms','price']]
dfdatanew['rooms'] = dfdatanew['rooms'].astype('str')
X = dfdatanew[['rooms']]
y = dfdatanew['price']
print("Rooms -> Price                  ",get_lr_result(X,y)[0])
X_encoded = pd.get_dummies(X,drop_first=True)
print("one hot encoded rooms -> Price  ",get_lr_result(X_encoded,y)[0])
coef1 = np.round(get_lr_result(X_encoded,y)[1].coef_)
tmp = X_encoded.sum(axis=1)
indices = tmp[tmp == 0].index
X_encoded.loc[indices,:] = -1.0
print("effect encoded rooms -> Price  ",get_lr_result(X_encoded,y)[0])
coef2 = np.round(get_lr_result(X_encoded,y)[1].coef_)
plt.figure(figsize=(18,4))
plt.plot(coef1,label='one-hot-encoding')
plt.plot(coef2,label='effect encoding')
plt.xticks(range(len(coef1)),X_encoded.columns,rotation=45)
plt.legend()
plt.show()
dfdatanew['rooms'] = dfdatanew['rooms'].astype('int')
dfdatanew['roomsnew'] = np.where(dfdatanew['rooms'] > 5,6,dfdatanew['rooms'])
fig,ax = plt.subplots(1,2,figsize=(18,5),sharey=True)
sns.countplot(x='rooms',data=dfdatanew,ax=ax[0])
sns.countplot(x='roomsnew',data=dfdatanew,ax=ax[1])
plt.show()
X_encoded = pd.get_dummies(dfdatanew['roomsnew'],drop_first=True)
print("with backoff bins -> Price  ",get_lr_result(X_encoded,y)[0])
X = dfdata['rooms']
X.value_counts()
X_encoded = pd.get_dummies(X, drop_first=True)
X_encoded.columns
from sklearn.feature_selection import VarianceThreshold
vt_filter = VarianceThreshold(threshold=0.01)
vt_filter.fit(X_encoded)
drop_list = [column for column in X_encoded.columns if column not in X_encoded.columns[vt_filter.get_support()]]
print(drop_list)
numeric_columns = dfdata.columns[dfdata.dtypes != 'object']
numeric_columns
corr_matrix = np.corrcoef(dfdata[numeric_columns],rowvar=False)
corr_matrix = dfdata[numeric_columns].corr()
sns.heatmap(corr_matrix,annot=True,fmt='.1g')
plt.show()
X = dfdata[['rooms','type','method','distance','bathroom','regionname']]
y = dfdata['price']
X_encoded = pd.get_dummies(X,drop_first=True)
X_encoded.columns
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=0)
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)

column_list = [column for column in X_encoded.columns if column not in X_encoded.columns[select.get_support()]]
print("X_train.shape: {}".format(X_train.shape))
print("X_train_selected.shape: {}".format(X_train_selected.shape))
print("Selected Features",column_list)
model = RandomForestRegressor(max_depth=10,random_state=0)
select = SelectFromModel(model,threshold="median")
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)

column_list = [column for column in X_encoded.columns if column not in X_encoded.columns[select.get_support()]]
print("X_train.shape: {}".format(X_train.shape))
print("X_train_selected.shape: {}".format(X_train_selected.shape))
print("Selected Features",column_list)
model = RandomForestRegressor(max_depth=10,random_state=0)
select = RFE(model,n_features_to_select=8)
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)

column_list = [column for column in X_encoded.columns if column not in X_encoded.columns[select.get_support()]]
print("X_train.shape: {}".format(X_train.shape))
print("X_train_selected.shape: {}".format(X_train_selected.shape))
print("Selected Features",column_list)
X = dfdata.select_dtypes(include=np.number).drop(['price'],axis=1)
X_scaled = pd.DataFrame(StandardScaler().fit_transform(X),columns = X.columns)
X_scaled.head()
pca = PCA(n_components = X_scaled.shape[1]).fit(X_scaled)
cumratio = np.round(100 * pca.explained_variance_ratio_.cumsum(),1)
sns.pointplot(x=X_scaled.columns.values,y=cumratio)
plt.xticks(rotation=45)
plt.show()
pca = PCA(n_components = 4)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)
sns.heatmap(pca.components_,cmap='viridis',annot=True,fmt='.1g')
plt.xticks(range(len(X.columns)),X.columns,rotation=45)
plt.show()
print("All Numeric -> Price          ",get_lr_result(X_scaled,y)[0])
print("PCA -> Price                  ",get_lr_result(X_pca,y)[0])
tsne_2d = TSNE(n_components=2, perplexity=20)
tsne_data = tsne_2d.fit_transform(X_scaled[['rooms']])
sns.scatterplot(tsne_data[:,0],tsne_data[:,1])
plt.show()
umap_data = umap.UMAP(n_neighbors=20).fit_transform(X_scaled[['rooms']])
sns.scatterplot(umap_data[:,0],umap_data[:,1])
plt.show()
