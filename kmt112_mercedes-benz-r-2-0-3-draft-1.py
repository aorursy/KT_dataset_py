import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import norm,skew
from matplotlib.pyplot import figure
from sklearn.linear_model import LinearRegression
train_df=pd.read_csv('../input/mercbenz/train.csv')
train_df.head()
test_df=pd.read_csv('../input/mercbenz/test.csv')
test_df.head()
train_df.drop("ID",axis=1,inplace =True)
test_df.drop("ID",axis=1, inplace=True)
train_df.describe()
test_df.describe()
plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.y.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.show()
train_df.loc[train_df['y'] > 250]
outliers = [883]
train_df= train_df.drop(train_df.index[outliers])
train_df.describe()
# replot the graph inorder to find out if there are any abnormlay left
plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.y.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.show()

train_df['y'].describe()
sns.distplot(train_df['y']);

#plot Skewness and Kurtosis
print("Skewness: %f" % train_df['y'].skew())
print("Kurtosis: %f" % train_df['y'].kurt())
fig =plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(train_df['y'], fit=norm);
(mu,sigma)= norm.fit(train_df['y'])
print('\n mu= {:.2f}\n'.format(mu,sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=${:.2f})'.format(mu,sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('y distribution')
plt.subplot(1,2,2)
res=stats.probplot(train_df['y'],plot=plt)
plt.suptitle('Before Transformation')

train_df.y=np.log1p(train_df.y)
y_train=train_df.y.values
y_train_orig=train_df.y

fig=plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(train_df['y'], fit=norm);
(mu,sigma)=norm.fit(train_df['y'])
print(' \n mu={:.2f} and sigma = {:.2f}\n'.format(mu,sigma))
plt.legend(['Normal dist. ($\mu=${:.2f} and $\sigma=$ {:.2f})'.format(mu,sigma)], loc='best')
plt.title('y Distribution')
plt.subplot(1,2,2)
res=stats.probplot(train_df['y'], plot=plt)
plt.suptitle('After Transformation')

#judging from the box cos tranformation we managed to bring down the spread slightly.
dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()
train_df.isnull().sum()
# there is no null values
test_df.isnull().sum()
unique_values_dict = {}
for col in train_df.columns:
    if col not in ["y", "X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]:
        unique_value = str(np.sort(train_df[col].unique()).tolist())
        tlist = unique_values_dict.get(unique_value, [])
        tlist.append(col)
        unique_values_dict[unique_value] = tlist[:]
for unique_val, columns in unique_values_dict.items():
    print("Columns containing the unique values : ",unique_val)
    print(columns)
    print("--------------------------------------------------")
train_df.X1.describe()
train_df = train_df.drop(columns=['X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X297', 'X330', 'X347'])
test_df = test_df.drop(columns=['X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X297', 'X330', 'X347'])
train_df.head()
test_df.head()
ranks = train_df.groupby("X0")["y"].median().sort_values(ascending=False)[::-1].index

var_name = "X0"
col_order = np.sort(train_df[var_name].unique()).tolist()
plt.figure(figsize=(12,6))
sns.boxplot(x=var_name, y='y', data=train_df, order=ranks)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()
ranks = train_df.groupby("X1")["y"].median().sort_values(ascending=False)[::-1].index

var_name = "X1"
col_order = np.sort(train_df[var_name].unique()).tolist()
plt.figure(figsize=(12,6))
sns.boxplot(x=var_name, y='y', data=train_df, order=ranks)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()
ranks = train_df.groupby("X2")["y"].median().sort_values(ascending=False)[::-1].index

var_name = "X2"
col_order = np.sort(train_df[var_name].unique()).tolist()
plt.figure(figsize=(12,6))
sns.boxplot(x=var_name, y='y', data=train_df, order=ranks)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()
ranks = train_df.groupby("X3")["y"].median().sort_values(ascending=False)[::-1].index

var_name = "X3"
col_order = np.sort(train_df[var_name].unique()).tolist()
plt.figure(figsize=(12,6))
sns.boxplot(x=var_name, y='y', data=train_df, order=ranks)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()
ranks = train_df.groupby("X4")["y"].median().sort_values(ascending=False)[::-1].index

var_name = "X4"
col_order = np.sort(train_df[var_name].unique()).tolist()
plt.figure(figsize=(12,6))
sns.boxplot(x=var_name, y='y', data=train_df, order=ranks)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()
ranks = train_df.groupby("X5")["y"].median().sort_values(ascending=False)[::-1].index

var_name = "X5"
col_order = np.sort(train_df[var_name].unique()).tolist()
plt.figure(figsize=(12,6))
sns.boxplot(x=var_name, y='y', data=train_df, order=ranks)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()
ranks = train_df.groupby("X6")["y"].median().sort_values(ascending=False)[::-1].index

var_name = "X6"
col_order = np.sort(train_df[var_name].unique()).tolist()
plt.figure(figsize=(12,6))
sns.boxplot(x=var_name, y='y', data=train_df, order=ranks)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()
ranks = train_df.groupby("X8")["y"].median().sort_values(ascending=False)[::-1].index
var_name = "X8"
col_order = np.sort(train_df[var_name].unique()).tolist()
plt.figure(figsize=(12,6))
sns.boxplot(x=var_name, y='y', data=train_df, order=ranks)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()
var_name = "X8"
col_order = np.sort(train_df[var_name].unique()).tolist()
plt.figure(figsize=(12,6))
ranks_1 = train_df.groupby("X0")["y"].median().sort_values(ascending=False)[::-1].index
ranks_2 = train_df.groupby("X2")["y"].median().sort_values(ascending=False)[::-1].index
plt.yticks(range(len(ranks_2)), ranks_2) # rearranging according to median y value
plt.xticks(range(len(ranks_1)),ranks_1)
sns.scatterplot(x='X0', y='X2', data=train_df, sizes=(40,400))
plt.xlabel('x0', fontsize=12)
plt.ylabel('x2', fontsize=12)
plt.title("Distribution of  variable with "+var_name, fontsize=15)
plt.show()
## insert weights for the nodes to tell how X0 and X2 look with overall build up time
column_list=['X10', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23', 'X24', 'X26', 'X27', 'X28', 'X29', 'X30', 'X31', 'X32', 'X33', 'X34', 'X35', 'X36', 'X37', 'X38', 'X39', 'X40', 'X41', 'X42', 'X43', 'X44', 'X45', 'X46', 'X47', 'X48', 'X49', 'X50', 'X51', 'X52', 'X53', 'X54', 'X55', 'X56', 'X57', 'X58', 'X59', 'X60', 'X61', 'X62', 'X63', 'X64', 'X65', 'X66', 'X67', 'X68', 'X69', 'X70', 'X71', 'X73', 'X74', 'X75', 'X76', 'X77', 'X78', 'X79', 'X80', 'X81', 'X82', 'X83', 'X84', 'X85', 'X86', 'X87', 'X88', 'X89', 'X90', 'X91', 'X92', 'X94', 'X95', 'X96', 'X97', 'X98', 'X99', 'X100', 'X101', 'X102', 'X103', 'X104', 'X105', 'X106', 'X108', 'X109', 'X110', 'X111', 'X112', 'X113', 'X114', 'X115', 'X116', 'X117', 'X118', 'X119', 'X120', 'X122', 'X123', 'X124', 'X125', 'X126', 'X127', 'X128', 'X129', 'X130', 'X131', 'X132', 'X133', 'X134', 'X135', 'X136', 'X137', 'X138', 'X139', 'X140', 'X141', 'X142', 'X143', 'X144', 'X145', 'X146', 'X147', 'X148', 'X150', 'X151', 'X152', 'X153', 'X154', 'X155', 'X156', 'X157', 'X158', 'X159', 'X160', 'X161', 'X162', 'X163', 'X164', 'X165', 'X166', 'X167', 'X168', 'X169', 'X170', 'X171', 'X172', 'X173', 'X174', 'X175', 'X176', 'X177', 'X178', 'X179', 'X180', 'X181', 'X182', 'X183', 'X184', 'X185', 'X186', 'X187', 'X189', 'X190', 'X191', 'X192', 'X194', 'X195', 'X196', 'X197', 'X198', 'X199', 'X200', 'X201', 'X202', 'X203', 'X204', 'X205', 'X206', 'X207', 'X208', 'X209', 'X210', 'X211', 'X212', 'X213', 'X214', 'X215', 'X216', 'X217', 'X218', 'X219', 'X220', 'X221', 'X222', 'X223', 'X224', 'X225', 'X226', 'X227', 'X228', 'X229', 'X230', 'X231', 'X232', 'X234', 'X236', 'X237', 'X238', 'X239', 'X240', 'X241', 'X242', 'X243', 'X244', 'X245', 'X246', 'X247', 'X248', 'X249', 'X250', 'X251', 'X252', 'X253', 'X254', 'X255', 'X256', 'X257', 'X258', 'X259', 'X260', 'X261', 'X262', 'X263', 'X264', 'X265', 'X266', 'X267', 'X269', 'X270', 'X271', 'X272', 'X273', 'X274', 'X275', 'X276', 'X277', 'X278', 'X279', 'X280', 'X281', 'X282', 'X283', 'X284', 'X285', 'X286', 'X287', 'X288', 'X291', 'X292', 'X294', 'X295', 'X296', 'X298', 'X299', 'X300', 'X301', 'X302', 'X304', 'X305', 'X306', 'X307', 'X308', 'X309', 'X310', 'X311', 'X312', 'X313', 'X314', 'X315', 'X316', 'X317', 'X318', 'X319', 'X320', 'X321', 'X322', 'X323', 'X324', 'X325', 'X326', 'X327', 'X328', 'X329', 'X331', 'X332', 'X333', 'X334', 'X335', 'X336', 'X337', 'X338', 'X339', 'X340', 'X341', 'X342', 'X343', 'X344', 'X345', 'X346', 'X348', 'X349', 'X350', 'X351', 'X352', 'X353', 'X354', 'X355', 'X356', 'X357', 'X358', 'X359', 'X360', 'X361', 'X362', 'X363', 'X364', 'X365', 'X366', 'X367', 'X368', 'X369', 'X370', 'X371', 'X372', 'X373', 'X374', 'X375', 'X376', 'X377', 'X378', 'X379', 'X380', 'X382', 'X383', 'X384', 'X385']
label_list=["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]
plot_df=[]
plot_df=train_df[column_list].sum(axis=1)
y=train_df.y
plot_df=pd.DataFrame(plot_df)
plot_df.insert(1,"y",train_df.y)
plot_df.columns=['Sumofflags','y'] #rename axis
total_bins=10
bins = np.linspace(plot_df.Sumofflags.min(),plot_df.Sumofflags.max(),total_bins)
delta=bins[1]-bins[0]
idx=np.digitize(plot_df.Sumofflags,bins)
running_median=[np.median(plot_df.y[idx==k]) for k in range(total_bins)]
sns.scatterplot(x='Sumofflags', y='y',data=plot_df)
plt.plot(bins-delta/2,running_median,'r--',lw=2.5,alpha=0.8)
plt.axis('tight')
plt.show()
print(plot_df.Sumofflags.describe())
train_df[column_list]
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
bin_df=train_df[column_list]
bin_df_2=test_df[column_list]
bin_df_std=StandardScaler().fit_transform(bin_df)
bin_df_std2=StandardScaler().fit_transform(bin_df_2)
cov_mat=np.cov(bin_df_std.T)
eigen_vals, eigen_vecs=np.linalg.eig(cov_mat)
import matplotlib.pyplot as plt

tot= sum(eigen_vals)
var_exp=[(i/tot) for i in sorted(eigen_vals,reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(0,356), var_exp, alpha=0.5,
        align='center', label='individual explained variance')
plt.step(range(0,356), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.show()
pca = PCA(n_components=155)
principalComponents = pca.fit_transform(bin_df_std)
principalDf= pd.DataFrame(data = principalComponents)
principalComponents_test = pca.fit_transform(bin_df_std2)
principalDf_2= pd.DataFrame(data=principalComponents_test)
principalDf_2.head()
train_df.head()
feature_list= train_df.columns[1:]

print("{} duplicate entries in training, out of {}, a {:.2f} %".format(
    len(train_df[train_df.duplicated(subset=feature_list, keep=False)]),
    len(train_df),
    100 * len(train_df[train_df.duplicated(subset=feature_list, keep=False)]) / len(train_df)
    ))
dup_df=train_df[train_df.duplicated(subset=feature_list, keep=False)].sort_values(by='X0')
dup_df
duplicate_std = train_df[train_df.duplicated(subset=feature_list,
                             keep=False)].groupby(list(feature_list.values))['y'].aggregate(['std', 'size']).reset_index(drop=False)

dup_df=duplicate_std.sort_values(by='std', ascending=False)
dup_df

tbd=dup_df.iloc[0:10]
tbd=tbd.drop(['std','size'], axis = 1)
tbd
plt.scatter(x="size",y="std", data=dup_df)
plt.ylabel('std deviation')
plt.xlabel('count')
def dataframe_difference(df1, df2, which='both'):
    """Find rows which are similar between two DataFrames."""
    comparison_df = df1.merge(df2,
                              indicator=True,
                              how='outer')
    if which is None:
        diff_df = comparison_df[comparison_df['_merge'] != 'both']
    else:
        diff_df = comparison_df[comparison_df['_merge'] == which]
    return diff_df
diff_df=dataframe_difference(train_df,tbd)
diff_df
dup = [635,636,693,694,1168,1169,1170,1617,1618,1619,2019,2020,2351,2352,2574,2575,3012,3013,3077,3078,3923,3924]
train_df = train_df.drop(columns=['X4'])
train_df=train_df.drop(columns=column_list)
test_df= test_df.drop(columns=['X4'])
test_df= test_df.drop(columns=column_list)
test_df.head()
train_df.head()
ID = [i for i in range(4208)]
ID2= [i for i in range(4209)]
principalDf.insert(0, 'ID', ID)
train_df.insert(0,'ID',ID)
principalDf_2.insert(0,'ID2',ID2,True)
test_df.insert(0,'ID2',ID2,True)
train_df_final=pd.merge(train_df, principalDf, how='outer', on=['ID'])
train_df_final

test_df_final=pd.merge(test_df, principalDf_2, how='outer', on=['ID2'])
test_df_final
from sklearn.model_selection import train_test_split
y=train_df_final.y.values
train_df_final=train_df_final.drop(columns='y')
final_features=pd.get_dummies(train_df_final)
print(final_features.shape)
X= final_features.iloc[:len(y),:]
X_test=pd.get_dummies(test_df_final)
print(X.shape, X_test.shape,y.shape)
overfit = []

for i in X.columns:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros/len(X) * 100 >99.95:
        overfit.append(i)


X=X.drop(overfit,axis=1).copy()
print(X.shape)
overfit = []

for i in X_test.columns:
    counts = X_test[i].value_counts()
    zeros = counts.iloc[0]
    if zeros/len(X) * 100 >99.95:
        overfit.append(i)

X_test=X_test.drop(overfit,axis=1).copy()
print(X_test.shape)
outlier=set(X_test.columns).symmetric_difference(set(X.columns))
X= X.drop(columns=['X2_an', 'X2_ah', 'X1_q', 'X2_av', 'ID', 'X1_d', 'X2_y', 'X2_at', 'X2_x'])

X_test=X_test.drop(columns=['ID2', 'X2_ab', 'X2_ad', 'X2_af', 'X5_g'])

from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error , make_scorer
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import LinearRegression

from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor
k_folds= KFold(n_splits=18, shuffle=True, random_state=42)

#model scoring and validation function

def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model,X,y,scoring="neg_mean_squared_error",cv=k_folds))
    return (rmse)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
## lightGBM

lightgbm=LGBMRegressor(objective='regression', num_leaves=4, learning_rate=0.01, n_estimators=9000, max_bin=200, bagging_fraction=0.75, bagging_freq=5,bagging_seed=7,feature_fraction=0.2, feature_fraction_seed=7, verbose=-1)

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt,cv=k_folds))
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7,alphas=alphas2, random_state=42, cv=k_folds))
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, 
                                                        alphas=e_alphas, cv=k_folds,l1_ratio=e_l1ratio))
stack_gen = StackingCVRegressor(regressors=(ridge,lasso,elasticnet,lightgbm), meta_regressor=elasticnet,use_features_in_secondary=True)

svr= make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003))

# store models, scores and prediction values 
models = {'Ridge': ridge,
          'Lasso': lasso, 
          'ElasticNet': elasticnet,
          'lightgbm': lightgbm,
          'Svd': svr}
predictions = {}
scores = {}
for name, model in models.items():
    
    model.fit(X,y)
    predictions[name]= np.expm1(model.predict(X))
    
    score= cv_rmse(model,X=X)
    scores[name] = (score.mean(), score.std())

print('---- Score with CV_RMSLE-----')
score = cv_rmse(ridge)
print("Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(lasso)
print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(elasticnet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(lightgbm)
print("lightgbm score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(svr)
print("SVR: {:.4f} ({:.4f})".format(score.mean(), score.std()))
print('----START Fit----',datetime.now())
print('Elasticnet')
elastic_model = elasticnet.fit(X, y)
print('Lasso')
lasso_model = lasso.fit(X, y)
print('Ridge')
ridge_model = ridge.fit(X, y)
print('lightgbm')
lgb_model_full_data = lightgbm.fit(X, y)
print('Svr')
svr_model_full_data = svr.fit(X, y)
print('Stack_gen_model')
stack_gen_model=stack_gen.fit(np.array(X), np.array(y))
def blend_models_predict(X):
    return ((0.16  * elastic_model.predict(X)) + \
            (0.16 * lasso_model.predict(X)) + \
            (0.11 * ridge_model.predict(X)) + \
            (0.2 * lgb_model_full_data.predict(X)) + \
            (0.1 * svr_model_full_data.predict(X)) + \
            (0.27 * stack_gen_model.predict(np.array(X))))
print('RMSLE score on the train data: ')
print(rmsle(y, blend_models_predict(X)))
print('Predict Submission')
submission = pd.read_csv('../input/mercbenz/train.csv')
submission.iloc[:,1] = (np.expm1(blend_models_predict(X_test)))
X.head()
X_test.head()
submission.to_csv("submission.csv", index=False)


