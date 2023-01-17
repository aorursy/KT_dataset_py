import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")
%matplotlib inline
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import skew
from scipy import stats
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_regression, f_regression
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
tx = pd.read_csv('/kaggle/input/house-prices-data/train.csv')
year_all = ['YearBuilt', 'YearRemodAdd','YrSold','MoSold','GarageYrBlt']
for i in tx:
    if tx[i].dtypes == object or i in year_all:
        tx[i] = tx[i].fillna(tx[i].mode()[0])
    else:
        tx[i] = tx[i].fillna(tx[i].mean())
tx = tx.select_dtypes(exclude=object).copy()

features = StandardScaler().fit_transform(tx)
# Create a PCA that will retain 90% of variance
pca = PCA(n_components=0.90, whiten=True)
# Conduct PCA
train = pca.fit_transform(features)
# Show results
print("Original number of features:", features.shape[1])
print("Reduced number of features:", train.shape[1])
train
# feature selection using SelectKBest
def select_features(X, Y, func):
  bestfeatures = SelectKBest(score_func=func, k='all')
  fit = bestfeatures.fit(X,Y)
  return fit,bestfeatures
fit,fs = select_features(tx, tx['SalePrice'], mutual_info_regression)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(tx.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score'] 

mutual_info = featureScores.nlargest(32,'Score')
mutual_info = list(mutual_info['Specs'])
print(len(mutual_info),'\n',mutual_info)
print(featureScores.nlargest(32,'Score'))
#Using Heatmap to see features importance and correlation with output

f,ax = plt.subplots(figsize=(50, 50))
corrmat = tx.corr()
k =10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(tx[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=False, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#Using GradientBoostingRegressor to see features importance
tx1 = tx.copy()
tx1.drop(['SalePrice'],axis=1,inplace=True)
ty1 = np.log1p(tx['SalePrice'])
tx1 = np.log1p(tx1)

X_train, X_test, y_train, y_test = train_test_split(tx1, ty1, test_size=0.2, random_state=13)

params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'ls'}
reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)

mse = mean_squared_error(y_test, reg.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
for i, y_pred in enumerate(reg.staged_predict(X_test)):
    test_score[i] = reg.loss_(y_test, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, reg.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
fig.tight_layout()
plt.show()
feature_importance = reg.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
#print(pos)
#print(np.array(tx.columns)[sorted_idx])
fig = plt.figure(figsize=(40, 30))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(tx.columns)[sorted_idx])
aa = (pos, np.array(tx.columns)[sorted_idx])
plt.title('Feature Importance (MDI)')
'''
result = permutation_importance(reg, X_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=2)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(result.importances[sorted_idx].T,vert=False, labels=np.array(tx.columns)[sorted_idx])
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.show() '''

l = 0
log_data = np.log1p(tx)
sqrt_data = np.sqrt(tx)
box_data = tx.copy()
for i in box_data:
  box_data[i],lam = stats.boxcox(box_data[i]+1)
f, axes = plt.subplots(12, 3, figsize=(50, 100), sharex=True)
c = 0
for i in range(12):
  for j in range(3):
    sns.kdeplot(log_data.iloc[:,c], color="red", cumulative=True, bw=1.5, ax=axes[i,j])
    c+=1
for i, ax in enumerate(axes.reshape(-1)):
    ax.text(x=0.97, y=0.97, transform=ax.transAxes, s="Skewness: %f" % log_data.iloc[:,i].skew(),\
        fontweight='demibold', fontsize=20, verticalalignment='top', horizontalalignment='right',\
        backgroundcolor='white', color='xkcd:poo brown')
    ax.text(x=0.97, y=0.91, transform=ax.transAxes, s="Kurtosis: %f" % log_data.iloc[:,i].kurt(),\
        fontweight='demibold', fontsize=20, verticalalignment='top', horizontalalignment='right',\
        backgroundcolor='white', color='xkcd:dried blood')
plt.tight_layout()
f, axes = plt.subplots(12, 3, figsize=(50, 100), sharex=True)
c = 0
for i in range(12):
  for j in range(3):
    sns.kdeplot(sqrt_data.iloc[:,c], color="red", cumulative=True, bw=1.5, ax=axes[i,j])
    c+=1
for i, ax in enumerate(axes.reshape(-1)):
    ax.text(x=0.97, y=0.97, transform=ax.transAxes, s="Skewness: %f" % log_data.iloc[:,i].skew(),\
        fontweight='demibold', fontsize=20, verticalalignment='top', horizontalalignment='right',\
        backgroundcolor='white', color='xkcd:poo brown')
    ax.text(x=0.97, y=0.91, transform=ax.transAxes, s="Kurtosis: %f" % log_data.iloc[:,i].kurt(),\
        fontweight='demibold', fontsize=20, verticalalignment='top', horizontalalignment='right',\
        backgroundcolor='white', color='xkcd:dried blood')
plt.tight_layout()
f, axes = plt.subplots(12, 3, figsize=(50, 100), sharex=True)
c = 0
for i in range(12):
  for j in range(3):
    sns.kdeplot(box_data.iloc[:,c], color="red", cumulative=True, bw=1.5, ax=axes[i,j])
    c+=1
for i, ax in enumerate(axes.reshape(-1)):
    ax.text(x=0.97, y=0.97, transform=ax.transAxes, s="Skewness: %f" % log_data.iloc[:,i].skew(),\
        fontweight='demibold', fontsize=20, verticalalignment='top', horizontalalignment='right',\
        backgroundcolor='white', color='xkcd:poo brown')
    ax.text(x=0.97, y=0.91, transform=ax.transAxes, s="Kurtosis: %f" % log_data.iloc[:,i].kurt(),\
        fontweight='demibold', fontsize=20, verticalalignment='top', horizontalalignment='right',\
        backgroundcolor='white', color='xkcd:dried blood')
plt.tight_layout()