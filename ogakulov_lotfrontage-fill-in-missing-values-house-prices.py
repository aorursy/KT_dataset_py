# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Special settings for Python notebook
%matplotlib inline

# Ignore FutureWarnings related to internal pandas and np code
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Read in the train and test datasets
raw_train = pd.read_csv('../input/train.csv')
raw_test = pd.read_csv('../input/test.csv')

# Drop SalePrice column from train dataset and merge into one data frame called all_data
raw_train = raw_train.drop('SalePrice', axis=1)
all_data = pd.concat([raw_train, raw_test], ignore_index=True).copy()
# Split into known and unknown LotFrontage records
test  = all_data[all_data.LotFrontage.isnull()]
train = all_data[~all_data.LotFrontage.isnull()]
target = train.LotFrontage
print("LotFrontage has {:} missing value, and {:} values avaialble.".format(test.shape[0], train.shape[0]))
fig, ax =plt.subplots(1,2, figsize=(16,4))
sns.distplot(target, ax=ax[0])
sns.boxplot(target, ax=ax[1]);
def idOutliers (dat):
    tile25 = dat.describe()[4]
    tile75 = dat.describe()[6]
    iqr = tile75 - tile25 
    out = (dat > tile75+1.5*iqr) | (dat < tile25-1.5*iqr)
    return out
# LotArea vs. LotFrontage
fig, ax =plt.subplots(1,2, figsize=(16,4))
ax[0].set_title('With LotArea outliers')
ax[1].set_title('Without LotArea outliers')
sns.regplot(train.LotArea.apply(np.sqrt), target, ax=ax[0])
ax[0].set(xlabel='sqrt(LotArea)')
sns.regplot(train.LotArea[~idOutliers(train.LotArea)].apply(np.sqrt), target[~idOutliers(train.LotArea)], ax=ax[1])
ax[1].set(xlabel='sqrt(LotArea)');
train_plot = train[~idOutliers(train.LotArea)].copy()
train_plot['sqrt_LotArea'] = train_plot['LotArea'].apply(np.sqrt)
sns.lmplot(x='sqrt_LotArea', y='LotFrontage', hue='BldgType', aspect=2, fit_reg=False, data=train_plot);
# Neighborhood vs. LotFrontage
plt.figure(figsize=(20,5))
sns.boxplot(x=train_plot['Neighborhood'], y=train_plot['LotFrontage'], width=0.7, linewidth=0.8);
# Alley vs. LotFrontage
train_plot['Alley'] = train_plot['Alley'].fillna('No Alley')
sns.boxplot(x=train_plot['Alley'], y=train_plot['LotFrontage'], linewidth=0.8);
# GarageCars vs. LotFrontage
sns.boxplot(x=train_plot['GarageCars'], y=train_plot['LotFrontage'], linewidth=0.8);
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
# Pull only the features for training the model. Define target variable
y_lotFrontage = train['LotFrontage']
X_train = train.loc[:,['LotArea', 'LotConfig', 'LotShape', 'Alley', 'MSZoning', 'BldgType', 'Neighborhood', 'Condition1', 'Condition2', 'GarageCars']]

# Dummify categorical variables and normalize the data
X_train = pd.get_dummies(X_train)
X_train = (X_train - X_train.mean())/X_train.std()
X_train = X_train.fillna(0)
# Classifier with tuned parameteres
clf = svm.SVR(kernel='rbf', C=100, gamma=0.001)

# Set initial scores
acc = 0
acc1 = 0
acc2 = 0

# Defien k-fold object for 10-fold validation
kf = KFold(n_splits=10, shuffle=True, random_state=3) 

# Main evaluator loop over the 10 folds
for trn, tst in kf.split(train):
    
    # Compute benchmark score prediction based on mean neighbourhood LotFrontage
    fold_train_samples = train.iloc[trn]
    fold_test_samples = train.iloc[tst]
    neigh_means = fold_train_samples.groupby('Neighborhood')['LotFrontage'].mean()
    all_mean = fold_train_samples['LotFrontage'].mean()
    y_pred_neigh_means = fold_test_samples.join(neigh_means, on = 'Neighborhood', lsuffix='benchmark')['LotFrontage']
    y_pred_all_mean = [all_mean] * fold_test_samples.shape[0]
    
    # Compute benchmark score prediction based on overall mean LotFrontage
    u1 = ((fold_test_samples['LotFrontage'] - y_pred_neigh_means) ** 2).sum()
    u2 = ((fold_test_samples['LotFrontage'] - y_pred_all_mean) ** 2).sum()
    v = ((fold_test_samples['LotFrontage'] - fold_test_samples['LotFrontage'].mean()) ** 2).sum()
    
    # Perform model fitting 
    clf.fit(X_train.iloc[trn], y_lotFrontage.iloc[trn])
    
    # Record all scores for averaging
    acc = acc + mean_absolute_error(fold_test_samples['LotFrontage'], clf.predict(X_train.iloc[tst]))
    acc1= acc1 + mean_absolute_error(fold_test_samples['LotFrontage'], y_pred_neigh_means)
    acc2 = acc2 + mean_absolute_error(fold_test_samples['LotFrontage'], y_pred_all_mean)

    
print('10-Fold Validation Mean Absolute Error results:')
print('\tSVR: {:.3}'.format(acc/10))
print('\tSingle mean: {:.3}'.format(acc2/10))
print('\tNeighbourhood mean: {:.3}'.format(acc1/10))
# Select columns for final prediction, dummify, and normalize
X_test = test.loc[:,['LotArea', 'LotConfig', 'LotShape', 'Alley', 'MSZoning', 'BldgType', 'Neighborhood', 'Condition1', 'Condition2', 'GarageCars']]
X_test = pd.get_dummies(X_test)
X_test = (X_test - X_test.mean())/X_test.std()
X_test = X_test.fillna(0)
# Make sure that dummy columns from training set are replicated in test set
for col in (set(X_train.columns) - set(X_test.columns)):
    X_test[col] = 0

X_test = X_test[X_train.columns]

# Assign predicted LotFrontage value in all_data 
all_data.loc[all_data.LotFrontage.isnull(), 'LotFrontage'] = clf.predict(X_test)

# Output to file
all_data.to_csv('housing_data_with_imputed_LotFrontage.csv')
# Appendix: Model Tuning
# Gridsearch for best model
#from sklearn.model_selection import GridSearchCV
#Cs = [0.1, 1, 10, 25, 100, 1000]
#gammas = [0.001, 0.01, 0.1]
#parameters = {'kernel':('linear', 'rbf'), 'C':Cs, 'gamma':gammas}
#clf_gd = GridSearchCV(svm.SVR(), parameters, cv=3, verbose=1)
#clf_gd.fit(X_train, y_lotFrontage)