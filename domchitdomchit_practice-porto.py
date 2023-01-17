import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold

from sklearn.feature_selection import SelectFromModel

from sklearn.utils import shuffle

from sklearn.ensemble import RandomForestClassifier
df_train = pd.read_csv('../input/porto-seguro-safe-driver-prediction/train.csv')

df_test = pd.read_csv('../input/porto-seguro-safe-driver-prediction/test.csv')
target_count = df_train.target.value_counts()

print('Class 0:', target_count[0])

print('Class 1:', target_count[1])

print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')



target_count.plot(kind='bar', title='Count (target)')



df_test.shape
df_train.isnull().any().any()
df_train_copy = df_train.replace(-1, np.NaN)
import missingno as msno

# Nullity or missing values by columns

msno.matrix(df=df_train_copy.iloc[:,2:39], figsize=(20, 14), color=(0.42, 0.1, 0.05))
sns.set(style="white")





# Compute the correlation matrix

corr = df_train.corr()





# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
data = []

for i in df_train.columns:

    # Defining the role

    if i == 'target':

        role = 'target'

    elif i == 'id':

        role = 'id'

    else:

        role = 'input'

         

    # Defining the level

    if 'bin' in i or i == 'target':

        level = 'binary'

    elif 'cat' in i or i == 'id':

        level = 'nominal'

    elif df_train[i].dtype == float:

        level = 'interval'

    elif df_train[i].dtype == int:

        level = 'ordinal'

        

    # Initialize keep to True for all variables except for id

    keep = True

    if i == 'id':

        keep = False

    

    # Defining the data type 

    dtype = df_train[i].dtype

    

    # Creating a Dict that contains all the metadata for the variable

    i_dict = {

        'varname': i,

        'role': role,

        'level': level,

        'keep': keep,

        'dtype': dtype

    }

    data.append(i_dict)

    

meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])

meta.set_index('varname', inplace=True)
pd.DataFrame({'count' : meta.groupby(['role', 'level'])['role'].size()}).reset_index()
v = meta[(meta.level == 'interval') & (meta.keep)].index

df_train[v].describe()
v = meta[(meta.level == 'ordinal') & (meta.keep)].index

df_train[v].describe()
v = meta[(meta.level == 'binary') & (meta.keep)].index

df_train[v].describe()
v = meta[(meta.level == 'nominal') & (meta.keep)].index

df_train[v].describe()
desired_apriori=0.10

from sklearn.utils import shuffle

# Get the indices per target value

idx_0 = df_train[df_train.target == 0].index

idx_1 = df_train[df_train.target == 1].index



# Get original number of records per target value

nb_0 = len(df_train.loc[idx_0])

nb_1 = len(df_train.loc[idx_1])



# Calculate the undersampling rate and resulting number of records with target=0

undersampling_rate = ((1-desired_apriori)*nb_1)/(nb_0*desired_apriori)

undersampled_nb_0 = int(undersampling_rate*nb_0)

print('Rate to undersample records with target=0: {}'.format(undersampling_rate))

print('Number of records with target=0 after undersampling: {}'.format(undersampled_nb_0))



# Randomly select records with target=0 to get at the desired a priori

undersampled_idx = shuffle(idx_0, random_state=37, n_samples=undersampled_nb_0)



# Construct list with remaining indices

idx_list = list(undersampled_idx) + list(idx_1)



# Return undersample data frame

train = df_train.loc[idx_list].reset_index(drop=True)
vars_with_missing = []



for f in train.columns:

    missings = train[train[f] == -1][f].count()

    if missings > 0:

        vars_with_missing.append(f)

        missings_perc = missings/train.shape[0]

        

        print('Variable {} has {} records ({:.2%}) with missing values'.format(f, missings, missings_perc))

        

print('In total, there are {} variables with missing values'.format(len(vars_with_missing)))
# Imputing with the mean or mode

mean_imp = Imputer(missing_values=-1, strategy='mean', axis=0)

mode_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)

train['ps_reg_03'] = mean_imp.fit_transform(train[['ps_reg_03']]).ravel()

train['ps_car_14'] = mean_imp.fit_transform(train[['ps_car_14']]).ravel()

train['ps_car_11'] = mode_imp.fit_transform(train[['ps_car_11']]).ravel()
v = meta[(meta.level == 'nominal') & (meta.keep)].index



for f in v:

    plt.figure()

    fig, ax = plt.subplots(figsize=(20,10))

    # Calculate the percentage of target=1 per category value

    cat_perc = train[[f, 'target']].groupby([f],as_index=False).mean()

    cat_perc.sort_values(by='target', ascending=False, inplace=True)

    # Bar plot

    # Order the bars descending on target mean

    sns.barplot(ax=ax, x=f, y='target', data=cat_perc, order=cat_perc[f])

    plt.ylabel('% target', fontsize=18)

    plt.xlabel(f, fontsize=18)

    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.show();
selector = VarianceThreshold(threshold=.01)

selector.fit(train.drop(['id', 'target'], axis=1)) # Fit to train without id and target variables



f = np.vectorize(lambda x : not x) # Function to toggle boolean array elements



v = train.drop(['id', 'target'], axis=1).columns[f(selector.get_support())]

print('{} variables have too low variance.'.format(len(v)))

print('These variables are {}'.format(list(v)))
train_drop2 =['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_car_10_cat', 'ps_car_12', 'ps_car_14']

train.drop(train_drop2, inplace=True, axis=1)
test_drop=['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_car_10_cat', 'ps_car_12', 'ps_car_14']

df_test.drop(test_drop, inplace = True, axis = 1)
unwanted = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(unwanted, axis=1)

test = df_test.drop(unwanted, axis =1)
def gini(actual, pred, cmpcol = 0, sortcol = 1):

    assert( len(actual) == len(pred) )

    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)

    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]

    totalLosses = all[:,0].sum()

    giniSum = all[:,0].cumsum().sum() / totalLosses

    

    giniSum -= (len(actual) + 1) / 2.

    return giniSum / len(actual)

 

def gini_normalized(a, p):

    return gini(a, p) / gini(a, a)



def gini_xgb(preds, dtrain):

    labels = dtrain.get_label()

    gini_score = gini_normalized(labels, preds)

    return 'gini', gini_score
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold

kfold = 5

skf = StratifiedKFold(n_splits=kfold, random_state=0)
from sklearn.datasets import load_iris 

from sklearn.model_selection import train_test_split 

from sklearn.metrics import accuracy_score 

from sklearn.ensemble import ExtraTreesClassifier 

from sklearn.ensemble import RandomForestClassifier 

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression #logistic regression

from sklearn import svm #support vector Machine

from sklearn import metrics #accuracy measure
params = {

    'min_child_weight': 10.0,

    'objective': 'binary:logistic',

    'max_depth': 7,

    'max_delta_step': 1.8,

    'colsample_bytree': 0.4,

    'subsample': 0.8,

    'eta': 0.025,

    'gamma': 0.65,

    'num_boost_round' : 700

    }
X = train.drop(['id', 'target'], axis=1).values

y = train.target.values

test_id = test.id.values

test = test.drop('id', axis=1)
sub = pd.DataFrame()

sub['id'] = test_id

sub['target'] = np.zeros_like(test_id)
import xgboost as xgb

for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    print('[Fold %d/%d]' % (i + 1, kfold))

    X_train, X_valid = X[train_index], X[test_index]

    y_train, y_valid = y[train_index], y[test_index]

    # Convert our data into XGBoost format

    d_train = xgb.DMatrix(X_train, y_train)

    d_valid = xgb.DMatrix(X_valid, y_valid)

    d_test = xgb.DMatrix(test.values)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]



    # Train the model! We pass in a max of 1,600 rounds (with early stopping after 70)

    # and the custom metric (maximize=True tells xgb that higher metric is better)

    mdl = xgb.train(params, d_train, 1600, watchlist, early_stopping_rounds=70, feval=gini_xgb, maximize=True, verbose_eval=100)



    print('[Fold %d/%d Prediciton:]' % (i + 1, kfold))

    # Predict on our test data

    p_test = mdl.predict(d_test, ntree_limit=mdl.best_ntree_limit)

    sub['target'] += p_test/kfold
sub.to_csv('StratifiedKFold.csv', index=False)
#from sklearn.ensemble import VotingClassifier

#ensemble_lin_rbf=VotingClassifier(estimators=[('RBF',svm.SVC(probability=True,kernel='rbf',C=0.5,gamma=0.1)),

#                                             ('RFor',RandomForestClassifier(n_estimators=100,random_state=0)),

 #                                             ('LR',LogisticRegression(C=0.05)),

  #                                            ('ET',ExtraTreesClassifier(random_state=0)),

   #                                           ('svm',svm.SVC(kernel='linear',probability=True))

    #                                         ], 

     #                  voting='soft')

#ensemble_lin_rbf.fit(train.drop(['id', 'target'],axis=1),train.target)

#print('The accuracy for ensembled model is:',ensemble_lin_rbf.score(test_X,test_Y.values.ravel()))

#cross=cross_val_score(ensemble_lin_rbf,X,Y, cv = 10,scoring = "accuracy")

#print('The cross validated score is',cross.mean())