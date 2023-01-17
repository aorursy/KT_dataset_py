import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
import gc
PATH="../input"

application_train = pd.read_csv(PATH+"/application_train.csv")
application_test = pd.read_csv(PATH+"/application_test.csv")
bureau = pd.read_csv(PATH+"/bureau.csv")
bureau_balance = pd.read_csv(PATH+"/bureau_balance.csv")
credit_card_balance = pd.read_csv(PATH+"/credit_card_balance.csv")
installments_payments = pd.read_csv(PATH+"/installments_payments.csv")
previous_application = pd.read_csv(PATH+"/previous_application.csv")
POS_CASH_balance = pd.read_csv(PATH+"/POS_CASH_balance.csv")
application_train.head()
application_test.head()
bureau.head()
bureau_balance.head()
credit_card_balance.head()
installments_payments.head()
previous_application.head()
def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data(application_train).head(10)
missing_data(application_test).head(10)
missing_data(bureau)
missing_data(bureau_balance)
missing_data(credit_card_balance)
missing_data(installments_payments)
import seaborn as sns
def plot_categorical(data, col, size=[8 ,4], xlabel_angle=0, title=''):
    '''use this for ploting the count of categorical features'''
    plotdata = data[col].value_counts()
    plt.figure(figsize = size)
    sns.barplot(x = plotdata.index, y=plotdata.values)
    plt.title(title)
    if xlabel_angle!=0: 
        plt.xticks(rotation=xlabel_angle)
    plt.show()
plot_categorical(data=application_train, col='TARGET', size=[8 ,4], xlabel_angle=0, title='train set: label')
plot_categorical(data=application_train, col='OCCUPATION_TYPE', size=[12 ,4], xlabel_angle=30, title='Occupation Type')
plot_categorical(data=application_train, col='NAME_INCOME_TYPE', size=[12 ,4], xlabel_angle=0, title='Income Type')
plot_categorical(data=application_train, col='NAME_HOUSING_TYPE', size=[12 ,4], xlabel_angle=0, title='House Type')
def plot_numerical(data, col, size=[8, 4], bins=50):
    '''use this for ploting the distribution of numercial features'''
    plt.figure(figsize=size)
    plt.title("Distribution of %s" % col)
    sns.distplot(data[col].dropna(), kde=True,bins=bins)
    plt.show()
plot_numerical(application_train, 'AMT_CREDIT')
plot_numerical(application_train, 'AMT_ANNUITY')
plot_numerical(application_train, 'DAYS_EMPLOYED')
def plot_categorical_bylabel(data, col, size=[12 ,6], xlabel_angle=0, title=''):
    '''use it to compare the distribution between label 1 and label 0'''
    plt.figure(figsize = size)
    l1 = data.loc[data.TARGET==1, col].value_counts()
    l0 = data.loc[data.TARGET==0, col].value_counts()
    plt.subplot(1,2,1)
    sns.barplot(x = l1.index, y=l1.values)
    plt.title('Default: '+title)
    plt.xticks(rotation=xlabel_angle)
    plt.subplot(1,2,2)
    sns.barplot(x = l0.index, y=l0.values)
    plt.title('Non-default: '+title)
    plt.xticks(rotation=xlabel_angle)
    plt.show()
plot_categorical_bylabel(application_train, 'CODE_GENDER', title='Gender')
plot_categorical_bylabel(application_train, 'NAME_EDUCATION_TYPE', size=[15 ,6], xlabel_angle=15, title='Education Type')
def plot_numerical_bylabel(data, col, size=[8, 4], bins=50):
    '''use this to compare the distribution of numercial features'''
    plt.figure(figsize=[12, 6])
    l1 = data.loc[data.TARGET==1, col]
    l0 = data.loc[data.TARGET==0, col]
    plt.subplot(1,2,1)
    sns.distplot(l1.dropna(), kde=True,bins=bins)
    plt.title('Default: Distribution of %s' % col)
    plt.subplot(1,2,2)
    sns.distplot(l0.dropna(), kde=True,bins=bins)
    plt.title('Non-default: Distribution of %s' % col)
    plt.show()
plot_numerical_bylabel(application_train, 'EXT_SOURCE_1', bins=50)
plot_numerical_bylabel(application_train, 'EXT_SOURCE_2', bins=50)
plot_numerical_bylabel(application_train, 'EXT_SOURCE_3', bins=50)
corr_mat = application_train.corr()
plt.figure(figsize=[15, 15])
sns.heatmap(corr_mat.values, annot=False)
plt.show()
application_train = pd.read_csv('../input/application_train.csv')
application_test= pd.read_csv('../input/application_test.csv')
# get positive sample
n_pos = application_train[application_train.TARGET==1].shape[0]
pos_data = application_train[application_train.TARGET==1]
# get negative sample, and select a subset
n_neg = application_train[application_train.TARGET==0].shape[0]
neg_data = application_train[application_train.TARGET==0].iloc[np.random.randint(1, n_neg, n_pos), :]
# combine them
application_train = pd.concat([pos_data, neg_data], axis=0)
del pos_data, neg_data
gc.collect()
application_train.shape
# use this if you want to convert categorical features to dummies(default)
def cat_to_dummy(train, test):
    train_d = pd.get_dummies(train, drop_first=True)
    test_d = pd.get_dummies(test, drop_first=True)
    # make sure that the number of features in train and test should be same
    for i in train_d.columns:
        if i not in test_d.columns:
            if i!='TARGET':
                train_d = train_d.drop(i, axis=1)
    for j in test_d.columns:
        if j not in train_d.columns:
            if j!='TARGET':
                test_d = test_d.drop(i, axis=1)
    return train_d, test_d
application_train, application_test = cat_to_dummy(application_train, application_test)
# use this if you want to convert categorical features to numerical ones
def cat_to_num(data):
    '''convert categorical features to numerical features'''
     #find categorical feature list
    cate_feature = [f for f in data.columns if data[f].dtype == 'object']
     #factorize all categorical features
    for feature in cate_feature:data[feature], b = pd.factorize(data[feature])
    return data

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
X = application_train.drop(['SK_ID_CURR', 'TARGET'], axis=1)
y = application_train.TARGET
del application_train
gc.collect()
X, y = shuffle(X, y)
lgbc = LGBMClassifier(n_estimators=720, learning_rate=0.01, num_leaves=6,colsample_bytree=.3,subsample=.8, min_split_gain=.01,
                     silent=-1, verbose=-1)
lgbc.fit(X, y)
feature_imp=pd.DataFrame({'feature name':X.columns,'feature importance':lgbc.feature_importances_}).sort_values('feature importance', ascending=False).iloc[:, [1,0]]
feature_imp.head()
n_show=20
plt.figure(figsize = [10, n_show/3])
ax = sns.barplot(x = 'feature importance', y='feature name', data=feature_imp.iloc[:n_show, :], label='Feature Importance')
ax.set_xlabel('feature name')
plt.show()
X_test = application_test.drop(['SK_ID_CURR'], axis=1)
y_pred = lgbc.predict_proba(X_test)[:, 1]
output = pd.DataFrame({'SK_ID_CURR':application_test.SK_ID_CURR, 'TARGET': y_pred})
output.to_csv('application_train.csv', index=False)