import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import os
import warnings
warnings.simplefilter('ignore')

from IPython.display import display, HTML

from imputer import Imputer
import lightgbm as lgb
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
print(os.listdir("../input"))
data_train = pd.read_csv('../input/application_train.csv')
#print( list(data_train.columns) ) 
data_train.info()
index = 0
for i in data_train.drop(columns='SK_ID_CURR').columns:
    if np.dtype(data_train[i]) == 'float64' and len(data_train[i].dropna())/data_train.shape[0] > 0.6:
        #if sum(data_train[i].dropna() == data_train[i].dropna().astype('int64')):
            index +=1
            plt.subplot(4, 5, index)
            #sns.boxplot(x = data_train.TARGET, y = data_train[i].dropna()[abs(stats.zscore(data_train[i].dropna())) < 3])
            curr_feature_1 = data_train[i][data_train.TARGET == 1].dropna()
            curr_feature_0 = data_train[i][data_train.TARGET == 0].dropna()
            sns.distplot(curr_feature_1[abs(stats.zscore(curr_feature_1)) < 3]) #to reduce most of outliers
            sns.distplot(curr_feature_0[abs(stats.zscore(curr_feature_0)) < 3])

plt.subplots_adjust(top=3, bottom=0, left=0, right=3, hspace=0.25, wspace=0.55)
plt.show()
index = 0
for i in data_train.drop(columns=['SK_ID_CURR', 'TARGET']).columns:
    if np.dtype(data_train[i]) == 'int64' and len(data_train[i].dropna())/data_train.shape[0] > 0.6 and len(data_train[i].unique()) > 50:
        index +=1
        plt.subplot(1, 3, index)
        curr_feature_1 = data_train[i][data_train.TARGET == 1].dropna()
        curr_feature_0 = data_train[i][data_train.TARGET == 0].dropna()
        #sns.boxplot(x = data_train.TARGET, y = data_train[i].dropna()[data_train.TARGET == 0])
        #sns.boxplot(x = data_train.TARGET, y = data_train[i].dropna()[data_train.TARGET == 1])
        sns.distplot(curr_feature_1)
        sns.distplot(curr_feature_0)
plt.subplots_adjust(top=1, bottom=0, left=0, right=3, hspace=0.25, wspace=0.55)
plt.show()
plt.subplot(211)
sns.distplot(np.log(data_train.AMT_INCOME_TOTAL[data_train.TARGET == 0] + 1))
plt.subplot(212)
sns.distplot(np.log(data_train.AMT_INCOME_TOTAL[data_train.TARGET == 1] + 1))
plt.show()
sum(np.log(data_train.AMT_INCOME_TOTAL + 1) > 14) #outliers
index = 0
for i in data_train.drop(columns=['SK_ID_CURR', 'TARGET']).columns:
    if np.dtype(data_train[i]) == 'O' and len(data_train[i].dropna())/data_train.shape[0] > 0.6:
        tab = pd.crosstab(data_train.TARGET, data_train[i], margins=True)
        display(HTML((tab/tab.loc[tab.index[-1]]).to_html()))
index = 0
for i in data_train.drop(columns=['SK_ID_CURR', 'TARGET']).columns:
    if np.dtype(data_train[i]) == 'int64' and len(data_train[i].dropna())/data_train.shape[0] > 0.6 and len(data_train[i].unique()) < 100:
        tab = pd.crosstab(data_train.TARGET, data_train[i], margins=True)
        display(HTML((tab/tab.loc[tab.index[-1]]).to_html()))
def chi_test(data, feature, target = 'TARGET', group_classes=False):
    
    if sum(pd.isna(data[feature])):
        data[feature].replace(np.nan, 'Unknown', inplace=True)

    cnt_table = pd.crosstab(data[target], data[feature])#to check if there are enough observations in each class
    
    if group_classes:
        tab = pd.crosstab(data[target], data[feature], margins=True)
        tab = tab/tab.loc[tab.index[-1]]
        labels = {}
        for i in cnt_table.columns:
            if tab[i][1] > tab['All'][1]:
                labels[i] = 'High risk'
            else:
                labels[i] = 'Low risk'
        cnt_bi_table = pd.crosstab(data[target], data[feature].replace(labels))
        chi = stats.chi2_contingency(cnt_bi_table)
        display(HTML(pd.crosstab(data[target], data[feature].replace(labels), margins=True).to_html()))
        print( { 'Chi-square statisitc': chi[0],
           'p-value': chi[1], 
          'df': chi[2]} )
        return labels
    else:
        chi = stats.chi2_contingency(cnt_table)                           
        display(HTML(pd.crosstab(data[target], data[feature], margins=True).to_html()))
        print( { 'Chi-square statisitc': chi[0],
           'p-value': chi[1], 
          'df': chi[2]} )
chi_test(data_train[data_train.CODE_GENDER != 'XNA'], 'CODE_GENDER')
inc_labels = chi_test(data_train, 'NAME_INCOME_TYPE', group_classes=True)
hsng_labels = chi_test(data_train, 'NAME_HOUSING_TYPE', group_classes=True)
occup_labels = chi_test(data_train, 'OCCUPATION_TYPE', group_classes=True)
orgn_labels = chi_test(data_train, 'ORGANIZATION_TYPE', group_classes=True)
data_train.shape
print( sum(data_train.FLAG_OWN_CAR == "Y") )
print( data_train.OWN_CAR_AGE.dropna().shape )
data_train.OWN_CAR_AGE.fillna(value = 0, inplace=True)
print( data_train.OCCUPATION_TYPE.unique() )
data_train.OCCUPATION_TYPE.fillna(value = 'Unknown', inplace=True)
print( sum(data_train.FLAG_MOBIL == 0) )
print( sum(data_train.FLAG_MOBIL == 1) )
print( sum(data_train.FLAG_CONT_MOBILE == 0) )
print( sum(data_train.FLAG_CONT_MOBILE == 1) )
data_train.drop(columns=['FLAG_MOBIL', 'FLAG_CONT_MOBILE'], inplace=True)
data_train.shape
data_train['DOC_COUNT'] = data_train.FLAG_DOCUMENT_2

for i in range(3, 22):
    data_train['DOC_COUNT'] = data_train['DOC_COUNT'] + data_train['FLAG_DOCUMENT_'+str(i)]
msno.matrix(data_train.iloc[:,0:42])
plt.show()
msno.matrix(data_train.iloc[:,42:89])
plt.show()
msno.matrix(data_train.iloc[:,89:])
plt.show()
for i in data_train.iloc[:, 42:56].columns:
    data_train[i] = -pd.isna(data_train[i])

for i in data_train.iloc[:, 84:89].columns:
    data_train[i] = -pd.isna(data_train[i])
data_train['HOUSE_INFO'] = (data_train.iloc[:, 84:89].sum(axis=1) + data_train.iloc[:, 42:56].sum(axis=1))/19
#for i in data_train.iloc[:, 114:120].columns:
#   data_train[i].fillna(data_train[i].median(), inplace = True)
data_train.drop(columns=data_train.iloc[:, 94:114].columns, inplace=True)
data_train.drop(columns=data_train.iloc[:, 42:89].columns, inplace=True)
data_train.drop(columns='EXT_SOURCE_1', inplace=True)
sns.distplot(data_train.EXT_SOURCE_3.dropna())
sns.distplot(data_train.EXT_SOURCE_3.fillna(data_train.EXT_SOURCE_3.dropna().median()))
plt.show()
data_train.info()
to_be_scaled = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH']
for i in data_train.columns:
    if data_train[i].dtype == 'float64':
        to_be_scaled.append(i)
#data_train.head()
for i in data_train.columns:
    if data_train[i].dtype == 'O':
        print( [i, data_train[i].unique()] )
sum(data_train.CODE_GENDER == 'XNA')
data_train.CODE_GENDER.replace('XNA', 'F', inplace=True)
data_train.NAME_INCOME_TYPE.replace(inc_labels, inplace=True)
data_train.OCCUPATION_TYPE.replace(occup_labels, inplace=True)
data_train.ORGANIZATION_TYPE.replace(orgn_labels, inplace=True)
data_train.WEEKDAY_APPR_PROCESS_START.replace({'WEDNESDAY': 'Week', 'MONDAY': 'Week', 'THURSDAY': 'Week', 'SUNDAY': 'Weekend',
                                               'SATURDAY': 'Weekend', 'FRIDAY': 'Week', 'TUESDAY': 'Week'}, inplace=True)
data_train.head()
binarizer = LabelBinarizer()
for i in data_train.columns:
    if data_train[i].dtype == 'O' and len(data_train[i].unique()) == 2:
        data_train[i] = binarizer.fit_transform(data_train[i])
#data_train.head()
encoder = LabelEncoder()
sum(data_train.NAME_TYPE_SUITE.isnull())
data_train.NAME_TYPE_SUITE = encoder.fit_transform(data_train.NAME_TYPE_SUITE.replace(np.nan, 'Unknown'))
print( encoder.classes_ )
sum(data_train.NAME_TYPE_SUITE == 7)
data_train.NAME_TYPE_SUITE.replace('Unknown', np.nan, inplace=True)
data_train.NAME_EDUCATION_TYPE = encoder.fit_transform(data_train.NAME_EDUCATION_TYPE)
encoder.classes_ #ordered
data_train = pd.get_dummies(data_train)
data_train.shape
impute = Imputer()
%%time
data_train_cl = pd.DataFrame(impute.knn(X=data_train, column='EXT_SOURCE_3', k = 3), columns=data_train.columns)
sns.distplot(data_train.EXT_SOURCE_3.dropna())
sns.distplot(data_train_cl.EXT_SOURCE_3)
plt.show()
%%time
for i in data_train.columns:
    if sum(data_train[i].isnull()):
        data_train_cl = pd.DataFrame(impute.knn(X=data_train_cl, column=i, k = 3), columns=data_train.columns)
data_train_cl.head()
std = RobustScaler()
for i in to_be_scaled:
    data_train_cl[i] = std.fit_transform(pd.DataFrame(data_train_cl[i], columns=[i]))
X_train, X_test, y_train, y_test = train_test_split(data_train_cl.drop(columns='TARGET'), 
                                                    data_train_cl.TARGET, test_size=0.2, random_state=23, stratify=data_train_cl.TARGET)
%%time
results = {}
for i in [0.001, 0.01, 0.1, 1, 2]:
    lgreg = LogisticRegression(C=i, class_weight='balanced', penalty='l2', max_iter=1000)
    lgreg.fit(X=X_train, y=y_train)
    pred = lgreg.predict(X_test)
    results[i] = roc_auc_score(y_test, pred)
results
forest = RandomForestClassifier(max_depth=10, n_estimators=200, class_weight='balanced', n_jobs=-1)
%%time
forest.fit(X_train, y_train)
pred2 = forest.predict(X_test)
roc_auc_score(y_test, pred2)
# for validation lgb
X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X_train, 
                                                    y_train, test_size=0.2, random_state=23)
train_data = lgb.Dataset(X_train_v, label=y_train_v)
test_data = lgb.Dataset(X_test_v, label=y_test_v)
parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 50,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 30,
    'learning_rate': 0.05,
    'verbose': 0
}

model = lgb.train(parameters,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=5000,
                       early_stopping_rounds=100)
pred4= model.predict(X_test)
roc_auc_score(y_test, pred4)