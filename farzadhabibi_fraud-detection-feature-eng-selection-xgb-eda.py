import pandas as pd

import os

import numpy as np

import seaborn as sns

sns.set()

import matplotlib.pyplot as plt
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
DATASET_PATH = '../input/ieee-fraud-detection'

train_transaction = pd.read_csv(os.path.join(DATASET_PATH, 'train_transaction.csv'), index_col='TransactionID')

train_transaction = reduce_mem_usage(train_transaction)

test_transaction = pd.read_csv(os.path.join(DATASET_PATH, 'test_transaction.csv'), index_col='TransactionID')

test_transaction = reduce_mem_usage(test_transaction)

train_identity = pd.read_csv(os.path.join(DATASET_PATH,'train_identity.csv'), index_col='TransactionID')

train_identity = reduce_mem_usage(train_identity)

test_identity = pd.read_csv(os.path.join(DATASET_PATH,'test_identity.csv'), index_col='TransactionID')

test_identity = reduce_mem_usage(test_identity)

sample_submission = pd.read_csv(os.path.join(DATASET_PATH,'sample_submission.csv'), index_col='TransactionID')

train_set = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

test_set = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)
train_set=reduce_mem_usage(train_set)

test_set=reduce_mem_usage(test_set)
del train_transaction

del train_identity

del test_transaction

del test_identity
import gc

gc.collect()
train_labels = train_set.select_dtypes([object])

train_numbers = train_set.select_dtypes([np.float64, np.float32, np.float16, np.int32, np.int16, np.int8, np.int64])

label_atrrs = train_labels.columns.values

numbers_attrs = train_numbers.columns.values
train_numbers.info(max_cols=400)
from sklearn.base import TransformerMixin, BaseEstimator

class PercentImputer(TransformerMixin, BaseEstimator):

    def __init__(self, percent=0.6):

        self.percent = percent

        self.not_labels = []

    def fit(self, X, y=None):

        length = len(X)

        nulls_count = X.isnull().sum()

        labels = X.columns.values

        self.new_labels = []

        for label in labels:

            if(nulls_count[label] < self.percent * length):

                self.new_labels.append(label)

            else:

                self.not_labels.append(label)

        return self

    def transform(self, X, y=None):

        X=X.replace(np.inf,-999)

        return X[self.new_labels]
p_number_imputer = PercentImputer(percent=0.9)

p_labels_imputer = PercentImputer(percent=0.9)

numbers_p_imputed = p_number_imputer.fit_transform(train_numbers)

labels_p_imputed = p_labels_imputer.fit_transform(train_labels)
train_numbers.info()

numbers_p_imputed.info()
from sklearn.impute import SimpleImputer

numbers_imputed = SimpleImputer(strategy='most_frequent', verbose=-1).fit_transform(numbers_p_imputed)
numbers_imputed = pd.DataFrame(numbers_imputed, columns=numbers_p_imputed.columns)
numbers_cols  =numbers_p_imputed.columns

del numbers_p_imputed
y = numbers_imputed['isFraud'].copy() 
numbers_imputed.drop(['isFraud'], axis=1, inplace=True)
sns.countplot(y)
plt.figure(figsize=(15, 10))

sns.boxenplot(x=y, y=numbers_imputed['TransactionAmt'])

plt.ylim(top=7000, bottom=0)
class NormalizeByLog(BaseEstimator, TransformerMixin):

    def __init__(self, feature_name):

        self.feature_name = feature_name

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        X = X.copy()

        X[self.feature_name] = np.log1p(X[self.feature_name])

        return X
numbers_ta_filtered = NormalizeByLog('TransactionAmt').transform(numbers_imputed)
del numbers_imputed
plt.figure(figsize=(15, 10))

sns.distplot(numbers_ta_filtered['TransactionAmt'])


plt.figure(figsize=(15, 5))

plt.subplot(121)

sns.boxenplot(x=y, y=numbers_ta_filtered['TransactionDT'])

plt.subplot(122)

sns.distplot(numbers_ta_filtered['TransactionDT'])
numbers_ta_filtered.info(max_cols=400)
plt.figure(figsize=(20, 10))

l = 0

for i in range(5):

    att = 'card' + str(i+1)

    if att in numbers_ta_filtered:

        l+=1

        plt.subplot(2, 2, l)

        sns.distplot(numbers_ta_filtered[att], )

numbers_ta_filtered['card3'].value_counts().head()
to_drop_numbers = ['card3']
sns.distplot(numbers_ta_filtered['dist1'])
numbers_ta_filtered['dist1'].value_counts()
plt.figure(figsize=(15, 5))

for i in range(2):

    plt.subplot(1, 2, i+1)

    sns.distplot(numbers_ta_filtered['addr' + str(i+1)])
np.unique(numbers_ta_filtered['addr2'].values, return_counts = True)


to_drop_numbers.append('addr2')
plt.figure(figsize=(20, 20))

for i in range(14):

    plt.subplot(4, 4, i+1)

    sns.distplot(numbers_ta_filtered['C' + str(i+1)])

plt.figure(figsize=(15, 25))

for i in range(14):

    counts = np.unique(numbers_ta_filtered['C' + str(i+1)], return_counts=True)

    less_than_ten = 0

    for j in range(len(counts[0])):

        if counts[0][j] <= 4:

            less_than_ten += counts[1][j]

    print('C' + str(i+1), less_than_ten)

    
class CategorizeCs(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        X = X.copy()

        for i in range(14):

            att = 'C'+str(i+1)

            if att in X :

                X[att] = X[att].apply(lambda l : l if l <=4 else 4)

        return X
numbers_cat = CategorizeCs().transform(numbers_ta_filtered)
plt.figure(figsize=(20, 20))

for i in range(14):

    plt.subplot(4, 4, i+1)

    sns.countplot(x=numbers_cat['C' + str(i+1)], hue=y)

#     sns.kdeplot(numbers_cat['C' + str(i+1)])
to_drop_numbers.append('C3')
class CategorizeCs(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        X = X.copy()

        for i in range(14):

            att = 'C'+str(i+1)

            if att == 'C7' and att in X : 

                X[att] = X[att].apply(lambda l : l if l <=2 else 2)

            elif att in X :

                X[att] = X[att].apply(lambda l : l if l <=4 else 4)

        return X

numbers_cat = CategorizeCs().transform(numbers_ta_filtered)
del numbers_ta_filtered
plt.figure(figsize=(20, 20))

j = 1;

for i in range(15):

    att = 'D' + str(i+1)

    if att in numbers_cat:

        plt.subplot(4, 4, j)

        sns.distplot(numbers_cat['D' + str(i+1)])

        j+=1
plt.figure(figsize=(20, 64))

j = 1;

for i in range(340):

    att = 'V' + str(i+1)

    if att in numbers_cat:

        plt.subplot(34, 10, j)

        sns.distplot(numbers_cat['V' + str(i+1)])

        j+=1
class DropExtraVs(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):

        self.to_drop = []

        for i in range(340):

            att = 'V' + str(i+1)

            if att in X :

                counts = np.unique(X[att], return_counts=True)[1]

                counts.sort()

                if (counts[len(counts) - 1] + counts[len(counts) - 2]) > 0.85 * len(X) :

                    self.to_drop.append(att)

        return self

    def transform(self, X, y=None):

        return X.drop(self.to_drop, axis=1) 
dropper = DropExtraVs()

numbers_v_droped = dropper.fit_transform(numbers_cat)
numbers_v_droped.head()
corr = numbers_v_droped.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(15, 12))

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, vmax=1, center=0,vmin=-1 , 

            square=True, linewidths=.005)
corr = corr.iloc[1:, 1:]

corr = corr.applymap(lambda x : 1 if x > 0.75 else -1 if x < -0.75 else 0)

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(15, 12))

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, vmax=1, center=0,vmin=-1 , 

            square=True, linewidths=.005)
numbers_v_droped['isFraud'] = y

numbers_v_droped.corr()['isFraud'].sort_values()
to_drop_numbers += ['V314', 'V315', 'V308', 'V306', 'V131', 'V130', 'V128', 'V127', 'V285', 'V96', 'V91', 'V82', 'V76',

                    'V49', 'V48', 'V36', 'V11', 'D2', 'C10', 'C7', 'C8', 'C11', 'C1']
class DataFrameDropper(BaseEstimator, TransformerMixin):

    def __init__(self, drop_attrs=[]):

        self.drop_attrs = drop_attrs

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        X = X.copy()

        X.drop(self.drop_attrs, axis=1, inplace=True, errors='ignore')

        return X
final_numbers = DataFrameDropper(drop_attrs=to_drop_numbers).transform(numbers_v_droped)
del numbers_v_droped
del numbers_cat
gc.collect()
labels_p_imputed.head()
class LabelImputer(TransformerMixin, BaseEstimator):

    def __init__(self, dummy=False):

      self.dummy = dummy

    def fit(self, X, y=None):

        self.tops = [[], []]

        for col in X:

            self.tops[0].append(str(col))

            self.tops[1].append(X[col].describe()['top'])

        return self

    def transform(self, X, y=None):

        X = X.copy()

        if self.dummy:

          return X.fillna('-9999')

        for i in range(len(self.tops[0])):

            X[self.tops[0][i]].fillna(self.tops[1][i], inplace=True)

        return X
label_imputer = LabelImputer(dummy=False)

label_imputed = label_imputer.fit_transform(labels_p_imputed)
plt.figure(figsize=(20, 5))

sns.countplot(y=label_imputed['ProductCD'].reset_index()['ProductCD'], hue=y, data=label_imputed)
del labels_p_imputed
plt.figure(figsize=(20, 10))

plt.subplot(211)

sns.countplot(y=label_imputed['card4'].reset_index()['card4'], hue=y)

plt.subplot(212)

sns.countplot(y=label_imputed['card6'].reset_index()['card6'], hue=y)
label_imputed[['card6', 'card4']].groupby('card6').agg(['count']).stack()
class ChangeToDebit(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        X = X.copy()

        X['card6'] = X['card6'].apply(lambda l: l if not l == 'debit or credit' and not l == 'charge card' else 'debit')

        return X
label_card6_changed = ChangeToDebit().transform(label_imputed)
plt.figure(figsize=(20, 20))

sns.countplot(y=label_card6_changed['P_emaildomain'].reset_index()['P_emaildomain'], hue=y)
class CategorizeEmail(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        X = X.copy()

        X['P_emaildomain'] = X['P_emaildomain'].apply(lambda l: l if l == 'gmail.com' or 

                                                      l == 'yahoo.com' or

                                                      l == 'anonymous.com' or 

                                                      l == 'hotmail.com' or

                                                      l == 'aol.com'

                                                      else 'others')

        return X
label_mail_changed = CategorizeEmail().transform(label_card6_changed)
plt.figure(figsize=(20, 5))

sns.countplot(y=label_mail_changed['P_emaildomain'].reset_index()['P_emaildomain'], hue=y)
plt.figure(figsize=(20, 5))

sns.countplot(y=label_mail_changed['M6'].reset_index()['M6'], hue=y)
from sklearn.preprocessing import OneHotEncoder

class OneHotGoodEncoder(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.encoder = OneHotEncoder()

    def fit(self, X, y=None): 

        self.encoder.fit(X)

        return self

    def transform(self, X, y=None):

        columns = X.columns

        X_transformed = self.encoder.transform(X).toarray()

        cats = self.encoder.categories_

        i = 0

        labels = []

        for cat in cats:

            for c in cat:

                labels.append(columns[i] + ' : ' + c)

            i = i+1

        return pd.DataFrame(X_transformed, columns=labels)
encoder = OneHotGoodEncoder()

encoder.fit(label_mail_changed)

label_encoded = encoder.transform(label_mail_changed)
from sklearn.feature_selection import f_regression

F, p_value = f_regression(label_encoded, y)

np.array(label_encoded.columns) + " = " + (p_value < 0.05).astype(str) 
from sklearn.preprocessing import LabelEncoder

class ModifiedLabelEncoder(TransformerMixin, BaseEstimator):

    def fit(self,X, y=None):

        self.cols=X.columns

        return self

    def transform(self,X, y=None):

        X=X.copy()

        for f in self.cols:

          if X[f].dtype=='object': 

            lbl = LabelEncoder()

            lbl.fit(list(X[f].values) )

            X[f] = lbl.transform(list(X[f].values))

        return X
del label_mail_changed

del label_encoded

del label_card6_changed
iden_numbers_imputed = final_numbers
id_to_drop = []
ids = ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_08', 'id_10', 

      'id_11', 'id_12', 'id_13', 'id_14', 'id_17', 'id_19', 'id_20', 'id_32']

cols = list(iden_numbers_imputed.columns)

plt.figure(figsize=(20, 20))

j = 1

for i in range(len(cols)):

    if(str(cols[i]) in ids):

      plt.subplot(4, 4, j)

      j+=1

      sns.distplot(iden_numbers_imputed[cols[i]])
train_numbers['id_03'].value_counts(dropna=False, normalize=True).head()
id_to_drop.append('id_3')
train_numbers['id_04'].value_counts(dropna=False, normalize=True).head()
id_to_drop.append('id_04')
train_numbers['id_09'].value_counts(dropna=False, normalize=True).head()
id_to_drop.append('id_09')
train_numbers['id_10'].value_counts(dropna=False, normalize=True).head()
id_to_drop.append('id_10')
corr = iden_numbers_imputed.corr()

corr = corr.iloc[1:, 1:]

corr = corr.applymap(lambda x : 1 if x > 0.75 else -1 if x < -0.75 else 0)

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(15, 12))

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, vmax=1, center=0,vmin=-1 , 

            square=True, linewidths=.005)
id_labels_to_drop = list()
iden_labels_imputed = label_imputed

del label_imputed
# y_i = pd.merge(train_identity, train_transaction, how='inner', on = 'TransactionID')['isFraud']
cols = ['id_12', 'id_15', 'id_16', 'id_28', 'id_29', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38']
plt.figure(figsize=(20, 20))

for i in range(len(cols)):

    plt.subplot(4, 3, i+1)

    sns.countplot(y=iden_labels_imputed[cols[i]].reset_index()[cols[i]], hue=y)

iden_labels_imputed['id_34'].value_counts()
class MeltMatchStatus(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        X = X.copy()

        X['id_34'] = X['id_34'].apply(lambda l: 'match_status:1' if l == 'match_status:0'

                                                                         or  l == 'match_status:-1' else l)

        return X
iden_label_melted = MeltMatchStatus().transform(iden_labels_imputed)
iden_label_melted['id_30'].value_counts().head()
class SimplifyOS(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

         return self

    def transform(self, X, y=None):

        X = X.copy()

        X['OS'] = X['id_30'].apply(lambda l: 'iOS' if l.find('iOS') is not -1 else 'Android' if l.find('Android') is not -1

                                  else 'Windows' if l.find('Windows') is not -1 else 'Mac' if l.find('Mac') is not -1 else 'Others')

        X.drop(['id_30'],axis=1, inplace=True)

        return X
iden_label_simple = SimplifyOS().transform(iden_label_melted)
plt.figure(figsize=(20, 5))

sns.countplot(y=iden_label_simple['OS'].reset_index()['OS'], hue=y)
iden_label_simple['OS'].value_counts().head()
iden_label_simple['DeviceInfo'].value_counts().head()
id_labels_to_drop.append('DeviceInfo')

iden_label_simple.drop(['DeviceInfo'], axis=1, inplace=True)
class SimplifyBrowser(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

         return self

    def transform(self, X, y=None):

        X = X.copy()

        X['Browser'] = X['id_31'].apply(lambda l: 'm_chrome' if l.find('for android') is not -1 else 'm_safari' if l.find('mobile safari') is not -1

                                  else 'ie' if l.find('ie') is not -1 else 'ie' if l.find('edge') is not -1

                                       else 'safari' if l.find('safari') is not -1 else 'chrome' if l.find('chrome') is not -1 

                                       else 'firefox' if l.find('firefox') is not -1 else 'others')

        X.drop(['id_31'],axis=1, inplace=True)

        return X
iden_simple_browser = SimplifyBrowser().transform(iden_label_simple)
iden_simple_browser['Browser'].value_counts()
plt.figure(figsize=(20, 5))

sns.countplot(y=iden_simple_browser['Browser'].reset_index()['Browser'], hue=y)
class ScreenSimplify(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        X = X.copy()

        X['Screen'] = X['id_33'].apply(lambda l : 'Big' if int(l[:l.find('x')]) *  int(l[l.find('x')+1:]) >= 2073600

                                       else 'Medium' if int(l[:l.find('x')]) *  int(l[l.find('x')+1:]) > 777040

                                       else 'Small')

        X.drop(['id_33'],axis=1, inplace=True)

        return X
iden_simple_screen = ScreenSimplify().transform(iden_simple_browser)
plt.figure(figsize=(20, 5))

sns.countplot(y=iden_simple_screen['Screen'].reset_index()['Screen'], hue=y)
encoder = OneHotGoodEncoder()

encoder.fit(iden_simple_screen)

label_encoded = encoder.transform(iden_simple_screen)
from sklearn.feature_selection import f_regression

F, p_value = f_regression(label_encoded, y)

np.set_printoptions(threshold=40)

np.array(label_encoded.columns) + " = " + (p_value < 0.05).astype(str)
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn import preprocessing

class FeatureEng(BaseEstimator, TransformerMixin):

  def fit(self, X, y=None):

    return self

  def transform(self, X, y=None):

    X=X.copy()

    

    ## Check proton mail

    X['P_isproton']=(X['P_emaildomain']=='protonmail.com')

    X['R_isproton']=(X['R_emaildomain']=='protonmail.com')

    

    ## number of nulls

    X['nulls1'] = X.isna().sum(axis=1)

    

    ## check latest browser or not

    a = np.zeros(X.shape[0])

    X["lastest_browser"] = a

    X.loc[X["id_31"]=="samsung browser 7.0",'lastest_browser']=1

    X.loc[X["id_31"]=="opera 53.0",'lastest_browser']=1

    X.loc[X["id_31"]=="mobile safari 10.0",'lastest_browser']=1

    X.loc[X["id_31"]=="google search application 49.0",'lastest_browser']=1

    X.loc[X["id_31"]=="firefox 60.0",'lastest_browser']=1

    X.loc[X["id_31"]=="edge 17.0",'lastest_browser']=1

    X.loc[X["id_31"]=="chrome 69.0",'lastest_browser']=1

    X.loc[X["id_31"]=="chrome 67.0 for android",'lastest_browser']=1

    X.loc[X["id_31"]=="chrome 63.0 for android",'lastest_browser']=1

    X.loc[X["id_31"]=="chrome 63.0 for ios",'lastest_browser']=1

    X.loc[X["id_31"]=="chrome 64.0",'lastest_browser']=1

    X.loc[X["id_31"]=="chrome 64.0 for android",'lastest_browser']=1

    X.loc[X["id_31"]=="chrome 64.0 for ios",'lastest_browser']=1

    X.loc[X["id_31"]=="chrome 65.0",'lastest_browser']=1

    X.loc[X["id_31"]=="chrome 65.0 for android",'lastest_browser']=1

    X.loc[X["id_31"]=="chrome 65.0 for ios",'lastest_browser']=1

    X.loc[X["id_31"]=="chrome 66.0",'lastest_browser']=1

    X.loc[X["id_31"]=="chrome 66.0 for android",'lastest_browser']=1

    X.loc[X["id_31"]=="chrome 66.0 for ios",'lastest_browser']=1

    

    ## check mail suffix

    emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

    us_emails = ['gmail', 'net', 'edu']

    for c in ['P_emaildomain', 'R_emaildomain']:

      X[c + '_bin'] = X[c].map(emails)

      X[c + '_suffix'] = X[c].map(lambda x: str(x).split('.')[-1])

      X[c + '_suffix'] = X[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

    del emails, us_emails

    

    

    ### new card, addr features

    X['card1_count_full'] = X['card1'].map(pd.concat([train_set['card1'], test_set['card1']], ignore_index=True).value_counts(dropna=False))

    X['card2_count_full'] = X['card2'].map(pd.concat([train_set['card2'], test_set['card2']], ignore_index=True).value_counts(dropna=False))

    X['card3_count_full'] = X['card3'].map(pd.concat([train_set['card3'], test_set['card3']], ignore_index=True).value_counts(dropna=False))

    X['card4_count_full'] = X['card4'].map(pd.concat([train_set['card4'], test_set['card4']], ignore_index=True).value_counts(dropna=False))

    X['card5_count_full'] = X['card5'].map(pd.concat([train_set['card5'], test_set['card5']], ignore_index=True).value_counts(dropna=False))

    X['card6_count_full'] = X['card6'].map(pd.concat([train_set['card6'], test_set['card6']], ignore_index=True).value_counts(dropna=False))

    X['addr1_count_full'] = X['addr1'].map(pd.concat([train_set['addr1'], test_set['addr1']], ignore_index=True).value_counts(dropna=False))

    X['addr2_count_full'] = X['addr2'].map(pd.concat([train_set['addr2'], test_set['addr2']], ignore_index=True).value_counts(dropna=False))

    

    ###  Transaction_amt , id02, and D15

    X['TransactionAmt_to_mean_card1'] = X['TransactionAmt'] / X.groupby(['card1'])['TransactionAmt'].transform('mean')

    X['TransactionAmt_to_mean_card4'] = X['TransactionAmt'] / X.groupby(['card4'])['TransactionAmt'].transform('mean')

    X['TransactionAmt_to_std_card1'] = X['TransactionAmt'] / X.groupby(['card1'])['TransactionAmt'].transform('std')

    X['TransactionAmt_to_std_card4'] = X['TransactionAmt'] / X.groupby(['card4'])['TransactionAmt'].transform('std')

    X['id_02_to_mean_card1'] = X['id_02'] / X.groupby(['card1'])['id_02'].transform('mean')

    X['id_02_to_mean_card4'] = X['id_02'] / X.groupby(['card4'])['id_02'].transform('mean')

    X['id_02_to_std_card1'] = X['id_02'] / X.groupby(['card1'])['id_02'].transform('std')

    X['id_02_to_std_card4'] = X['id_02'] / X.groupby(['card4'])['id_02'].transform('std')

    X['D15_to_mean_card1'] = X['D15'] / X.groupby(['card1'])['D15'].transform('mean')

    X['D15_to_mean_card4'] = X['D15'] / X.groupby(['card4'])['D15'].transform('mean')

    X['D15_to_std_card1'] = X['D15'] / X.groupby(['card1'])['D15'].transform('std')

    X['D15_to_std_card4'] = X['D15'] / X.groupby(['card4'])['D15'].transform('std')

    

    ###  time of transactions

    X['Transaction_day_of_week'] = np.floor((X['TransactionDT'] / (3600 * 24) - 1) % 7)

    X['Transaction_hour_of_day'] = np.floor(X['TransactionDT'] / 3600) % 24

    X['TransactionAmt_decimal'] = ((X['TransactionAmt'] - X['TransactionAmt'].astype(int)) * 1000).astype(int)  

    

    ### some combinations

    for feature in ['id_02__id_20', 'id_02__D8', 'D11__DeviceInfo', 'DeviceInfo__P_emaildomain', 'P_emaildomain__C2', 

                    'card2__dist1', 'card1__card5', 'card2__id_20', 'card5__P_emaildomain', 'addr1__card1']:

      f1, f2 = feature.split('__')

      X[feature] = X[f1].astype(str) + '_' + X[f2].astype(str)

      le =preprocessing.LabelEncoder()

      le.fit(list(X[feature].astype(str).values))

      X[feature] = le.transform(list(X[feature].astype(str).values))

    for feature in ['id_01', 'id_31', 'id_33', 'id_35']:

    

    # Count encoded separately for train and test

      X[feature + '_count_dist'] = X[feature].map(X[feature].value_counts(dropna=False))



    category_features=["ProductCD","P_emaildomain",

                       "R_emaildomain","M1","M2","M3","M4","M5","M6","M7","M8","M9","DeviceType","DeviceInfo","id_12",

                       "id_13","id_14","id_15","id_16","id_17","id_18","id_19","id_20","id_21","id_22","id_23","id_24",

                       "id_25","id_26","id_27","id_28","id_29","id_30","id_32","id_34", 'id_36'

                       "id_37","id_38"]

    for c in category_features:

      X[feature + '_count_full'] = X[feature].map(pd.concat([train_set[feature], test_set[feature]], ignore_index=True).value_counts(dropna=False))



    del le

    return X



train_set = FeatureEng().transform(train_set)

test_set = FeatureEng().transform(test_set)
class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attrs):

        self.attrs = attrs

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        return X.loc[:, self.attrs]

class ToDataFrame(BaseEstimator, TransformerMixin):

    def __init__(self, columns):

        self.columns = columns

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        return pd.DataFrame(X, columns=self.columns)

        
from sklearn.pipeline import Pipeline

from sklearn.pipeline import FeatureUnion

from sklearn.preprocessing import StandardScaler



numbers_pipeline = Pipeline([

    ('select', DataFrameSelector(numbers_attrs)),

    ('p_imputer', PercentImputer(percent=0.9)),

    ('s_imputer', SimpleImputer(strategy='mean')),

    ('to_dataFrame', ToDataFrame(columns=numbers_cols)),

    ('drop_dt_fraud', DataFrameDropper(drop_attrs=['isFraud'])),

    ('normalize', NormalizeByLog('TransactionAmt')),

    ('categorize', CategorizeCs()),

    ('dropVs',  DropExtraVs()),

    ('drop1', DataFrameDropper(drop_attrs=to_drop_numbers)),

    ('drop2', DataFrameDropper(drop_attrs=id_to_drop)),

    ('std_scale', StandardScaler())

])

# We use lgb and xgb, so reducing data does not make it better. We can 

# do not use some of our reductions and droppings. Do to that we have more

# features, therefore it got more time to train, yet we will have a better

# score.

labels_pipeline = Pipeline([

    ('select', DataFrameSelector(label_atrrs)),

    ('p_imputer', PercentImputer(percent=0.9)),

    ('l_imputer', LabelImputer(dummy=True)),

    ('change_debit', ChangeToDebit()),

#     ('categorize_email', CategorizeEmail()), 

    ('melt', MeltMatchStatus()),

#     ('os', SimplifyOS()),

#     ('browser', SimplifyBrowser()),

#     ('screen', ScreenSimplify()),

#     ('drop', DataFrameDropper(drop_attrs=id_labels_to_drop)),

    ('encode', ModifiedLabelEncoder()),

    ('std_scale', StandardScaler())

])



pipeline = FeatureUnion([

    ('numbers', numbers_pipeline),

    ('labels', labels_pipeline),

])
gc.collect()
del train_labels

del train_numbers
X_train = pipeline.fit_transform(train_set)

X_test = pipeline.transform(test_set)
y_train = y

del y
X_train
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.model_selection import cross_val_predict

def show_fpr_tpr(fpr, tpr):

    plt.plot(fpr, tpr)

    plt.xlabel("False Positive Rate")

    plt.plot([0, 1], [0, 1], 'k--')

    plt.axis([0, 1, 0, 1])

    plt.ylabel("True Positive Rate")

    plt.show()



def analys_model(model, x=X_train, y=y_train):

    y_probs = cross_val_predict(model, x, y, cv=3, method="predict_proba", n_jobs=-1)

    y_score = y_probs[:, -1]

    fpr, tpr, threshold = roc_curve(y, y_score)

    show_fpr_tpr(fpr, tpr)

    print(roc_auc_score(y, y_score))
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier()

rf_clf.fit(X_train, y_train)
analys_model(rf_clf, x=X_train, y=y_train)
from sklearn.model_selection import TimeSeriesSplit,KFold

n_fold = 4

folds = KFold(n_splits=n_fold,shuffle=True)



print(folds)
X_train
lgb_submission=sample_submission.copy()

lgb_submission['isFraud'] = 0

import lightgbm as lgb

from sklearn.metrics import roc_auc_score

for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train)):

    print(fold_n)

    

    X_train_, X_valid = X_train[train_index], X_train[valid_index]

    y_train_, y_valid = y_train[train_index], y_train[valid_index]

    dtrain = lgb.Dataset(X_train, label=y_train)

    dvalid = lgb.Dataset(X_valid, label=y_valid)

    

    lgbclf = lgb.LGBMClassifier(

        num_leaves= 512,

        n_estimators=512,

        max_depth=9,

        learning_rate=0.064,

        subsample=0.85,

        colsample_bytree=0.85,

        boosting_type= "gbdt",

        reg_alpha=0.3,

        reg_lamdba=0.243,

        verbosity=-1,

    )

    

    X_train_, X_valid = X_train[train_index], X_train[valid_index]

    y_train_, y_valid = y_train[train_index], y_train[valid_index]

    lgbclf.fit(X_train_,y_train_)

    

    del X_train_,y_train_

    print('finish train')

    pred=lgbclf.predict_proba(X_test)[:,1]

    val=lgbclf.predict_proba(X_valid)[:,1]

    print('finish pred')

#     del lgbclf, X_valid

    print('ROC accuracy: {}'.format(roc_auc_score(y_valid, val)))

    del val,y_valid

    lgb_submission['isFraud'] = lgb_submission['isFraud']+pred/n_fold

    del pred
lgb_submission.to_csv('./lgb.csv')

xgb_submission=sample_submission.copy()

xgb_submission['isFraud'] = 0

import xgboost as xgb

from sklearn.metrics import roc_auc_score

for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train)):

    print(fold_n)

    xgbclf = xgb.XGBClassifier(

        n_estimators=512,

        max_depth=16,

        learning_rate=0.014,

        subsample=0.85,

        colsample_bytree=0.85,

        missing=-999,

        tree_method='gpu_hist',

        reg_alpha=0.3,

        reg_lamdba=0.243

    )

    

    X_train_, X_valid = X_train[train_index], X_train[valid_index]

    y_train_, y_valid = y_train[train_index], y_train[valid_index]

    xgbclf.fit(X_train_,y_train_)

    del X_train_,y_train_

    pred=xgbclf.predict_proba(X_test)[:,1]

    val=xgbclf.predict_proba(X_valid)[:,1]

    del xgbclf, X_valid

    print('ROC accuracy: {}'.format(roc_auc_score(y_valid, val)))

    del val,y_valid

    xgb_submission['isFraud'] = xgb_submission['isFraud']+pred/n_fold

    del pred

    gc.collect()
xgb_submission.to_csv( './xgb.csv')
y_final = sample_submission.copy()

y_final['isFraud'] = 0.5 * xgb_submission['isFraud'] + 0.5 * lgb_submission['isFraud']
y_final.to_csv('./prediction.csv')
y_final.head()