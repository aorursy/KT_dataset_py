import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
path = '../input/ieee-fraud-detection/'



train_identity = pd.read_csv(f'{path}train_identity.csv')

train_transaction = pd.read_csv(f'{path}train_transaction.csv')

test_identity = pd.read_csv(f'{path}test_identity.csv')

test_transaction = pd.read_csv(f'{path}test_transaction.csv')
print('train identity shape:', train_identity.shape)

print('train transaction shape:',train_transaction.shape)

print('-'*40)

print('test identity shape:',test_identity.shape)

print('train transaction shape:',test_transaction.shape)
train = pd.merge(train_transaction, train_identity, how='inner', on='TransactionID')

test = pd.merge(test_transaction, test_identity, how='inner', on='TransactionID')



print('Training Data shape:',train.shape)

print('Training Data shape:',test.shape)
del train_identity, train_transaction, test_identity, test_transaction 
y = train.isFraud

train.drop(['isFraud'], axis=1, inplace=True)
train.head()
test.head()
print(train.info())

print()

print(test.info())
train.describe()
train_id_cols = [ 'id_01', 'id_02','id_05','id_06','id_11', 'id_12', 'id_13', 

                 'id_15', 'id_16','id_17', 'id_19', 'id_20', 'id_28', 'id_29',

                 'id_31', 'id_35', 'id_36', 'id_37','id_38']



test_id_cols = [ 'id-01', 'id-02','id-05','id-06','id-11', 'id-12', 'id-13', 

                 'id-15', 'id-16','id-17', 'id-19', 'id-20', 'id-28', 'id-29',

                 'id-31', 'id-35', 'id-36', 'id-37','id-38']



test = test.rename(columns = dict(zip(test_id_cols, train_id_cols)))
def NullPercentage(df):

    cols = [col for col in df.columns if (df[col].isnull().sum() / len(df) * 100) <= 15]

    return df[cols]



train = NullPercentage(train)

test = test[train.columns]
train.columns == test.columns
print(train.info())

print()

print(test.info())
train_cat = train.select_dtypes(include='object')

test_cat = test.select_dtypes(include='object')



print(train_cat.info())

print('\n', '*'*50, '\n')

print(test_cat.info())
for cat in train_cat:

    print(cat)

    print(train_cat[cat].value_counts())

    print('*'*40, '\n')
from sklearn.impute import SimpleImputer



imp = SimpleImputer(strategy='most_frequent')

imp.fit(train_cat)



train_cat = pd.DataFrame(imp.transform(train_cat), columns=train_cat.columns)

test_cat = pd.DataFrame(imp.transform(test_cat), columns=test_cat.columns)
pd.concat([train_cat,test_cat],axis=0, ignore_index=True)
df = pd.concat([train_cat,test_cat],axis=0, ignore_index=True)



for cat_col in df.columns:

    temp = pd.get_dummies(df[cat_col], drop_first=True, prefix=cat_col)

    df = pd.concat([df, temp], axis=1)

    df.drop([cat_col], axis=1, inplace= True)

       

train_cat = df[:len(train)]

test_cat = df[len(train):] 



print(train_cat.shape)

print(test_cat.shape)

train_cat.head()
test_cat.head()
train_num = train.select_dtypes(exclude='object')

test_num = test.select_dtypes(exclude='object')



print('Numerical Training Data shape:',train_num.shape)

print('Numerical Training Data shape:',test_num.shape)
train_num.describe()
from sklearn.impute import SimpleImputer



imp = SimpleImputer(strategy='mean')

imp.fit(train_num)



train_num = pd.DataFrame(imp.transform(train_num), columns=train_num.columns)

test_num = pd.DataFrame(imp.transform(test_num), columns=test_num.columns)
train_num.drop(['TransactionID'], axis=1, inplace=True)

test_num.drop(['TransactionID'], axis=1, inplace=True)
train = pd.concat([train_num, train_cat], axis=1)

test = pd.concat([test_num, test_cat], axis=1)
del train_num, train_cat, test_num, test_cat
y.value_counts()
from sklearn.preprocessing import StandardScaler



df = pd.concat([train ,test],axis=0, ignore_index=True)



sc = StandardScaler()

sc.fit(df)

df = pd.DataFrame(sc.transform(df), columns=df.columns)



train= df[:len(train)]

test = df[len(train):] 
del df
from imblearn.over_sampling import RandomOverSampler

from imblearn.under_sampling import RandomUnderSampler



rus = RandomUnderSampler(random_state=0)

X_resampled, y_resampled =  rus.fit_resample(train, y)
print('X resampled shape', X_resampled.shape)

print('y resampled shape', y_resampled.shape)
import statsmodels.api as sm



log = sm.GLM(y_resampled, (sm.add_constant(X_resampled)),family=sm.families.Binomial())

print(log.fit().summary())
from sklearn.linear_model import  LogisticRegression

logreg = LogisticRegression()



from sklearn.feature_selection import RFE

rfe =RFE(logreg, 50) 

rfe = rfe.fit(X_resampled, y_resampled)



print(rfe.support_)

print(rfe.ranking_)
dict1 = dict(zip(X_resampled.columns,rfe.ranking_))

print(dict1)



arr1=np.asarray(list(dict1.items()))



dfarr=pd.DataFrame(arr1,columns=['words','number'])

print(dfarr.columns)



df2=dfarr[dfarr.number == '1']

arr2=np.array(df2.words)

print(arr2)
print(arr2)
X_resampled = X_resampled[arr2]
def vif_cal(input_data, dependent_col):



  vif_df = pd.DataFrame(columns = ['var', 'vif'])

  x_vars = input_data.drop([dependent_col], axis=1)

  xvar_names = x_vars.columns



  for i in range(0, xvar_names.shape[0]):

    y = x_vars[xvar_names[i]]

    x = x_vars[xvar_names.drop(xvar_names[i])]



    rsq = sm.OLS(y, x).fit().rsquared

    vif = round( 1 / (1-rsq) , 2)

    vif_df.loc[i] = [xvar_names[i], vif]



  return vif_df.sort_values(by='vif', axis=0, ascending=True, inplace=False, ignore_index=True)    
VIF_DATA = pd.concat([X_resampled, y_resampled], axis=1)

VIF_DATA.head()
vif_cal(input_data=VIF_DATA, dependent_col='isFraud')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
from xgboost import XGBClassifier

model = XGBClassifier()



model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_test.value_counts()
from sklearn import metrics



sm = metrics.confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = sm.ravel()



print(tn, fp, fn, tp)

print(sm)
def draw_roc(actual , probs):

  fpr,tpr,thresholds = metrics.roc_curve(actual, probs, drop_intermediate=False)



  auc_score = metrics.roc_auc_score(actual, probs)

  plt.figure(figsize=(6,4))

  plt.plot(fpr,tpr, label='ROC curve ( area = %0.2f)'% auc_score)

  plt.plot([0,1],[0,1],'k--')



  plt.xlim([0.0,1.0])

  plt.ylim([0.0,1.05])

  plt.xlabel('False Positive Rate or [1- True Negative Rate]')

  plt.ylabel('True Positive Rate')

  plt.title('Receiver operating Characterstics example')

  plt.legend(loc='lower right')

  plt.show()



  return fpr ,tpr , thresholds



draw_roc(y_test, y_pred)
from catboost import CatBoostClassifier

model = CatBoostClassifier()



model.fit(X_train, y_train)

y_pred = model.predict(X_test)
draw_roc(y_test, y_pred)
from lightgbm import LGBMClassifier

model = LGBMClassifier()



model.fit(X_train, y_train)

y_pred = model.predict(X_test)



draw_roc(y_test, y_pred)