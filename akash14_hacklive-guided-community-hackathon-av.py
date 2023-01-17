!wget https://datahack-prod.s3.amazonaws.com/test_file/Test_jPKyvmK.csv

!wget https://datahack-prod.s3.amazonaws.com/train_file/Train_eP48B9k.csv

!wget https://datahack-prod.s3.amazonaws.com/sample_submission/SampleSubmission_XkIpo3X.csv
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

!pip install rfpimp

!pip install catboost

from sklearn.metrics import mean_absolute_error,accuracy_score

import lightgbm as lgb

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,GroupKFold,train_test_split,StratifiedShuffleSplit

from rfpimp import *

from tqdm import tqdm

from catboost import *

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder
train = pd.read_csv('Train_eP48B9k.csv')

test = pd.read_csv('Test_jPKyvmK.csv')

sub = pd.read_csv('SampleSubmission_XkIpo3X.csv')
train.head(5)
test.head(5)
df=train.append(test,ignore_index=True)
df.isnull().sum(),df.nunique()
df['id']=df.id.str.extract('(\d+)').astype(int)
df['customer_age']=df['customer_age'].fillna(method='bfill')
df['marital']=df['marital'].fillna('other')
df['balance']=df['balance'].fillna(df['balance'].mean())
df['personal_loan']=df['personal_loan'].fillna('other')
df['num_contacts_in_campaign']=df['num_contacts_in_campaign'].fillna(-1)
df['days_since_prev_campaign_contact']=df['days_since_prev_campaign_contact'].fillna(method='bfill')

df['days_since_prev_campaign_contact']=df['days_since_prev_campaign_contact'].fillna(method='ffill')
df['last_contact_duration']=df['last_contact_duration'].fillna(method='bfill')
x=['job_type','marital','education','default','housing_loan','personal_loan','communication_type','month','prev_campaign_outcome']

from sklearn.preprocessing import LabelEncoder

for i in x:

  le = LabelEncoder()

  df[i] = le.fit_transform(df[i])

  df[i]=df[i]+1
df.head()
x=['job_type','marital','education','default','housing_loan','personal_loan','communication_type','month','prev_campaign_outcome']

df = pd.get_dummies(df, columns=x)
train = df[df['term_deposit_subscribed'].isnull()==False]

test = df[df['term_deposit_subscribed'].isnull()==True]

del test['term_deposit_subscribed']
#bad_labels = train[train['term_deposit_subscribed'] == 0].sample(15000).index

#train = train[~train.index.isin(bad_labels)]
train['term_deposit_subscribed'].value_counts()
train_df=train.copy()

test_df=test.copy()
from math import sqrt 

from sklearn.metrics import f1_score
X = train_df.drop(labels=['term_deposit_subscribed'], axis=1)

y = train_df['term_deposit_subscribed'].values



#from imblearn.over_sampling import SMOTE

#sm = SMOTE(random_state=2)

#X, y = sm.fit_sample(X, y.ravel())



from sklearn.model_selection import train_test_split

X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.25, random_state=101)
X_train.shape, y_train.shape, X_cv.shape, y_cv.shape
categorical_features_indices = np.where(X_train.dtypes == 'category')[0]

categorical_features_indices
from catboost import CatBoostClassifier

cat = CatBoostClassifier(loss_function='MultiClass', 

                         eval_metric='TotalF1', 

                         classes_count=2,

                         depth=10,

                         random_seed=121, 

                         iterations=1000, 

                         learning_rate=0.1,

                         leaf_estimation_iterations=1,

                         l2_leaf_reg=1,

                         bootstrap_type='Bayesian', 

                         bagging_temperature=1, 

                         random_strength=1,

                         od_type='Iter', 

                         border_count=100,

                        #task_type = 'GPU',

                         od_wait=500)

cat.fit(X_train, y_train, verbose=100,

        use_best_model=True,

        cat_features=categorical_features_indices,

        eval_set=[(X_train, y_train),(X_cv, y_cv)],

        plot=False)

predictions = cat.predict(X_cv)

print('accuracy:', f1_score(y_cv, predictions, average='binary'))
print('accuracy:', f1_score(y_cv, predictions, average='binary'))
import seaborn as sns

feature_imp = pd.DataFrame(sorted(zip(cat.feature_importances_, X.columns), reverse=True)[:50], 

                           columns=['Value','Feature'])

plt.figure(figsize=(15,15))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('Catboost Features')

plt.tight_layout()

plt.show()
Xtest = test_df
from sklearn.model_selection import KFold



errcat = []

y_pred_totcat = []



fold = KFold(n_splits=10, shuffle=True, random_state=101)



for train_index, test_index in fold.split(X):

    X_train, X_test = X.loc[train_index], X.loc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    cat = CatBoostClassifier(loss_function='MultiClass', 

                         eval_metric='TotalF1', 

                         classes_count=2,

                         depth=6,

                         random_seed=121, 

                         iterations=3500, 

                         learning_rate=0.1,

                         leaf_estimation_iterations=1,

                         l2_leaf_reg=1,

                         bootstrap_type='Bayesian', 

                         bagging_temperature=0.8, 

                         random_strength=1,

                         od_type='Iter', 

                         border_count=100,

                         od_wait=500)

    cat.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0, early_stopping_rounds=200, cat_features=categorical_features_indices)



    y_pred_cat = cat.predict(X_test)

    print("Accuracy: ", f1_score(y_test,y_pred_cat, average='binary'))



    errcat.append(f1_score(y_test,y_pred_cat, average='binary'))

    p = cat.predict(Xtest)

    y_pred_totcat.append(p)
np.mean(errcat,0)
cat_final = np.mean(y_pred_totcat,0).round().astype(int)

cat_final
xxx = pd.DataFrame(data=cat_final, columns=['term_deposit_subscribed'])
submission = pd.DataFrame({

        "id":sub['id'],

        "term_deposit_subscribed": xxx['term_deposit_subscribed']

    })

submission.to_csv('./submission.csv', index=False)

print(submission)