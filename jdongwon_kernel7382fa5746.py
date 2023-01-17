import numpy as np

import pandas as pd

from scipy.stats import randint, uniform

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import font_manager, rc

import matplotlib.font_manager as fm

import warnings

from imblearn.combine import *

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, classification_report

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import cross_val_score, KFold

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC 

from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder

from sklearn.pipeline import make_pipeline

from sklearn.datasets import make_classification

from collections import Counter

from sklearn.pipeline import Pipeline

from sklearn.pipeline import FeatureUnion

from sklearn.decomposition import PCA

from sklearn.feature_selection import SelectKBest

from sklearn.metrics import confusion_matrix

from imblearn.under_sampling import TomekLinks
df = pd.read_csv('../input/111112/human.csv',encoding='cp949')

df.head()
df.rename(columns={'아이디':'ID', '나이':'age','노동 계급':'Working Class', '학력':'Academic Background', 

                   '교육 수':'Education Number', '혼인 상태':'Married','직업':'Job', '관계':'Relationship',

                   '인종':'Race', '성별':'Sex', '자본 이득':'Capital Gain', '자본 손실':'Capital Loss',

                   '주당 시간':'Hours Per Week', '모국':'Country' }, inplace=True)
df.head()
df.info()
df.isnull().sum()
df.dropna()

#df.fillna(0, inplace = True)
f,ax=plt.subplots(1,1,figsize=(20,10))

sns.countplot('Academic Background',hue='Sex',data=df)

plt.show()
f,ax=plt.subplots(1,1,figsize=(18,8))

sns.countplot('Job',hue='Sex',data=df)

plt.show()
f,ax=plt.subplots(1,1,figsize=(18,8))

sns.countplot('Working Class',hue='Sex',data=df)

plt.show()
df['age'].describe()
df['age_level'] = df['age'].apply(lambda x : 1 if x < 20

                                 else 2 if 20 <= x < 40

                                 else 3 if 40 <= x < 60

                                 else 4)
df['Hours Per Week'].describe()
df['Hours Per Week_level'] = df['Hours Per Week'].apply(lambda x : 1 if x < 25

                                 else 2 if 25 <= x < 40

                                 else 3 if 40 <= x < 60

                                 else 4)
np.var(df['Capital Gain'])
np.var(df['Capital Loss'])
df['Capital Gain'].describe()
df['Capital Loss'].describe()
a = df['Capital Gain'] > 0

b = df['Capital Loss'] > 0

df['Capital'] = (a & b)

df['Capital'].value_counts()
df['Capital'] = df['Capital Gain'] - df['Capital Loss']
df['Capital'] = df['Capital'].apply(lambda x: 2 if x > 0

                                   else 1 if x < 0

                                   else 0)
# 2 : 'Capital Gain' > 0

# 1 : 'Capital Loss' > 0

# 0 : 'Capital Gain' & 'Capital Loss' == 0

df['Capital'].value_counts()
obj1 = ['Working Class', 'Academic Background', 'Married', 'Job', 'Relationship', 'Race', 'Country', 'Sex']

df[obj1] = df[obj1].apply(lambda x: x.astype('category').cat.codes)

df.head()
df.corr()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from imblearn.combine import SMOTETomek
dfX = df.drop(['ID','Sex'],axis=1)

dfy = df['Sex']

X_train, X_test, y_train, y_test = train_test_split(

     dfX,dfy,random_state=0)
tree3 = DecisionTreeClassifier(max_depth=6, random_state=0)



tree3.fit(X_train, y_train)

y_pred3 = tree3.predict(X_test)



print(classification_report(y_test, y_pred3))
X_resampled, y_resampled = SMOTEENN(random_state=0).fit_sample(dfX, dfy)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=0)

print(dfX.shape, X_resampled.shape, X_train.shape, X_test.shape)
tree3 = DecisionTreeClassifier(max_depth=6, random_state=0)



tree3.fit(X_train, y_train)

y_pred3 = tree3.predict(X_test)



print(classification_report(y_test, y_pred3))
from sklearn.ensemble import GradientBoostingClassifier



gb = GradientBoostingClassifier(random_state=0)

gb.fit(X_train,y_train)
pred = gb.predict(X_test)

print("정확도 : {0: 3f}".format(accuracy_score(y_test, pred)))
gb_param_grid = {

    'n_estimators' : [100, 200, 300, 400],

    'max_depth' : [6, 8, 10, 12],

    'min_samples_leaf' : [3, 5, 7, 10],

    'min_samples_split' : [2, 3, 5, 10],

    'learning_rate' : [0.05, 0.1, 0.2]

}
#gb_grid = GridSearchCV(gb, param_grid = gb_param_grid, scoring='accuracy', n_jobs = -1, verbose = 1)

#gb_grid.fit(X_train,y_train)
from sklearn.ensemble import GradientBoostingClassifier



gbm = GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None, learning_rate=0.1,

                                 loss='deviance', max_depth=3, max_features=None, max_leaf_nodes=None,

                                 min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, 

                                 min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=100,

                                 random_state=0, subsample=1.0, tol=0.0001, validation_fraction=0.1,

                                 verbose=0, warm_start=False)

gbm.fit(X_train, y_train).score(X_test, y_test)
!pip install lightgbm
import lightgbm as lgbm

from lightgbm import LGBMClassifier



lgbm = LGBMClassifier(n_estimators=1000, num_leaves=50, subsample=0.8,

                      min_child_samples=60, max_depth=20)



evals = [(X_test, y_test)]



lgbm.fit(X_train, y_train, early_stopping_rounds=100, eval_metric='auc',

         eval_set=evals, verbose=True).score(X_test, y_test)
!pip install catboost
from catboost import CatBoostClassifier

catboost = CatBoostClassifier(iterations=1000, 

                           task_type="GPU",

                           devices='0:1')

catboost.fit(X_train, y_train).score(X_test, y_test)
from sklearn.metrics import auc

from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, gbm.predict_proba(X_test)[:,1])

auc(fpr, tpr)
def plot_roc_curve(fpr, tpr, model, color=None) :

    model = model + ' (auc = %0.3f)' % auc(fpr, tpr)

    plt.plot(fpr, tpr, label=model, color=color)

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

    plt.axis([0,1,0,1])

    plt.xlabel('FPR (1 - specificity)')

    plt.ylabel('TPR (recall)')

    plt.title('ROC curve')

    plt.legend(loc="lower right")
fpr_gbm, tpr_gbm, _ = roc_curve(y_test, 

                                  gbm.predict_proba(X_test)[:,1])

plot_roc_curve(fpr_gbm, tpr_gbm, 'gbm', 'darkgreen')



fpr_lgbm, tpr_lgbm, _ = roc_curve(y_test, 

                                  lgbm.predict_proba(X_test)[:,1])

plot_roc_curve(fpr_lgbm, tpr_lgbm, 'lgbm', 'hotpink')
new = pd.read_csv('kaggle_data/human_new.csv',encoding='cp949')

new.head()
new.rename(columns={'아이디':'ID'}, inplace=True)

new.rename(columns={'나이':'age'}, inplace=True)

new.rename(columns={'노동 계급':'Working Class'}, inplace=True)

new.rename(columns={'학력':'Academic Background'}, inplace=True)

new.rename(columns={'교육 수':'Education Number'}, inplace=True)

new.rename(columns={'혼인 상태':'Married'}, inplace=True)

new.rename(columns={'직업':'Job'}, inplace=True)

new.rename(columns={'관계':'Relationship'}, inplace=True)

new.rename(columns={'인종':'Race'}, inplace=True)

new.rename(columns={'성별':'Sex'}, inplace=True)

new.rename(columns={'자본 이득':'Capital Gain'}, inplace=True)

new.rename(columns={'자본 손실':'Capital Loss'}, inplace=True)

new.rename(columns={'주당 시간':'Hours Per Week'}, inplace=True)

new.rename(columns={'모국':'Country'}, inplace=True)
new.dropna()
new['age_level'] = new['age'].apply(lambda x : 1 if x < 20

                                 else 2 if 20 <= x < 40

                                 else 3 if 40 <= x < 60

                                 else 4)
new['Hours Per Week_level'] = new['Hours Per Week'].apply(lambda x : 1 if x < 25

                                 else 2 if 25 <= x < 40

                                 else 3 if 40 <= x < 60

                                 else 4)
new['Capital'] = new['Capital Gain'] - new['Capital Loss']
new['Capital'] = new['Capital'].apply(lambda x: 2 if x > 0

                                   else 1 if x < 0

                                   else 0)
obj1 = ['Working Class', 'Academic Background', 'Married', 'Job', 'Relationship', 'Race', 'Country']

new[obj1] = new[obj1].apply(lambda x: x.astype('category').cat.codes)

new.head()
best_model = lgbm

lgbm.fit(X_train, y_train, early_stopping_rounds=100, eval_metric='auc',

         eval_set=evals, verbose=True).score(X_test, y_test)
new['Sex'] = best_model.predict(new.loc[:,'age':'Capital'])
print(best_model.predict_proba(new.loc[:,'age':'Capital']))

new['pred_prob'] = best_model.predict_proba(new.loc[:,'age':'Capital'])[:,1]

new
new['Sex'].value_counts()
new.to_csv("predict.csv", index=False, encoding = 'cp949')
new1 = new[['ID', 'Sex']]
new1.to_csv("predict7.csv", index=False, encoding = 'cp949')