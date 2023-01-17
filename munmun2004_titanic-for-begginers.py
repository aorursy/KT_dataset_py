import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = 'Malgun Gothic'
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

sample_submission = pd.read_csv('../input/titanic/gender_submission.csv')
train.head(3)
test.head()
train.info()
test.info()
train.isnull().sum()
test.isnull().sum()
fe_name = list(test)

df_train = train[fe_name]

df = pd.concat((df_train,test))
print(train.shape, test.shape, df.shape)
target = train['Survived']
def stack_plot(feature):

    survived = train[train['Survived'] == 1][feature].value_counts()

    dead = train[train['Survived'] == 0][feature].value_counts()

    df = pd.DataFrame([survived,dead])

    df.index = ['survived','dead']

    df.plot(kind='bar', stacked= True, figsize = (10,5))
lable = ['survived','dead']

plt.title('생존 수')

plt.pie(train['Survived'].value_counts(),labels= lable,autopct='%.f%%')
stack_plot('Pclass')
Pclass_encoded = pd.get_dummies(df['Pclass'],prefix= 'Pclass')

df = pd.concat((df,Pclass_encoded), axis=1)

df = df.drop(columns = 'Pclass')
stack_plot('Sex')
sex_encoded = pd.get_dummies(df['Sex'],prefix= 'Sex')

df = pd.concat((df,sex_encoded), axis=1)

df = df.drop(columns = 'Sex')
df.drop('Sex_female', axis=1, inplace=True)
stack_plot('SibSp')
stack_plot('Parch')
df['Travelpeople']=df["SibSp"]+df["Parch"]

df['TravelAlone']=np.where(df['Travelpeople']>0, 0, 1)
df.drop('SibSp', axis=1, inplace=True)

df.drop('Parch', axis=1, inplace=True)
df.drop('Travelpeople', axis=1, inplace=True)
df['New_name']  = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)

train['New_name']  = train['Name'].str.extract('([A-Za-z]+)\.', expand=False)
train ['New_name'] =  train['New_name'].map({"Mr": 0 , "Mrs":2, "Miss":1,"Dr":3,"Rev":3,

                                             "Mlle":3,"Major":3,"Col":3,"Ms":3,"Jonkheer":3,

                                             "Sir" :3,"Lady":3,"Mme":3,"Capt":3,"Don":3,"Countess":3})

df['New_name'] =  df['New_name'].map({"Mr": 0 , "Mrs":2, "Miss":1,"Dr":3,"Rev":3,

                                             "Mlle":3,"Major":3,"Col":3,"Ms":3,"Jonkheer":3,

                                             "Sir" :3,"Lady":3,"Mme":3,"Capt":3,"Don":3,"Countess":3})
stack_plot('New_name')
df['New_name'] = df['New_name'].fillna('0')
df = df.astype({'New_name':'float'})
df = df.drop(columns = 'Name')
New_name_encoded = pd.get_dummies(df['New_name'],prefix= 'New_name')

df = pd.concat((df,New_name_encoded), axis=1)

df = df.drop(columns = 'New_name')
df['Age'].hist(bins = 15)
df['Age'].fillna(28, inplace = True)
sns.countplot(x= 'Embarked', data= df)
df['Embarked'].fillna('S',inplace=True)
Embarked_encoded = pd.get_dummies(df['Embarked'],prefix= 'Embarked')

df = pd.concat((df,Embarked_encoded), axis=1)

df = df.drop(columns = 'Embarked')
from scipy.stats import norm
sns.distplot(train['Fare'],fit = norm)
df['Fare'] = df['Fare'].map(lambda i : np.log(i) if i >0 else 0)
sns.distplot(df['Fare'],fit = norm)
df['Cabin'].value_counts()
df = df.drop(columns = 'Cabin')
df = df.drop(['PassengerId','Ticket'],axis = 1)
df.isnull().sum()
from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
new_train = df[:train.shape[0]]

new_test = df[train.shape[0]:]
cols=["Age", "Fare", "TravelAlone", "Pclass_1", "Pclass_2","Embarked_C","Embarked_S",

      "Sex_male",'New_name_0.0','New_name_1.0', 'New_name_2.0'] 
X = new_train[cols]

Y = train['Survived']
import statsmodels.api as sm

from scipy import stats

stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

logit_model=sm.Logit(Y,X)

result=logit_model.fit()

print(result.summary())
cols2 = ["Age", "TravelAlone", "Pclass_1", "Pclass_2","Embarked_S",

      "Sex_male",'New_name_1.0', 'New_name_2.0'] 
X2=new_train[cols2]

Y=train['Survived']



logit_model=sm.Logit(Y,X2)

result=logit_model.fit()

print(result.summary())
cols3=["Age", "Pclass_1", "Pclass_2","Embarked_S",

      "Sex_male",'New_name_1.0', 'New_name_2.0'] 
X3=new_train[cols3]

Y=train['Survived']



logit_model=sm.Logit(Y,X3)

result=logit_model.fit()

print(result.summary())
f_test = new_test[cols3]
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(X3, Y)



print("모델 Accuracy : {:.2f}%".format(logreg.score(X3, Y)*100))
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
kfold = StratifiedKFold(n_splits=8)
random_state = 1

clf = []



clf.append(XGBClassifier(random_state = random_state))

clf.append(LGBMClassifier(random_state = random_state))

clf.append(KNeighborsClassifier())

clf.append(RandomForestClassifier(random_state=random_state))

clf.append(GradientBoostingClassifier(random_state=random_state))

clf.append(DecisionTreeClassifier(random_state=random_state))

clf.append(LogisticRegression(random_state = random_state))

clf.append(SVC(random_state=random_state))
clf_results = []

for classifier in clf :

    clf_results.append(cross_val_score(classifier, new_train, y = Y, scoring = "accuracy", cv = kfold, n_jobs=4))
clf_means = []

clf_std = []

for clf_result in clf_results:

    clf_means.append(clf_result.mean())

    clf_std.append(clf_result.std())
clf_re = pd.DataFrame({"CrossValMeans":clf_means,"CrossValerrors": clf_std})

clf_re
# XGBoost 파라미터 튜닝 

XGB = XGBClassifier()

xgb_param_grid = {'learning_rate': [1,0.1,0.01,0.001],

              'n_estimators': [50, 100, 200, 500, 1000],

              'max_depth' : [1,3,5,10,50]}

gsXGB = GridSearchCV(XGB,param_grid = xgb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsXGB.fit(new_train,Y)

XGB_best = gsXGB.best_estimator_



# 최고 점수

gsXGB.best_score_
#LGBMClassifier 파라미터 튜닝

LGB = LGBMClassifier()

lgb_param_grid = {

    'n_estimators': [400, 700, 1000], 

    'max_depth': [15,20,25],

    'num_leaves': [50, 100, 200],

    'min_split_gain': [0.3, 0.4],

}

gsLGB = GridSearchCV(LGB,param_grid = lgb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsLGB.fit(new_train,Y)

LGB_best = gsLGB.best_estimator_



# 최고 점수

gsLGB.best_score_
# RandomForestClassifier 파라미터 튜닝 

RFC = RandomForestClassifier()



rf_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 7],

              "min_samples_split": [2, 3, 7],

              "min_samples_leaf": [1, 3, 7],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}

gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(new_train,Y)

RFC_best = gsRFC.best_estimator_



# 최고 점수

gsRFC.best_score_
# Gradient boosting 파라미터 튜닝

GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1] 

              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsGBC.fit(new_train,Y)

GBC_best = gsGBC.best_estimator_



# 최고 점수

gsGBC.best_score_
test_Survived_XGB = pd.Series(XGB_best.predict(new_test), name="XGB")

test_Survived_LGB = pd.Series(LGB_best.predict(new_test), name="LGB")

test_Survived_RFC = pd.Series(RFC_best.predict(new_test), name="RFC")

test_Survived_GBC = pd.Series(GBC_best.predict(new_test), name="GBC")



ensemble_results = pd.concat([test_Survived_XGB,test_Survived_LGB,

                              test_Survived_RFC,test_Survived_GBC],axis=1)

g= sns.heatmap(ensemble_results.corr(),annot=True)
from sklearn.ensemble import VotingClassifier
votingC = VotingClassifier(estimators=[('XGB', XGB_best), ('LGB', LGB_best),

('RFC', RFC_best), ('GBC',GBC_best)], voting='soft', n_jobs=4)

votingC = votingC.fit(new_train, Y)  
test_Survived = pd.Series(votingC.predict(new_test), name="Survived")
submission = pd.DataFrame({

    "PassengerId" :test["PassengerId"],

    "Survived": test_Survived

})
submission.to_csv('voting_titanic.csv',index=False) 
from mlxtend.classifier import StackingClassifier

from sklearn.utils.testing import ignore_warnings
clf1 = XGB_best

clf2 = LGB_best

clf3 = RFC_best

clf4 = GBC_best



lr = LogisticRegression()

st_clf = StackingClassifier(classifiers=[clf1, clf1, clf2, clf3, clf4], meta_classifier=lr)

params = {'meta_classifier__C': [0.1,1.0,5.0,10.0] ,

          #'use_probas': [True] ,

          #'average_probas': [True] ,

          'use_features_in_secondary' : [True, False]

         }

with ignore_warnings(category=DeprecationWarning):

    st_clf_grid = GridSearchCV(estimator=st_clf, param_grid=params, cv=5, refit=True)

    st_clf_grid.fit(new_train, Y)

    st_clf_grid.best_score_
with ignore_warnings(category=DeprecationWarning):    

    pred_all_stack = st_clf_grid.predict(new_test)



submission1 = pd.DataFrame({

    "PassengerId" :test["PassengerId"],

    "Survived": pred_all_stack

})

#submission1.to_csv('stack_clf.csv',index=False)