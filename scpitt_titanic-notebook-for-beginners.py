import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from mlxtend.classifier import StackingCVClassifier



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
# Store our passenger ID for easy access

PassengerId = test['PassengerId']

y_train = train['Survived'].reset_index(drop=True)

X_train = train.drop(['Survived'], axis=1)

# Combine train and test set

df = pd.concat((X_train, test)).reset_index(drop=True)
sns.countplot(x = 'Survived', data = train)
df.isnull().sum()
plt.subplots(figsize=(10,10))

sns.set(font_scale=1.25)

sns.heatmap(df.corr(),square=True,annot=True)
age_by_pclass_sex = df.groupby(['Pclass', 'Sex']).median()['Age']

age_by_pclass_sex
df['Age'] = df.groupby(['Pclass','Sex'])['Age'].apply(lambda x: x.fillna(x.median()))
df['Fare'] = df.groupby(['Pclass'])['Fare'].apply(lambda x: x.fillna(x.median()))
df[df['Embarked'].isnull()]
df['Embarked'] = df['Embarked'].fillna('S')
df['title'] = df['Name'].str.split(', ', expand = True)[1].str.split('.',expand=True)[0]

df['title'].value_counts()
# combine titles

df['title'] = df['title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Mme', 'Mr', 'Master'], 'Ordinary')

df['title'] = df['title'].replace(['Lady', 'the Countess', 'Dona', 'Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Noble')
df['FamSize'] = df['SibSp'] + df['Parch'] + 1
sns.distplot(df['FamSize'])
df['IsAlone'] = df['FamSize'].apply(lambda x: 1 if x==1 else 0)
df['has_cabin'] = df['Cabin'].apply(lambda x: 1 if type(x) == str else 0)
df.info()
drop_cols = ['PassengerId', 'Name', 'Ticket','Cabin', 'SibSp', 'Parch']

df = df.drop(drop_cols, axis = 1)
df = pd.get_dummies(df).reset_index(drop=True)

# Split train and test

X_train = df.iloc[:len(y_train), :]

X_test = df.iloc[len(y_train):, :]

X_train.shape, X_test.shape, y_train.shape
X_train = StandardScaler().fit_transform(X_train)

y_train = y_train.values

X_test = StandardScaler().fit_transform(X_test)
SEED = 42



# random forest

rf = RandomForestClassifier(criterion='gini', 

                           n_estimators=1700,

                           max_depth=6,

                           min_samples_split=6,

                           min_samples_leaf=4,

                           max_features='auto',

                           random_state=SEED,

                           n_jobs=-1,

                           verbose=1)



#logistic regression

lr = LogisticRegression(penalty='l2', 

                        dual=False, 

                        tol=0.0001, 

                        C=1.0, 

                        fit_intercept=True, 

                        intercept_scaling=1,  

                        random_state=SEED, 

                        solver='lbfgs', 

                        max_iter=100, 

                        multi_class='auto', 

                        verbose=0, 

                        n_jobs=-1)



# support vector machine

svm = SVC(C=1, 

       kernel='rbf', 

       gamma='scale', 

       coef0=0.0, 

       cache_size=200, 

       class_weight=None, 

       verbose=False, 

       max_iter=-1, 

       decision_function_shape='ovr', 

       random_state=SEED,

         probability=True)



# gradient boosting

gb = GradientBoostingClassifier(loss='deviance', 

                                learning_rate=0.2, 

                                n_estimators=100, 

                                criterion='friedman_mse', 

                                min_samples_split=2, 

                                min_samples_leaf=2, 

                                max_depth=3, 

                                random_state=SEED,  

                                verbose=0)



# decision tree

dt = DecisionTreeClassifier(criterion='gini', 

                            max_depth = 4,

                            min_samples_split=2, 

                            min_samples_leaf=1, 

                            random_state=SEED)



# naive bayes

nb = GaussianNB()



# XGBoost

xgb = XGBClassifier(learning_rate=0.01,

                       n_estimators=6000,

                       max_depth=4,

                       min_child_weight=0,

                       gamma=0.6,

                       subsample=0.7,

                       colsample_bytree=0.7,

                       objective='binary:logistic',

                       nthread=-1,

                       scale_pos_weight=1,

                       seed=242,

                       reg_alpha=0.00006,

                       random_state=42)



# KNN

knn = KNeighborsClassifier(n_neighbors=10, 

                           weights='uniform', 

                           algorithm='auto', 

                           leaf_size=30, 

                           p=2, 

                           metric='minkowski', 

                           metric_params=None, 

                           n_jobs=None)
models = [rf, lr, svm, gb, dt, nb, xgb, knn]

scores = []

for model in models:

    score = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 5, scoring='roc_auc')

    scores.append(score.mean())
names = ['lr', 'svm', 'gb', 'dt', 'nb','rf','xgb', 'knn']

cv_score = pd.DataFrame(columns=['model', 'avg_cv_score'])

cv_score['model'] = names

cv_score['avg_cv_score'] = scores

cv_score
"""stack = StackingCVClassifier(classifiers = [rf, lr, dt, nb, knn, xgb, gb, svm],

                            meta_classifier = lr,

                            random_state = SEED)

accuracies = cross_val_score(estimator = stack, X = X_train, y = y_train, cv = 5)

accuracies.mean()

stack_model = stack.fit(X_train, y_train)

stack_model.predict(X_test)"""
best_model = svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])

submission_df['PassengerId'] = PassengerId

submission_df['Survived'] = y_pred

submission_df.to_csv('submissions.csv', header=True, index=False)