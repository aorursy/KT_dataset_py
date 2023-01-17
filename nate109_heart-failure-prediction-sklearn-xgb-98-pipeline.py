# import packages

import numpy as np #n-dimensional array operations

import scipy as sp #fundamental scientific computing

import pandas as pd #data manipulation and analysis

from matplotlib import pyplot as plt #plotting and visualization

import seaborn as sns #easy and beautiful data visualization

%matplotlib inline



# data augmentation

from sklearn.utils import resample



# preprocess categorical data

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder



# preprocess numerical data

from sklearn.preprocessing import StandardScaler, RobustScaler # use robust because we are dealing with outliers



from sklearn.model_selection import train_test_split



# prediction models

from sklearn.pipeline import Pipeline

from sklearn.compose import make_column_selector as selector

from sklearn.compose import ColumnTransformer

from xgboost import XGBClassifier

import lightgbm

from sklearn.ensemble import ExtraTreesClassifier



# model selection

from sklearn.model_selection import GridSearchCV



# report

from sklearn.metrics import classification_report, plot_confusion_matrix, roc_auc_score, accuracy_score
df = pd.read_csv("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")

cat_cols = ['anaemia', 'diabetes', 'high_blood_pressure', 'smoking', 'sex']

num_cols = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']

y_col = ['DEATH_EVENT']

all_cols = cat_cols.copy()

all_cols.extend(num_cols)

all_cols.sort()

# changing categorical columns to pandas 'category' datatype

df[cat_cols] = df[cat_cols].astype('category')
df.head()
df.describe()
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

sns.boxplot(df["creatinine_phosphokinase"], ax=axes[0, 0], palette="Reds")

sns.boxplot(df["ejection_fraction"], ax=axes[0, 1], palette="Reds")

sns.boxplot(df["platelets"], ax=axes[1, 0], palette="Reds")

sns.boxplot(df["serum_creatinine"], ax=axes[1,1], palette="Reds")
df.DEATH_EVENT.value_counts()
X, y = df[all_cols], df[y_col]
forest = ExtraTreesClassifier()

forest.fit(X, y)

print(forest.feature_importances_)

importances = pd.Series(forest.feature_importances_, index=X.columns)

importances.nlargest(12).plot(kind='barh', colormap='Reds_r')

plt.show()
features = ['time','ejection_fraction','serum_creatinine','age', 'serum_sodium']

X, y = df[features], df[y_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2698)
y_train.DEATH_EVENT.value_counts()
train = pd.concat([X_train, y_train])

train_maj = df[df.DEATH_EVENT==0]

train_min = df[df.DEATH_EVENT==1]

train_maj_upsampled = resample(train_maj,

                           replace=True,

                           n_samples=506,

                           random_state=1)

train_min_upsampled = resample(train_min,

                           replace=True,

                           n_samples=506,

                           random_state=1)

train = pd.concat([train_maj_upsampled, train_min_upsampled])

X_train = train[features]

y_train = pd.DataFrame(train.DEATH_EVENT)
y_train.DEATH_EVENT.value_counts()
cat_transformer = Pipeline(steps=[

    #('ordinal', OrdinalEncoder()),

    ('onehot', OneHotEncoder())

])

num_transformer = Pipeline(steps=[

    ('scaler', RobustScaler()),

])

preprocessor = ColumnTransformer(

    transformers=[

        ('num', num_transformer, selector(dtype_exclude='category')),

        ('cat', cat_transformer, selector(dtype_include='category'))

    ]

)



clf = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('classifier', XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

                        gamma=0, learning_rate=0.05, max_delta_step=0,

                        max_depth = 3, n_estimators=100, colsample_bytree=0.3, 

                        random_state=0

                        ))

])
param_grid = {'classifier__learning_rate':[0.05],

              'classifier__colsample_bytree':[0.3],

              'classifier__max_depth':[3, 5, 15]

              

             }  

grid_search = GridSearchCV(clf, param_grid, cv=10)

grid_search.fit(X_train, y_train.values.ravel())

print("best {estimator} from grid search: {accuracy}".format(estimator='XGB', accuracy=grid_search.score(X_test, y_test)))
pred = grid_search.predict(X_test)

prob = grid_search.predict_proba(X_test)

roc_auc = roc_auc_score(y_test, pred)

print("AUC: ", roc_auc)

print("Accuracy: ", accuracy_score(y_test, pred))

plot_confusion_matrix(grid_search, X_test, y_test, cmap='Reds')
print(grid_search.best_estimator_)