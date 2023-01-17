! pip install vtreat
! pip install pygam
#load packages

import sys #access to system parameters https://docs.python.org/3/library/sys.html

print("Python version: {}". format(sys.version))



import numpy as np

print("numpy version: {}". format(np.__version__))



import pandas as pd

print("pandas version: {}". format(pd.__version__))



import vtreat

print("vtreat version: {}". format(vtreat.__version__))



import ipykernel 

print("ipykernel version: {}". format(ipykernel.__version__))



import plotly 

print("plotly version: {}". format(plotly.__version__))



import seaborn as sns

print("seaborn version: {}". format(sns.__version__))



import shap

print("shap version: {}". format(shap.__version__))



import xgboost as xgb

print("xgboost version: {}". format(xgb.__version__))



import pandas_profiling

print("pandas_profiling: {}". format(pandas_profiling.__version__))
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer, MissingIndicator

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import cross_val_score, RandomizedSearchCV

from pygam import LogisticGAM, s, f, l
! ls ../input/titanic/
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

submission = pd.read_csv('../input/titanic/gender_submission.csv')

submission.dtypes
train.iloc[:100,:].profile_report()
# test.profile_report()
print('Features:')

print(train.columns.intersection(test.columns))

print('-'*80)

print('Target:')

print(train.columns.difference(test.columns))
train['Survived'].value_counts()
train = train.drop(['PassengerId'], axis = 1)

test = test.drop(['PassengerId'], axis = 1)
def downsample(df, label_col_name, mini_threshold=.1):

    nummin = df[label_col_name].value_counts().min()

    nummax = df[label_col_name].value_counts().max()

    nammax = df[label_col_name].value_counts().idxmax()

    nammin = df[label_col_name].value_counts().idxmin()

    

    if nummin/(nummin+nummax) < mini_threshold:

        return (pd.concat([df[df[label_col_name]==nammax].

                       sample(int(round(nummin*(1-mini_threshold)/mini_threshold))),

                       df[df[label_col_name]==nammin]]).

                       reset_index(drop=True))

    else:

        return df
train = downsample(train, 'Survived')
train.Survived.value_counts()
train.info()
train['title'] = train.Name.str.split(expand=True).iloc[:,1]

test['title'] = test.Name.str.split(expand=True).iloc[:,1]
class fulfill(ABC):

    

    def __init__(self, df):

        self.df = df

        super().__init__()

    

    @abstractmethod

    def impute():

        pass
class numfill(fulfill):

    

    def impute(self):

        num = self.df.select_dtypes(include=np.number)

        transform = vtreat.UnsupervisedTreatment(

            # cols_to_copy=[''],

            params=vtreat.unsupervised_parameters({

                "missingness_imputation": np.median,

            })

        )

        num_treated = transform.fit_transform(num)

        return num_treated

    

class charfill(fulfill):

    

    def impute(self):

        char = self.df.select_dtypes(include=np.object)

        return char.fillna('missing')
train_imputed = pd.concat([numfill(train).impute(), charfill(train).impute()], axis=1)

test_imputed = pd.concat([numfill(test).impute(), charfill(test).impute()], axis=1)
add = [var for var in train_imputed.columns if '_is_bad' in var]
char = test_imputed.select_dtypes(include=np.object)

num = test_imputed.select_dtypes(include=np.number)

minus = char.columns[char.nunique()>10].tolist()
char = char.drop(minus, axis=1)
char_dummy = pd.DataFrame()

for column in char.columns:

    dummy = pd.get_dummies(char[column])

    minus.append(column+'-'+dummy.columns[0])

    char_dummy = pd.concat([dummy.iloc[:,1:], char_dummy], axis=1)
test_trim = pd.concat([char_dummy, num], axis=1)
char = train_imputed.select_dtypes(include=np.object)

num = train_imputed.select_dtypes(include=np.number)

minus = char.columns[char.nunique()>10].tolist()
dropoff = [i.split('-')[-1] for i in minus]
char = char.drop(dropoff, axis=1)
char_dummy = pd.DataFrame()

for column in char.columns:

    dummy = pd.get_dummies(char[column])

    minus.append(column+'-'+dummy.columns[0])

    char_dummy = pd.concat([dummy.iloc[:,1:], char_dummy], axis=1)
train_trim = pd.concat([char_dummy, num], axis=1)
for i in train_trim.columns:

    if train_trim[i].value_counts(normalize=True).reset_index(drop=True)[0]>.95:

        minus.append(i)

        train_trim.drop(i, axis=1, inplace=True)
# for i in test_trim.columns:

#     if test_trim[i].value_counts(normalize=True).reset_index(drop=True)[0]>.95:

#         minus.append(i)

#         test_trim.drop(i)
test_clean = test_trim.drop(['Fare_is_bad'], axis=1)
minus
X, y = train_trim.drop(['Survived'], axis=1), train_trim['Survived']
steps = [('scaler', StandardScaler()), ('xgb_model', xgb.XGBClassifier())]

xgb_pipeline = Pipeline(steps)
xgb_param_grid = {

    'xgb_model__learning_rate': np.arange(.05, 1, .05),

    'xgb_model__max_depth': np.arange(3,10, 1),

    'xgb_model__n_estimators': np.arange(50, 200, 50)

}



# Perform RandomizedSearchCV

randomized_roc_auc = RandomizedSearchCV(estimator=xgb_pipeline,

                                        param_distributions=xgb_param_grid,

                                        n_iter=5, scoring='accuracy', cv=5, verbose=1)
randomized_roc_auc.estimator.get_params().keys()
# Fit the estimator

randomized_roc_auc.fit(X, y)



# Compute metrics

print(randomized_roc_auc.best_score_)

print(randomized_roc_auc.best_estimator_)
randomized_roc_auc.refit
submission['Survived'] = pd.DataFrame(randomized_roc_auc.predict(test_clean)).astype('int')

print(submission.dtypes)

submission.to_csv('submission1.csv', index=False)
# load JS visualization code to notebook

shap.initjs()



# train XGBoost model

model = xgb.train({"learning_rate": 0.01}, xgb.DMatrix(X, label=y), 100)



# explain the model's predictions using SHAP

# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)



# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)

shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
shap.force_plot(explainer.expected_value, shap_values[0:10,:], X.iloc[0:10,:])
gam = LogisticGAM(f(0) + f(1) + f(2) + f(3) + l(4) + s(5) + s(6) + s(7) + s(8)).gridsearch(X.to_numpy(), y.to_numpy())



fig, axs = plt.subplots(1, 9)

titles = ['Q', 'S', 'male', 'Age_is_bad', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']



for i, ax in enumerate(axs):

    XX = gam.generate_X_grid(term=i)

    pdep, confi = gam.partial_dependence(term=i, width=.95)



    ax.plot(XX[:, i], pdep)

    ax.plot(XX[:, i], confi, c='r', ls='--')

    ax.set_title(titles[i]);
gam.accuracy(X, y)
gam.summary()
# gam.predict(test_clean.to_numpy())
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

submission = pd.read_csv('../input/titanic/gender_submission.csv')

submission.dtypes
train.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)

test.drop(['PassengerId','Ticket','Cabin'],axis=1,inplace=True)
X, y = train.drop(['Survived'], axis=1), train['Survived']
categorical_feature_mask =X.dtypes==object

categorical_features = X.columns[categorical_feature_mask].tolist()



numeric_feature_mask = X.dtypes!=object

numeric_features = X.columns[numeric_feature_mask].tolist()
features = X.columns.to_list()
categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy="most_frequent")),

    ('onehot', OneHotEncoder(handle_unknown='ignore')),

])



numeric_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler()),

])



miss_ind = Pipeline(steps=[

    ('indicator', MissingIndicator(error_on_new=False)),

])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numeric_transformer, numeric_features),

        ('cat', categorical_transformer, categorical_features),

        ('ind', miss_ind, features)

    ]

)
class RemoveLowInfo(BaseEstimator, TransformerMixin):

    def __init__(self, threshold):

        self.threshold = threshold



    def fit(self, X, y=None):

        return self



    def transform(self, X):

        df = pd.DataFrame(X)

        keep = [column for column in df.columns if df[column].value_counts(normalize=True).reset_index(drop=True)[0]<self.threshold]

        return df[keep].to_numpy()
transformer = Pipeline([

         ('preprocessor', preprocessor),

         ('removelowinfo', RemoveLowInfo(threshold=0.99))])
pd.DataFrame(transformer.fit_transform(X))
from sklearn.linear_model  import LogisticRegression, SGDClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.decomposition import PCA

import xgboost as xgb

import lightgbm as lgb

import catboost as cgb
cv = 5

methods = [

           ('logistic', LogisticRegression(solver='lbfgs')), 

           ('sgd', SGDClassifier()), 

           ('tree', DecisionTreeClassifier()),

           ('bag', BaggingClassifier()),

           ('xgb', xgb.XGBClassifier(max_depth=3)),

           ('lgb', lgb.LGBMClassifier(max_depth=3)),

           ('cgb', cgb.CatBoostClassifier(max_depth=3,silent=True)),

           ('ada', AdaBoostClassifier()),

           ('gbm', GradientBoostingClassifier()),

           ('rf', RandomForestClassifier(n_estimators=100)),

           ('svc', LinearSVC()),

           ('rbf', SVC()),

           ('nb', Pipeline([('pca', PCA()), ('gnb', GaussianNB())])),

           ('nn', MLPClassifier()),

           ('knn', KNeighborsClassifier()),

          ]
results = []



for method in methods:

    clf = Pipeline([

         ('transformer', transformer),

#          ('pca', PCA()),

         method

    ])



    # Perform cross-validation

    cross_val_scores = cross_val_score(clf, X, y, scoring="accuracy", cv=cv)

    

    results.append([method[0], clf, cross_val_scores])

    

    # Print avg. AUC

    print(method[0], " ", cv, "-fold AUC: ", np.mean(cross_val_scores), sep="")
names = [result[0] for result in results]

scores = [result[2] for result in results]
# boxplot algorithm comparison

fig = plt.figure(figsize=(15,6))

fig.suptitle('Classifier Algorithm Comparison', fontsize=22)

ax = fig.add_subplot(111)

sns.boxplot(x=names, y=scores)

ax.set_xticklabels(names)

ax.set_xlabel("Algorithmn", fontsize=20)

ax.set_ylabel("Accuracy of Models", fontsize=18)

ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

plt.show()
from sklearn.ensemble import VotingClassifier
cv = 5

methods = [

#            ('logistic', LogisticRegression(solver='lbfgs')), 

#            ('sgd', SGDClassifier()), 

#            ('tree', DecisionTreeClassifier()),

#            ('bag', BaggingClassifier()),

           ('xgb', xgb.XGBClassifier(max_depth=3)),

           ('lgb', lgb.LGBMClassifier(max_depth=3)),

#            ('cgb', cgb.CatBoostClassifier(max_depth=3,silent=True)),

           ('ada', AdaBoostClassifier()),

           ('gbm', GradientBoostingClassifier()),

           ('rf', RandomForestClassifier(n_estimators=100)),

#            ('svc', LinearSVC()),

           ('rbf', SVC(gamma='auto')),

#            ('nb', Pipeline([('pca', PCA()), ('gnb', GaussianNB())])),

           ('nn', MLPClassifier()),

           ('knn', KNeighborsClassifier()),

          ]
ensemble = VotingClassifier(

        methods,

#         voting='soft', 

        weights=[1,1,1,1,2,2,1,1],

        flatten_transform=True,

)
clf = Pipeline([

     ('transformer', transformer),

     ('ensemble', ensemble)

])

np.mean(cross_val_score(clf, X, y, scoring="accuracy", cv=cv))
clf.fit(X,y)
submission['Survived'] = pd.DataFrame(clf.predict(test))

print(submission.dtypes)

submission.to_csv('submission2.csv', index=False)
# !pip install kaggle --upgrade
# !kaggle competitions submit -c titanic -f submission.csv -m "Message"
# # Use pickle to save model for next usage.

# filename = 'model_v1.pk'

# with open('./'+filename, 'wb') as file:

#     pickle.dump(pipe, file) 

# # Open saved model, and directly make the prediction with new data

# with open('./'+filename ,'rb') as f:

#     loaded_model = pickle.load(f)

# loaded_model.predict(X_test.loc[0:15])