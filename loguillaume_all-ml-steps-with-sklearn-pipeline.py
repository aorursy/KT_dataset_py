#utils

import pandas as pd

import numpy as np

from math import ceil

import warnings

warnings.filterwarnings('ignore')



#visualization

import matplotlib.pyplot as plt

import seaborn as sns



#pipeline

from sklearn.pipeline import make_pipeline, make_union

from sklearn.compose import make_column_transformer, make_column_selector

from sklearn import set_config                      



#preprocessing

from sklearn.impute import KNNImputer, SimpleImputer

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, RobustScaler, StandardScaler, FunctionTransformer, KBinsDiscretizer



#feature selection

from sklearn.decomposition import PCA



#evaluation

from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve, GridSearchCV

from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc



#models

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from xgboost import XGBClassifier
filepath = '../input/titanic/train.csv'
data = pd.read_csv(filepath)

data.head(5)
print(f'The train set is composed of {data.shape[0]} observations and {data.shape[1]} features.')
print(f'Number of duplicated observations: {data.duplicated().sum()}')
print('Feature | % of dictinct values\n')

for feature in data.columns:

    print(f'{feature:>12}: {round(data[feature].nunique()/len(data), 3):>.2%}')
print('Feature | % of null values\n')

for feature in data.columns:

    print(f'{feature:>12}: {round(data[feature].isna().sum()/len(data), 3):>.2%}')
drop_features = ['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin']

X = data.drop(columns = drop_features)

y = data['Survived']
cat_feature = ['Pclass', 'Sex', 'Embarked']

num_feature = ['Age', 'SibSp', 'Parch', 'Fare']
plt.figure(figsize = (10, 10))

plt.suptitle('Discrete variables', weight = 'bold')

i = 0

for feature in cat_feature:

    i += 1 

    plt.subplot(ceil(len(cat_feature)/2), ceil(len(cat_feature)/2), i)

    sns.countplot(X[feature], hue = y)

    plt.title(feature, weight = 'bold')

plt.show()
plt.figure(figsize = (10, 10))

plt.suptitle('Continous variables', weight = 'bold')

i = 0

for feature in num_feature:

    i += 1 

    plt.subplot(ceil(len(cat_feature)/2), ceil(len(cat_feature)/2), i)

    sns.distplot(a = X[feature][y.to_numpy(dtype = bool)], hist = False, label = 'Survived', kde_kws = {'bw' : 2})

    sns.distplot(a = X[feature][np.invert(y.to_numpy(dtype = bool))], hist = False, label = 'Not survived', kde_kws = {'bw' : 2})

    plt.legend()

    plt.title(feature, weight = 'bold')

plt.show()
plt.figure(figsize = (16, 8))

plt.suptitle('Boxplot of continuous variables', weight = 'bold')

i = 0

for feature in num_feature:

    i += 1 

    plt.subplot(ceil(len(cat_feature)/2), ceil(len(cat_feature)/2), i)

    sns.boxplot(X[feature])

plt.show()
plt.figure()

plt.title('Features correlations heatmap', weight = 'bold')

sns.heatmap(pd.concat([X, y], axis = 1).corr())

plt.show()
pp = sns.pairplot(pd.concat([X, y], axis = 1))

pp.set(xticklabels=[])
X.info()
X = X.astype({name:'category' for name in cat_feature})
X.info()
cat_pipe = make_pipeline(SimpleImputer(strategy='most_frequent'),       

                                        OneHotEncoder(drop = 'first')) # Categorical features pipeline



num_pipe = make_pipeline(KNNImputer()) # Numerical features pipeline 



preprocessor = make_column_transformer((num_pipe, make_column_selector(dtype_exclude="category")),

                                       (cat_pipe, make_column_selector(dtype_include="category"))) #preprocessing pipeline
# from sklearn import set_config

set_config(display='diagram')

preprocessor
def plot_results(model_name, y_test, y_pred, N, train_score, val_score, fpr, tpr, roc_auc):

    plt.figure(figsize = (14, 6))

    plt.suptitle(f'{model_name} (accuracy: {round(accuracy_score(y_test, y_pred), 3)})', weight = 'bold')

    

    plt.subplot(1, 2, 1)

    plt.plot(N, train_score.mean(axis = 1), label = 'train', color = 'navy')

    plt.plot(N, val_score.mean(axis = 1), label = 'validation', color='darkorange')

    plt.title('Learning curve')

    plt.xlabel('Observations')

    plt.ylabel('Accuracy')

    plt.legend()



    plt.subplot(1, 2, 2)

    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC curve')

    plt.legend(loc="lower right")

    plt.show()   

    

    print(classification_report(y_test, y_pred))



def evaluation(X, y, preprocessing_pipeline, models: dict, feature_selection = 'passthrough', random_state = 0, test_size = 0.2):

    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state, stratify = y)

    cv = StratifiedKFold(n_splits=5, shuffle = True, random_state=random_state)

    

    for key, sub_key in models.items():

        

        pipe = make_pipeline(preprocessing_pipeline, feature_selection, sub_key.get('model')) #As default feature_selection is passthrough       

        param_grid = sub_key.get('param_grid')

        search = GridSearchCV(pipe, param_grid , cv = cv)

        search.fit(X_train, y_train)

        y_pred = search.best_estimator_.predict(X_test)

        y_pred_proba = search.best_estimator_.predict_proba(X_test)

                

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:,1])

        roc_auc = auc(fpr, tpr)

                

        N, train_score, val_score = learning_curve(search.best_estimator_,

                                                   X_train,

                                                   y_train,

                                                   cv = cv,

                                                   scoring = 'accuracy', 

                                                   train_sizes = np.linspace(0.1, 1, 10))



        plot_results(key, y_test, y_pred, N, train_score, val_score, fpr, tpr, roc_auc)

        print(f'Parameters: {search.best_params_}')

seed = 42

lr = {'LogisticRegression' : {'model': LogisticRegression(random_state = seed),

                              'param_grid': {}}}
evaluation(X, y, preprocessor, models = lr, random_state = seed)
X['Name'] = data.Name.astype(dtype = 'category')
title_names = X.Name.str.split(',', expand = True)[1].str.split('.', expand = True)[0].value_counts() > 10

print(title_names)
title_list = title_names.index[title_names].tolist()
def custom_title_transformer(x):

    x = x.iloc[:,0].str.split(',',expand=True)[1].str.split(".", expand=True)[0]

    x = x.apply(lambda x: 'Misc' if x not in title_list else x)

    return x.to_frame(name = 'Title') # .to_frame() because make_column_transformer need a dataframe object in output
title_transformer = make_column_transformer((FunctionTransformer(custom_title_transformer), ['Name']),

                                            remainder = 'passthrough')
is_alone_pipe = make_pipeline(FunctionTransformer(lambda x: x.sum(axis = 1).to_frame()),

                              FunctionTransformer(lambda x: (x >= 1).astype(int)))
cat_pipe = make_pipeline(title_transformer,

                         SimpleImputer(strategy = 'most_frequent'), 

                         OneHotEncoder(drop='first'))
family_transformer = make_column_transformer((FunctionTransformer(lambda x: x.sum(axis = 1).to_frame(name = 'Family')), ['SibSp', 'Parch']),

                                             remainder = 'passthrough') 

# .to_frame() because make_column_transformer need a dataframe object in output

# remainder = 'passthrough' to indicate not to drop the columns untransformed
num_2_cat_pipe = make_pipeline(KNNImputer(),

                               KBinsDiscretizer(n_bins = 5, strategy = 'quantile'))
num_pipe = make_pipeline(family_transformer,

                         KNNImputer(),

                         'passthrough')
feature_engineering = make_column_transformer((num_pipe, make_column_selector(dtype_exclude="category")),

                                              (cat_pipe, make_column_selector(dtype_include="category")))
param_grid_fe = {'columntransformer__pipeline-1__knnimputer__n_neighbors': np.arange(2, 11),

                 'columntransformer__pipeline-1__passthrough': [MinMaxScaler(),

                                                                StandardScaler(),

                                                                RobustScaler()]}
lr_feature_engineering = {'LogisticRegression': {'model': LogisticRegression(random_state=seed),

                                                 'param_grid': param_grid_fe}}
feature_engineering
evaluation(X, y, feature_engineering, models = lr_feature_engineering, random_state = seed)
pca = PCA()

pca_range = np.arange(1, feature_engineering.fit_transform(X).shape[1] + 1) # range until the number of feature after preprocessing

lr_pca = {'LogisticRegression' : {'model': LogisticRegression(random_state = seed),

                                'param_grid': {'columntransformer__pipeline-1__knnimputer__n_neighbors': [3], # I keep best parameters

                                               'columntransformer__pipeline-1__passthrough': [StandardScaler()],

                                               'pca__n_components': pca_range}}}
evaluation(X, y, feature_engineering, models = lr_pca, feature_selection= pca, random_state = seed)
param_grid_ms= {'columntransformer__pipeline-1__knnimputer__n_neighbors': [3],

                'columntransformer__pipeline-1__passthrough': [StandardScaler()]}
ensemble_models = {'RandomForestClassifier': {'model': RandomForestClassifier(random_state = seed),

                                              'param_grid': param_grid_ms},

                   'ExtraTreesClassifier': {'model': ExtraTreesClassifier(random_state = seed),

                                            'param_grid': param_grid_ms},

                   'XGBClassifier': {'model': XGBClassifier(random_state = seed),

                                     'param_grid': param_grid_ms}}
evaluation(X, y, feature_engineering, models = ensemble_models, feature_selection=pca, random_state = seed)
param_grid_xgbc = {'columntransformer__pipeline-1__knnimputer__n_neighbors': [3],

                   'columntransformer__pipeline-1__passthrough': [StandardScaler()],

                   'xgbclassifier__n_estimators': [50, 100, 1000],

                   'xgbclassifier__max_depth': [10, 50, 100],

                   'xgbclassifier__colsample_bytree':[0.8, 0,9, 1],

                   'xgbclassifier__learning_rate': [0.1, 0.05],

                   'xgbclassifier__subsample': [1, 0.8, 0.6]}                   
XGBC = {'XGBClassifier': {'model': XGBClassifier(random_state = seed),

                          'param_grid': param_grid_xgbc}}
evaluation(X, y, feature_engineering, models = XGBC, random_state = seed)
test = pd.read_csv('../input/titanic/test.csv')
final_model = XGBClassifier(random_state = 42,

                            learning_rate = 0.05,

                            max_depth = 10,

                            n_estimators = 100,

                            subsample = 0.6,

                            colsample_bytree = 0.8)
X_test = test.drop(columns = ['PassengerId', 'Cabin', 'Ticket'])

X_test = X_test.astype({name:'category' for name in ['Name', 'Embarked', 'Sex']})
num_pipe_test = make_pipeline(family_transformer, KNNImputer(n_neighbors=3), StandardScaler())
num_pipe_test = make_pipeline(family_transformer, KNNImputer(n_neighbors=3), StandardScaler())



feature_engineering_test = make_column_transformer((num_pipe, make_column_selector(dtype_exclude="category")),

                                                   (cat_pipe, make_column_selector(dtype_include="category")))



test_pipe = make_pipeline(feature_engineering_test, final_model)
test_pipe.fit(X, y)

y_pred = test_pipe.predict(X_test)
commit = pd.DataFrame(test['PassengerId'])

commit['Survived'] = y_pred
commit.to_csv('../working/submission.csv', index = False)