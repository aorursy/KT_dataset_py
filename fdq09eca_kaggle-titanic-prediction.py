# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
print('train set has {r} row, {c} columns'.format(r = df_train.shape[0], c = df_train.shape[1]))

df_train.head()
print('test set has {r} row, {c} columns'.format(r = df_test.shape[0], c = df_test.shape[1]))

df_test.head()
def missing_rate(df):

    miss_obs = df.isnull().sum()

    miss_col_rate = (df.isnull().sum()/df.shape[0])[miss_obs>0]

    return miss_col_rate.sort_values(ascending=False)
print ('df_train missing rate')

missing_rate(df_train)
print ('df_test missing rate:')

missing_rate(df_test)
num_cols = df_train._get_numeric_data().columns

non_num_cols = [c for c in df_train.columns if c not in num_cols]

print('numeric columns: ', num_cols.tolist())

print('non-numeric columns: ', non_num_cols)
df_train['Survived'].value_counts(normalize=True)
# exploring the non numeric columns

df_train['Cabin'].value_counts()

# It could be futher extract information to Class and seat number
df_train.columns.tolist().index('Cabin')

df_train.iloc[:,10]
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.compose import ColumnTransformer



def n_px(df):

#     cabin = df.columns.tolist().index('Cabin')

#     cabin_list = df.iloc[:, cabin].str.split()

#     imp_50p.fit_transform(df['Cabin'])

    cabin_list = df['Cabin'].str.split()

    df['n_px'] = cabin_list.map(len, na_action='ignore')

    return df

def px_seat(df):

#     cabin = df.columns.tolist().index('Cabin')

#     cabin_list = df.iloc[:, cabin].str.split()

    cabin_list = df['Cabin'].str.split()

    df['seat_r'] = cabin_list.map(lambda x: x[0][0], na_action='ignore')

    return df

def title(df):

    df['title'] = df['Name'].str.extract(r',\s(.*?).\s')

    return df

def alone(df):

    df['fam_n'] = df[['SibSp','Parch']].sum(axis = 'columns')

    df['alone'] = df['fam_n']==0

    return df



add_new_features = Pipeline(steps = [

    ('add passenger number', FunctionTransformer(n_px, validate = False)),

    ('add seat alpha', FunctionTransformer(px_seat, validate = False)),

    ('add title', FunctionTransformer(title, validate = False)),

    ('add alone', FunctionTransformer(alone, validate = False)) 

])





num_pip = Pipeline(steps = [

    ('impute', SimpleImputer(strategy = 'most_frequent')),

    ('std_scaler', StandardScaler())

])



cat_pip = Pipeline(steps = [

    ('impute', SimpleImputer(strategy = 'most_frequent')),

    ('OneHot', OneHotEncoder(handle_unknown='ignore'))

])



cols_tr = ColumnTransformer([

    ('num', num_pip, ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'fam_n']),

    ('cat', cat_pip, ['Sex', 'Embarked','alone', 'title'])

])



full_pip = Pipeline(steps = [

    ('new_feats', add_new_features),

    ('cols_transform', cols_tr)

])



full_pip.fit(df_train.drop(columns = ['Survived']))



df_train_tr = full_pip.transform(df_train.drop(columns = ['Survived']))

full_pip.transform(df_test)



cat_cols = cols_tr.named_transformers_['cat'].named_steps['OneHot'].get_feature_names().tolist()

df_train_tr = pd.DataFrame(df_train_tr, columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'fam_n'] + cat_cols)
def report_clf(y, y_hat):

    for func in [f1_score, recall_score, precision_score, accuracy_score]:

        print(f'{func.__name__}: {func(y, y_hat)}')

    return pd.DataFrame(confusion_matrix(y, y_hat), columns = ['p_0','p_1'], index= ['a_0','a_1'])
from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix
lr = LogisticRegression(random_state=42, max_iter = 500)

rf = RandomForestClassifier(random_state=42)

svm = SVC(random_state=42)

knn = KNeighborsClassifier()



pipe = Pipeline([

    ('pca', PCA(random_state=42)),

    ('est', lr)

])



lr_params = {

    'est': [lr],

    'est__C': np.linspace(.1,1,10),

    'est__solver': ['newton-cg', 'lbfgs', 'liblinear'],

    'est__class_weight': ['balanced', None]

            }



rf_params = {

    'est': [rf],

    'est__n_estimators': range(100, 601, 200),

    'est__max_depth': range(5, 10, 2),

    'est__criterion': ['gini', 'entropy']

}



svm_params = {

    'est': [svm],

    'est__kernel': ['rbf','poly','sigmoid'],

    'est__gamma': ['scale','auto'],

    'est__decision_function_shape': ['ovo','ovr']

}



knn_params = {

    'est': [knn],

    'est__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],

    'est__p': [1, 2],

    'est__n_neighbors': range(5,15)

}

# svm_params = {'est': [svm(random_state=42)]}



search_space = [lr_params, svm_params, rf_params, knn_params]

gs = GridSearchCV(pipe, search_space, cv = 10, scoring = 'accuracy')
X_train, X_test, y_train, y_test = train_test_split(df_train_tr, df_train['Survived'], test_size=0.3, random_state=42, stratify = df_train['Survived'])

gs.fit(X_train, y_train)

print(f"'best_est': {gs.best_estimator_[1]}")

print(f"best_score: {gs.scoring} - {gs.best_score_}")

y_hat = gs.best_estimator_.predict(X_test)

report_clf(y_test, y_hat)
print(f'false positive rate over all prediction: {19/(146+19+25+78)}')
gs.cv_results_['std_test_score']
df_test_tr = full_pip.transform(df_test)

y_hat_final = gs.best_estimator_.predict(df_test_tr)

y_hat_proba_final = gs.best_estimator_.predict(df_test_tr)
submission = pd.DataFrame.from_dict({

    'PassengerId': df_test.PassengerId,

    'Survived': y_hat_final

})

submission.head(6)
print(f'predicted survival rate in test set {round(submission.Survived.value_counts(normalize=True)[1], 2)}')

print(f'observrd survival rate in train set {round(df_train.Survived.value_counts(normalize=True)[1], 2)}')
submission.to_csv('Submission.csv', index = False)