# визуализация

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# машинное обучение

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
import numpy as np # линейная алгебра

import pandas as pd # работа с данными

# обзор файлов для обучения и теста

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
f'train shape: {train_df.shape},   test shape: {test_df.shape},   submission shape: {submission.shape}'
submission.head(3)
train_df.head(3)
test_df.head(3)
train_df.info()

print('_'*40)

test_df.info()
train_df.describe()
train_df.describe(include=['O'])
test_df.describe()
test_df.describe(include=['O']) 
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_df[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Fare', bins=20)
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
f,ax = plt.subplots(figsize=(22,16))

sns.heatmap(train_df.drop(['PassengerId',  'Name', 'Ticket', 'Cabin'], axis=1).corr(),annot=True, linewidths=.1, fmt='.1f', ax=ax)

plt.show()
train_df.Name.shape, train_df.Name.nunique(), test_df.Name.shape, test_df.Name.nunique(), train_df.Name.append(test_df.Name).shape, train_df.Name.append(test_df.Name).nunique()
train_df.Ticket.shape, train_df.Ticket.nunique(), test_df.Ticket.shape, test_df.Ticket.nunique(), train_df.Ticket.append(test_df.Ticket).shape, train_df.Ticket.append(test_df.Ticket).nunique()
test_df.Ticket.nunique(), test_df[~test_df.Ticket.isin(train_df.Ticket)].Ticket.nunique()
train_df.Cabin.shape, train_df.Cabin.nunique(), test_df.Cabin.shape, test_df.Cabin.nunique(), train_df.Cabin.append(test_df.Cabin).shape, train_df.Cabin.append(test_df.Cabin).nunique()
test_df.Cabin.nunique(), test_df[~test_df.Cabin.isin(train_df.Cabin)].Cabin.nunique()
print("До удаления", train_df.shape, test_df.shape)

# ['PassengerId',  'Name', 'Ticket', 'Cabin']

train_df = train_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

train_df = train_df.set_index('PassengerId')

test_df = test_df.set_index('PassengerId')

"После", train_df.shape, test_df.shape
# не будем применять преобразования к признакам, которые категориальные, но реально имеющих зависимость от порядка цифр.

# То есть, к примеру, класс билета, является категориальным признаком, но при этом реально важно 1-й клас иил 3-й. А вот для пола порта это не важно

def cat_encode(df):

    df = df.copy()

    df = pd.concat([df, pd.get_dummies(df.Sex)], axis=1)

    df.drop('Sex', axis=1, inplace=True)

    df = pd.concat([df, pd.get_dummies(df.Embarked)], axis=1)

    df.drop('Embarked', axis=1, inplace=True)

    return df
def make_features():

    train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1

    test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1

    train_df['IsAlone'] = 0

    train_df.loc[train_df['FamilySize'] == 1, 'IsAlone'] = 1

    test_df['IsAlone'] = 0

    test_df.loc[test_df['FamilySize'] == 1, 'IsAlone'] = 1
def fill_nan():

    test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

    test_df['Age'].fillna(test_df['Age'].dropna().median(), inplace=True)

    train_df['Age'].fillna(train_df['Age'].dropna().median(), inplace=True)
train
from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

s = TSNE(n_components=2).fit_transform(X_train)

s1 = PCA(n_components=2).fit_transform(X_train)

cluster_colors = {0: 'black', 1: 'red',}

cluster_names = {0:'не выжил',1:'выжил',}



def make_pic(c, df, column='Survived'):

    x, y = c[:,0 ], c[:,1]

    plt.scatter(x, y,

                color=df[column].map(cluster_colors), marker='o')

    plt.title('Распределение выживших и не выживших')

    plt.grid(True, linestyle='-', color='0.75')

    plt.show()



make_pic(s, train)

make_pic(s1, train)
fill_nan()

train = cat_encode(train_df)

test = cat_encode(test_df)

X_train = train.drop("Survived", axis=1)

Y_train = train["Survived"]

X_test  = test

X_train.shape, Y_train.shape, X_test.shape
from sklearn.preprocessing import RobustScaler
# Logistic Regression



def logred_clf(X_train, Y_train, X_test):

    logreg = LogisticRegression()

    logreg.fit(X_train, Y_train)

    Y_pred = logreg.predict(X_test)

    acc_log = round(logreg.score(X_train, Y_train) * 100, 2)



    print(acc_log)

    return logreg, acc_log

logred, acc_log = logred_clf(X_train, Y_train, X_test)

# после нормализации данных

rs = RobustScaler().fit(X_train)

r_logred, r_acc_log = logred_clf(rs.transform(X_train), Y_train, rs.transform(X_test))

coeff_df = pd.DataFrame(train.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
# Support Vector Machines

def svc_clf(X_train, Y_train, X_test):

    svc = SVC()

    svc.fit(X_train, Y_train)

    Y_pred = svc.predict(X_test)

    acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

    print(acc_svc)

    return svc, acc_svc

svc, acc_svc = svc_clf(X_train, Y_train, X_test)

# после нормализации данных

r_svc, r_acc_svc = svc_clf(rs.transform(X_train), Y_train, rs.transform(X_test))

def knn_clf(X_train, Y_train, X_test):

    knn = KNeighborsClassifier(n_neighbors = 3)

    knn.fit(X_train, Y_train)

    Y_pred = knn.predict(X_test)

    acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

    print(acc_knn)

    return knn, acc_knn

knn, acc_knn = knn_clf(X_train, Y_train, X_test)

# после нормализации данных

r_knn, r_acc_knn = knn_clf(rs.transform(X_train), Y_train, rs.transform(X_test))
def naive_clf(X_train, Y_train, X_test):

    gaussian = GaussianNB()

    gaussian.fit(X_train, Y_train)

    Y_pred = gaussian.predict(X_test)

    acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

    print(acc_gaussian)

    return gaussian, acc_gaussian

naive, acc_gaussian = naive_clf(X_train, Y_train, X_test)

# после нормализации данных

r_naive, r_acc_gaussian = naive_clf(rs.transform(X_train), Y_train, rs.transform(X_test))
def dt_clf(X_train, Y_train, X_test):

    decision_tree = DecisionTreeClassifier()

    decision_tree.fit(X_train, Y_train)

    Y_pred = decision_tree.predict(X_test)

    acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

    print(acc_decision_tree)

    return decision_tree, acc_decision_tree

    

dt, acc_decision_tree = dt_clf(X_train, Y_train, X_test)

# после нормализации данных

r_dt, r_acc_decision_tree = dt_clf(rs.transform(X_train), Y_train, rs.transform(X_test))
def rf_clf(X_train, Y_train, X_test):

    random_forest = RandomForestClassifier(n_estimators=100)

    random_forest.fit(X_train, Y_train)

    Y_pred = random_forest.predict(X_test)

    random_forest.score(X_train, Y_train)

    acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

    print(acc_random_forest)

    return random_forest, acc_random_forest

random_forest, acc_random_forest = rf_clf(X_train, Y_train, X_test)

# после нормализации данных

r_random_forest, r_acc_random_forest = rf_clf(rs.transform(X_train), Y_train, rs.transform(X_test))
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
from sklearn import metrics

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split



from sklearn.model_selection import GridSearchCV
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
dt = DecisionTreeClassifier()

dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)



print(metrics.classification_report(y_test, y_pred))

print(metrics.confusion_matrix(y_test, y_pred))



def plot_roc_curve(clf, y_test, x_test):

    sns.set(font_scale=1.5)

    sns.set_color_codes("muted")



    plt.figure(figsize=(10, 8))

    fpr, tpr, thresholds = metrics.roc_curve(y_test, clf.predict_proba(x_test)[:,1], pos_label=1)



    lw = 2

    plt.plot(fpr, tpr, lw=lw, label='ROC curve ')

    plt.plot([0, 1], [0, 1])

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC curve')

    plt.savefig("ROC.png")

    plt.show()

plot_roc_curve(dt, y_test, x_test)
dt = DecisionTreeClassifier()

cv = cross_val_score(dt, X_train, Y_train, cv=5)

print(cv, np.mean(cv))
param_grid = {

    

    "criterion": ['entropy', 'gini'],

    "min_samples_split": [2, 3, 4,6, 8],

    "max_depth": [2,5,10,12,14, 16],

    "min_samples_leaf":[1, 2, 3, 4, 6]

    }

dt = GridSearchCV(DecisionTreeClassifier(random_state = 42), param_grid, n_jobs=-1, cv=5)

dt.fit(x_train, y_train)

best_params = dt.best_params_

print(dt.best_params_)

dt = dt.best_estimator_

y_pred = dt.predict(x_test)



print(metrics.classification_report(y_test, y_pred))

print(metrics.confusion_matrix(y_test, y_pred))

plot_roc_curve(dt, y_test, x_test)
dt_finall = DecisionTreeClassifier(**best_params)

dt_finall.fit(X_train, Y_train)

predict = dt_finall.predict(X_test)
predict
submission.shape, predict.shape
submission['Survived'] = predict

submission.head()
submission.to_csv('submission.csv', index=False)
import lightgbm as lgb

import catboost as cb
train_df

cat_feats = ['Pclass']

train_data = lgb.Dataset(X_train, label = Y_train, categorical_feature=cat_feats, free_raw_data=False)

fake_valid_inds = np.random.choice(len(X_train), 1000000)

fake_valid_data = lgb.Dataset(X_train.iloc[fake_valid_inds], label = Y_train.iloc[fake_valid_inds],categorical_feature=cat_feats,

                             free_raw_data=False)  

params = {

        "objective" : "poisson",

        "metric" :"rmse",

        "force_row_wise" : False,

        "learning_rate" : 0.075,

        "sub_row" : 0.75,

        "bagging_freq" : 1,

        "lambda_l2" : 0.1,

        "metric": ["binary_logloss"],

    'verbosity': 1,

    'num_iterations' : 500,

#     'device' : 'gpu'

}
%%time



m_lgb = lgb.train(params, train_data, valid_sets = [fake_valid_data], verbose_eval=50) 
predict = m_lgb.predict(X_test)

submission['Survived'] = predict

submission['Survived'] = submission['Survived'].apply(lambda r: 0 if r <=0.5 else 1)

submission.to_csv('submission_lg.csv', index=False)

submission.head()