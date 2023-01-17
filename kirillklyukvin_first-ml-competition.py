#common
import numpy as np
import pandas as pd 
import IPython
from IPython.display import display
import warnings
warnings.simplefilter('ignore')

#visualisation
import seaborn as sns
import matplotlib.pyplot as plt

#metrics and preprocessing
from sklearn.metrics import SCORERS
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, KFold, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import shuffle, resample
from sklearn.decomposition import PCA, IncrementalPCA

#classifiers
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


#tensorflow
import keras
from keras import initializers

from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from keras import utils
from keras.layers import Activation
from keras.regularizers import l2

from keras.callbacks import LearningRateScheduler
from keras.callbacks import History

from keras import losses
from sklearn.utils import shuffle
subm = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
train_df.head()
RND_ST = 42
train_ind = train_df['PassengerId'].index
test_ind = test_df['PassengerId']
test_ind = test_ind.tolist()
test_ind.append(891)
test_ind = np.sort(test_ind)
test_ind[:5]
full_df = pd.concat([train_df] + [test_df])
full_df.head()
train_df.info()
test_df.info()
total = full_df.isnull().sum().sort_values(ascending = False)
percent = round(full_df.isnull().sum().sort_values(ascending = False)/len(full_df)*100,2)
pd.concat([total, percent], axis=1, keys=['Total_missing','Percent_missing']).style.background_gradient(cmap='Reds')
full_df['Cabin'].unique()
print('Number of females in a train set is {}'.format(len(train_df.loc[train_df['Sex'] == 'female'])))
print('Number of males in a train set is {}'.format(len(train_df.loc[train_df['Sex'] == 'male'])))
temp_sex = train_df[['Survived', 'Sex']].groupby(['Sex'], as_index=False).mean()

sns.set_style('whitegrid')
plt.figure(figsize=(8,6))
sns.barplot(x='Sex', y='Survived', data=temp_sex)
plt.title('Difference in survived between males and women', size=15);
plt.figure(figsize=(10,6))

sns.distplot(full_df['Age'], color='darkgreen', bins=100)

plt.title('Distribution of passengers age', size=15);
plt.figure(figsize=(10,6))


sns.kdeplot(full_df.loc[full_df['Survived']==1,'Age'], label='Survived', shade=True)
sns.kdeplot(full_df.loc[full_df['Survived']==0,'Age'], label='Died', shade=True)

plt.title('Comparison of sirvived and died ages', size=15);
plt.figure(figsize=(10,8))
sns.scatterplot(x=train_df['Fare'], y=train_df['Age'], hue=train_df['Survived'], style=train_df['Sex'])
plt.title('Correlations between sex, ticket fare and age', size=15);
temp_pclass = train_df[['Pclass', 'Survived', 'Sex']].groupby(['Pclass', 'Sex'], as_index=False).mean()

plt.figure(figsize=(8,6))
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=temp_pclass)
plt.title('Difference in survival probability between Pclasses', size=15);
temp_pclass = train_df[['Pclass', 'Survived', 'Sex']].groupby(['Pclass', 'Sex'], as_index=False).mean()

sns.set_style('whitegrid')
plt.figure(figsize=(8,6))
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=temp_pclass)
plt.title('Difference in survival probability between classes', size=15);
temp_pclass = train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean()

plt.figure(figsize=(8,5))
sns.barplot(x='Parch', y='Survived', data=temp_pclass, color='darkgreen')

plt.xlabel('Number of parents/children')
plt.ylabel('Surviving probability')
plt.title('Survival probability of families with childer', size=15);
temp_sibsp = train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean()

plt.figure(figsize=(8,5))
sns.barplot(x='SibSp', y='Survived', data=temp_sibsp, color='darkgreen')

plt.xlabel('Number of siblings / spouses')
plt.ylabel('Surviving probability')
plt.title('Survival probability in different family` sizes', size=15);
full_df.loc[full_df['Sex'] == 'male', 'Sex'] = 1
full_df.loc[full_df['Sex'] == 'female', 'Sex'] = 0

full_df['Sex'] = full_df['Sex'].astype('int')
import re
def title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""


full_df['Title'] = full_df['Name'].apply(title)
full_df['Title'].value_counts()
full_df['Title'] = full_df['Title'].replace(['Don', 'Rev', 'Dr','Major', 'Col', 'Capt', 'Countess','Jonkheer', 'Dona'], 'untitled')
full_df['Title'].unique()
full_df.loc[full_df['Title']=='Master', 'Title'] = 'Mr'
full_df.loc[full_df['Title']=='Mlle', 'Title'] = 'Miss'
full_df.loc[full_df['Title']=='Mme', 'Title'] = 'Miss'
full_df.loc[full_df['Title']=='Ms', 'Title'] = 'Miss'
full_df.loc[full_df['Title']=='Sir', 'Title'] = 'Mr'
full_df.loc[full_df['Title']=='Lady', 'Title'] = 'Miss'
full_df['Title'].value_counts()
temp_title = full_df[['Title', 'Survived']].groupby('Title', as_index=False).mean()
temp_title
title_dict = dict(Mrs = 0, Miss = 1, untitled = 2, Mr = 3)

full_df['Title'] = full_df['Title'].replace(title_dict)

full_df = full_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
full_df.head()
full_df.query('Fare == "NaN"')
full_df['Fare'] = full_df.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(value=x.median()))
plt.figure(figsize=(12,6))

sns.distplot(full_df['Fare'], kde=False, bins=200, color='darkgreen')
plt.xlim(-1,100)
plt.title('Fare distribution in range 0-100 dollars', size=15);
def fare_into_bins(row):
    
    fare = row['Fare']
    
    if 0 <= fare < 15:
        return 'cheap'
    elif 15 <= fare < 42:
        return 'moderate'
    elif 42 <= fare < 65:
        return 'expensive'
    elif 65 <= fare < 100:
        return 'business'
    else:
        return 'luxury'
    
full_df['Fare_bins'] = full_df.apply(fare_into_bins, axis=1)
full_df[['Fare_bins', 'Survived']].groupby('Fare_bins',as_index=False).mean().sort_values(by='Survived', ascending=False)
full_df.loc[full_df['Embarked'].isna()]
full_df[['Fare_bins','Embarked','Survived']].groupby(['Embarked','Fare_bins'], as_index=False).mean().query('Fare_bins == "business"')
full_df['Embarked'] = full_df['Embarked'].fillna('C')
full_df_non_dum = full_df.copy()
full_df_non_dum = full_df_non_dum.drop('Fare', axis=1)
full_df_non_dum.head()
full_df = pd.get_dummies(full_df, drop_first=True)

full_df = full_df.drop('Fare', axis=1)
full_df.head()
full_df.loc[full_df['Age'] < 1, 'Age'] = 1
full_df['Age'] = full_df.groupby(['Sex','Pclass'])['Age'].transform(lambda x: x.fillna(value=x.median()))
full_df['Age'] = full_df['Age'].astype('int')
full_df.head()
full_df['Fam_size'] = full_df['SibSp'] + full_df['Parch'] + 1
full_df[['Fam_size', 'Survived']].groupby('Fam_size', as_index=False).mean()
full_df['Fam_size'] = full_df['Fam_size'].replace([5,6,7,8,11],1)
full_df[['Fam_size', 'Survived']].groupby('Fam_size', as_index=False).mean()
def alone(row):
    
    size=row['Fam_size']
    
    if size == 1:
        return 1
    else:
        return 0
    
full_df['Lon_trav'] = full_df.apply(alone, axis=1)
full_df[['Lon_trav','Survived']].groupby('Lon_trav', as_index=False).mean()
full_df[['Pclass', 'Title', 'Survived']].groupby(['Pclass', 'Title'], as_index=False).mean().sort_values(by='Survived', ascending=False)
pf = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
res = pf.fit_transform(full_df[['Pclass', 'Title']]).astype('int')
poly_features = pd.DataFrame(res, columns=['Pclass^2', 'Title^2', 'Pc_T'])


full_df = full_df.reset_index(drop=True)
full_df = full_df.join(poly_features)
full_df['Pclass^2'] = full_df['Pclass^2']**2
full_df['Title^2'] = full_df['Title^2']**2
full_df.head()
full_df = full_df.drop(['SibSp','Parch'], axis=1)
full_df.isna().sum()
y_train = full_df.loc[:len(train_ind)-1, 'Survived'].astype('int')

X_train = full_df.query('index in @train_ind').drop(['PassengerId', 'Survived'], axis=1)
X_test = full_df.query('index in @test_ind').drop(['PassengerId', 'Survived'], axis=1)
X_train_aged = X_train.drop('Age', axis=1)
X_test_aged = X_train.drop('Age', axis=1)
MinMax = MinMaxScaler()

MinMax.fit(X_train)

X_train_scaled = MinMax.transform(X_train)
X_test_scaled = MinMax.transform(X_test)

MinMax.fit(X_train_aged)
X_train_aged_sc = MinMax.transform(X_train_aged)
X_test_aged_sc = MinMax.transform(X_test_aged)
#### Logistic Regression

lr = LogisticRegression(n_jobs=-1)


### Random Forest

rfc = RandomForestClassifier(random_state=RND_ST, n_jobs=-1)

params_rfc = dict(
                  n_estimators=range(150,600,25),
                  max_features=range(2,8),
                  max_depth=range(3,7), 
                  min_samples_split=[2,3,4])


### Nearest Neighbors

knn = KNeighborsClassifier(n_jobs=-1)

params_knn = dict(
                  metric = ['manhattan', 'minkowski'],
                  n_neighbors = range(5,15),
                  leaf_size=range(30,50,2))


### Ada Boost

ada = AdaBoostClassifier(random_state=RND_ST)

params_ada = dict(
                  n_estimators=range(100,500,25))


### Support Vector

svc = SVC(random_state=RND_ST)

params_svc = dict(
                  gamma = np.logspace(-6, -1, 5),
                  C = [0.1,1,10,100,1000],
                  tol=[1e-3, 1e-4, 1e-5])

### Linear Support Vector

lsv = LinearSVC(random_state=RND_ST)

params_lsv = dict(
                  C = [0.1, 1, 10, 100], 
                  penalty = ['l1', 'l2'],
                  max_iter = [1000,1500,2000])


### Gradient Boosting

gbc = GradientBoostingClassifier(random_state=RND_ST, max_depth=3)

params_gbc = dict(
                  n_estimators=range(10,500,10),
                  learning_rate=[0.01, 0.1, 1, 5, 10],
                  max_leaf_nodes=range(1,6))

### Stochaic Gradient Descent

sgd = SGDClassifier(random_state=RND_ST)

params_sgd = dict(
                  max_iter=[500,1000,1500],
                  tol=[1e-3, 1e-4, 1e-5])
clf = [rfc, lsv, knn, ada, svc, gbc, sgd]
clf_names = ['random_forest', 'linear_vector', 'near_neighbours', 'ada_boost', 'support_vector', 'gradient_boosting', 'stochaic_gradient_descent']
params = [params_rfc, params_lsv, params_knn, params_ada, params_svc, params_gbc, params_sgd]
def rs(model, params, feat, targ):
    
    search = RandomizedSearchCV(model, params, cv=7, scoring='f1', n_jobs=-1)
    search.fit(feat, targ)

    print(search.best_score_)
    print(search.best_params_)
def gs(model, params, feat, targ):
    
    gs = GridSearchCV(model, params, cv=7, scoring='f1', n_jobs=-1)
    
    gs.fit(feat, targ)

    print(gs.best_score_)
    print(gs.best_params_)
for clf_, name, param in zip(clf, clf_names, params):
    
    print(name)
    rs(clf_, param, X_train, y_train)
    print()
for clf_, name, param in zip(clf, clf_names, params):
    
    print(name)
    rs(clf_, param, X_train_aged, y_train)
    print()
gs(knn, params_knn, X_train_scaled, y_train)
gs(svc, params_svc, X_train, y_train)
gs(lsv, params_lsv, X_train_scaled, y_train)
gs(lsv, params_lsv, X_train, y_train)
gbc = GradientBoostingClassifier(random_state=RND_ST, max_depth=3)

params_gbc_estim = dict(
                  n_estimators=range(10,500,5))

params_gbc_learn_rate = dict(
                  learning_rate=[0.01, 0.1, 0.5, 1, 10],
                  min_samples_split=[2,3,4],
                  min_samples_leaf=[1,2,3],
                  )
gbc_1 = GradientBoostingClassifier(random_state=RND_ST, max_depth=3, n_estimators=85)
gs(gbc_1, params_gbc_learn_rate, X_train, y_train)

# 0.7803240907336154
model_gbc_new = GradientBoostingClassifier(random_state=RND_ST, max_depth=3, n_estimators=85, learning_rate=0.5, min_samples_leaf=3)
model_lsv = LinearSVC(C=0.1, max_iter=2000, penalty='l2', random_state=RND_ST)
model_lsv_1 = LinearSVC(C=0.1, max_iter=1000, penalty='l2', random_state=RND_ST)
model_svc = SVC(tol=0.0001, gamma=0.1, random_state=RND_ST)
model_gdc_1 = GradientBoostingClassifier(
    n_estimators=275, min_samples_split=2, min_samples_leaf=4, max_depth=4, learning_rate=0.1, random_state=RND_ST)
model_gdc_1 = GradientBoostingClassifier(
    n_estimators=130, max_depth=3, learning_rate=0.1, random_state=RND_ST)
model_svc_1 = SVC(C=1, gamma=0.1, tol=0.001, random_state=RND_ST)
def plot_feature_importances(model, feat, targ):
    
    model.fit(feat, targ)
    n_features = feat.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), feat.columns)
    plt.xlabel("Features importance")
    plt.ylabel("Features")

plot_feature_importances(model_gdc_1, X_train, y_train)
from catboost import CatBoostClassifier, Pool, cv
import lightgbm as lgb
full_df_cat = (
    full_df[['PassengerId', 'Survived','Pclass','Sex','Age','Title','Fam_size','Lon_trav','Pclass^2','Title^2','Pc_T']]
    .merge(full_df_non_dum[['PassengerId','Fare_bins']], on='PassengerId')
)
full_df_cat.head(3)
X_train_cat = full_df_cat.query('index in @train_ind').drop(['PassengerId', 'Survived'], axis=1)
X_test_cat = full_df_cat.query('index in @test_ind').drop(['PassengerId', 'Survived'], axis=1)

X_train_cat_age = full_df_cat.query('index in @train_ind').drop(['PassengerId', 'Survived', 'Age'], axis=1)
X_test_cat_age = full_df_cat.query('index in @test_ind').drop(['PassengerId', 'Survived', 'Age'], axis=1)

y_train_cat = full_df_cat.loc[:len(train_ind)-1, 'Survived'].astype('int')
cat_features = ['Pclass', 'Sex', 'Title', 'Fam_size', 'Lon_trav', 'Pclass^2',
       'Title^2', 'Pc_T', 'Fare_bins']
train_pool = Pool(X_train_cat,
                 y_train_cat,
                 cat_features=cat_features)

test_pool = Pool(X_test_cat,
                 cat_features=cat_features)

train_pool_age = Pool(X_train_cat_age,
                 y_train_cat,
                 cat_features=cat_features)

test_pool_age = Pool(X_test_cat_age,
                 cat_features=cat_features)
cbr = CatBoostClassifier(iterations=2500,
                         depth=2,
                         learning_rate=1,
                         loss_function='Logloss',
                         random_seed=RND_ST,
                         verbose=200)

cbr.fit(train_pool)

y_pred = cbr.predict(X_train_cat)

f1_score(y_train_cat, y_pred)
cbr.fit(train_pool_age)

y_pred_ = cbr.predict(X_train_cat_age)

f1_score(y_train_cat, y_pred)
len(X_train_cat_age)
y_proba = cbr.predict_proba(train_pool)[:,1]
roc_auc_score(y_train, y_proba)
y_proba = cbr.predict_proba(train_pool_age)[:,1]
roc_auc_score(y_train, y_proba)
pred_final = pd.DataFrame(cbr.predict(test_pool_age), columns=['Survived'])
submission = pd.DataFrame(test_df['PassengerId'])
submission = submission.join(pred_final)
submission.to_csv('/kaggle/working/titanic_sub_catboost_age.csv', index=False)
submission.head()
full_df.columns
feat_1 = full_df[['Pclass', 'Sex', 'Age', 'Title',
       'Embarked_Q', 'Embarked_S', 'Fare_bins_cheap', 'Fare_bins_expensive',
       'Fare_bins_luxury', 'Fare_bins_moderate', 'Fam_size', 'Lon_trav']]

X_train_1 = feat_1.query('index in @train_ind')
X_test_1 = feat_1.query('index in @test_ind')
ss = StandardScaler()
ss.fit(X_train_1)

X_train_1_ss = ss.transform(X_train_1)
X_test_1_ss = ss.transform(X_test_1)
ss.fit(X_train)

X_train_ss = ss.transform(X_train)
X_test_ss = ss.transform(X_test)
def plot_hist(history):

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid()
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid()
    plt.show()
optimizer = RMSprop(learning_rate=0.0001,
    rho=0.9,
    epsilon=1e-07)
optimizer = Adam(lr=0.0001)
kernel_initializer='lecun_uniform'
kernel_initializer='RandomNormal'
X_train_ss.shape
try:
    del model3
    print('refined')
except:
    print('next')

optimizer = optimizer

model3 = Sequential()

model3.add(Dense(200, input_dim=15, activation='relu', kernel_initializer=kernel_initializer))
model3.add(Dense(100, activation='relu', kernel_initializer=kernel_initializer))
model3.add(Dense(50, activation='relu', kernel_initializer=kernel_initializer))

model3.add(Dense(1, activation='sigmoid', kernel_initializer=kernel_initializer))

model3.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
history = model3.fit(X_train_ss, y_train, epochs=1000, validation_split=0.1, batch_size=100, verbose=0)
plot_hist(history)
pred = model3.predict(X_train_ss).round()
roc_auc_score(y_train, pred).round(4)
model3.summary()
def prediction(model, feat_tr, feat_test, targ):
    
    model.fit(feat_tr, targ)
    

    pred_final = pd.DataFrame(model.predict(feat_test), columns=['Survived'])
 
    return pred_final
pred = prediction(model_gbc_new, X_train, X_test, y_train)
submission = pd.DataFrame(test_df['PassengerId'])

submission = submission.join(pred)

submission.to_csv('/kaggle/working/gbc_new.csv', index=False)
pred_final = model3.predict(X_test_ss).round().astype('int')

submission = pd.DataFrame(test_df['PassengerId'])

submission['Survived'] = pred_final

submission.to_csv('/kaggle/working/keras09.csv', index=False)
submission
