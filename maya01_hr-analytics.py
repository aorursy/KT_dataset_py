import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import matplotlib as mlp

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder,StandardScaler

from sklearn.pipeline import Pipeline

import plotly.express as px

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,VotingClassifier

from sklearn.svm import SVC

from sklearn.metrics import  accuracy_score, confusion_matrix, roc_auc_score, roc_curve

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import roc_auc_score,roc_curve,auc

from sklearn.model_selection import StratifiedKFold

from datetime import datetime

from random import random

from sklearn.compose import ColumnTransformer

try:

    import xgboost

except ImportError as ex:

    print('Xgboost in not installed on your system')

    xgboost = None
path = '../input/avhranalytics/'

train = pd.read_csv(path+'train_jqd04QH.csv')

test = pd.read_csv(path+'test_KaymcHn.csv')
MLP_XKCD_COLOR = mlp.colors.XKCD_COLORS

MLP_BASE_COLOR = mlp.colors.BASE_COLORS

MLP_CNAMES = mlp.colors.cnames

MLP_CSS4 = mlp.colors.CSS4_COLORS

MLP_HEX = mlp.colors.hexColorPattern

MLP_TABLEAU = mlp.colors.TABLEAU_COLORS

print('I like COLORS :>')

def random_color_generator(color_type=None):

    if color_type is None:

        colors = sorted(MLP_CNAMES.items(), key=lambda x: random())

    else:

        colors = sorted(color_type.items(), key=lambda x: random())

    return dict(colors)
def timer(start_time=None):

    if not start_time:

        start_time = datetime.now()

        return start_time

    elif start_time:

        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)

        tmin, tsec = divmod(temp_sec, 60)

        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
train.head(2)
colors = random_color_generator(MLP_BASE_COLOR)

train['target'].value_counts().plot(kind='bar',color=colors)

plt.title('Target Distribution')

plt.show()
fig = px.scatter(train, x="training_hours", y="city_development_index", color="target",

                 size='training_hours', hover_data=['city_development_index'])

fig.show()
print(train.shape)

print(test.shape)
full = train.append(test)

full.shape
full.info()
full.isnull().sum()
def fillna(col,typeX=None):

    if typeX:

        full[col].fillna('-1',inplace=True)

    else:

        full[col].fillna('Unknown',inplace=True)

    return full[col]
fillna('gender')

full.gender.value_counts()
fillna('relevent_experience')

full.relevent_experience.value_counts()
fillna('enrolled_university')

full.enrolled_university.value_counts()
fillna('education_level')

full.education_level.value_counts()
fillna('major_discipline')

full.major_discipline.value_counts()
fillna('experience','int')

full.experience = full['experience'].replace('>20','21')

full.experience = full['experience'].replace('<1','0')

full.experience = full['experience'].astype('int')

full.experience.value_counts()
full.experience.describe()
bin_labels = ['unknown','low', 'medium', 'high']

bins= [-1,0,7,14,25]

full['experience_cut'] = pd.cut(full['experience'], bins=bins, labels=bin_labels, right=False)
fillna('company_size','int')

full.company_size = full['company_size'].replace('<10','1')

full.company_size = full['company_size'].replace('10/49','2')

full.company_size = full['company_size'].replace('50-99','3')

full.company_size = full['company_size'].replace('100-500','4')

full.company_size = full['company_size'].replace('500-999','5')

full.company_size = full['company_size'].replace('1000-4999','6')

full.company_size = full['company_size'].replace('5000-9999','7')

full.company_size = full['company_size'].replace('10000+','8')

full.company_size = full['company_size'].astype('int')

full.company_size.value_counts()
bins= [-1,0,3,6,10]

labels = ['Unknown','small','medium','large']

full['company_size_cut'] = pd.cut(full['company_size'], bins=bins, labels=labels, right=False)
fillna('company_type')

full.company_type.value_counts()
fillna('last_new_job','int')

full.last_new_job = full['last_new_job'].replace('>4','5')

full.last_new_job = full['last_new_job'].replace('never','0')

full.last_new_job.value_counts()
full.isnull().sum()
full.city.value_counts()
full.city_development_index.value_counts()
full.city_development_index.describe()
bins= [0,.25,.75,1.]

labels = ['low','medium','high']

full['city_development_index_cut'] = pd.qcut(full['city_development_index'], q=bins, labels=labels)
full.training_hours = np.log(full.training_hours)
train.target.value_counts()
fig = px.box(train, x="target", y="city_development_index", points="all",color="target")

fig.show()
full.head(2)
train_1 = full[:train.shape[0]]

test_1 = full[train_1.shape[0]:]

print(train_1.shape)

print(test_1.shape)
sns.countplot(train_1['gender'],palette='Paired')

plt.title('Gender Distribution')

plt.show()
sns.countplot(train_1['relevent_experience'])

plt.title('Experience Distribution')

plt.show()
colors = random_color_generator()

train_1.enrolled_university.value_counts().plot(kind='bar',color=colors)

plt.title('Enrolled University Distribution')

plt.show()
colors = random_color_generator(MLP_XKCD_COLOR)

train_1.education_level.value_counts().plot(kind='bar',color=colors)

plt.title('Education Level Distribution')

plt.show()
colors = random_color_generator(MLP_TABLEAU)

train_1.major_discipline.value_counts().plot(kind='bar',color=colors)

plt.title('Major Discipline Distribution')

plt.show()
plt.figure(figsize=(12,8))

colors = random_color_generator(MLP_XKCD_COLOR)

train_1.experience.value_counts().plot(kind='bar',color=colors)

plt.title('Experience wise count')

plt.show()
colors = random_color_generator(MLP_XKCD_COLOR)

train_1.experience_cut.value_counts().plot(kind='bar',color=colors)

plt.title('Experience wise count')

plt.show()
colors = random_color_generator(MLP_TABLEAU)

train_1.company_size.value_counts().plot(kind='bar',color=colors)

plt.title('Company Size Distribution')

plt.show()
colors=random_color_generator(MLP_CNAMES)

train_1.company_type.value_counts().plot(kind='bar',color=colors)

plt.title('Company Type Distribution')

plt.show()
colors=random_color_generator(MLP_BASE_COLOR)

train_1.last_new_job.value_counts().plot(kind='bar',color=colors)

plt.title('Applicant Experience Distribution')

plt.show()
colors=random_color_generator()

fig = train_1.groupby('relevent_experience')['target'].value_counts().unstack().plot(kind='bar',color=colors)

plt.title('Applicant Experience Distribution')

fig.plot()
colors=random_color_generator()

train_1.groupby('enrolled_university')['target'].value_counts().unstack().plot.bar(stacked=True,color=colors)

plt.title('Enrolled University Distribution')

plt.plot()
colors=random_color_generator()

train_1.groupby('education_level')['target'].value_counts().unstack().plot.bar(stacked=True,color=colors)

plt.title('Education Level wise Distribution')

plt.plot()
colors=random_color_generator()

train_1.groupby('major_discipline')['target'].value_counts().unstack().plot.bar(stacked=True,color=colors)

plt.title('Major Discipline wise  Distribution')

plt.plot()
colors=random_color_generator()

train_1.groupby('experience')['target'].value_counts().unstack().plot.bar(stacked=True,color=colors)

plt.title('Experience wise Distribution')

plt.plot()
colors=random_color_generator()

train_1.groupby('experience_cut')['target'].value_counts().unstack().plot.bar(stacked=True,color=colors)

plt.plot()
colors=random_color_generator()

train_1.groupby('company_size')['target'].value_counts().unstack().plot(kind='bar',color=colors)

plt.title('Company Size wise Distribution')

plt.plot()
colors=random_color_generator()

train_1.groupby('company_size_cut')['target'].value_counts().unstack().plot.bar(stacked=True,color=colors)

plt.plot()
colors=random_color_generator()

train_1.groupby('company_type')['target'].value_counts().unstack().plot(kind='bar',color=colors)

plt.title('Company Type wise Distribution')

plt.plot()
colors=random_color_generator()

train_1['training_hours'].hist(bins=50,color=list(colors.keys())[1])

plt.xlabel('Training hours')

plt.ylabel('Count')

plt.show()
colors=random_color_generator()

train_1.plot(kind='scatter',x='city_development_index',y='training_hours',color=list(colors.keys())[1])

plt.plot()
colors=random_color_generator()

train_1.groupby('city_development_index_cut')['target'].value_counts().unstack().plot(kind='bar',color=colors)

plt.title('City Development Index Distribution')

plt.plot()
x_train_cleaned = train_1.drop(['enrollee_id','city','target'],axis=1)

y_train_cleaned = train_1['target']

test_cleaned = test_1.drop(['enrollee_id','city','target'],axis=1)
x_train_cleaned.head()
x_train_cleaned.shape
x_train_cleaned.isnull().sum()
test_cleaned.head(2)
test_cleaned.shape
cat_attr = ['gender','relevent_experience','enrolled_university','education_level','major_discipline','company_type']

num_attr = ['city_development_index','experience','company_size','last_new_job','training_hours']
train_pipeline = ColumnTransformer([('num',StandardScaler(),num_attr),

                                   ('cat',OneHotEncoder(),cat_attr),])



train_prepared = train_pipeline.fit_transform(x_train_cleaned)
test_pipeline = ColumnTransformer([('num',StandardScaler(),num_attr),

                                   ('cat',OneHotEncoder(),cat_attr),])



test_prepared = test_pipeline.fit_transform(test_cleaned)
print(train_prepared.shape)

print(test_prepared.shape)
y = y_train_cleaned.astype(np.int)

x_train,x_val,y_train,y_val = train_test_split(train_prepared,y,test_size=0.2,random_state=42)

print(x_train.shape)

print(x_val.shape)

print(y_train.shape)

print(y_val.shape)
rf_clf = RandomForestClassifier(n_estimators=1000, random_state=42)

extra_trees_clf = ExtraTreesClassifier(n_estimators=1000, random_state=42)

mlp_clf = MLPClassifier(random_state=42)

svc_clf = SVC(gamma='scale',probability=True , random_state=42)
start_time = timer(None)

y_pred_track=[]

estimators = [rf_clf,extra_trees_clf,mlp_clf,svc_clf]

print('------------- ROC-AUC Scores -------------')

for estimator in estimators:

    estimator.fit(x_train,y_train)

    y_pred = estimator.predict_proba(x_val)[:,1]

    y_pred_track.append(y_pred)

    print(estimator.__class__.__name__,'-->',roc_auc_score(y_val,y_pred))

timer(start_time)
plt.figure(figsize=(10,8))

plt.title('Reciever Operating Characteristics Curve')

for y_pred,estimator in zip(y_pred_track,estimators):

    colors=random_color_generator()

    frp,trp, threshold = roc_curve(y_val,y_pred)

    roc_auc_ = auc(frp,trp)

    plt.plot(frp,trp,'r',label = '%s AUC = %0.3f' %(estimator.__class__.__name__,roc_auc_),color=list(colors.keys())[1])

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'b--')

plt.ylabel('True positive rate')

plt.xlabel('False positive rate')

plt.show()
named_estimators = [

    ("random_forest_clf", rf_clf),

    ("extra_trees_clf", extra_trees_clf),

    ("mlp_clf", mlp_clf)

]
voting_clf = VotingClassifier(named_estimators,voting='soft')

start_time = timer(None)

voting_clf.fit(x_train, y_train)

y_pred = voting_clf.predict_proba(x_val)[:,1]

print('ROC-AUC Score -->',roc_auc_score(y_val,y_pred))

timer(start_time)
[estimator.score(x_val, y_val) for estimator in voting_clf.estimators_]
xgb = xgboost.XGBClassifier(

 n_estimators=100,

 max_depth=5,

 min_child_weight=1,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=42)

xgb.fit(x_train,y_train,eval_set=[(x_val,y_val)])

y_pred = xgb.predict_proba(x_val)[:,1]

aoc_auc = roc_auc_score(y_val, y_pred) # Not shown

print("AOC ROC Score", aoc_auc) 
param_test1 = {

 'max_depth':range(3,10,2),

 'min_child_weight':range(1,6,2)

}



folds=10

param_comb = 5



skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 42)



random_search = RandomizedSearchCV(xgb, param_distributions=param_test1, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf, verbose=2, random_state=42 )



start_time = timer(None)

random_search.fit(x_train, y_train)

random_search.best_score_

timer(start_time) 
random_search.best_estimator_
random_search.best_score_
err = []

y_pred_tot_xgb = []

fold = StratifiedKFold(n_splits=15)

i = 1

for train_index, test_index in fold.split(train_prepared, y):

    x_train, x_val = train_prepared[train_index], train_prepared[test_index]

    y_train, y_val = y[train_index], y[test_index]

    m = random_search.best_estimator_

    m.fit(x_train, y_train,

          eval_set=[(x_train,y_train),(x_val, y_val)],

          early_stopping_rounds=200,

          eval_metric='auc',

          verbose=200)

    pred_y = m.predict_proba(x_val)[:,1]

    print("err_xgb: ",roc_auc_score(y_val,pred_y))

    err.append(roc_auc_score(y_val, pred_y))

    pred_test = m.predict_proba(test_prepared)[:,1]

    i = i + 1

    y_pred_tot_xgb.append(pred_test)
np.mean(err,0)
colors = random_color_generator()

feat_imp = pd.Series(m.feature_importances_).sort_values(ascending=False)

feat_imp.plot(kind='bar', title='Feature Importances',color=colors)

plt.ylabel('Feature Importance Score')
predictions = test[['enrollee_id']]

predictions['prediction'] = m.predict(test_prepared)
predictions.head()
predictions.prediction.value_counts()