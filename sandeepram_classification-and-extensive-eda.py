import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno
from warnings import filterwarnings
filterwarnings('ignore')
data = pd.read_csv(r'../input/student-grade-prediction/student-mat.csv')
data.head().transpose()
cardinality = {'columns' : data.columns,
               'cardinal' : []}
for i in cardinality['columns']:
    cardinality['cardinal'].append(data[i].nunique())
len(data)
missingno.matrix(data)
plt.figure(figsize = (10,10))
data['G3'].value_counts().sort_values().plot(kind = 'barh', width = 0.8, color = sns.color_palette("RdBu", 40))
b = sns.countplot(data['G3'])
b.set_xlabel('Final Grade')
b.set_ylabel('Count')
corr = data.corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr, cmap = 'coolwarm', annot = True)
sns.distplot(data['age'], kde= False, color = 'r')
sns.kdeplot(data['age'], shade = True, color = 'r')
plt.figure(figsize = (7,5))
sns.countplot(data['age'], hue = data['sex'])
data['sex'].unique()
sns.kdeplot(data.loc[data['sex'] == 'F', 'G3'], label='Female', shade = True)
sns.kdeplot(data.loc[data['sex'] == 'M', 'G3'], label='Male', shade = True)
plt.title('Does gender affect your graders?', fontsize = 20)
plt.show()
sns.kdeplot(data.loc[data['romantic'] == 'yes', 'G3'], label='Relationship', shade = True)
sns.kdeplot(data.loc[data['romantic'] == 'no', 'G3'], label='Single', shade = True)
plt.title('Does relationship affect studies?', fontsize = 20)
plt.show()
sns.kdeplot(data.loc[data['address'] == 'U', 'G3'], label='Urban', shade = True)
sns.kdeplot(data.loc[data['address'] == 'R', 'G3'], label='Rural', shade = True)
plt.title('Do urban students score higher than rural students?', fontsize = 20)
plt.xlabel('Grade', fontsize = 20);
plt.ylabel('Density', fontsize = 20)
plt.show()
sns.kdeplot(data.loc[data['address'] == 'U', 'age'], label='Urban', shade = True)
sns.kdeplot(data.loc[data['address'] == 'R', 'age'], label='Rural', shade = True)
plt.title('Do urban students attend more years of school?', fontsize = 20)
plt.show()
copy_set = data.copy()
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
binary_cols = []

for col in data.columns:
    if data[col].nunique()==2:
        binary_cols.append(col)
for i in binary_cols:
    data[i] = LabelBinarizer().fit_transform(data[i])
data.head().transpose()
categorical_cols = [column for column in data.columns if (data[column].nunique()<=30)]
str_col = [col for col in categorical_cols if data[col].dtype =='O']
str_col
for i in str_col:
    print(i, ' :', data[i].unique())
data_ = data.copy()
for i in str_col:
    print(i)
    data = pd.concat([data.drop(i, axis = 1), pd.get_dummies(data[i], prefix=i, drop_first = True)], axis = 1)
data['pass'] = data['G3'].copy()
def classify(x):
    if x >= 10:
        return 1
    else:
        return 0

data['pass'] = data['pass'].apply(classify)
data_clf = data.drop(['G1', 'G2', 'G3'], axis = 1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_clf.drop('pass', 1), data_clf['pass'], random_state = 42, test_size = 0.2)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm  import SVC
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
scores_df = {'name': [],
             'train_score': [],
             'test_score' : []}
def base_score(model_info):
    
    model, name = model_info
    model.fit(X_train, y_train)
    scores_df['name'].append(name)
    scores_df['train_score'].append( model.score(X_train, y_train))
    scores_df['test_score'].append(model.score(X_test, y_test))
models = [(RandomForestClassifier(),'rf'), (GradientBoostingClassifier(), 'gbc'), (LogisticRegression(), 'lr'),
          (BernoulliNB(), 'naive_b'), (GaussianNB(), 'naive_g'), (SVC(), 'svc'), (xgb.XGBClassifier(), 'xgb')]
for i in models:
    base_score(i)
scores_df = pd.DataFrame(scores_df)
scores_df.set_index('name', inplace = True)
scores_df
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
model_rf = RandomForestClassifier()
model_gbc = GradientBoostingClassifier()
model_xgb = xgb.XGBClassifier()
model_lr = LogisticRegression()
params = {
    model_rf: 
    {
        'n_estimators' : np.arange(10,100,10),
        'max_features' : [0.2, 0.5, 1],
        'max_depth' : [2,3,5,7],
    },
    model_gbc: 
    {
        'n_estimators' : np.arange(10,100,10),
        'learning_rate' : np.arange(0.01, 0.05, 1),
        'subsample' : [0.2, 0.5, 0.8, 1],
        'max_depth' : [2, 3, 5]
        
    },
    model_xgb:
    {
        'max_depth' : [2, 3, 5],
        'subsample' : [0.2, 0.5, 1],
        'n_estimators' : np.arange(40,150,10),
        'learning_rate': np.arange(0.01, 0.5, 1),
    },
    model_lr:
    {
        'penalty': ['l2', 'l1'],
        'C': np.arange(0.1, 1, 0.1),
    }
         }
best_estimators = []
for model in params.keys():
    clf = RandomizedSearchCV(model, params[model], cv = 3, n_jobs = -1, random_state = 42)
    search = clf.fit(X_train, y_train)
    best_estimators.append(search.best_estimator_)
def scoring(estimator):
    estimator.fit(X_train, y_train)
    print(estimator.score(X_test, y_test))
    
for estimator in best_estimators:
    print(estimator)
    scoring(estimator)
model_rf = RandomForestClassifier(max_depth=5, max_features=0.5, n_estimators=50)
model_lr = LogisticRegression(C=0.1)
model_xgb = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                              importance_type='gain', interaction_constraints='',
                              learning_rate=0.01, max_delta_step=0, max_depth=5,
                              min_child_weight=1, monotone_constraints='()',
                              n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,
                              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=0.2,
                              tree_method='exact', validate_parameters=1, verbosity=None)
model_rf.fit(X_train, y_train)
model_xgb.fit(X_train, y_train)
model_lr.fit(X_train, y_train)
def Voting(data):
    
    preds_1 = np.array(model_lr.predict(data))
    preds_2 = np.array(model_xgb.predict(data))
    preds_3 = np.array(model_rf.predict(data))
    
    pred = preds_1 + preds_2 + preds_3
    prediction = []
    
    for i in pred:
        if i<=1.5:
            prediction.append(0)
        elif i>=1.5:
            prediction.append(1)
    
    return np.array(prediction)
preds = Voting(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, preds)
rf_imps = model_rf.feature_importances_
xgb_imps = model_xgb.feature_importances_
cols = X_train.columns
df = {'columns' : cols,
      'rf_imp': rf_imps,
      'xgb_imps': xgb_imps}
df = pd.DataFrame(df)
df['mean_importance'] = (df['rf_imp'] + df['xgb_imps'])/2
df = df.sort_values(by=['mean_importance'], ascending = False)
df_copy = df.copy()
num_cols = 15
df = df_copy.copy()
df = df.head(num_cols)
df.shape
columns = df['columns']
X_train_2 = X_train[columns]
X_test_2 = X_test[columns]
model_rf = RandomForestClassifier(max_depth=5, max_features=0.5, n_estimators=50)
model_lr = LogisticRegression(C=0.1)
model_xgb = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                              importance_type='gain', interaction_constraints='',
                              learning_rate=0.01, max_delta_step=0, max_depth=5,
                              min_child_weight=1, monotone_constraints='()',
                              n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,
                              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=0.2,
                              tree_method='exact', validate_parameters=1, verbosity=None)
X_train_2.shape

model_rf.fit(X_train_2, y_train)
model_xgb.fit(X_train_2, y_train)
model_lr.fit(X_train_2, y_train)
preds = Voting(X_test_2)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, preds)
