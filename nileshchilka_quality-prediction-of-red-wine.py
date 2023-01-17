import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

pd.set_option('display.max_columns',None)
df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df
df.isnull().sum()
df.quality.value_counts()
df['quality'] = np.where(df['quality']<=4,0,df['quality'])
df['quality'] = np.where((df['quality']<=6) & (df['quality']!=0 ),1,df['quality'])
df['quality'] = np.where( df['quality']>=7,2,df['quality'])
df.quality.value_counts()
from imblearn.combine import SMOTETomek
smk = SMOTETomek(random_state=0)
X,y=smk.fit_sample(df.drop('quality',axis=1),df['quality'])
df = pd.concat([X,y],axis=1)
df.quality.value_counts()
df.head()
features = [feature for feature in df.columns if feature!='quality']
for feature in features:

    sns.boxplot(x=feature,data=df)

    plt.xlabel(feature)

    plt.show()
dic = {}

for feature in features:

    IQR = df[feature].quantile(0.75) - df[feature].quantile(0.25)

    upper_bond = df[feature].quantile(0.75) + (IQR * 1.5)

    lower_bond = df[feature].quantile(0.25) - (IQR * 1.5)

    

    df[feature] = np.where(df[feature]>upper_bond,upper_bond,df[feature])

    df[feature] = np.where(df[feature]<lower_bond,lower_bond,df[feature])
for feature in features:

    sns.boxplot(x=feature,data=df)

    plt.xlabel(feature)

    plt.show()
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2
selectk = SelectKBest(score_func=chi2,k=7)
Best = selectk.fit(df.drop('quality',axis=1),df['quality'])
Best.scores_
features
dfscores = pd.DataFrame(Best.scores_)

dffeatures = pd.DataFrame(features)
features_scores = pd.concat([dffeatures,dfscores],axis=1)
features_scores.columns = ['feature','scores']
features_scores.sort_values(by='scores',ascending=False)
Best_features = features_scores[features_scores['scores']>30]['feature']
plt.figure(figsize=(10,10))

sns.heatmap(df.corr(),annot=True)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df[Best_features],df['quality'],test_size=0.2,random_state=0)
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,classification_report
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
y_predict_proba_train = model.predict_proba(X_train)
y_predict_proba_test = model.predict_proba(X_test)
roc_auc_score(y_train,y_predict_proba_train,multi_class='ovo')
roc_auc_score(y_test,y_predict_proba_test,multi_class='ovo')
confusion_matrix(y_test,y_predict)
accuracy_score(y_test,y_predict)
print(classification_report(y_test,y_predict))
from sklearn.model_selection import cross_val_score
cross_val_score(model,df[Best_features],df['quality'],scoring='accuracy',n_jobs=-1).mean()
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
y_predict_proba_train = model.predict_proba(X_train)
y_predict_proba_test = model.predict_proba(X_test)
roc_auc_score(y_train,y_predict_proba_train,multi_class='ovo')
roc_auc_score(y_test,y_predict_proba_test,multi_class='ovo')
confusion_matrix(y_test,y_predict)
accuracy_score(y_test,y_predict)
print(classification_report(y_test,y_predict))
cross_val_score(model,df[Best_features],df['quality'],scoring='accuracy',n_jobs=-1).mean()
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
y_predict_proba_train = model.predict_proba(X_train)
y_predict_proba_test = model.predict_proba(X_test)
roc_auc_score(y_train,y_predict_proba_train,multi_class='ovo')
roc_auc_score(y_test,y_predict_proba_test,multi_class='ovo')
confusion_matrix(y_test,y_predict)
accuracy_score(y_test,y_predict)
print(classification_report(y_test,y_predict))
cross_val_score(model,df[Best_features],df['quality'],scoring='accuracy',n_jobs=-1).mean()
params = {

    'n_estimators' : list(np.arange(5,101,1)) ,

    'max_depth' : list(np.arange(3,16,1)) ,

    'min_child_weight' : [1,3,4,5,6,7,8] ,

    'learning_rate' : list(np.arange(0.05,0.35,0.05)) ,

    'colsample_bytree' : [0.4,0.5,0.6,0.7],

    'gamma' : [0.0,0.1,0.2,0.3,0.4]    

}
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(model,param_distributions=params,n_jobs=-1,scoring='accuracy',verbose=3,cv=5)
random_search.fit(df[Best_features],df['quality'])
random_search.best_estimator_
model = XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.4, gamma=0.4, gpu_id=-1,

              importance_type='gain', interaction_constraints=None,

              learning_rate=0.3, max_delta_step=0, max_depth=10,

              min_child_weight=1, monotone_constraints=None,

              n_estimators=37, n_jobs=0, num_parallel_tree=1,

              objective='multi:softprob', random_state=0, reg_alpha=0,

              reg_lambda=1, scale_pos_weight=None, subsample=1,

              tree_method=None, validate_parameters=False, verbosity=None)
cross_val_score(model,df[Best_features],df['quality'],scoring='accuracy',n_jobs=-1).mean()