import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
pd.set_option('display.max_columns',None)
df = pd.read_csv('../input/heart-disease-uci/heart.csv')
df
df.isnull().sum()
features = [feature for feature in df.columns if feature!= 'target']
dis_feature = [ feature for feature in features if len(df[feature].unique()) < 10 ]
dis_feature
for feature in dis_feature:
    sns.countplot(x=feature,data=df,hue='target')
    plt.xlabel(feature)
    plt.ylabel('count')
    plt.show()
for feature in dis_feature:
    df.groupby(feature)['target'].mean().plot()
    plt.xlabel(feature)
    plt.show()
for feature in dis_feature:
    mean = df.groupby(feature)['target'].mean()
    index = mean.sort_values().index
    ordered_labels = { k:i for i,k in enumerate(index,0) }
    df[feature] = df[feature].map(ordered_labels)
    
for feature in dis_feature:
    df.groupby(feature)['target'].mean().plot()
    plt.xlabel(feature)
    plt.show()
con_feature = [ feature for feature in features if feature not in dis_feature]
con_feature
for feature in con_feature:
    df[feature].hist(bins=10)
    plt.xlabel(feature)
    plt.show()
for feature in con_feature:
    sns.boxplot(x=feature,data=df)
    plt.show()
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
selectk = SelectKBest(score_func=chi2,k=9)
feature_scores = selectk.fit(df.drop('target',axis=1),df['target'])
feature_scores.scores_
df_scores = pd.DataFrame(feature_scores.scores_)
df_features = pd.DataFrame(features)
features_scores = pd.concat([df_features,df_scores],axis=1)
features_scores.columns = ['features','scores']
features_scores.sort_values(by='scores',ascending=False,inplace=True)
features_scores
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)
Best_features = features_scores[features_scores['scores']>18]['features'].values
Best_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
model = RandomForestClassifier()
cross_val_score(model,df[Best_features],df['target'],cv=10).mean()
from sklearn.model_selection import RandomizedSearchCV
params = {
    'n_estimators' : list(np.arange(10,101,1)),
    'max_depth' :  list(np.arange(3,30,1)),
    'min_samples_leaf' :  list(np.arange(1,10,1)),
    'min_samples_split' :  list(np.arange(1,10,1))
}
random_search = RandomizedSearchCV(model,param_distributions=params,n_jobs=-1,n_iter=10,scoring='f1_macro',cv=5,verbose=3)
random_search.fit(df[Best_features],df['target'])
random_search.best_estimator_
model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=22, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=4, min_samples_split=7,
                       min_weight_fraction_leaf=0.0, n_estimators=41,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
cross_val_score(model,df[Best_features],df['target'],cv=10).mean()