import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import numpy as np
import statsmodels.api as sm
%matplotlib inline
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt   
from sklearn import metrics
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
df = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
df.head()
df.isnull().sum().sum()
test.isnull().sum().sum()
df.describe()
df.info()
df.columns
df.dtypes
df
!pip install pydotplus
X = df.drop(columns='label')
y = df['label']


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(X,y)

from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus



features = X.columns
dot_data = export_graphviz(dt,out_file=None,feature_names=features)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
dt = DecisionTreeClassifier(max_depth=3,min_samples_leaf=11)
dt.fit(X,y)
features = X.columns
dot_data = export_graphviz(dt,out_file=None,feature_names=features)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
test1 = test
y_pred = dt.predict(test1)
y_prob = dt.predict_proba(test1)
y_pred_train = dt.predict(X)
y_prob_train = dt.predict_proba(X)
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,f1_score
print('Accuracy of Decision Tree on Train',accuracy_score(y_pred_train,y))
print('F1 score on train',f1_score(y,y_pred_train,average='weighted')*100)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10,random_state=1)


rfc.fit(X,y)
test1 = test
y_pred = rfc.predict(test1)
y_prob = rfc.predict_proba(test1)
y_pred_train = rfc.predict(X)
y_prob_train = rfc.predict_proba(X)
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,f1_score
print('Accuracy of Decision Tree on Train',accuracy_score(y_pred_train,y))
print('F1 score on train',f1_score(y,y_pred_train,average='weighted')*100)
samp = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
samp.head()
test.shape
data = pd.DataFrame(y_pred,index=samp['ImageId']).rename(columns={0:'Label'})
data = data.reset_index().rename(columns={'index':'ImageId'})
data = data.set_index('ImageId')
data.head()
data.shape
data.to_csv('submission2.csv')
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV


rfc= RandomForestClassifier(random_state=1)

params = {'n_estimators':sp_randint(5,25),'criterion':['gini','entropy'],
    'max_depth':sp_randint(2,10),
    'min_samples_split':sp_randint(2,20),
    'min_samples_leaf':sp_randint(1,20),'max_features':sp_randint(2,10)}

rand_search_rfc = RandomizedSearchCV(rfc,param_distributions=params,cv=3,random_state=1)

rand_search_rfc.fit(X,y)

print(rand_search_rfc.best_params_)
from sklearn.model_selection import train_test_split

rfc = RandomForestClassifier(**rand_search_rfc.best_params_)

rfc.fit(X,y)

test1 = test
y_pred = rfc.predict(test1)
y_prob = rfc.predict_proba(test1)
y_pred_train = rfc.predict(X)
y_prob_train = rfc.predict_proba(X)

from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,f1_score

print('Accuracy of Decision Tree on Train',accuracy_score(y_pred_train,y))
print('F1 score on train',f1_score(y,y_pred_train,average='weighted')*100)
data = pd.DataFrame(y_pred,index=samp['ImageId']).rename(columns={0:'Label'})
data = data.reset_index().rename(columns={'index':'ImageId'})
data = data.set_index('ImageId')
data.head()

data.to_csv('submission3.csv')
##LGBM:
import lightgbm as lgb
lgbc = lgb.LGBMClassifier()

lgbc.fit(X,y)

test1 = test
y_pred = lgbc.predict(test1)
y_prob = lgbc.predict_proba(test1)
y_pred_train = lgbc.predict(X)
y_prob_train = lgbc.predict_proba(X)

from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,f1_score

print('Accuracy of Decision Tree on Train',accuracy_score(y_pred_train,y))
print('F1 score on train',f1_score(y,y_pred_train,average='weighted')*100)

cols = X.columns
lgbc.feature_importances_

fi = pd.DataFrame(index=cols,data=lgbc.feature_importances_,columns=['Importance'])
fi

#Feature importance is available for bagging and boosting both

fi['Importance'].sort_values(ascending=False).plot.bar()
data = pd.DataFrame(y_pred,index=samp['ImageId']).rename(columns={0:'Label'})
data = data.reset_index().rename(columns={'index':'ImageId'})
data = data.set_index('ImageId')
data.head()

data.to_csv('submission7.csv')
#hyperparameter tuning of LGBM

from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV

lgbc= lgb.LGBMClassifier(random_state=1)

params = {'n_estimators':sp_randint(5,250),
    'max_depth':sp_randint(2,20),
    'min_child_samples':sp_randint(1,20),'num_leaves':sp_randint(5,50)}

rand_search_lgbc = RandomizedSearchCV(lgbc,param_distributions=params,cv=3,random_state=1)

rand_search_lgbc.fit(X,y)

print(rand_search_lgbc.best_params_)



lgbc= lgb.LGBMClassifier(**rand_search_lgbc.best_params_,random_state=1)

lgbc.fit(X,y)

test1 = test
y_pred = lgbc.predict(test1)
y_prob = lgbc.predict_proba(test1)
y_pred_train = lgbc.predict(X)
y_prob_train = lgbc.predict_proba(X)

from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,f1_score

print('Accuracy of Decision Tree on Train',accuracy_score(y_pred_train,y))
print('F1 score on train',f1_score(y,y_pred_train,average='weighted')*100)

cols = X.columns
lgbc.feature_importances_

fi = pd.DataFrame(index=cols,data=lgbc.feature_importances_,columns=['Importance'])
fi

#Feature importance is available for bagging and boosting both

fi['Importance'].sort_values(ascending=False).plot.bar()
data = pd.DataFrame(y_pred,index=samp['ImageId']).rename(columns={0:'Label'})
data = data.reset_index().rename(columns={'index':'ImageId'})
data = data.set_index('ImageId')
data.head()
data.to_csv('submission8.csv')
##Support Vector Machines
# Stacking algorithms

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
dt = DecisionTreeClassifier()
lgbc= lgb.LGBMClassifier(**rand_search_lgbc.best_params_)
rfc = RandomForestClassifier(**rand_search_rfc.best_params_)


clf = VotingClassifier(estimators=[('dt',dt),('lgbc',lgbc),('rfc',rfc),('lr',lr)],voting='soft')
clf.fit(X,y)


test1 = test
y_pred = clf.predict(test1)
y_prob = clf.predict_proba(test1)
y_pred_train = clf.predict(X)
y_prob_train = clf.predict_proba(X)

from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,f1_score

print('Accuracy of Decision Tree on Train',accuracy_score(y_pred_train,y))
print('F1 score on train',f1_score(y,y_pred_train,average='weighted')*100)
data = pd.DataFrame(y_pred,index=samp['ImageId']).rename(columns={0:'Label'})
data = data.reset_index().rename(columns={'index':'ImageId'})
data = data.set_index('ImageId')
data.head()
data.to_csv('submission11.csv')
