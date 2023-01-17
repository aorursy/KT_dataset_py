import pandas as pd
train_df=pd.read_csv('/kaggle/input/eval-lab-2-f464/train.csv')
train_df.head()
#train_df = train_df[train_df['class']!=6]

#train_df = train_df[train_df['class']!=5]

#train_df = train_df[train_df['class']!=7]
%matplotlib inline



corr = train_df.corr()

corr.style.background_gradient(cmap='coolwarm')
test_df=pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')
test_df.head()
train_df.shape
from sklearn.linear_model import LogisticRegression  

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier

import lightgbm as lgb

from sklearn.naive_bayes import MultinomialNB

from imblearn.ensemble import BalancedRandomForestClassifier
#X_train = train_df.drop(['chem_2','chem_3','chem_7','attribute','class'],axis=1)

#X_train = train_df.drop(['id','class'],axis=1)

X_train=train_df[['chem_1','chem_2','chem_4','chem_6']]
y_train = train_df['class']
y_train.value_counts()
from sklearn.model_selection import cross_val_score
#clf=LogisticRegression(multi_class='multinomial',solver='newton-cg',class_weight='balanced',max_iter=50)

clf=RandomForestClassifier(n_estimators=2000,n_jobs=-1,max_depth=9)

#clf=XGBClassifier()

#clf=DecisionTreeClassifier(max_depth=15)

#clf = lgb.LGBMClassifier(max_depth=5,objective='multiclassova',num_class=4,learning_rate=0.1,n_estimators=50)

#clf=MultinomialNB()

#clf=BalancedRandomForestClassifier(n_estimators=50,max_depth=3,random_state=42)

#clf=ExtraTreesClassifier()
scores = cross_val_score(clf, X_train, y_train, cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#X_test=test_df.drop(['id'],axis=1)

X_test=test_df[['chem_1','chem_2','chem_4','chem_6']]
clf.fit(X_train,y_train)

preds=clf.predict(X_test)
sub=pd.read_csv('/kaggle/input/eval-lab-2-f464/sample_submission.csv')
sub = sub[0:0]

sub['id'] = test_df['id']
sub['class'] = preds
sub.head()
sub['class'].value_counts()
sub.to_csv('sub18.csv',index=False)
train_df = train_df[train_df['class']!=6]

train_df = train_df[train_df['class']!=5]
clf_2=RandomForestClassifier(n_estimators=2000,n_jobs=-1,max_depth=9).fit(X_train,y_train)
scores_2 = cross_val_score(clf_2, X_train, y_train, cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_2.mean(), scores_2.std() * 2))
#clf_2.fit(X_train,y_train)

preds_2=clf_2.predict(X_test)
sub['class'] = preds_2
sub.to_csv('sub20.csv',index=False)