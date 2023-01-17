import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("../input/train.csv")
df1=pd.read_csv("../input/test.csv")
df.isnull().sum()
df.head()
a=pd.DataFrame(df.Id)
del df['Id']
df.dtypes
X=df.iloc[:,:-1]
Y=df.Cover_Type 
Y
from sklearn import model_selection,metrics
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import accuracy_score
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier

seed = 7
scoring = 'accuracy'



models = []
models.append(('LR', LogisticRegression(C=2,random_state=0)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('RFC', RandomForestClassifier(criterion='entropy',max_depth=10,max_features='auto',min_samples_split=10,n_estimators=1000, oob_score=True,random_state=101)))
#models.append(('NB', GaussianNB()))
#models.append(('KNN', KNeighborsClassifier(n_neighbors=10)))
#models.append(('SVC', SVC(C=1,decision_function_shape='ovr',kernel='rbf', degree=3,gamma='auto',random_state=101)))
models.append(('ETC', ExtraTreesClassifier(criterion='entropy', n_estimators=1000,random_state=42)))


# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    
    msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std()*100)
    print(name+'\n',cv_results.mean())

df1.head()
b=pd.DataFrame(df1.Id)
del df1['Id']
df1.corr()
df1.isnull().sum()
etc=ExtraTreesClassifier(criterion='gini', n_estimators=1000,random_state=42)
etc.fit(X,Y)
etc_pred1=etc.predict(df1)
submission_a = pd.DataFrame({'Id' : b.Id, 'Cover_Type ' : etc_pred1})

submission_a.head(5)
submission_a.to_csv('submissiona.csv', index=False)

#import lightgbm as lgb
#lgb_train = lgb.Dataset(X_train, label = Y_train)
#params = {
#            'task': 'train',
#            'boosting_type': 'gbdt',
#            'objective': 'multiclass',
 #           'metric':{'multi_logloss'},
 #           'num_class': 10,
#            'num_leaves': 30,
#            'min_data_in_leaf': 1,
 #           'learning_rate': 0.1,
 #           'boost_from_average': True
 #       }
#lgb_cls = lgb.train(params,  lgb_train,  100)
#lgb_y_pred = lgb_cls.predict(X_test)
#lgb_y_orig = []
#for p in lgb_y_pred:
 #   lgb_y_orig.append(np.argmax(p))
#print('LightG: ', accuracy_score(Y_test, lgb_y_orig))


#lgb_y_pred = lgb_cls.predict(df1)
#lgb_y_original = []
#for p in lgb_y_pred:
   # lgb_y_original.append(np.argmax(p))
#submission_b= pd.DataFrame({'Id' : b.Id, 'Cover_Type' : lgb_y_original})
#submission_b.head(5)

#submission_b.to_csv('submissionb.csv', index=False)
