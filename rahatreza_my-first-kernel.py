import pandas as pd
train=pd.read_csv("/kaggle/input/av-comp/train.csv")
test=pd.read_csv("/kaggle/input/av-comp/test.csv")
sample=pd. read_csv ("/kaggle/input/av-comp/sample.csv")
train.head() 
train=train.drop(["id"],axis=1)
test=test.drop(["id"],axis=1)

 
train=train.drop(['age'],axis=1)
test=test.drop(['age'],axis=1)

train=train.replace({'difficulty_level': {'easy' :0,'intermediate' :1,'hard':2,'vary hard' :3},
                     'education' :{'No Qualification' :0,'Matriculation' :1,'High School Diploma' :2,'Bachelors' :3,'Masters':4}, 
                    'program_type':{'S':0,'T':1,'U':2,'V':3,'W':4,'X':5,'Y':6,'Z':7},
                    'test_type':{'offline':0,'online':1},
                    'gender':{'M':1,'F':0}, 
                    'is_handicapped':{'N':0,'Y':1} 
                    })

test=test.replace({'difficulty_level': {'easy' :0,'intermediate' :1,'hard':2,'vary hard' :3},
                     'education' :{'No Qualification' :0,'Matriculation' :1,'High School Diploma' :2,'Bachelors' :3,'Masters':4}, 
                    'program_type':{'S':0,'T':1,'U':2,'V':3,'W':4,'X':5,'Y':6,'Z':7},
                    'test_type':{'offline':0,'online':1},
                    'gender':{'M':1,'F':0}, 
                    'is_handicapped':{'N':0,'Y':1} 
                    })

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5, weights="uniform")
col=['trainee_id','gender','education','city_tier', 'trainee_engagement_rating'] 
X_tr=train[col]
X_test=test[col] 
X_tr=imputer.fit_transform(X_tr)
X_test=imputer.transform(X_test) 
train=train.join(pd.DataFrame({'imp':X_tr[:,- 1]}))
test=test.join(pd.DataFrame({'imp':X_test[:,- 1]}))

train=train.drop(['trainee_engagement_rating'],axis=1)
test=test.drop(['trainee_engagement_rating'],axis=1)

Id=train.program_id.unique()
for i in range(len(Id)):
    train=train.replace({Id[i]:i})
    test=test.replace({Id[i]:i})


train.head() 

from catboost import CatBoostClassifier,Pool
from sklearn.model_selection import train_test_split

train_y=train.is_pass
train=train.drop(["is_pass"],axis=1)
x_train,x_valid,y_train,y_valid=train_test_split(train,train_y,train_size =0.8)
cat_features=[0,1,3,6] 
#these are the indices of "program_id", "program_type",
#"test_id", "trainee_id" 
train_pool=Pool(train, train_y, cat_features=cat_features)
test_pool=Pool(test, cat_features =cat_features) 
weight=[1, 0.43]
model = CatBoostClassifier(
        iterations=5000,logging_level="Silent",early_stopping_rounds=500,
    use_best_model=True,custom_loss=["AUC"], class_weights=weight, 
        eval_metric="AUC")
model.fit(  x_train,y_train, 
        cat_features=cat_features,
        eval_set=(x_valid,y_valid),
        plot=True 
        )
from sklearn.metrics import roc_auc_score
roc_auc_score(y_valid, model.predict_proba(x_valid) [:,1]) 
feature_importances=model.get_feature_importance(train_pool) 
feature_names=train.columns
for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    print('{}: {}'.format(name, score))

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import plot_confusion_matrix, confusion_matrix 
Cm=confusion_matrix(y_valid,model.predict(x_valid))
plot_confusion_matrix(model,x_valid, y_valid, cmap=plt.cm.Blues ) 
Cm
final_model = CatBoostClassifier(
        iterations=1500,logging_level="Silent",class_weights=weight, 
        eval_metric="AUC")
final_model.fit(train, train_y, cat_features=cat_features )
final_pred=final_model.predict_proba(test)[:,1]
sample.is_pass=final_pred
sample.to_csv("best_model.csv",index=None)

#from imblearn.under_sampling import RandomUnderSampler
#import warnings

#rus = RandomUnderSampler(random_state=0, sampling_strategy=0.8)
#train_,train_y_ = rus.fit_resample(train,train_y)

