#Importing Data Set

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

class color:

    BOLD='\033[1m'

    END='\033[0m'

    BLUE='\033[94m'

    

pd.set_option('display.max_columns',100)

pd.set_option('display.max_rows',100)

import warnings

warnings.simplefilter(action='ignore',category=FutureWarning)
train_df=pd.read_csv('../input/hr-analytics-av/train_hr.csv')

test_df=pd.read_csv('../input/hr-analytics-av/test_hr.csv')

train_df.head()
test_df.head()
train_df.shape,test_df.shape
train_df['Data']='train'

test_df['Data']='test'

test_df['is_promoted']=np.nan
train_df.isnull().sum()
test_df.isnull().sum()
train_df.previous_year_rating.value_counts()
test_df.previous_year_rating.value_counts()
train_df.education.value_counts()
test_df.education.value_counts()
train_df.info()
train_df.shape[0]+test_df.shape[0]
df=pd.concat([train_df,test_df],ignore_index=True,axis=0)

df.shape
df.head()
df.isnull().sum()
df.recruitment_channel.value_counts()
pd.DataFrame(df.groupby('education')['department'])
df.department.value_counts()
df.education.value_counts()
pd.crosstab([df.department,df.education],df.recruitment_channel)
pd.crosstab(df.department,df.recruitment_channel)
df.previous_year_rating=df.previous_year_rating.fillna(method='bfill')

df.previous_year_rating=df.previous_year_rating.fillna(method='ffill')
df.isnull().sum()
df.loc[(df.department=='Analytics') & (df.recruitment_channel=="other") & (df['education'].isnull()),'education']="Bachelor's"

df.loc[(df.department=='Analytics') & (df.recruitment_channel=="sourcing") & (df['education'].isnull()),'education']="Master's & above"

df.loc[(df.department=='Analytics') & (df.recruitment_channel=="referred") & (df['education'].isnull()),'education']="Master's & above"

df.loc[(df.department=='Finance') & (df.recruitment_channel=="other") & (df['education'].isnull()),'education']="Bachelor's"

df.loc[(df.department=='Finance') & (df.recruitment_channel=="sourcing") & (df['education'].isnull()),'education']="Below Secondary"

df.loc[(df.department=='Finance') & (df.recruitment_channel=="referred") & (df['education'].isnull()),'education']="Master's & above"
df.loc[(df.department=='HR') & (df.recruitment_channel=="other") & (df['education'].isnull()),'education']="Bachelor's"

df.loc[(df.department=='HR') & (df.recruitment_channel=="sourcing") & (df['education'].isnull()),'education']="Below Secondary"

df.loc[(df.department=='HR') & (df.recruitment_channel=="referred") & (df['education'].isnull()),'education']="Master's & above"
df.loc[(df.department=='Legal') & (df.recruitment_channel=="other") & (df['education'].isnull()),'education']="Bachelor's"

df.loc[(df.department=='Legal') & (df.recruitment_channel=="sourcing") & (df['education'].isnull()),'education']="Below Secondary"

df.loc[(df.department=='Legal') & (df.recruitment_channel=="referred") & (df['education'].isnull()),'education']="Master's & above"
df.loc[(df.department=='Operations') & (df.recruitment_channel=="other") & (df['education'].isnull()),'education']="Bachelor's"

df.loc[(df.department=='Operations') & (df.recruitment_channel=="sourcing") & (df['education'].isnull()),'education']="Below Secondary"

df.loc[(df.department=='Operations') & (df.recruitment_channel=="referred") & (df['education'].isnull()),'education']="Master's & above"
df.loc[(df.department=='Procurement') & (df.recruitment_channel=="other") & (df['education'].isnull()),'education']="Bachelor's"

df.loc[(df.department=='Procurement') & (df.recruitment_channel=="sourcing") & (df['education'].isnull()),'education']="Below Secondary"

df.loc[(df.department=='Procurement') & (df.recruitment_channel=="referred") & (df['education'].isnull()),'education']="Master's & above"
df.loc[(df.department=='R&D') & (df.recruitment_channel=="other") & (df['education'].isnull()),'education']="Bachelor's"

df.loc[(df.department=='R&D') & (df.recruitment_channel=="sourcing") & (df['education'].isnull()),'education']="Below Secondary"

df.loc[(df.department=='R&D') & (df.recruitment_channel=="referred") & (df['education'].isnull()),'education']="Master's & above"
df.loc[(df.department=='Sales & Marketing') & (df.recruitment_channel=="other") & (df['education'].isnull()),'education']="Bachelor's"

df.loc[(df.department=='Sales & Marketing') & (df.recruitment_channel=="sourcing") & (df['education'].isnull()),'education']="Below Secondary"

df.loc[(df.department=='Sales & Marketing') & (df.recruitment_channel=="referred") & (df['education'].isnull()),'education']="Master's & above"
df.loc[(df.department=='Technology') & (df.recruitment_channel=="other") & (df['education'].isnull()),'education']="Bachelor's"

df.loc[(df.department=='Technology') & (df.recruitment_channel=="sourcing") & (df['education'].isnull()),'education']="Below Secondary"

df.loc[(df.department=='Technology') & (df.recruitment_channel=="referred") & (df['education'].isnull()),'education']="Master's & above"
df.previous_year_rating.value_counts()
df.isnull().sum()

df.region.value_counts().sort_index
df['region_num']=df.region.apply(lambda x:str(x).split('_')[1]).astype(int)
df.head()
df.region_num.value_counts(sort=False).sort_index
df.recruitment_channel.value_counts()
df.info()
df.head()
def gender(x):

    if x=='m':

        x=1

    else:

        x=0

    return x

df.gender=df.gender.apply(gender)
df.gender.value_counts()
#drop employee_id,region,data
df=df.drop(['employee_id','region'],axis=1)
categor_ = df.select_dtypes(['object']).copy()

categor_.columns
df=pd.get_dummies(df,columns=['department', 'education', 'recruitment_channel'])

df.info()
df.head()
X = df[df.Data=='train']

y=df.is_promoted.dropna()

test = df[df.Data=='test']
df.is_promoted.dropna().value_counts()
X=X.drop(['is_promoted','Data'],axis=1)

test=test.drop(['is_promoted','Data'],axis=1)
X.head()
test.head()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier

from xgboost.sklearn import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier as KNN

from imblearn.over_sampling import SMOTE



X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
rf=RandomForestClassifier()

xgb=XGBClassifier()

et=ExtraTreesClassifier()

knn=KNN()
rf.fit(X_train,y_train)

xgb.fit(X_train,y_train)

knn.fit(X_train,y_train)

et.fit(X_train,y_train)
#Accuracy

print(color.BOLD+'RandomForestClassifier'+color.END,rf.score(X_train,y_train))

print(color.BOLD+'XGBClassifier'+color.END,xgb.score(X_train,y_train))

print(color.BOLD+'ExtraTreesClassifier'+color.END,et.score(X_train,y_train))

print(color.BOLD+'KNN'+color.END,knn.score(X_train,y_train))
#Accuracy

print(color.BOLD+'RandomForestClassifier'+color.END,rf.score(X_test,y_test))

#print(color.BOLD+'XGBClassifier'+color.END,xgb.score(X_test,y_test))

print(color.BOLD+'ExtraTreesClassifier'+color.END,et.score(X_test,y_test))

print(color.BOLD+'KNN'+color.END,knn.score(X_test,y_test))
#f1_score RandomForestClassifier

predrf=rf.predict(X_test)

rf.score(X_test,predrf)

seg_rf=rf.predict(test)



from sklearn.metrics import f1_score

print('f1-score of RandomForestClassifier: ',f1_score(y_test,predrf,average='weighted'))



#f1_score XGBClassifier

'''predxgb=xgb.predict(X_test)

xgb.score(X_test,predxgb)

seg_xgb=xgb.predict(test)



from sklearn.metrics import f1_score

print('f1-score of ExtraTreesClassifier: ',f1_score(y_test,predxgb,average='weighted'))'''

#f1_score ExtraTreesClassifier

predet=et.predict(X_test)

et.score(X_test,predet)

seg_et=et.predict(test)



from sklearn.metrics import f1_score

print('f1-score of ExtraTreesClassifier: ',f1_score(y_test,predet,average='weighted'))



#f1_score KNN

predknn=knn.predict(X_test)

knn.score(X_test,predet)

seg_knn=knn.predict(test)



from sklearn.metrics import f1_score

print('f1-score of KNN: ',f1_score(y_test,predet,average='weighted'))
id=pd.DataFrame(test_df.employee_id,columns=['employee_id'])
seg_rf=pd.DataFrame(seg_rf)

seg_rf.columns=['is_promoted']



#seg_xgb=pd.DataFrame(seg_xgb)

#seg_xgb.columns=['is_promoted']



seg_et=pd.DataFrame(seg_et)

seg_et.columns=['is_promoted']



seg_knn=pd.DataFrame(seg_knn)

seg_knn.columns=['is_promoted']
rf_1=pd.concat([id,seg_rf],axis=1).astype(int)

rf_1.to_csv('rf_1.csv',index=False)
knn_1=pd.concat([id,seg_knn],axis=1).astype(int)

knn_1.to_csv('knn_1.csv',index=False)
et_1=pd.concat([id,seg_et],axis=1).astype(int)

et_1.to_csv('et_1.csv',index=False)
import warnings

warnings.simplefilter(action='ignore',category=FutureWarning)



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,cohen_kappa_score,confusion_matrix
from imblearn.over_sampling import SMOTE

smote=SMOTE('auto')

X_sm,y_sm=smote.fit_sample(X_train,y_train)

print(X_sm.shape,y_sm.shape)
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier

from xgboost.sklearn import XGBClassifier

rf=RandomForestClassifier()

rf.fit(X_sm,y_sm)

y_pred_rf=rf.predict(X_test)

print(color.BOLD+'Train Accuracy -:'+color.END,rf.score(X_sm,y_sm))

print(color.BOLD+'Test Accuracy -:'+color.END,rf.score(X_test,y_test))

print('\n'*2)

y_train_pred=rf.predict(X_sm)

y_test_pred=rf.predict(X_test)

print(color.BOLD+'Classification_report Train-:\n'+color.END,classification_report(y_sm,y_train_pred))

print(color.BOLD+'Accuracy_score Train-:'+color.END,accuracy_score(y_sm,y_train_pred))

print(color.BOLD+'Confusin Matrix Train-:\n'+color.END,confusion_matrix(y_sm,y_train_pred))



print('======================================================================================')

print(color.BOLD+'Classification_report Test-:\n'+color.END,classification_report(y_test,y_test_pred))

print(color.BOLD+'Accuracy_score Test-:'+color.END,accuracy_score(y_test,y_test_pred))

print(color.BOLD+'Confusin Matrix Test-:\n'+color.END,confusion_matrix(y_test,y_test_pred))

print(color.BOLD+'Cohen_Kappa_score is-:'+color.END,cohen_kappa_score(y_test_pred,y_test,weights='quadratic'))

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier



from sklearn.model_selection import RandomizedSearchCV
randsv=RandomizedSearchCV(estimator=RandomForestClassifier(),

                          param_distributions=[{'n_estimators':np.arange(10,100,5),

                                                'max_depth':np.arange(5,30,2),

                                                'min_samples_leaf':np.arange(5,10,2),

                                               'min_samples_split':np.arange(5,10,2)}])
rand_fit=randsv.fit(X_sm,y_sm)

print(rand_fit)
print(rand_fit.best_estimator_)
rand_score=rand_fit.score(X_sm,y_sm)

rand_score
rfscv=RandomForestClassifier(max_depth=29,min_samples_leaf=5,min_samples_split=9,n_estimators=70)

rfscv.fit(X_sm,y_sm)

y_pred_rf=rfscv.predict(X_test)

print(color.BOLD+'Train Accuracy -:'+color.END,rfscv.score(X_sm,y_sm))

print(color.BOLD+'Test Accuracy -:'+color.END,rfscv.score(X_test,y_test))

print('\n'*2)

y_train_pred=rfscv.predict(X_sm)

y_test_pred=rfscv.predict(X_test)

print(color.BOLD+'Classification_report Train-:\n'+color.END,classification_report(y_sm,y_train_pred))

print(color.BOLD+'Accuracy_score Train-:'+color.END,accuracy_score(y_sm,y_train_pred))

print(color.BOLD+'Confusin Matrix Train-:\n'+color.END,confusion_matrix(y_sm,y_train_pred))



print('======================================================================================')

print(color.BOLD+'Classification_report Test-:\n'+color.END,classification_report(y_test,y_test_pred))

print(color.BOLD+'Accuracy_score Test-:'+color.END,accuracy_score(y_test,y_test_pred))

print(color.BOLD+'Confusin Matrix Test-:\n'+color.END,confusion_matrix(y_test,y_test_pred))

print(color.BOLD+'Cohen_Kappa_score is-:'+color.END,cohen_kappa_score(y_test_pred,y_test,weights='quadratic'))
