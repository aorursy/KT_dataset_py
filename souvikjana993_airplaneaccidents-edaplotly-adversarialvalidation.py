# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import gc

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.float_format', lambda x: '%.3f' % x)



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV,StratifiedKFold,KFold,GroupKFold,train_test_split

from sklearn.metrics import f1_score,roc_auc_score,classification_report,confusion_matrix

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#importing plotly

import chart_studio

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import cufflinks as cf

cf.go_offline()



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/train.csv')

test_df = pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/test.csv')

sample_sub_df = pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/sample_submission.csv')
train_df.head()
test_df.head()
sample_sub_df.head()
print(f'Shape of training data: {train_df.shape}')

print(f'Shape of testing data: {test_df.shape}')
train_df.isna().sum()
(train_df==0).sum()
test_df.isna().sum()
(test_df==0).sum()
safetysummary=train_df.groupby('Severity')[['Safety_Score','Control_Metric','Adverse_Weather_Metric','Max_Elevation','Violations','Cabin_Temperature','Turbulence_In_gforces','Total_Safety_Complaints','Days_Since_Inspection']].mean()





lists=[]

for idx,i in enumerate(safetysummary.columns):

    bool_array=[False]*len(safetysummary.columns)

    bool_array[idx]=True

    lists.append(

        dict(label=str(i),

             method="update",

             args=[{"visible":bool_array},

                   {"title":i}]))



layout=dict(

    updatemenus=list([

        dict(

            active=0,

            buttons=lists,

        )

    ])

)



safetysummary.iplot(kind='bar', xTitle='Severity', yTitle='Magnitude',title='Severity to mean scores',layout=layout)
train_df.head()
#correlation b/w safety scores and days since inspection

import plotly.graph_objects as go

from cufflinks import tools

import chart_studio.plotly as py

safetysummary1=train_df.groupby('Days_Since_Inspection')[['Safety_Score']].mean()

safetysummary1.iplot(kind='bar', xTitle='Days_Since_Inspection',fill=True, yTitle='Magnitude',title='Severity to mean scores')

train_data=train_df[['Safety_Score','Control_Metric','Adverse_Weather_Metric','Max_Elevation','Cabin_Temperature','Turbulence_In_gforces','Total_Safety_Complaints','Days_Since_Inspection']]



lists=[]

for idx,i in enumerate(train_data.columns):

    bool_array=[False]*len(train_data.columns)

    bool_array[idx]=True

    lists.append(

        dict(label=str(i)+" histogram",

             method="update",

             args=[{"visible":bool_array},

                   {"title":i}]))



layout=dict(

    updatemenus=list([

        dict(

            active=0,

            buttons=lists,

        )

    ])

)



#X_train[['Safety_Score','Control_Metric','Adverse_Weather_Metric']].iplot(kind="hist",title="ass",layout=layout)

train_df[['Safety_Score','Control_Metric','Adverse_Weather_Metric','Max_Elevation','Cabin_Temperature','Turbulence_In_gforces','Total_Safety_Complaints','Days_Since_Inspection']].iplot(kind="hist",title="ass",layout=layout)
num_columns=[i for i in train_df.columns if train_df[i].dtype in [np.int64,np.float64]]

train_df[num_columns].corr()
#train_df['Days_Since_Inspection'].hist(by=train_df['Severity']);

box_age = train_df[['Safety_Score', 'Severity']]

box_age.pivot(columns='Severity', values='Safety_Score').iplot(kind='box')
#outlier processing clipping to 99th percentile

percentiles = train_df['Total_Safety_Complaints'].quantile([0.01,0.99]).values

train_df['Total_Safety_Complaints'] = np.clip(train_df['Total_Safety_Complaints'], percentiles[0], percentiles[1])
#train_df['Days_Since_Inspection'].hist(by=train_df['Severity']);

box_age = train_df[['Total_Safety_Complaints', 'Severity']]

box_age.pivot(columns='Severity', values='Total_Safety_Complaints').iplot(kind='box')
train_df.columns
train_df[['Turbulence_In_gforces','Control_Metric']].corr()

#train_df[['Turbulence_In_gforces','Control_Metric']].iplot(kind='scatter', xTitle='Control_Metric', yTitle='Turbulence_In_gforces',title='Turbulence to Control')
codesummary=train_df.groupby('Accident_Type_Code')[['Severity','Safety_Score','Control_Metric','Adverse_Weather_Metric','Max_Elevation','Cabin_Temperature','Turbulence_In_gforces','Total_Safety_Complaints','Days_Since_Inspection','Violations']].mean()





lists=[]

for idx,i in enumerate(codesummary.columns):

    bool_array=[False]*len(codesummary.columns)

    bool_array[idx]=True

    lists.append(

        dict(label=str(i),

             method="update",

             args=[{"visible":bool_array},

                   {"title":i}]))



layout=dict(

    updatemenus=list([

        dict(

            active=0,

            buttons=lists,

        )

    ])

)



codesummary.iplot(kind='bar', xTitle='Accident Type Code', yTitle='Magnitude',title='Accident type code exploration',layout=layout)
train_df.columns
train_df.head()
pd.crosstab(train_df['Violations'],train_df['Severity'],normalize='index')
# X_train['Total_Safety_Complaints'] = np.power(2, X_train['Total_Safety_Complaints'])

# X_train['Days_Since_Inspection'] = np.power(2, X_train['Days_Since_Inspection'])

# X_train['Safety_Score'] = np.power(2, X_train['Safety_Score'])
def flag_features(train_df):

    train_df['Violations_FLAG']=np.where(train_df['Violations']>0,1,0)

    train_df['Total_Safety_Complaints_FLAG']=np.where(train_df['Total_Safety_Complaints']>0,1,0)

    return train_df.drop(['Accident_ID'],axis=1)



#def grouped_features(df):

    

    



train_df=flag_features(train_df)

test_df2=flag_features(test_df)
train_df.head()
class_map = {

    'Minor_Damage_And_Injuries': 0,

    'Significant_Damage_And_Fatalities': 1,

    'Significant_Damage_And_Serious_Injuries': 2,

    'Highly_Fatal_And_Damaging': 3

}

inverse_class_map = {

    0: 'Minor_Damage_And_Injuries',

    1: 'Significant_Damage_And_Fatalities',

    2: 'Significant_Damage_And_Serious_Injuries',

    3: 'Highly_Fatal_And_Damaging'

}
train_df2=train_df.iloc[:9000,:]

sample_base=train_df.iloc[9001:,:]
X=train_df2.drop(['Severity'],axis=1)

y=train_df2['Severity'].map(class_map)





rf=RandomForestClassifier(random_state=123)

n_splits = 10



kf=StratifiedKFold(n_splits=n_splits,random_state=123)





train_oof = np.zeros((train_df.shape[0],))

test_preds = 0





for i,(train_index,test_index) in enumerate(kf.split(X,y)):

    #print("TRAIN:", train_index, "TEST:", test_index)

    X_train, X_test = X.iloc[train_index].values, X.iloc[test_index].values

    y_train, y_test = y.iloc[train_index].values, y.iloc[test_index].values

    rf.fit(X_train,y_train)

    y_pred=rf.predict(X_test)

    print("Fold F1Score:", f1_score(y_test, y_pred,average='weighted'))

    test_preds += rf.predict(test_df2)/n_splits

    del X_train, X_test, y_train, y_test

    gc.collect()
test_preds2=np.round(test_preds).astype(int)

submission_file=pd.DataFrame(test_df['Accident_ID'])

submission_file['Severity']=test_preds2

submission_file['Severity']=submission_file['Severity'].map(inverse_class_map)
sample_base2=sample_base.drop(['Severity'],axis=1)

y_predoof=rf.predict(sample_base2)

y_oof=sample_base['Severity'].map(class_map)

print(f1_score(y_oof,y_predoof,average='weighted'))

print(confusion_matrix(y_oof,y_predoof))
#submission_file.to_csv('submission1.csv', index=False)
import xgboost as xgb

xtrain = pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/train.csv')

id_train = xtrain['Accident_ID']

ytrain = xtrain['Severity']

xtrain.drop(['Severity','Accident_ID'], axis = 1, inplace = True)

xtrain.fillna(-999, inplace = True)





xtest = pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/test.csv')

id_test = xtest['Accident_ID']

xtest.drop(['Accident_ID'], axis = 1, inplace = True)

xtest.fillna(-999, inplace = True)





# add identifier and combine

xtrain['istrain'] = 1

xtest['istrain'] = 0

xdat = pd.concat([xtrain, xtest], axis = 0)



# convert non-numerical columns to integers

df_numeric = xdat.select_dtypes(exclude=['object'])

df_obj = xdat.select_dtypes(include=['object']).copy()

    

for c in df_obj:

    df_obj[c] = pd.factorize(df_obj[c])[0]

    

xdat = pd.concat([df_numeric, df_obj], axis=1)

y = xdat['istrain']; xdat.drop('istrain', axis = 1, inplace = True)



skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 44)

xgb_params = {

        'learning_rate': 0.05, 'max_depth': 4,'subsample': 0.9,

        'colsample_bytree': 0.9,'objective': 'binary:logistic',

        'silent': 1, 'n_estimators':100, 'gamma':1,

        'min_child_weight':4

        }   

clf = xgb.XGBClassifier(**xgb_params, seed = 10)







for train_index, test_index in skf.split(xdat, y):

        x0, x1 = xdat.iloc[train_index], xdat.iloc[test_index]

        y0, y1 = y.iloc[train_index], y.iloc[test_index]        

        print(x0.shape)

        clf.fit(x0, y0, eval_set=[(x1, y1)],

               eval_metric='logloss', verbose=False,early_stopping_rounds=10)

                

        prval = clf.predict(x1)

        print(roc_auc_score(y1,prval,average='weighted'))