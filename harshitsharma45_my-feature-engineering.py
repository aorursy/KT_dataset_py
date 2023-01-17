import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd



data=pd.read_csv("/kaggle/input/kickstarter-projects/ks-projects-201801.csv",parse_dates=['launched','deadline'])

data
data=data.query('state!="live"')

data=data.assign(outcome=(data.state=="successful").astype(int))

data
data=data.assign(hour=data.launched.dt.hour,

           day=data.launched.dt.day,

           month=data.launched.dt.month,

           year=data.launched.dt.year)

data
cat_features=['country','currency','category']



from sklearn.preprocessing import LabelEncoder



encoder=LabelEncoder()



df=data[cat_features].apply(encoder.fit_transform)

df
df1=data[['hour','day','month','year','outcome']].join(df)

df1
valid_fraction=0.1

valid_size=int(len(df1)*valid_fraction)

def train_test_spli(df1):

    train=df1[:-2*valid_size]

    valid=df1[-2*valid_size:-valid_size]

    test=df1[-valid_size:]

    return train,valid,test

train,valid,test=train_test_spli(df1)
X_train=train.drop(columns=['outcome'])

y_train=train.outcome

X_test=test.drop(columns=['outcome'])

y_test=test.outcome
import lightgbm as lgb



def model_test(X_train,y_train,X_test,y_test):

    dtrain=lgb.Dataset(X_train,label=y_train)

    dtest=lgb.Dataset(X_test,label=y_test)



    param={'num_leaves':64,'objective':'binary','metric':'auc'}

    num_round=1000



    bst=lgb.train(param,dtrain,num_round,verbose_eval=False)



    from sklearn import metrics



    predictions=bst.predict(X_test)

    score=metrics.roc_auc_score(y_test,predictions)

    return score

print(model_test(X_train,y_train,X_test,y_test))
from category_encoders import CountEncoder



count_encoder=CountEncoder()



cat_features=['currency','category','country']

c_encoded=count_encoder.fit_transform(data[cat_features])

c_encoded
df2=data[['hour','day','month','year','outcome']].join(c_encoded)
train,valid,test=train_test_spli(df2)



X_train=train.drop(columns=['outcome'])

y_train=train.outcome

X_test=test.drop(columns=['outcome'])

y_test=test.outcome

print(model_test(X_train,y_train,X_test,y_test))


from category_encoders import TargetEncoder



cat_features=['country','currency','category']



train,valid,test=train_test_spli(data)



t_encoder=TargetEncoder(cols=cat_features)



a=t_encoder.fit_transform(train[cat_features],train.outcome)

b=t_encoder.fit_transform(test[cat_features],test.outcome)



a1=data[['hour','day','month','year','outcome']].join(a)

b1=data[['hour','day','month','year','outcome']].join(b)
X_train=a1.drop(columns=['outcome'])

y_train=a1.outcome

X_test=b1.drop(columns=['outcome'])

y_test=b1.outcome
print(model_test(X_train,y_train,X_test,y_test))
from category_encoders import CatBoostEncoder



train,valid,test=train_test_spli(data)



cb_encoder=CatBoostEncoder(cols=cat_features)



c1=cb_encoder.fit_transform(train[cat_features],train.outcome)

d1=cb_encoder.fit_transform(test[cat_features],test.outcome)



x=data[['hour','day','month','year','outcome']].join(c1)

y=data[['hour','day','month','year','outcome']].join(d1)



X_train=x.drop(columns=['outcome'])

y_train=x.outcome

X_test=y.drop(columns=['outcome'])

y_test=y.outcome



print(model_test(X_train,y_train,X_test,y_test))
#Feature Generation-interactions(joining of cat valus)

from sklearn.preprocessing import LabelEncoder



encoder=LabelEncoder()



interaction=data['country']+'-'+data['category']

data=data.assign(country_category=(encoder.fit_transform(interaction)))

bala=pd.Series(data.index,index=data.launched).sort_index()

bala
count_7_days=bala.rolling('7d').count()-1

count_7_days
count_7_days.index=bala.values

count_7_days.reindex(data.index)
def time(series):

    return series.diff().dt.total_seconds()/3600



dj=data[['category','launched']].sort_values('launched')



timedeltas=dj.groupby('category').transform(time)

timedeltas
dj1=timedeltas.fillna(timedeltas.mean()).reindex(data.index)

dj1
#Overfit of higher range in values not properly distributed



import seaborn as sns

import matplotlib.pyplot as plt



plt.hist(data.goal,range=(0,100000),bins=50)







import numpy as np



plt.hist(np.sqrt(data.goal),range=(1,400),bins=50)
plt.hist(np.log(data.goal),range=(0,25),bins=50)
df1
from sklearn.feature_selection import SelectKBest,f_classif



cat_features=df1.columns.drop('outcome')



feature1=df1.drop(columns=['outcome'])

feature2=df1.outcome





selected=SelectKBest(f_classif,k=5)



e=selected.fit_transform(feature1,feature2)

e
selected_feature=pd.DataFrame(selected.inverse_transform(e),index=df1.index,columns=cat_features)

selected_feature
selected_columns=selected_feature.columns[selected_feature.var()!=0]



df1[selected_columns]


from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectFromModel



X,y=df1[df1.columns.drop('outcome')],df1['outcome']



logistic=LogisticRegression(penalty='l1',solver='liblinear',random_state=7).fit(X,y)

model=SelectFromModel(logistic,prefit=True)

zx=model.transform(X)

zx
selected_feature=pd.DataFrame(model.inverse_transform(zx),index=X.index,columns=X.columns)

selected_cols=selected_feature.columns[selected_feature.var()!=0]

df1[selected_cols]