import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
train = pd.read_csv("../input/widsdatathon2020/training_v2.csv")

test = pd.read_csv("../input/widsdatathon2020/unlabeled.csv")
train.shape, test.shape
train.head()
test.head()
train.describe()
test.describe()
train.isnull().sum()/len(train)*100
f,ax=plt.subplots(1,2,figsize=(18,8))

train['hospital_death'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('hospital_death')

ax[0].set_ylabel('')

sns.countplot('hospital_death',data=train,ax=ax[1])

ax[1].set_title('hospital_death')

plt.show()
train.drop("readmission_status",inplace=True,axis=1)

test.drop("readmission_status",inplace=True,axis=1)
test.drop("hospital_death",inplace=True,axis=1)

y_train = train[['encounter_id', 'patient_id', 'hospital_id',"hospital_death"]].copy()

train.drop("hospital_death",inplace=True,axis=1)
non_categorical = train.loc[:,train.dtypes!="object"].columns
categorical = [c for c in train[non_categorical].columns if (train[c].nunique()<10)]

non_categorical = [c for c in train[non_categorical].columns if (train[c].nunique()>=10)]
print(train[non_categorical].isnull().sum()/len(train))

a = train[non_categorical].isnull().sum()/len(train)>0.40 ## editable

missing_m40 = train[non_categorical].loc[:,a].columns



a = train[non_categorical].isnull().sum()/len(train)<=0.40 ## editable

missing_l40 = train[non_categorical].loc[:,a].columns

del a
a = train.isnull().sum()/len(train)!= 0 ## editable

missing = train.loc[:,a].columns

del a

for i in missing:

    train[str(i)+"_Na"]=pd.get_dummies(train[i].isnull(),prefix=i).iloc[:,0]

    test[str(i)+"_Na"]=pd.get_dummies(test[i].isnull(),prefix=i).iloc[:,0]



for i in missing_l40:

    for j in train.hospital_id.unique():

        train[i][train.hospital_id==j]=train[i][train.hospital_id==j].fillna(train[i][train.hospital_id==j].median())

    for k in test.hospital_id.unique():

        test[i][test.hospital_id==k]=test[i][test.hospital_id==k].fillna(test[i][test.hospital_id==k].median())
train["apache_4a_hospital_death_prob"]=train["apache_4a_hospital_death_prob"].replace({-1:np.nan})

test["apache_4a_hospital_death_prob"]=test["apache_4a_hospital_death_prob"].replace({-1:np.nan})



train["apache_4a_icu_death_prob"]=train["apache_4a_icu_death_prob"].replace({-1:np.nan})

test["apache_4a_icu_death_prob"]=test["apache_4a_icu_death_prob"].replace({-1:np.nan})
a = train[non_categorical].isnull().sum()/len(train)<=0.40 ## editable

missing_l40 = train[non_categorical].loc[:,a].columns

for i in missing_l40:

    train[i] = train[i].fillna(train[i].median())

    

a = test[non_categorical].isnull().sum()/len(test)<=0.40 ## editable

missing_l40 = test[non_categorical].loc[:,a].columns

for i in missing_l40:

    test[i] = test[i].fillna(train[i].median())

del a, missing_l40, missing_m40
categorical=np.concatenate([train.loc[:,train.dtypes=="object"].columns.tolist(),categorical])
train[categorical].isnull().sum()/len(train)
test[categorical].isnull().sum()/len(test)
train[categorical].nunique()
## imputador gender

train["gender"][train.height>167]=train["gender"][train.height>167].fillna("M")

train["gender"][train.height<=167]=train["gender"][train.height<=167].fillna("F")
for i in categorical:

    train[i] = train[i].fillna(train[i].value_counts().index[0])

    test[i] = test[i].fillna(train[i].value_counts().index[0])
categorical = train.loc[:,train.dtypes=="object"].columns.tolist()
train[categorical].nunique()
train["hospital_admit_source"]=train["hospital_admit_source"].replace({'Other ICU':"ICU",'ICU to SDU':"SDU",

                                       'Step-Down Unit (SDU)':"SDU",

                                      'Acute Care/Floor':"Floor",

                                      'Other Hospital':"Other"})

test["hospital_admit_source"]=test["hospital_admit_source"].replace({'Other ICU':"ICU",'ICU to SDU':"SDU",

                                       'Step-Down Unit (SDU)':"SDU",

                                      'Acute Care/Floor':"Floor",

                                      'Other Hospital':"Other"})

train["apache_2_bodysystem"] = train["apache_2_bodysystem"].replace({'Undefined Diagnoses':"UD",

                                                                    'Undefined diagnoses':"UD"})

test["apache_2_bodysystem"] = test["apache_2_bodysystem"].replace({'Undefined Diagnoses':"UD",

                                                                    'Undefined diagnoses':"UD"})
train = train.join(pd.get_dummies(train[categorical]).drop("gender_F",axis=1))

test = test.join(pd.get_dummies(test[categorical]).drop("gender_F",axis=1))

train.drop(categorical,axis=1,inplace=True)

test.drop(categorical,axis=1,inplace=True)
non = ['encounter_id', 'patient_id', 'hospital_id','icu_id']
correlated_features = set()

train1 = train.drop(non,axis=1) 

correlation_matrix = train1.corr()

del train1
for i in range(len(correlation_matrix.columns)):

     for j in range(i):

            if abs(correlation_matrix.iloc[i, j]) ==  1:

                colname = correlation_matrix.columns[i]

                correlated_features.add(colname)

correlated_features=list(correlated_features)
train.drop(correlated_features,axis=1,inplace=True)

test.drop(correlated_features,axis=1,inplace=True)
train.shape, test.shape
train.drop("hospital_admit_source_Observation",axis=1,inplace=True)
train = train.set_index("encounter_id")

test = test.set_index("encounter_id")

y_train = y_train.set_index("encounter_id")
test = test.fillna(0)
from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold, GroupKFold

drop_cols = ['patient_id', 'hospital_id','icu_id']

#drop_cols = np.concatenate([drop_cols,perdidos])

gf = GroupKFold(n_splits=4)

groups = np.array(train.hospital_id)

test_probs = []

train_probs = []



for i,(a,b) in enumerate(gf.split(train,y_train.loc[train.index, "hospital_death"],groups)) :

    Xt = train.iloc[a,:]

    yt = y_train.loc[Xt.index, "hospital_death"]

    Xt = Xt.drop(drop_cols, axis=1)

    Xt = Xt.fillna(0)

    

    Xv = train.iloc[b,:]

    yv = y_train.loc[Xv.index, "hospital_death"]

    Xv = Xv.drop(drop_cols, axis=1)

    Xv = Xv.fillna(0)

    print("*+*+*+*+*entrenando fold: {} ".format(i+1))

    

    learner = LGBMClassifier(n_estimators=10,learning_rate=0.03,num_iterations=3400,lambda_l2=7 ,lambda_l1 =7

                                 ,num_leaves =7,max_depth=5,min_data_in_leaf =500,early_stopping_rounds=200,feature_fraction= 0.8

                            ,bagging_fraction=0.85,bagging_freq=10)

    

    learner.fit(Xt, yt  , eval_metric="auc",eval_set= [(Xt, yt),(Xv, yv)], verbose=50)

    

    

    train_probs.append(pd.Series(learner.predict_proba(Xv)[:, -1],

                                index=Xv.index, name="probs"+ str(i)))

    test_probs.append(pd.Series(learner.predict_proba(test.drop(drop_cols, axis=1))[:, -1],

                                index=test.index, name="fold_" + str(i)  ))

      

test_probs = pd.concat(test_probs, axis=1).mean(axis=1)

train_probs = pd.concat(train_probs, axis=1)
test_probs = pd.DataFrame(test_probs.rename("hospital_death"))

test_probs.to_csv("lightgbm_baseline.csv", header=True)