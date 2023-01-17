from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import MinMaxScaler



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")



scoring = 'accuracy'

seed=8



df = pd.read_csv('/kaggle/input/west-nile-virus-wnv-mosquito-test-results.csv')



# For this iteration, just drop when location information is missing

df = df.dropna()



# Scale numerical values

scaler = MinMaxScaler(feature_range=(0, 1))

lst_scaler = ['Wards','Census Tracts', 'Zip Codes', 'Community Areas','Historical Wards 2003-2015']

df[lst_scaler] = scaler.fit_transform(df[lst_scaler])



# One-hot encode categorical values

lst_onehot = ['SEASON YEAR','WEEK','SPECIES','TRAP_TYPE']

df_s = df[lst_onehot]

df_o = pd.get_dummies(df_s)

df = df.drop(lst_onehot,axis = 1)

df = pd.concat([df,df_o], axis=1)



# Remove outliers

df = df[df['NUMBER OF MOSQUITOES'] < 50] 



# Convert target to numerical values

df['RESULT'] = df['RESULT'].map({'positive': 1,'negative': 0})



y = df['RESULT']

X = df.drop(['TEST ID','BLOCK','TRAP','TEST DATE','RESULT','LOCATION'], axis=1)



X_train,X_test,Y_train,Y_test= train_test_split(X,y,random_state=seed,test_size=0.3)



def list_model():

    models = []

    models.append(('sklearn.neighbors.KNeighborsClassifier', KNeighborsClassifier()))

    models.append(('sklearn.ensemble.GradientBoostingClassifier', GradientBoostingClassifier()))

    models.append(('sklearn.ensemble.RandomForestClassifier', RandomForestClassifier()))

    models.append(('xgboost.XGBClassifier', XGBClassifier()))

    models.append(('lightgbm.LGBMClassifier', LGBMClassifier()))

    return models



def train_model(models):

    results = []

    for name, model in models:

        kfold = KFold(n_splits=10, random_state=seed)  

        cv = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

        results.append([name,cv.mean(),cv.std()])

    return(results)



model_candicates = list_model()

pd.DataFrame(train_model(model_candicates),columns=['model','mean','std'])