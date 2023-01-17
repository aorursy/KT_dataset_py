import pandas as pd

df = pd.read_csv('../input/train.csv')
from catboost import CatBoostClassifier

import numpy as np



df1 = df.copy().drop('id',1)



for column in df1:

    if df[column].isnull().sum() !=0:

        df1[column] = df[column].fillna(df[column].median())

        

df1.isnull().sum().sum()
df1.head(1)
X = df1.drop('gender',1)

X.head()



y = df1['gender']

y = pd.factorize(y)[0]
def rata(X,feat1,feat2):

    avg = (X[feat1]+X[feat2])/2

    X_avg = pd.concat([X.drop([feat1,feat2],1),avg],1)

    return X_avg
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.75, random_state=42)



model = CatBoostClassifier(

    iterations = 1000,

    custom_loss=['Accuracy'],

    random_seed=42,

    logging_level='Silent',

    loss_function = 'MultiClass'

)



def predict_score(name, X_train, y_train, X_validation, y_validation):

    model.fit(X_train, y_train)

    y_predict = model.predict(X_validation)

    print(name, accuracy_score(y_predict, y_validation))



predict_score('ga diapa2in',X_train, y_train, X_validation, y_validation)



predict_score('drop poi 1',

              X_train.drop('poi_1',1), y_train, 

              X_validation.drop('poi_1',1), y_validation)



predict_score('drop poi 3',

              X_train.drop('poi_3',1), y_train, 

              X_validation.drop('poi_3',1), y_validation)



predict_score('rata2 poi 1 & poi 3',

              rata(X_train,'poi_1','poi_3'), y_train, 

              rata(X_validation, 'poi_1', 'poi_3'), y_validation)





predict_score('drop fac 4',

              X_train.drop('fac_4',1), y_train, 

              X_validation.drop('fac_4',1), y_validation)



predict_score('drop fac 5',

              X_train.drop('fac_5',1), y_train, 

              X_validation.drop('fac_5',1), y_validation)



predict_score('rata2 fac4 & fac5',

              rata(X_train,'fac_4','fac_5'), y_train, 

              rata(X_validation,'fac_4','fac_5'), y_validation)





predict_score('drop fac 1',

              X_train.drop('fac_1',1), y_train, 

              X_validation.drop('fac_1',1), y_validation)



predict_score('drop price month',

              X_train.drop('price_monthly',1), y_train, 

              X_validation.drop('price_monthly',1), y_validation)



predict_score('rata2 fac1 & price_monthly',

              rata(X_train,'fac_1','price_monthly'), y_train, 

              rata(X_validation,'fac_1','price_monthly'), y_validation)

for col in X:

    X_temp_train = X_train.drop(col,1)

    X_temp_val = X_validation.drop(col,1)

    model.fit(X_temp_train, y_train)

    y_predict = model.predict(X_temp_val)

    print("drop",col)

    score = accuracy_score(y_predict, y_validation)

    print("score",score)
explored = []

for col in X:

    explored.append(col)

    for col2 in X:

        if col2 not in explored:

            X_temp_train = X_train.drop(col,1).drop(col2,1)

            X_temp_val = X_validation.drop(col,1).drop(col2,1)

            model.fit(X_temp_train, y_train)

            y_predict = model.predict(X_temp_val)

            print("drop",col,col2)

            score = accuracy_score(y_predict, y_validation)

            print("score",score)
from sklearn.model_selection import RandomizedSearchCV

params = {'depth':[4,5,6,7,8,9,10],

          'iterations':[250,100,500,1000],

          'learning_rate':[0.03,0.001,0.01,0.1,0.2,0.3], 

          'l2_leaf_reg':[3,1,5,10,100],

          'border_count':[32,5,10,20,50,100,200],

          'ctr_border_count':[50,5,10,20,100,200],

         }



rf_random = RandomizedSearchCV(

    estimator=model,

    param_distributions=params,

)



rf_random.fit(X_train,y_train)

best_random = rf_random.best_estimator_



print(rf_random.best_params_)
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test_data.csv')



train_df = train_df.drop('id',1)

X_test = test_df.drop('id',1)



X_train = train_df.drop('gender',1)

y_train = train_df['gender']





# custom

y_train, label = pd.factorize(y_train)

X_train = X_train.drop('fac_7',1)



X_test = X_test.drop('fac_7',1)



# custom

model = CatBoostClassifier(

    learning_rate= 0.03,

    l2_leaf_reg= 1,

    iterations= 1000,

    depth= 6,

    border_count= 100,

    custom_loss=['Accuracy'],

    random_seed=42,

    logging_level='Silent',

    loss_function = 'MultiClass'

)

model.fit(X_train, y_train)

y_submit = pd.DataFrame(label[model.predict(X_test).astype(int)])

out = pd.concat([test_df['id'],y_submit],axis=1)

out.to_csv('../output/submit9.csv', index=False, header=['id','gender'])