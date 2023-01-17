# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import sklearn

import os

print(os.listdir("../input"))



from sklearn.model_selection import train_test_split

from catboost import Pool, CatBoostClassifier, cv



# Any results you write to the current directory are saved as output.
path = '../input/'



hacker_numeric = pd.read_csv(path+'HackerRank-Developer-Survey-2018-Numeric.csv',na_values='#NULL!',low_memory=False)

hacker_qna = pd.read_csv(path+'HackerRank-Developer-Survey-2018-Codebook.csv')

df = pd.read_csv(path+ 'HackerRank-Developer-Survey-2018-Values.csv',na_values='#NULL!',low_memory=False)

print('Number of rows and columns in Hacker value data set',df.shape)
df = df[df.q3Gender.notnull()]
df = df[df.q3Gender!='Non-Binary']
X=df.drop(columns=['RespondentID','StartDate','EndDate','q3Gender'])

X=X.fillna(-1)



y=df['q3Gender'].map({'Male': 1, 'Female': 0})





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 

                                                    random_state=42)



X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, 

                                                    random_state=52)
categorical_features_indices = [i for i in range(len(X.columns))]
model = CatBoostClassifier(

        random_seed = 400,

    

    class_weights=[3.0,1.0],

        iterations=500,

    

    )

    

model.fit(

    X_train, y_train,

    cat_features = categorical_features_indices,

    eval_set=(X_valid, y_valid),

    

    use_best_model=True,

    verbose=True

    )



pred_train = model.predict(X_train)

pred_test = model.predict(X_test)



k = pd.DataFrame(sklearn.metrics.confusion_matrix(y_test,pred_test))

print((k[0][0]+k[1][1])/(k[0][0]+k[1][1]+k[0][1]+k[1][0]))

model = CatBoostClassifier(

        random_seed = 400,

    

    class_weights=[3.0,1.0],

        iterations=500,

    

    )

    

model.fit(

    X_train, y_train,

    cat_features = categorical_features_indices,

    eval_set=(X_valid, y_valid),

    

    use_best_model=True,

    verbose=True

    )



pred_train = model.predict(X_train)

pred_test = model.predict(X_test)



k = pd.DataFrame(sklearn.metrics.confusion_matrix(y_test,pred_test))

print((k[0][0]+k[1][1])/(k[0][0]+k[1][1]+k[0][1]+k[1][0]))

model = CatBoostClassifier(

        random_seed = 400,

    

    class_weights=[2.0,1.0],

        iterations=500,

    

    )

    

model.fit(

    X_train, y_train,

    cat_features = categorical_features_indices,

    eval_set=(X_valid, y_valid),

    

    use_best_model=True,

    verbose=True

    )



pred_train = model.predict(X_train)

pred_test = model.predict(X_test)



k = pd.DataFrame(sklearn.metrics.confusion_matrix(y_test,pred_test))

print((k[0][0]+k[1][1])/(k[0][0]+k[1][1]+k[0][1]+k[1][0]))

model = CatBoostClassifier(

        random_seed = 400,

    

    class_weights=[1.0,1.0],

        iterations=500,

    

    )

    

model.fit(

    X_train, y_train,

    cat_features = categorical_features_indices,

    eval_set=(X_valid, y_valid),

    

    use_best_model=True,

    verbose=True

    )



pred_train = model.predict(X_train)

pred_test = model.predict(X_test)



k = pd.DataFrame(sklearn.metrics.confusion_matrix(y_test,pred_test))

print((k[0][0]+k[1][1])/(k[0][0]+k[1][1]+k[0][1]+k[1][0]))