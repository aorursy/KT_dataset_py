import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



def concat_df(train_data, test_data):

    # Returns a concatenated df of training and test set on axis 0

    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)



def label_encoder(df,cat_features):

    for feature in cat_features:

        df[feature] = LabelEncoder().fit_transform(df[feature])

    return df



def grid_search_cv_print(X_train,y_train):

    param_grid = {'n_estimators': range(10, 71, 10),

                  'max_depth': range(3, 14, 2),

                  'min_samples_split': range(50, 201, 20)}

    gsearch = GridSearchCV(estimator=RandomForestClassifier(max_features='sqrt'),

                           param_grid=param_grid, scoring='roc_auc', cv=5, return_train_score=True)

    clf = gsearch.fit(X_train, y_train)

    print("best param:{0}\nbest score:{1}".format(clf.best_params_, clf.best_score_))
df_train = pd.read_csv("../input/titanic-machine-learning-from-disaster/train.csv")

df_test = pd.read_csv("../input/titanic-machine-learning-from-disaster/test.csv")

df_all = concat_df(df_train, df_test)



df_all['Embarked'] = df_all['Embarked'].fillna('S')

cat_features = ['Embarked', 'Sex', 'Pclass']

df_all = label_encoder(df_all,cat_features)



df_train = df_all.loc[df_all['Survived'].isin([np.nan]) == False]

df_test = df_all.loc[df_all['Survived'].isin([np.nan]) == True]
df_test_filter = df_test.filter(regex='Embarked|Sex|Pclass')

train_data = df_train.filter(regex='Embarked|Sex|Pclass')

train_labels = df_train['Survived']

# grid_search_cv_print(train_data,train_labels)
clf = RandomForestClassifier(min_samples_split=190,n_estimators=50,max_depth=9,max_features='sqrt',oob_score=True)

clf.fit(train_data,train_labels)

print(clf.oob_score_)
predictions = clf.predict(np.array(df_test_filter))

result = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': predictions})

result.to_csv('logistic_regression_predictions.csv', index=False, float_format='%1d')

result.head()