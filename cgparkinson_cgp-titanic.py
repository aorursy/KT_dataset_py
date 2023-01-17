# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df.head()
from sklearn.impute import SimpleImputer

def proc_df(df):
    df.set_index('PassengerId', inplace = True)
    cabin_letter = pd.Series([str(elt)[:1] for elt in df['Cabin']])
    cabin_letter = cabin_letter.astype('category',categories=list('ABCDEFGTn'))
    df['Cabin_Letter'] = cabin_letter
    df = df.drop(['Name', 'Cabin', 'Ticket'], axis=1)
    df = pd.get_dummies(df)
    
    #my_imputer = SimpleImputer()
    #df = pd.DataFrame(my_imputer.fit_transform(df), columns=df.columns, index=df.index)
    return df

def drop_label(df):
    X = df.drop(['Survived'], axis=1)
    return X
df = proc_df(df)
X = drop_label(df)
y = df.Survived.astype(int)

from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)
X.head()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
def evaluate_model(pre_pipe, model, test_X, test_y):
    print(pre_pipe)
    print(model)
    
    pipeline = make_pipeline(pre_pipe, model)

    scores = cross_validate(pipeline, X, y, scoring='accuracy', cv=5)
    print("Test scores: {}".format(scores['test_score']))

    # model = RandomForestRegressor().fit(X, y)
    pipeline.fit(X, y)
    print("Accuracy score: {}".format(accuracy_score(pipeline.predict(X).astype(int), y)))

    # pre_pipe is workaround for PermutationImportance lack of detection of Imputer https://github.com/TeamHG-Memex/eli5/issues/262
    perm = PermutationImportance(pipeline, random_state=1)
    perm.fit(test_X, test_y)
    display(eli5.show_weights(perm, feature_names = X.columns.tolist()))
    return pipeline
pre_pipe =  make_pipeline(SimpleImputer())
test_X=pre_pipe.fit_transform(test_X)
rf_model = evaluate_model(pre_pipe, RandomForestClassifier(n_estimators=100), test_X, test_y)
xgb_model = evaluate_model(pre_pipe, XGBClassifier(), test_X, test_y)
logit_model = evaluate_model(pre_pipe, LogisticRegression(solver='lbfgs', max_iter=1000), test_X, test_y)
df_test = pd.read_csv('../input/test.csv')
df = proc_df(df_test)
df.columns
results_combined_model = (rf_model.predict(df) + xgb_model.predict(df) + logit_model.predict(df))

results = pd.Series([round(item/3) for item in results_combined_model])
def save_for_kaggle(df, results):
#     df['result'] = pd.Series(results, index=df.index)
#     print(df['result'])
    data_to_submit = pd.DataFrame({
        'PassengerId':df.index,
        'Survived':results.astype(int)
    })
    data_to_submit.to_csv('csv_to_submit.csv', index = False)
save_for_kaggle(df, results)