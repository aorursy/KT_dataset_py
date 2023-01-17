from matplotlib import pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeRegressor

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sklearn.tree import export_graphviz

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectFromModel

from os.path import join as pjoin
PATH_TO_DATA = '../input/lab12-classification-problem'
train = pd.read_csv(pjoin(PATH_TO_DATA, 'train.csv'))

labels = train['Label']

train.head()
test = pd.read_csv(pjoin(PATH_TO_DATA, 'test.csv'))

test.head()
def char_range(c1, c2):

    for c in range(ord(c1), ord(c2)+1):

        yield chr(c)



def check_symbol(symbol, with_capital=False):

    from_l = 'а'

    if with_capital:

        from_l = 'А'

    for num,letter in enumerate(char_range(from_l, 'я')):

        if symbol==letter:

            return num

    return -1



def generate_data(df):

    # is fits latter capital

    data = df.copy()

    data['is_firs_letter_capital'] = data['Word'].map(lambda x: x[0].isupper())



    # vowels and consonants     

    vowels = 'аоиеёэыуюя'

    data['count_vowels'] = data['Word'].map(lambda x: len(set(x).intersection(set(vowels))))

    consonants = 'бвгджзйклмнпрстфхцчшщ'

    data['count_consonants'] = data['Word'].map(lambda x: len(set(x).intersection(set(consonants))))



    data['first_letter'] = (data['Word'].map(lambda x: check_symbol(x[0], with_capital=True))).astype('int')

    data['last_letter'] = (data['Word'].map(lambda x: check_symbol(x[-1]))).astype('int')

    data['penultimate_letter'] = (data['Word'].map(lambda x: check_symbol(x[-2]) if len(x) > 2 else -1)).astype('int')

    data['before_the_penultimate_letter'] = (data['Word'].map(lambda x: check_symbol(x[-3]) if len(x) > 3 else -1)).astype('int')



    df = data.drop(['Word'], axis=1)

    return df



def metrics(y_holdout, predict):

    return {

    "accuracy": accuracy_score(y_holdout, predict),

    "f1": f1_score(y_holdout, predict),

    'precision_score': precision_score(y_holdout, predict),

    'recall_score': recall_score(y_holdout, predict)

}
train = generate_data(train)

train = train.drop(['Label'], axis=1)

train.head()
# generate train and holdout data 

X_train, X_holdout, y_train, y_holdout = train_test_split(train, labels, test_size=0.30,random_state=17)
# fit single tree

tree = DecisionTreeClassifier()

# tree = DecisionTreeClassifier(max_depth=best_tree_params['max_depth'], max_features=best_tree_params['max_features'])

tree.fit(train, labels)



tree_pred = tree.predict(X_holdout)

metrics(y_holdout, tree_pred)
# fit tree to get best params 

tree = DecisionTreeClassifier()

tree_params = {'max_depth': [None, 3, 5 ,10, 15, 20, 25, 30],

               'max_features': range(1,7)}



# tree cross validation

tree_grid = GridSearchCV(tree, tree_params, cv=5, n_jobs=-1, verbose=True)



tree_grid.fit(train, labels)



best_tree_params = tree_grid.best_params_



best_tree = DecisionTreeClassifier(max_depth=best_tree_params['max_depth'], max_features=best_tree_params['max_features'])

best_tree.fit(train, labels)



best_tree_pred = best_tree.predict(X_holdout)

output = metrics(y_holdout, best_tree_pred)

output.update(best_tree_params)

output
# fit single knn

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)



knn_pred = knn.predict(X_holdout)

metrics(y_holdout, knn_pred)
# fit knn to get best params 

# use StandardScaler for normalize params

knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_jobs=-1))])

knn_params = {'knn__n_neighbors': [10, 30, 50, 70, 90, 110, 130, 150]}



knn_grid = GridSearchCV(knn_pipe, knn_params, cv=5, n_jobs=-1, verbose=True)



knn_grid.fit(X_train, y_train)



best_params = knn_grid.best_params_



best_knn = KNeighborsClassifier(n_neighbors=best_params['knn__n_neighbors'])

best_knn.fit(X_train, y_train)



best_knn_pred = best_knn.predict(X_holdout)

output = metrics(y_holdout, knn_pred)

output.update(best_params)

output
# fit single forst

forest = RandomForestClassifier()

forest.fit(train, labels)



forest_predict = forest.predict(X_holdout)

metrics(y_holdout, forest_predict)
# fit forest to best params

forest_params = {'n_estimators': [10, 40, 80, 150, 250, 350, 450],

                'max_depth': [ None, 2, 10, 15, 20, 25, 30],

                }



forest_grid = GridSearchCV(forest, forest_params,cv=3, n_jobs=-1, verbose=True)



forest_grid.fit(train, labels)



best_forest_params = forest_grid.best_params_



best_forest = RandomForestClassifier(n_estimators=best_forest_params['n_estimators'], 

                                max_depth = best_forest_params['max_depth'], 

                                n_jobs=-1, 

                                random_state=17)

best_forest.fit(train, labels)



best_forest_predict = best_forest.predict(X_holdout)

output = metrics(y_holdout, best_forest_predict)

output.update(best_params)

output
test_data = generate_data(test)

test_data.head()
predict = best_forest.predict_proba(test_data)[:,1]

predict
predicted = pd.DataFrame(predict, columns =['Prediction'])

predicted['Id'] = predicted.index

predicted.head()
predicted.to_csv('predict.csv', index=False)