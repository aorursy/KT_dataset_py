import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

import xgboost as xgb

warnings.filterwarnings(action='ignore', category=DeprecationWarning)



from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.externals import joblib

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, make_scorer



from keras import layers

from keras import optimizers

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import Adam, SGD, RMSprop, Adagrad

def plot_history(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(x, acc, 'b', label='Training acc')

    plt.plot(x, val_acc, 'r', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(x, loss, 'b', label='Training loss')

    plt.plot(x, val_loss, 'r', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()



def train_to_test_split(train_x, train_y):

    sc = MinMaxScaler()

    train_x = sc.fit_transform(train_x)

#     joblib.dump(sc.fit(train_x), scaler_filename) 

    return train_test_split(train_x, train_y, test_size=0.3, random_state=5)





def kfold(clf):

    kf = KFold(len(dt.index), n_folds=10, shuffle=True, random_state=111)

    outcomes = []

    fold = 0

    for train_index, test_index in kf:

        fold += 1

        X_train, X_test = dt.loc[:,'age':'thal'].values[train_index], dt.loc[:,'age':'thal'].values[test_index]

        y_train, y_test = dt['target'].values[train_index], dt['target'].values[test_index]

        clf.fit(X_train, y_train)

        prediction = clf.predict(X_test)

        accuracy = accuracy_score(y_test, prediction)

        outcomes.append(accuracy)

        print("Fold {0} accuracy: {1}".format(fold, accuracy))     

    mean_outcome = np.mean(outcomes)

    print("Mean Accuracy: {0}".format(mean_outcome)) 



    

def grid_search(clf, parameters, X_train, y_train):

    acc_scorer = make_scorer(accuracy_score)

    grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer, n_jobs=-1, cv=5)

    grid_obj = grid_obj.fit(X_train, y_train.values.ravel())

    clf = grid_obj.best_estimator_

    return(clf)

    

    

def classifier(clf):

    

    clf_name = clf.__class__.__name__

    parameters = parameter_set(clf_name)

    print(parameters)

    # return predictions from gird search best model

    clf = grid_search(clf, parameters, X_train, y_train)

    

    # fit best model

    clf.fit(X_train, y_train.values.ravel())

    

    predictions = clf.predict(X_test) 

    if clf_name == 'XGBClassifier':

        predictions = [value for value in predictions]

    return(predictions)



def parameter_set(clf_name):

    if clf_name == 'RandomForestClassifier':

        parameters = {'n_estimators': [5, 10, 50, 100, 150, 200], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

#               'max_depth': list(range(2,10)), 

#               'min_samples_split': list(range(2,5)),

#               'min_samples_leaf': list(range(1,5)),

              'verbose': [0]

             }

    if clf_name == 'DecisionTreeClassifier':

        parameters = {

              'max_depth': list(range(2,10)), 

              'min_samples_split': list(range(2,10))

             }

    if clf_name == 'AdaBoostClassifier':

        parameters = {

            "n_estimators" : [5, 10, 50, 100, 150, 200],

            "algorithm" :  ["SAMME", "SAMME.R"],

            'learning_rate':[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]

             }

    if clf_name == 'GradientBoostingClassifier':

        parameters = {

            "loss":["deviance"],

            "learning_rate": [0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7],

#             "min_samples_split": list(range(2,5)),

#             "min_samples_leaf": list(range(1,5)),

            "max_depth": [3,5,8],

            "max_features": ["log2","sqrt"],

            "criterion": ["friedman_mse",  "mae"],

            "subsample": [0.5, 0.8, 0.9, 1.0],

            "n_estimators": [5, 10, 50, 100, 150, 200]

             }

    if clf_name == 'XGBClassifier':

        parameters = {

            'learning_rate': np.linspace(0.01, 0.5, 9),

#             'max_depth': list(range(5,10)),

#             'min_child_weight': list(range(3,10)),

            'gamma': np.linspace(0, 0.5, 11),

#             'subsample': [0.8, 0.9],

#             'colsample_bytree': [0.3, 0.4, 0.5 , 0.7, 0.8, 0.9],

            'objective': ['binary:logistic']

        }

    return(parameters)
dt = pd.read_csv("../input/heart.csv")

X_train, X_test, y_train, y_test = train_to_test_split(dt[dt.columns[0:13]] ,dt[['target']])
model = Sequential()

model.add(Dense(units=512, activation='relu', input_dim=np.shape(X_train)[1]))

model.add(Dense(units=512, activation='relu'))

model.add(Dropout(0.3, noise_shape=None, seed=None))

model.add(Dense(units=128, activation='relu'))

model.add(Dense(units=128, activation='relu'))

model.add(Dropout(0.3, noise_shape=None, seed=None))

model.add(Dense(units=32, activation='relu'))

model.add(Dropout(0.3, noise_shape=None, seed=None))

model.add(Dense(units=16, activation='relu'))

model.add(Dropout(0.3, noise_shape=None, seed=None))

model.add(Dense(units=1, activation='sigmoid'))
model.summary()
opt = optimizers.SGD()

model.compile(loss="binary_crossentropy", optimizer=opt, metrics = ["accuracy"])
result = model.fit(X_train, y_train ,epochs=100, batch_size=16, validation_split=0.1, shuffle=True)
MLP_acc = accuracy_score(y_test, model.predict_classes(X_test))

MLP_loss = log_loss(y_test, model.predict_classes(X_test))

print(MLP_acc)

print(MLP_loss)
plot_history(result)
classifiers = [

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    xgb.XGBClassifier()

]
# Logging for Visual Comparison# Logging for Visual Comparison

log_cols=["Classifier", "Accuracy"]

log = pd.DataFrame([['MLP', MLP_acc*100]],columns=log_cols)



for clf in classifiers:

    

    name = clf.__class__.__name__

    clf.fit(X_train, y_train.values.ravel())

    print("="*30)

    print(name)

    

    train_predictions = clf.predict(X_test)

    acc = accuracy_score(y_test, train_predictions)

    print("Accuracy: {:.2%}".format(acc))

    

    log_entry = pd.DataFrame([[name, acc*100]], columns=log_cols)

    log = log.append(log_entry)

    

print("="*30)
# Grid Search

for clf in classifiers:

    name = clf.__class__.__name__ + 'Grid'

    print("="*30)

    print(name)

    train_predictions = classifier(clf)

    acc = accuracy_score(y_test, train_predictions)

    print("Accuracy: {:.2%}".format(acc))

    log_entry = pd.DataFrame([[name, acc*100]], columns=log_cols)

    log = log.append(log_entry)

print("="*30)

sns.set_color_codes("muted")

g=sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")



plt.xlabel('Accuracy %')

plt.title('Classifier Accuracy')  



for p in g.patches:

    x = p.get_x() + p.get_width() +.3

    y = p.get_y() + p.get_height()/2 + .1

    g.annotate("%.2f %%" % (p.get_width()), (x, y))



plt.show()



clf = classifiers[1] 

clf.fit(X_train, y_train.values.ravel())

# model_name = '../model/'+clf.__class__.__name__+'_00001.pkl'

# joblib.dump(clf, model_name) 

print('Loading ' + clf.__class__.__name__)

# clf = joblib.load(model_name) 

# scaler = joblib.load(scaler_filename) 



input_dt = [37,1,3,130,250,1,0,187,0,2.3,0,0,1]

input_dt = np.array(input_dt).reshape(1, -1)

# input_dt = scaler.transform(input_dt)

train_predictions = clf.predict(input_dt)





print('Result is: ' + str(float(train_predictions)))