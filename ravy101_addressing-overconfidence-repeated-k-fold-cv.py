import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time

import statistics

import os



import matplotlib.pyplot as plt

import xgboost

import lightgbm

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from catboost import CatBoostClassifier



from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict

from sklearn import linear_model

from sklearn import metrics
heart_df = pd.read_csv("/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")

heart_df.head()
selected_features = ['time','ejection_fraction','serum_creatinine','age']

X = heart_df[selected_features]

X_all_features = heart_df[heart_df.columns.difference(['DEATH_EVENT'])]

y = heart_df['DEATH_EVENT']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2698)



r_clf = RandomForestClassifier(max_features=0.5, max_depth=15, random_state=1)

r_clf.fit(x_train, y_train)

acc =  r_clf.score(x_test,y_test)

print(f"Random Forest Test Accuracy: {round(acc*100, 3)}")
# Same parameters for the RF Classifier

r_clf = RandomForestClassifier(max_features=0.5, max_depth=15, random_state=1)



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

r_clf.fit(x_train, y_train)

acc =  r_clf.score(x_test,y_test)

print(f"Random Forest Test 1 Accuracy: {round(acc*100, 3)}")

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4321)

r_clf.fit(x_train, y_train)

acc =  r_clf.score(x_test,y_test)

print(f"Random Forest Test 2 Accuracy: {round(acc*100, 3)}")

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)

r_clf.fit(x_train, y_train)

acc =  r_clf.score(x_test,y_test)

print(f"Random Forest Test 3 Accuracy: {round(acc*100, 3)}")
#start with initial split

x_train_val, x_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=1111)
for i in range(1000):

    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, random_state=i)

    # Same parameters for the RF Classifier

    r_clf = RandomForestClassifier(max_features=0.5, max_depth=15, random_state=1)

    r_clf.fit(x_train, y_train)

    acc =  r_clf.score(x_val,y_val)*100   

    if acc > 94:

        print(f"Random Forest Val Accuracy: {round(acc, 3)}")

        print(f"High Accuracy at: {i}")

        break

acc = r_clf.score(x_test,y_test)*100

print(f"Random Forest Test Accuracy: {round(acc, 3)}")
np.random.seed(19566390)

FOLDS = 5

REPEATS = 5
def repeat_cross_validation_accuracy(model, x, y, n_folds = 5, n_repeats = 5, metric = 'accuracy'):

    oof_acc = []

    oof_predictions = []

    for i in range(n_repeats):

        kf = StratifiedKFold(n_folds, shuffle=True)

        acc = cross_val_score(model, x.values, y=y.values,scoring=metric, cv = kf)

        oof_acc.append(acc.mean())

        predictions = cross_val_predict(model, x, y=y.values, cv=kf)

        oof_predictions.append(predictions)

    return oof_acc, oof_predictions 


selected_features = ['time','ejection_fraction','serum_creatinine','age']



X = heart_df[selected_features]

#X_all_features = heart_df[heart_df.columns.difference(['DEATH_EVENT'])]

y = heart_df['DEATH_EVENT']
rf_model = RandomForestClassifier(max_features=0.5, max_depth=15)

knn_model = KNeighborsClassifier(n_neighbors=6)

dt_model = DecisionTreeClassifier(max_leaf_nodes=10, criterion='entropy')

gb_model = GradientBoostingClassifier(max_depth=2 )

xgb_model = xgboost.XGBRFClassifier(max_depth=3 )

lgb_model = lightgbm.LGBMClassifier(max_depth=2)

cat_model = CatBoostClassifier(verbose=0)



models = dict()

models['Random Forest'] = rf_model

models['KNN'] = knn_model

models['Decision Tree'] = dt_model

models['Gradient Boosting'] = gb_model

models['XGB'] = xgb_model

models['LGB'] = lgb_model

models['Cat Boost'] = cat_model
accuracies = []

training_times = []



print(f"Fitting models with {REPEATS} iterations of {FOLDS} fold CV\n")

for k in models.keys():

    print(f"####################################\nTraining Model: {k}")

    start = time.time()

    acc, preds = repeat_cross_validation_accuracy(models[k], X, y, n_folds = FOLDS, n_repeats = REPEATS)

    end = time.time()

    elapsed = end - start

    print(f"Total Training Time: {round(elapsed,4)} seconds")

    print(f"Mean OOF Accuracy: {round(statistics.mean(acc)*100, 2)} %")

    print(f"####################################\n\n")

    accuracies.append(statistics.mean(acc))

    training_times.append(elapsed)
res_df = pd.DataFrame({'Model': list(models.keys()), 'Accuracy': accuracies, f"Train Time ({FOLDS} by {REPEATS})": training_times})

res_df