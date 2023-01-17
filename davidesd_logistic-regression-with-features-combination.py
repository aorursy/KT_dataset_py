import pandas as pd

import time

from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss

from sklearn.linear_model import LogisticRegressionCV



def encode_df(train, test):

    le = LabelEncoder().fit(train.species) 

    labels = le.transform(train.species)           # encode species strings

    classes = list(le.classes_)                    # save column names for submission

    test_ids = test.id                             # save test ids for submission

    

    train = train.drop(['species', 'id'], axis=1)  

    test = test.drop(['id'], axis=1)

    

    cols_s = [c for c in train.columns if 'shape' in c]

    cols_t = [c for c in train.columns if 'texture' in c]

    cols_m = [c for c in train.columns if 'margin' in c]

    cols_st = [c for c in train.columns if ('shape' in c or 'texture' in c)]

    cols_mt = [c for c in train.columns if ('margin' in c or 'texture' in c)]

    cols_sm = [c for c in train.columns if ('margin' in c or 'shape' in c)]



    train_set = [train[cols_s], train[cols_m], train[cols_t], train[cols_st], train[cols_sm], train[cols_mt], train]

    test_set = [test[cols_s], test[cols_m], test[cols_t], test[cols_st], test[cols_sm], test[cols_mt], test]

    

    feat_order = ["Shape", "Margin", "Texture", "Shape-Texture", "Shape-Margin", "Margin-Texture", "All Features"]

    

    return labels, test_ids, classes, train_set, test_set, feat_order



# load the data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

labels, test_ids, classes, train_set, test_set, feat_order = encode_df(train, test)



# Logging for Visual Comparison

log_cols=["Features", "Accuracy", "Log Loss", "Score", "Elapsed Time"]

logFrame = pd.DataFrame(columns=log_cols)



i = 0

for tr, te in zip(train_set, test_set):

    log_row = []

    log_row.append(feat_order[i])

    print("="*30)

    print(feat_order[i])

    i += 1



    sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)

    for train_index, test_index in sss:

        X_train, X_test = tr.values[train_index], tr.values[test_index]

        y_train, y_test = labels[train_index], labels[test_index]

        

    start_t = time.clock()

    logist_regr = LogisticRegressionCV()

    logist_regr.fit(X_train, y_train)

    train_predictions = logist_regr.predict(X_test)

    acc = accuracy_score(y_test, train_predictions)

    log_row.append(acc)

    train_predictions = logist_regr.predict_proba(X_test)

    ll = log_loss(y_test, train_predictions)

    log_row.append(ll)

    log_row.append(logist_regr.score(X_test, y_test))

    end_t = time.clock()

    log_row.append(end_t - start_t)

    

    log_entry = pd.DataFrame([log_row], columns=log_cols)

    logFrame = logFrame.append(log_entry)  
p1 = logFrame.plot(kind='barh', x="Features", y="Accuracy", legend=False, color="green")

p1.set_xlabel("Accuracy")
p2 = logFrame.plot(kind='barh', x="Features", y="Log Loss", legend=False, color="darkorange")

p2.set_xlabel("Log Loss")
p3 = logFrame.plot(kind='barh', x="Features", y='Elapsed Time', legend=False, color="royalblue")

p3.set_xlabel("Elapsed Time [s]")