import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split,StratifiedKFold

from sklearn.metrics import roc_auc_score, roc_curve,confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print("train shape:", train.shape, "test.shape:", test.shape)
sns.countplot(train['target'])
total1 = train["target"].value_counts()[1]

print("There are {} target values with 1, is about {}% of total data".format(total1, 100 * total1/train.shape[0]))
train0 = train.loc[train['target'] == 0]

train1 = train.loc[train['target'] == 1]

print(train0.shape, train1.shape)

splitNum = 2

t0PerSplit = train0.shape[0] // splitNum

print(t0PerSplit)

splits = []

for i in range(splitNum-1):

    splits.append(pd.concat([train0[i*t0PerSplit:(i+1)*t0PerSplit], train1]).sample(frac=1))

    print(splits[i].shape)

splits.append(pd.concat([train0[(splitNum-1)*t0PerSplit:], train1]).sample(frac=1))

print(splits[splitNum-1].shape)

splits[splitNum-1].iloc[:,1].values[:100]
features = train.columns.values[2:202]

scaler = StandardScaler()

x_test = scaler.fit_transform(test[features])
X_test = x_test.reshape(200000,20,10,1)

predictions = np.zeros(len(X_test)) 

print(X_test.shape, predictions.shape)

for split_ in range(splitNum):

    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=2019)

    #train_s = splits[split_][features]

    train_s =  scaler.transform(splits[split_][features])

    target_s = splits[split_]["target"]

    print(train_s.shape, target_s.shape)

    prediction = np.zeros(len(test))

    oof = np.zeros(len(train_s))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_s, target_s)):

        X_train, y_train = train_s[trn_idx], target_s.iloc[trn_idx]

        X_train = X_train.reshape(trn_idx.shape[0], 20, 10, 1)

        X_valid, y_valid = train_s[val_idx], target_s.iloc[val_idx]

        X_valid = X_valid.reshape(val_idx.shape[0], 20, 10, 1)

        print("Split:", split_,  "Fold:",fold_, trn_idx.shape, val_idx.shape, X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)

        #create model

        model = Sequential()

        #add model layers

        model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(20,10,1)))

        model.add(Conv2D(32, kernel_size=3, activation='relu'))

        model.add(Flatten())

        model.add(Dense(1, activation='sigmoid'))

        #compile model using accuracy to measure model performance

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        model.summary()

        #train the model

        model.fit(X_train, y_train, batch_size=100, validation_data=(X_valid, y_valid), epochs=10)

    

        y_pred = model.predict_classes(X_valid)

        oof = model.predict_classes(X_valid)

        # Confusion matrix

        tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()

    

        # Accuracy: Overall, how often is the classifier correct?  (tp+tn) / (tn+fp+fn+tp)

        # Precision score, When it predicts yes, how often is it correct?  tp/(fp + tp)

        # Specificity, True Negative Rate: When it's actually no, how often does it predict no? tn/(tn+fp)

        # Recall score, or Sensitivity, True positive rate, When it's actually yes, how often does it predict yes? tp/(fn+tp)

    

        print("precision_score:", tp/(fp + tp), "specificity_score:", tn/(tn+fp), "recall_score:", tp/(fn+tp))

        # F1 score

        print("f1_score", f1_score(y_valid,y_pred))

        print("CV score: {:<8.5f}".format(roc_auc_score(y_valid, oof)))

        # Cohen's kappa score

        #cohen_kappa_score(y_valid, y_pred)

    

        prediction += model.predict_proba(X_test)[:,0] / folds.n_splits 

        #print("CV score: {:<8.5f}".format(roc_auc_score(target_s, oof)))

    predictions += prediction / splitNum   
print(predictions.shape)

predictions[:50]
sub = pd.DataFrame() 

sub["ID_code"] = test["ID_code"] 

sub["target"] = predictions

sub.to_csv("submission-cnn-ksplit.csv", index=False)