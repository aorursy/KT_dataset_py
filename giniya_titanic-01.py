import numpy as np

import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.head(10)
train.isna().sum(axis=0)
test.isna().sum(axis=0)
sel_nan = np.isnan(train.loc[:,'Age'])

np.sum(sel_nan)
mean_age = np.mean(train.loc[:,'Age'])

print(mean_age)
train.loc[sel_nan,'Age'] = mean_age

train.isna().sum(axis=0)
sel_nan = np.isnan(test.loc[:,'Age'])

test.loc[sel_nan,'Age'] = mean_age

test.isna().sum(axis=0)
train.head()
X_num = train.iloc[:, [5]].values

X_cat = train.iloc[:, [2, 4]].values.astype('str')

y = train.iloc[:, 1].values
encoder = OneHotEncoder(sparse=False)

encoder.fit(X_cat)

X_enc = encoder.transform(X_cat)



print(X_enc.shape)
X_train_num, X_valid_num, y_train, y_valid = train_test_split(X_num, y, test_size=0.3, random_state=1, stratify=y)

X_train_enc, X_valid_enc, y_train, y_valid = train_test_split(X_enc, y, test_size=0.3, random_state=1, stratify=y)

print(y_train.shape)

print(y_valid.shape)
scaler = MinMaxScaler()

scaler.fit(X_train_num)

X_train_sc = scaler.transform(X_train_num)

X_valid_sc = scaler.transform(X_valid_num)

print(X_train_sc[:5])
X_train = np.hstack([X_train_sc, X_train_enc])

X_valid = np.hstack([X_valid_sc, X_valid_enc])

print(X_train.shape)

print(X_valid.shape)
tr_acc = []

va_acc = []



# i did not use l1 as for solver 'lbfgs' , only ['none', 'l2']  can be used

penalty = ['none', 'l2'] 



for p in penalty:

    temp_mod = LogisticRegression(solver='lbfgs',penalty=p,random_state=1)

    temp_mod.fit(X_train, y_train)

    tr_acc.append(temp_mod.score(X_train, y_train))

    va_acc.append(temp_mod.score(X_valid, y_valid))



plt.figure(figsize=([9, 6]))

plt.plot(penalty, tr_acc, label='Training Accuracy')

plt.plot(penalty, va_acc, label='Validation Accuracy')

plt.xlabel('penalty')

plt.ylabel('Accuracy')

plt.xticks(penalty)

plt.legend()

plt.show()
# l2 has higher validation and training accuracy scores
logreg_model = LogisticRegression(solver='lbfgs', penalty='l2')

logreg_model.fit(X_train, y_train)



print('Training Accuracy:  ', round(logreg_model.score(X_train, y_train),4))

print('Validation Accuracy:', round(logreg_model.score(X_valid, y_valid),4))
valid_pred = logreg_model.predict(X_valid)

valid_acc = np.mean(valid_pred == y_valid)

print(valid_acc)
tr_acc = []

va_acc = []



depth_list = range(1,21)



for d in depth_list:

    temp_mod = DecisionTreeClassifier(max_depth=d, random_state=1)

    temp_mod.fit(X_train, y_train)

    tr_acc.append(temp_mod.score(X_train, y_train))

    va_acc.append(temp_mod.score(X_valid, y_valid))



plt.figure(figsize=([9, 6]))

plt.plot(depth_list, tr_acc, label='Training Accuracy')

plt.plot(depth_list, va_acc, label='Validation Accuracy')

plt.xlabel('Maximum Depth')

plt.ylabel('Accuracy')

plt.xticks(depth_list)

plt.legend()

plt.show()
ix_best = np.argmax(va_acc)

best_md = depth_list[ix_best]

print('Optimal Value of max_depth:', best_md)
tree_model = DecisionTreeClassifier(max_depth=1, random_state=1)

tree_model.fit(X_train, y_train)



print('Training Accuracy:  ', round(tree_model.score(X_train, y_train),4))

print('Validation Accuracy:', round(tree_model.score(X_valid, y_valid),4))
tr_acc1 = []

va_acc1 = []



depth_list1 = range(1,21)



for d in depth_list1:

    temp_mod1 = RandomForestClassifier(n_estimators=500,max_depth=d, random_state=1)

    temp_mod1.fit(X_train, y_train)

    tr_acc1.append(temp_mod1.score(X_train, y_train))

    va_acc1.append(temp_mod1.score(X_valid, y_valid))



plt.figure(figsize=([9, 6]))

plt.plot(depth_list1, tr_acc1, label='Training Accuracy')

plt.plot(depth_list1, va_acc1, label='Validation Accuracy')

plt.xlabel('Maximum Depth')

plt.ylabel('Accuracy')

plt.xticks(depth_list1)

plt.legend()

plt.show()
ix_best1 = np.argmax(va_acc1)

best_md1 = depth_list1[ix_best1]

print('Optimal Value of max_depth:', best_md1)
forest_mod = RandomForestClassifier(n_estimators=500, max_depth=4, random_state=1)

forest_mod.fit(X_train, y_train)



print('Training Accuracy:  ', forest_mod.score(X_train, y_train))

print('Validation Accuracy:', forest_mod.score(X_valid, y_valid))
valid_pred = logreg_model.predict(X_valid)



cm = confusion_matrix(y_valid, valid_pred)



cm_df = pd.DataFrame(cm)

cm_df
print(classification_report(y_valid, valid_pred))
# Process the training set



X_num = train.iloc[:,[5]]

X_cat = train.iloc[:,[2,4]]

y_train = train.iloc[:,1]



encoder = OneHotEncoder(sparse=False)

encoder.fit(X_cat)

scaler = MinMaxScaler()

scaler.fit(X_num)



X_enc = encoder.transform(X_cat)

X_sc = scaler.transform(X_num)



X_train = np.hstack([X_enc, X_sc])
# Process test data

X_test_num = test.iloc[:, [4]]

X_test_cat = test.iloc[:, [1, 3]]

X_test_enc = encoder.transform(X_test_cat)

X_test_sc = scaler.transform(X_test_num)

X_test = np.hstack([X_test_enc, X_test_sc])

print(X_test.shape)
final_model = LogisticRegression(solver='lbfgs', penalty='l2')

final_model.fit(X_train, y_train)



print(final_model.score(X_train, y_train))
test_pred = final_model.predict(X_test)

test_pred[:5]
submission = pd.DataFrame({

    'PassengerID':test.PassengerId,

    'Survived':test_pred

})

submission.head()
submission.to_csv('my_submission.csv', index=False)