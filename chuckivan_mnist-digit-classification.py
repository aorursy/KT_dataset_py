import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.columns.difference(test.columns)
target = 'label'
labels = train[target]
train = train.drop(target, axis=1)
train.shape
some_digit = train.values[50].reshape(28,28)
plt.imshow(some_digit, cmap = matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()
def std_scaled(data, scaler, is_train):
    # Don't scale row Id
    preds = data.select_dtypes(include=[np.int, np.float]).columns.tolist()
    if is_train:
        for pred in preds:
            data[pred] = scaler.fit_transform(data[[pred]])
    else:
        for pred in preds:
            data[pred] = scaler.transform(data[[pred]])            
    return data
%%capture
scaler = StandardScaler()
train_scaled = std_scaled(train, scaler, True)
test_scaled = std_scaled(test, scaler, False)
train_scaled[:1]
svc = SVC(C=1, kernel='poly', degree=9, coef0=1, decision_function_shape='ovo')
svc.fit(train_scaled, labels)
some_data = train_scaled[:10]
some_labels = labels[:10]
print("Prediction:", list(svc.predict(some_data)))
print("Actual lab:", list(some_labels))
def accuracy(preds, labs):
    n = len(preds)
    if n != len(labs):
        raise ValueError
    wrong = 0
    for i in range(n):
        if preds[i] != labs[i]:
            wrong += 1
    return (n-wrong) / n
svc_preds = list(svc.predict(train_scaled))
labs = list(labels)
print("SVC accuracy on train set: " + str(accuracy(svc_preds, labs)))
# F1 is a harmonic mean of precision and recall, so it's higher if both are good
F1 = precision_recall_fscore_support(labs, svc_preds)[2]
F1
# GridSearchCV for slack variable sum (C) and basis function (degree)
# param_grid = [{'C': [0.1, 1, 5]}]
# grid_search = GridSearchCV(svc, param_grid, n_jobs=-1, cv=5)
# grid_search.fit(train_scaled, labels)
# grid_search.best_params_
# C=1 on train is best
# Submit
predictions = svc.predict(test).tolist()
test['ImageId'] = range(1, len(test)+1)
submission = pd.DataFrame({'ImageId': test.ImageId, target: predictions})
submission.to_csv('submission.csv', index=False)
