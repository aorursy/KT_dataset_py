import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data_train_file = "/kaggle/input/minor-project-2020/train.csv"

data_test_file = "/kaggle/input/minor-project-2020/test.csv"



df = pd.read_csv(data_train_file)

df_test = pd.read_csv(data_test_file)
X=df.loc[:, 'col_0':'col_87']

y=df[['target']]
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=121)
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state = 2)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(solver ='newton-cg',max_iter=1500, penalty ='l2' ,C=10,n_jobs=-1)

model_res = clf.fit(X_train_res, np.ravel(y_train_res))
y_pred = clf.predict(X_test)
# from sklearn.linear_model import LogisticRegression

# model = LogisticRegression(max_iter=1500)

# solvers2 = [ 'lbfgs','liblinear','newton-cg']

# penalty2 = ['l2','l1']

# c_values2 = [10,1, 0.1,0.01]

# # define grid search

# grid = dict(solver=solvers2,penalty=penalty2,C=c_values2)
# from sklearn.model_selection import GridSearchCV

# grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=5, scoring='roc_auc')

# grid_result = grid_search.fit(X_train_res, y_train_res)
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# means = grid_result.cv_results_['mean_test_score']

# stds = grid_result.cv_results_['std_test_score']

# params = grid_result.cv_results_['params']

# for mean, stdev, param in zip(means, stds, params):

#     print("%f (%f) with: %r" % (mean, stdev, param))
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

print("Confusion Matrix: ")



print(confusion_matrix(y_test, y_pred))

plot_confusion_matrix(clf, X_test, y_test, cmap = plt.cm.Blues)

print("Classification Report: ")

print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_curve, auc

plt.style.use('seaborn-pastel')



FPR, TPR, _ = roc_curve(y_test, y_pred)

ROC_AUC = auc(FPR, TPR)

print (ROC_AUC)



plt.figure(figsize =[11,9])

plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)

plt.plot([0,1],[0,1], 'k--', linewidth = 4)

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.05])

plt.xlabel('False Positive Rate', fontsize = 18)

plt.ylabel('True Positive Rate', fontsize = 18)

plt.title('ROC for Minor Project Classification', fontsize= 18)

plt.show()
ids = df_test["id"]

Xtestdata = df_test.loc[:, 'col_0':'col_87']

y_predict = clf.predict_proba(Xtestdata)[:,1]

output = pd.DataFrame(data={"id": ids, "target":y_predict})

output.to_csv("trial1_predictions.csv", index=False)