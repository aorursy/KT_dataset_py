# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2,f_regression, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
data_y=pd.read_csv("../input/data0709/data_csv_0608_y_notime_c5.csv")
data_x_raw=pd.read_csv("../input/data0709/data_csv_0608_x_notime_LU_ratio.csv")
data_x=data_x_raw
scaler = StandardScaler()
data_x_scaled=scaler.fit_transform(data_x)
data_x=pd.DataFrame(data_x_scaled,columns=list(data_x.columns))

y_field="cluster_0"

data_x_sel=data_x.loc[(data_y[y_field] == 1) | (data_y[y_field] == 5)]
data_y_sel=data_y.loc[(data_y[y_field] == 1) | (data_y[y_field] == 5)]

train_x, test_x, train_y, test_y  = train_test_split(data_x_sel,data_y_sel[y_field],random_state=3,test_size=0.5,stratify=data_y_sel[y_field]) 
scores = ['recall']

tuned_parameters = [
    {'C': [0.1,1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [1,0.1,0.01,0.001, 0.0001],'class_weight': [None,'balanced']}]

K=range(4,data_x_sel.shape[1]+1)
for nk in K:
    selector = SelectKBest(score_func= f_regression, k=nk) #f_classif f_regression
    selector.fit(test_x, test_y)

    mask = selector.get_support() 
    print("K="+str(nk))
    print(mask)

    train_x_selected = selector.transform(train_x)
    test_x_selected = selector.transform(test_x)



    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            SVC(), tuned_parameters, scoring='%s_macro' % score
        )
        clf.fit(train_x_selected, train_y)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        print()
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = test_y, clf.predict(test_x_selected)
        print(classification_report(y_true, y_pred))
        print()
C = 1000
kernel = 'rbf'
gamma  = 0.01
w=None

K=7
selector = SelectKBest(score_func=f_regression, k=K) 
selector.fit(test_x, test_y)

mask = selector.get_support()   
print(mask)

train_x_selected = selector.transform(train_x)
test_x_selected = selector.transform(test_x)


svm = SVC(C=C, kernel=kernel, gamma=gamma,probability=True,class_weight=w)
svm.fit(train_x_selected, train_y)
pred_y_svm = svm.predict(test_x_selected)

print(y_field+'_train_SVM: {:.5f}'.format(svm.score(train_x_selected, train_y)))
print(y_field+'_test_SVM: {:.5f}'.format(svm.score(test_x_selected, test_y)))

cf_matrix = confusion_matrix(test_y, pred_y_svm)
print(cf_matrix)

y_true, y_pred = test_y, svm.predict(test_x_selected)
print(classification_report(y_true, y_pred))
aa=list(data_x_raw.columns)
list_sel=[]
for i in range(len(mask)):
    if mask[i]==True:
        list_sel.append(aa[i])
print(aa)
print(list_sel)
import eli5
from eli5.sklearn import PermutationImportance

data_x_selected=data_x[list_sel]
print(data_x_selected.shape)

val=data_x_selected.columns.tolist()
perm = PermutationImportance(svm).fit(test_x_selected, test_y)
eli5.show_weights(perm, top = 100, feature_names = val, show_feature_values = True)
import shap
shap.initjs()
data_x=pd.DataFrame(data_x_scaled,columns=list(data_x.columns))
test_x_selected=pd.DataFrame(test_x_selected,columns=list(data_x_selected))

explainer = shap.KernelExplainer(svm.predict_proba, train_x_selected,link='logit')
shap_values = explainer.shap_values(test_x_selected,l1_reg="aic")
shap.summary_plot(shap_values[0],test_x_selected)
