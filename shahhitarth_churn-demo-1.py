import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

farm = pd.read_csv("../input/FARMER_CHURN.csv")

farm.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

farm['Gender']= le.fit_transform(farm['Gender']) 

farm['Partner']= le.fit_transform(farm['Partner'])

farm['Dependents']= le.fit_transform(farm['Dependents'])

farm['Catt_Feed']= le.fit_transform(farm['Catt_Feed'])

farm['Ani_Hus']= le.fit_transform(farm['Ani_Hus'])

farm['Loans']= le.fit_transform(farm['Loans'])

farm['Curr_Status']= le.fit_transform(farm['Curr_Status'])

farm['Fat_Quality']= le.fit_transform(farm['Fat_Quality'])

farm['PaymentMethod']= le.fit_transform(farm['PaymentMethod'])

farm['Join_Date']= le.fit_transform(farm['Join_Date'])

farm['Leav_Date']= le.fit_transform(farm['Leav_Date'])

farm
farm.shape
farm['Churn'].unique
from sklearn import preprocessing

label_encoder=preprocessing.LabelEncoder()

farm['Churn']=label_encoder.fit_transform(farm['Churn'])

farm['Churn'].unique
count_no_churn=(farm['Churn']==0).sum()

print("Farmers that did not churned:", count_no_churn)

count_yes_churn=(farm['Churn']==1).sum()

print("Farmers that churned:", count_yes_churn)
X=farm.loc[:, farm.columns != 'Churn']

y=farm.loc[:, farm.columns == 'Churn']

print("Shape of X is: {}".format(X.shape))

print("Shape of y is: {}".format(y.shape))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



print("Number Transactions X_train datasets: ", X_train.shape)

print("Number Transactions X_test datasets: ", X_test.shape)

print("Number Transactions y_train datasets: ", y_train.shape)

print("Number Transactions y_test datasets: ", y_test.shape)
farm.dtypes
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=0)

X_train_res, y_train_res = sm.fit_sample(X_train,y_train)



print("After Oversampling the shape of train_X: {}".format(X_train_res.shape))

print("After Oversampling the shape of train_y: {} \n".format(y_train_res.shape))

print("After Oversampling the counts of label '1': {}".format(list(y_train_res['Churn']).count(1)))

print("After Oversampling the counts of label '0': {}".format(list(y_train_res['Churn']).count(0)))

print(y_train_res)
from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
models = [('knn', KNN), 

          ('logistic', LogisticRegression),

          ('tree', DecisionTreeClassifier),

          ('forest', RandomForestClassifier)

         ]



param_choices = [

    {

        'n_neighbors': range(1, 12)

    },

    {

        'C': np.logspace(-3,6, 12),

        'penalty': ['l1', 'l2']

    },

    {

        'max_depth': [1,2,3,4,5],

        'min_samples_leaf': [3,6,10]

    },

    {

        'n_estimators': [50, 100, 200],

        'max_depth': [1,2,3,4,5],

        'min_samples_leaf': [3,6,10]

    }

]



grids = {}

for model_info, params in zip(models, param_choices):

    name, model = model_info

    grid = GridSearchCV(model(), params)

    grid.fit(X_train_res, y_train_res)

    s = f"{name}: Best Score: {grid.best_score_}"

    print(s)

    grids[name] = grid
param_grid = { 

    'n_estimators': [50, 100, 200],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [2,4,5,6,7,8],

    'criterion' :['gini', 'entropy']

}
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(random_state=42)



rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)

rfc.fit(X_train_res, y_train_res)
rfc.best_params_
rfc_best=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 50, max_depth=8, criterion='gini')
rfc_best.fit(X_train_res, y_train_res)
y_pred_rfc=rfc_best.predict(X_test)
from sklearn.metrics import confusion_matrix





confusion_matrix_forest = confusion_matrix(y_test, y_pred_rfc)

print(confusion_matrix_forest)
import seaborn as sns



#plotting a confusion matrix

labels = ['Not Churned', 'Churned']

plt.figure(figsize=(7,5))

ax= plt.subplot()

sns.heatmap(confusion_matrix_forest,cmap="Blues",annot=True,fmt='.1f', ax = ax); #annot=True to annotate cells



# labels, title and ticks

ax.set_xticklabels(labels)

ax.set_yticklabels(labels)

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')

ax.set_title('Confusion Matrix Random Forests')
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_rfc))
from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve,auc



y_pred_rfc=rfc_best.predict(X_test)

y_score_rfc = rfc_best.predict_proba(X_test)[:,1]

fpr, tpr,_ = roc_curve(y_test, y_score_rfc)

roc_auc_forests = auc(fpr, tpr)

print(roc_auc_forests)
from sklearn import metrics

logreg = LogisticRegression()

logreg.fit(X_train_res, y_train_res)
from sklearn.model_selection import GridSearchCV

# Setup the hyperparameter grid

c_space = np.logspace(-3,6, 12)

param_grid = {'C': c_space}



# Instantiate a logistic regression classifier: logreg

logreg = LogisticRegression()



# Instantiate the GridSearchCV object: logreg_cv

logreg_cv = GridSearchCV(logreg, param_grid, cv=10)



# Fit it to the data

best_model = logreg_cv.fit(X_train_res, y_train_res)



#examine the best model

print(best_model.best_score_)

print(best_model.best_params_)

print(best_model.best_estimator_)
best_lr1 = LogisticRegression(C=0.8)

best_lr1.fit(X_train_res, y_train_res)
from sklearn.metrics import confusion_matrix

y_pred_log = best_lr1.predict(X_test)

confusion_matrix_log = confusion_matrix(y_test, y_pred_log)

print(confusion_matrix_log)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_log))
import seaborn as sns

labels = ['Not Churned', 'Churned']



plt.figure(figsize=(7,5))

ax= plt.subplot()

sns.heatmap(confusion_matrix_log,cmap="Blues",annot=True,fmt='.1f', ax = ax); #annot=True to annotate cells



# labels, title and ticks

ax.set_xticklabels(labels)

ax.set_yticklabels(labels)

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')

ax.set_title('Confusion Matrix ')
from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve,auc



y_pred_log=best_lr1.predict(X_test)

y_score_log = best_lr1.predict_proba(X_test)[:,1]

fpr, tpr,_ = roc_curve(y_test, y_score_log)

roc_auc_log = auc(fpr, tpr)

print(roc_auc_log)
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve,auc

from seaborn import despine



logit_roc_auc = roc_auc_score(y_test, best_lr1.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, best_lr1.predict_proba(X_test)[:,1])



roc_auc = auc(fpr,tpr)



plt.figure(figsize=(6,6))

# Plotting our Baseline..

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr, tpr,'g',label='AUC = %0.3f'% roc_auc)

plt.legend(loc='lower right')

plt.title('ROC Curve',fontsize = 15)

plt.xlabel('False Positive Rate',fontsize = 15)

plt.ylabel('True Positive Rate',rotation=0,labelpad=45,fontsize = 15)

despine()
logit_roc_auc = roc_auc_score(y_test, best_lr1.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, best_lr1.predict_proba(X_test)[:,1])

roc_auc = auc(fpr,tpr)

plt.figure(figsize=(6,6))

# Plotting our Baseline..

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr, tpr, 'g',label='AUC = %0.3f'% roc_auc)

plt.annotate('optimal threshold', xy=(0.59, 0.96), xytext=(0.5, 0.8),

            arrowprops=dict(facecolor='black', shrink=0.05),

            )

plt.legend(loc='lower right')

plt.title('ROC Curve')

plt.xlabel('False Positive Rate',fontsize = 15)

plt.ylabel('True Positive Rate',rotation=0,labelpad=45,fontsize = 15)

despine()
#y_pred1 = model_svm2.predict(X_test)
def makecost(obs,prob,falsepos_cost,falseneg_cost,truepos_cost):

    def cost(cutoff):

        pred = np.array(prob > cutoff)

        fpos = pred * (1 - obs) 

        fneg = (1 - pred) * obs

        tpos=pred*obs

        return np.sum(fpos * falsepos_cost + fneg * falseneg_cost+ tpos*truepos_cost)

    return np.vectorize(cost)
cut = np.linspace(0,1,100)

cost = np.zeros_like(cut)

from sklearn.model_selection import KFold, cross_val_predict

obs = np.ravel(y_train)



K = 10

for j in range(K):

    folds = KFold(n_splits=5,shuffle=True)

    prob = cross_val_predict(best_lr1,X_train,obs,cv=folds,method='predict_proba',n_jobs=5)[:,1]

    getcost = makecost(obs,prob,falsepos_cost=100,falseneg_cost=500,truepos_cost=200)

    currentcost = getcost(cut)/X_train.shape[0]

    cost += currentcost

    plt.plot(cut, currentcost,c='C0',alpha=0.05)

cost /= K



plt.plot(cut,cost,c='C0')

plt.xlabel('cutoff',fontsize = 15)

plt.ylabel('Expected cost per data point',labelpad=69,fontsize = 15,rotation=0)

plt.title("Optimal Threshold",fontsize=15)

despine()
bestcut = cut[np.argmin(cost)]

bestcut
min(cost)
pd.set_option('display.max_rows', 500)

cutoff_list = pd.DataFrame(np.column_stack([tpr, fpr, thresholds]), 

                               columns=['tpr', 'fpr', 'thresholds'])

cutoff_list
best_lr1.predict_proba(X_test)[0:10]
y_pred_prob = best_lr1.predict_proba(X_test)[:, 1]
import matplotlib.pyplot as plt



plt.rcParams['font.size'] = 12

plt.hist(y_pred_prob)



# x-axis limit from 0 to 1

plt.xlim(0,1)

plt.title('Histogram of predicted probabilities')

plt.xlabel('Predicted probability of a Farmer Churning')

plt.ylabel('Frequency')

despine()
no_churn = (y_test['Churn'] == 0).sum()

yes_churn=(y_test['Churn'] == 1).sum()

print("Not Churned:",no_churn)

print("Yes Churned:",yes_churn)