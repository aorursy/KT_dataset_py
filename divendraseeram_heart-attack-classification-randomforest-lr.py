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
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



from sklearn.metrics import classification_report, roc_auc_score, roc_curve



import seaborn as sns

sns.set(style="whitegrid")
df = pd.read_csv('../input/health-care-data-set-on-heart-attack-possibility/heart.csv')

df.head()
df.select_dtypes(exclude=['object']).isnull().sum()
df.dtypes
plt.figure(figsize = (10,7))

ax1 = sns.countplot(x = 'target', data = df, palette = ["C2", "C3"])

ax1.set_xticklabels(["Low","High"])

plt.title("Heart Attack Chance Patient Counts", weight = 'bold', fontsize = 15)

plt.xlabel('Heart Attack Chance')

plt.ylabel("Patient Count")
plt.figure(figsize = (8,5))

ax2 = sns.countplot(x = 'sex', data = df, palette = ["C2", "C3"], hue = 'target')

ax2.set_xticklabels(["Female","Male"])

plt.title("Heart Attack Chance by Sex", weight = 'bold', fontsize = 15)

plt.xlabel('Sex')

plt.ylabel("Patient Count")

plt.legend(title = "Heart Attack Chance by Sex",labels=['Low', 'High'], loc = 'upper left')

plt.show()



plt.figure(figsize = (8,5))

ax2 = sns.countplot(x = 'sex', data = df,)

ax2.set_xticklabels(["Female","Male"])

plt.title("Sex Sampling Counts", weight = 'bold', fontsize = 15)

plt.xlabel('Sex')

plt.ylabel("Patient Count")

plt.show()

#plt.legend(title = "Heart Attack Chance",labels=['Low', 'High'], loc = 'upper left')
plt.figure(figsize= (8,5))

sns.distplot(df['age'])

plt.title("Heart Attack Dataset Age Distribution", weight = 'bold', fontsize = 15)
corr = df.corr()



plt.figure(figsize = (18,18))

sns.heatmap(corr, annot = True, cmap = 'coolwarm', vmin = -1, vmax=1)
X = df.drop('target', axis = 1)

y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, shuffle = True)
rf_model = RandomForestClassifier(random_state = 42)
def full_report(y_test,y_hat,y_hat_probs,name = ''):

    if name != '':

        print(name)

    print(classification_report(y_test, y_hat))

    print("ROC AUC = ",roc_auc_score(y_test, y_hat_probs),'\n\n')

    

def roc_plot_label():

    plt.xlabel('False positive rate')

    plt.ylabel('True positive rate')

    plt.title('ROC curve')

    plt.legend(loc="best")

    
rf_model.fit(X_train, y_train)

yhat_forest = rf_model.predict(X_test)



confusion_matrix = pd.crosstab(y_test, yhat_forest, rownames=['Actual'], colnames=['Predicted'])

sns.heatmap(confusion_matrix, annot=True, fmt = 'g', cmap = 'Reds')

plt.show()



yhat_forest_probs = rf_model.predict_proba(X_test)

yhat_forest_probs = yhat_forest_probs[:,1]



full_report(y_test,yhat_forest, yhat_forest_probs, name = "Base Model")



fpr, tpr, _ = roc_curve(y_test, yhat_forest_probs)



plt.figure(figsize=(10,7))

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr, label = "Base", color = "Blue")

roc_plot_label()

plt.show()
print(rf_model.get_params())
random_grid = {'max_depth': [5,10,25,50,100,250,500,None],

               'max_features': ['auto', 'sqrt', 'log2', None],

               'min_samples_leaf': np.linspace(0.1, 0.5, 5, endpoint=True),

               'min_samples_split': np.linspace(0.1, 1.0, 10, endpoint=True),

               'n_estimators': [2, 4, 8, 16, 32, 64, 100, 200, 500]}
from sklearn.model_selection import RandomizedSearchCV



rf_random = RandomizedSearchCV(estimator = rf_model, param_distributions = random_grid, n_iter = 100, cv = 5, random_state = 42, n_jobs = -1, verbose = 2)

rf_random.fit(X_train, y_train)

best = rf_random.best_params_

rf_random.best_params_
rf_model2 = RandomForestClassifier(random_state = 42, 

                                   n_estimators = best['n_estimators'], 

                                   min_samples_split = best['min_samples_split'], 

                                   min_samples_leaf = best['min_samples_leaf'], 

                                   max_features = best['max_features'], 

                                   max_depth = best['max_depth'])

rf_model2.fit(X_train, y_train)

yhat_forest2 = rf_model2.predict(X_test)



confusion_matrix = pd.crosstab(y_test, yhat_forest2, rownames=['Actual'], colnames=['Predicted'])

sns.heatmap(confusion_matrix, annot=True, fmt = 'g', cmap = 'Reds')

plt.show()



yhat_forest_probs2 = rf_model2.predict_proba(X_test)

yhat_forest_probs2 = yhat_forest_probs2[:,1]



full_report(y_test,yhat_forest,yhat_forest_probs, name = "Base Model")

full_report(y_test,yhat_forest2,yhat_forest_probs2, name = "RandomSearchCV Tuned Model")



fpr2, tpr2, _ = roc_curve(y_test, yhat_forest_probs2)



plt.figure(figsize=(10,7))

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr, label = "Base", color = 'blue')

plt.plot(fpr2, tpr2, label = "RandomTuned", color = 'green')

roc_plot_label()

plt.show()
from sklearn.model_selection import GridSearchCV



param_grid={'max_depth': [24, 25, 28, 32, None],

            'max_features': ['auto', 'sqrt', 'log2', None],

            'min_samples_leaf': [1, 2, 3],

            'min_samples_split': [1, 2, 3],

            'n_estimators': [100, 300, 500]}
grid = GridSearchCV(rf_model, param_grid = param_grid, cv = 5, verbose=2, n_jobs=-1)

grid.fit(X_train, y_train)
best2 = grid.best_params_

best2
rf_model3 = RandomForestClassifier(random_state = 42, 

                                   n_estimators = best2['n_estimators'], 

                                   min_samples_split = best2['min_samples_split'], 

                                   min_samples_leaf = best2['min_samples_leaf'], 

                                   max_features = best2['max_features'], 

                                   max_depth = best2['max_depth'])

rf_model3.fit(X_train, y_train)

yhat_forest3 = rf_model3.predict(X_test)



confusion_matrix = pd.crosstab(y_test, yhat_forest3, rownames=['Actual'], colnames=['Predicted'])

sns.heatmap(confusion_matrix, annot=True, fmt = 'g', cmap = 'Reds')

plt.show()



yhat_forest_probs3 = rf_model3.predict_proba(X_test)

yhat_forest_probs3 = yhat_forest_probs3[:,1]



full_report(y_test,yhat_forest,yhat_forest_probs, name = "Base Model")

full_report(y_test,yhat_forest2,yhat_forest_probs2, name = "RandomSearchCV Tuned Model")

full_report(y_test,yhat_forest3,yhat_forest_probs3, name = "GridSearchCV Tuned Model")



fpr3, tpr3, _ = roc_curve(y_test, yhat_forest_probs3)



plt.figure(figsize=(10,7))

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr, label = "Base", color = 'blue')

plt.plot(fpr2, tpr2, label = "RandomTuned", color = 'green')

plt.plot(fpr3, tpr3, label = "GridSearch", color = 'purple')

roc_plot_label()

plt.show()
from sklearn.metrics import average_precision_score

from sklearn.metrics import precision_recall_curve, auc

from sklearn.metrics import plot_precision_recall_curve



precision_base, recall_base, _ = precision_recall_curve(y_test, yhat_forest_probs)

precision_r, recall_r, _ = precision_recall_curve(y_test, yhat_forest_probs2)

precision_g, recall_g, _ = precision_recall_curve(y_test, yhat_forest_probs3)



plt.figure(figsize=(10,7))

sns.lineplot(recall_base, precision_base, label="Base Model", color = 'blue', ci = None)

sns.lineplot(recall_r, precision_r, label="RandomSearchCV Tuning", color = 'green', ci = None)

sns.lineplot(recall_g, precision_g, label="GridSearchCV Tuning", color = 'purple', ci = None)



plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision-Recall Curves", weight ='bold', fontsize = 15)

plt.legend(loc="best")



auc_score = auc(recall_base, precision_base)

auc_score_r = auc(recall_r, precision_r)

auc_score_g = auc(recall_g, precision_g)



print("P-R AUC (Base Model):", auc_score)

print("P-R AUC (RandomSearchCV):", auc_score_r)

print("P-R AUC (GridSearchCV):", auc_score_g)
from sklearn.pipeline import Pipeline



steps = [('scaler', StandardScaler()), ('LR', LogisticRegression())]

pipe = Pipeline(steps)



pipe.fit(X_train, y_train)
y_hat_lr = pipe.predict(X_test)

y_hat_lr_probs = pipe.predict_proba(X_test)

y_hat_lr_probs = y_hat_lr_probs[:,1]



full_report(y_test,yhat_forest2,yhat_forest_probs2, name = "RandomSearchCV Tuned Model")

full_report(y_test,yhat_forest3,yhat_forest_probs3, name = "GridSearchCV Tuned Model")

full_report(y_test,y_hat_lr,y_hat_lr_probs, name = 'Logistic Regression')



fpr_lr, tpr_lr, _ = roc_curve(y_test, y_hat_lr_probs)



plt.figure(figsize=(10,7))

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_lr, tpr_lr,color = 'red', label = "Logistic Regression")

plt.plot(fpr2, tpr2,color = 'green', label = "RandomTuned RF")

plt.plot(fpr3, tpr3, color = 'purple', label = "GridSearch RF")

roc_plot_label()

plt.show()

precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_hat_lr_probs)



plt.figure(figsize=(10,7))

sns.lineplot(recall_lr, precision_lr, color = 'red', label="Logistic Regression", ci = None)

sns.lineplot(recall_r, precision_r, color = 'green', label="RandomSearchCV Tuning", ci = None)

sns.lineplot(recall_g, precision_g, color = 'purple', label="GridSearchCV Tuning", ci = None)



plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision-Recall Curves", weight ='bold', fontsize = 15)

plt.legend(loc="best")



auc_score_lr = auc(recall_lr, precision_lr)

auc_score_r = auc(recall_r, precision_r)

auc_score_g = auc(recall_g, precision_g)



print("P-R AUC (LogisticRegression):",auc_score_lr)

print("P-R AUC (RandomSearchCV):", auc_score_r)

print("P-R AUC (GridSearchCV):", auc_score_g)