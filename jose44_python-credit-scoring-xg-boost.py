# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from xgboost import XGBClassifier

from xgboost import plot_importance  



# Selection model

from sklearn.model_selection import train_test_split, KFold, cross_val_score,GridSearchCV,StratifiedKFold 



# Metrics

from sklearn.metrics import accuracy_score, roc_auc_score,f1_score, confusion_matrix, classification_report



# Visualization

import matplotlib.pyplot as plt

%matplotlib inline
def evalua(y_pred,y_test):

    

    # Evaluate of predictions 

    accuracy = accuracy_score(y_test, y_pred) 

    roc = roc_auc_score(y_test, y_pred)

    f1=f1_score(y_test, y_pred)



    # Data test results

    print('Evaluation of predictions: \n')

    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    print("Area ROC: %.2f%%" % (roc * 100.0))

    print("F1 Score: %.2f%%" % (f1 * 100.0))
original = pd.read_excel('../input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx',"Data")
original.head()
original.info()
df = original

df.drop(['ID'],axis = 1, inplace = True)

x = df.drop(['Personal Loan'], axis = 1)

y = df['Personal Loan']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
# Grid Search 

model = XGBClassifier() 



learning_rate = [0.01, 0.1, 0.2] 

max_depth=[6]

booster=['gbtree']

n_estimators=[100]

nthread=[6]

min_child_weight=[1,5,10,20,30]



param_grid = dict(min_child_weight=min_child_weight,nthread=nthread,learning_rate=learning_rate,max_depth=max_depth,booster=booster,n_estimators=n_estimators)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7) 

grid_search = GridSearchCV(model, param_grid, scoring="accuracy", n_jobs = 10, cv=kfold) 

grid_result = grid_search.fit(x_train, y_train)
# Summarize results   

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score'] 

stds = grid_result.cv_results_['std_test_score'] 

params = grid_result.cv_results_['params']
# Evaluate of predictions 

model_best = grid_search.best_estimator_

y_pred=model_best.predict(x_test)

evalua(y_pred,y_test)



# Plot feature importance 

plot_importance(model_best) 

plt.show()
# Classification report

print('Classification Report:\n')

print(classification_report(y_test, y_pred),'\n')
# Confusion matrix

cm =confusion_matrix(y_test,y_pred)



plt.clf()

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)

classNames = ['Negative','Positive']

plt.title('Confusion Matrix - Test Data')

plt.ylabel('True label')

plt.xlabel('Predicted label')

tick_marks = np.arange(len(classNames))

plt.xticks(tick_marks, classNames, rotation=45)

plt.yticks(tick_marks, classNames)

s = [['TN','FP'], ['FN', 'TP']]

 

for i in range(2):

    for j in range(2):

        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))

plt.show()