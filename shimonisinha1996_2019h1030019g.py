# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/minor-project-2020/train.csv")

test = pd.read_csv("../input/minor-project-2020/test.csv")
data.head()
data.describe()
data['target'].value_counts()
target_val_counts = pd.value_counts(data['target'], sort = True).sort_index()

target_val_counts.plot(kind = 'bar')

plt.title("Target values histogram")

plt.xlabel("Target")

plt.ylabel("Frequency")
mask = np.zeros_like(data.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.set_style('whitegrid')

plt.subplots(figsize = (100,100))

sns.heatmap(data.corr(), 

            annot=True,

            mask = mask,

            cmap = 'RdBu', ## in order to reverse the bar replace "RdBu" with "RdBu_r"

            linewidths=.9, 

            linecolor='white',

            fmt='.2g',

            center = 0,

            square=True)
#X = data.iloc[:, 1:-1]

X = data.iloc[:, data.columns != 'target']

y = data.iloc[:, data.columns == 'target']
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

Xtr_sm , ytr_sm = SMOTE(random_state=140).fit_sample(X_train, y_train)

Xte_sm, yte_sm = SMOTE(random_state=90).fit_sample(X_test, y_test)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

from sklearn.metrics import precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score 
lr = LogisticRegression()

lr.fit(Xtr_sm , ytr_sm.values.ravel())

y_pred = lr.predict(Xte_sm)
print("Confusion Matrix: ")



print(confusion_matrix(yte_sm, y_pred))
plot_confusion_matrix(lr, Xte_sm, yte_sm, cmap = plt.cm.Blues)
print("Classification Report: ")

print(classification_report(yte_sm, y_pred))
lr = LogisticRegression()

y_pred_score = lr.fit(Xtr_sm ,ytr_sm.values.ravel()).decision_function(Xte_sm.values)





fpr, tpr, thresholds = roc_curve(yte_sm.values.ravel(),y_pred_score)

roc_auc = auc(fpr,tpr)



# Plot ROC

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.0])

plt.ylim([-0.1,1.01])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
Y_pred = lr.predict_proba(test)
acc_decision_tree = round(lr.score(Xtr_sm, ytr_sm) * 100, 2)

print(round(acc_decision_tree,2,), "%")
submission = pd.DataFrame({

        "id": test["id"],

        "target": Y_pred[:,1]

    })

submission.to_csv('submission2.csv', index=False)