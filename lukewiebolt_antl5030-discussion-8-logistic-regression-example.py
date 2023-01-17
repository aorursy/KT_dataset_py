# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/the-ultimate-halloween-candy-power-ranking/candy-data.csv')

data.head(5)
print('Shape of Data', data.shape)
#Let's drop the competitorname

data.dtypes
df = data[['chocolate', 'fruity', 'caramel', 'peanutyalmondy', 'nougat', 'crispedricewafer',

               'hard', 'bar', 'pluribus', 'sugarpercent', 'pricepercent', 'winpercent']]



df.head()
X = df.values[:, 1:12]

Y = df.values[:, 0]



print(X.shape)

print(Y.shape)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split (X, Y, test_size = 0.2, random_state = 42)
from matplotlib import pyplot as plt

import seaborn as sns

fig, ax = plt.subplots(figsize=(11,11));

sns.heatmap(df.corr(), ax=ax, annot=True, linewidths=.5, cmap = "YlGnBu");

plt.xlabel('');

plt.ylabel('');

plt.title('Pearson Correlation matrix heatmap');
#We see that our groups are similar

df['chocolate'].value_counts()
import statsmodels.api as sm

logit_model=sm.Logit(y_train, x_train)

result=logit_model.fit()

print(result.summary2())
df_slim = data[['chocolate', 'fruity', 'caramel', 'peanutyalmondy',

                'hard','pluribus', 'sugarpercent', 'pricepercent', 'winpercent']]



X_s = df_slim.values[:, 1:9]

Y_s = df_slim.values[:, 0]



x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split (X_s, Y_s, test_size = 0.2, random_state = 42)

import statsmodels.api as sm

logit_model=sm.Logit(y_train_1, x_train_1)

result=logit_model.fit()

print(result.summary2())
df_final = data[['chocolate', 'fruity', 'pluribus', 'winpercent']]



X_f = df_final.values[:, 1:4]

Y_f = df_final.values[:, 0]



x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split (X_f, Y_f, test_size = 0.2, random_state = 42)
import statsmodels.api as sm

logit_model=sm.Logit(y_train_2, x_train_2)

result=logit_model.fit()

print(result.summary2())
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(x_train_2, y_train_2)

y_pred = logreg.predict(x_test_2)

print('Accuracy of logistic regression  classifier on test set: {:.2f}'.format(logreg.score(x_test_2, y_test_2)))
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)

print('Confusion Matrix')

print(confusion_matrix)

print('This means we have 8 + 7 = 15 correct predictions')

print('and')

print('This means we have 1 + 1 = 2 incorrect predictions')
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

print('read more from the documentation on each of these metrics')

print('https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html')
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test_2, logreg.predict(x_test_2))

fpr, tpr, thresholds = roc_curve(y_test_2, logreg.predict_proba(x_test_2)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()