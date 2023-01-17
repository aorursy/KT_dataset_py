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
import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns 

import warnings

warnings.filterwarnings("ignore")

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

import collections

from sklearn.linear_model import LogisticRegression
df = pd.read_csv('/kaggle/input/advertisement/advertising.csv')
df.head()
df['Clicked on Ad'].value_counts()
sns.jointplot(x='Age', y='Daily Time Spent on Site', data=df, kind='kde', color='blue')
sns.jointplot(x='Area Income', y='Daily Internet Usage', data=df, kind='kde', color='purple')
df.Timestamp =df.Timestamp.apply(lambda x: x.split()[1])

df.Timestamp = df.Timestamp.apply(lambda x: x.split(':')[0])

df.Timestamp = df.Timestamp.astype(int)

df.head()
sns.jointplot(x='Timestamp', y='Clicked on Ad', data=df, kind='kde', color='purple')
country_map =df.Country.value_counts().to_dict()

df['country_encoding'] = df['Country'].map(country_map)
word_list = df['Ad Topic Line'].apply(lambda x: x.split())

a=collections.Counter()

for i in range(len(word_list)):

    b = collections.Counter(word_list[i])

    a = a+ b 

k = a.most_common(len(a))
word = []

freq = []

for i in range(len(k)):

    word.append(k[i][0])

    freq.append(k[i][1])

dict_1 = dict(zip(word, freq))
plus = []

for i in df['Ad Topic Line']:

    k = i.split()

    p = 0

    for j in range(len(k)):

        q = k[j]

        p = dict_1[q]+p

    plus.append(p/len(k))

df['Ad_encoding'] = plus
drop = ['Ad Topic Line','City','Country','Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns = drop), 

                                                   df['Clicked on Ad'], test_size=0.3, 

                                                    random_state=100)

from sklearn.model_selection import GridSearchCV

c_n=np.logspace(-3,3,7)

logit_param_grid = {'C':c_n,}

grid = GridSearchCV(LogisticRegression(),param_grid=logit_param_grid, cv=10)

grid.fit(X_train, y_train)

best_c = grid.best_params_

grid.best_params_, grid.best_score_
logmodel = LogisticRegression(fit_intercept=True, solver='liblinear',C=grid.best_params_['C'])

logmodel.fit(X_test,y_test)
predictions = logmodel.predict(X_test)

probs = logmodel.predict_proba(X_test)

print("accuracy: {:.2f}".format(logmodel.score(X_test, y_test)))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, probs[:,1])

roc_auc = auc(false_positive_rate, true_positive_rate)

from sklearn.metrics import plot_confusion_matrix

disp = plot_confusion_matrix(logmodel, X_test, y_test,

                                 cmap=plt.cm.Blues)
plt.title('Click Rate Characteristic')

plt.plot(false_positive_rate, true_positive_rate, 'b',

label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
coef = logmodel.coef_[0]

features = X_train.columns.tolist()

coef_table = pd.DataFrame({'feature': features, 'coefficient': coef})

print(coef_table)
coef_table
plt.rcParams['figure.figsize'] = (10,5)

plt.barh(coef_table.feature,coef_table.coefficient,height = 0.5 ,align='center',color = 'tan')

plt.xlabel('Importances')

plt.ylabel('Features')

#plt.savefig('fig13.png',dpi = 300,bbox_inches='tight')
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))