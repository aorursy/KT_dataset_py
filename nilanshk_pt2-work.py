# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import numpy as np

import itertools

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.express as px

import statsmodels.api as sm



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data=pd.read_csv('../input/credit/Credit (1).csv')
data.head()
data.info()
data.isnull().sum()
data.nunique()
categorical=["NoofTime30_59DaysPastDue", "NorOfTimes90DaysLate", "NoRealEstateLoansOrLines", "NoOfTime60_89DaysPastDue", "NoOfDependents"]
data.describe()
data[categorical].columns
for col in data[categorical].columns:

#     sns.countplot(data[col])

    fig = px.histogram(x=data[col], color=data[col])

    fig.update_layout(title_text=col)

    fig.show()
sns.pairplot(data)
coll=[]

for i in data.columns:

    if data[i].nunique()>15:

        coll.append(i)
coll
import scipy.stats as stats

plt.subplots(figsize=(12,9))



sns.distplot(data['RevolvingUtiOfUnsecuredLines'], fit=stats.norm)



# Get the fitted parameters used by the function



(mu, sigma) = stats.norm.fit(data['RevolvingUtiOfUnsecuredLines'])



# plot with the distribution



plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

plt.ylabel('Frequency')



#Probablity plot



fig = plt.figure()

stats.probplot(data['RevolvingUtiOfUnsecuredLines'], plot=plt)

plt.show()
for i in coll:

    plt.subplots(figsize=(12,9))



    sns.distplot(data[i], fit=stats.norm)



    # Get the fitted parameters used by the function



    (mu, sigma) = stats.norm.fit(data[i])



    # plot with the distribution



    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

    plt.ylabel('Frequency')



    #Probablity plot



    fig = plt.figure()

    stats.probplot(data[i], plot=plt)

    plt.show()
from scipy.stats import norm, skew #for some statistics

# Check the skew of all numerical features

skewed_feats = data.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness
skewness.plot()

skewness = skewness[abs(skewness) > 0.75]

new_data=data.iloc[:,1:]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.05

for feat in new_data.columns:

    new_data[feat] = boxcox1p(new_data[feat], lam)
new_data.head()
for i in coll:

    plt.subplots(figsize=(12,9))



    sns.distplot(new_data[i], fit=stats.norm)



    # Get the fitted parameters used by the function



    (mu, sigma) = stats.norm.fit(new_data[i])



    # plot with the distribution



    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

    plt.ylabel('Frequency')



    #Probablity plot



    fig = plt.figure()

    stats.probplot(new_data[i], plot=plt)

    plt.show()
data
Y=data.iloc[:,0:1]

X1=data.iloc[:,1:]

X2=new_data
X2
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

Y
clf1 = DecisionTreeClassifier(random_state=10)

clf2 = KNeighborsClassifier(n_neighbors=5)

from sklearn import metrics

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X1, Y, test_size=0.2, random_state=10)
clf1.fit(X_train, y_train)

y_pred1=clf1.predict(X_test)

acc1=metrics.accuracy_score(y_test, y_pred1)

prec1=metrics.precision_score(y_test, y_pred1, average='weighted')

recall1=metrics.recall_score(y_test, y_pred1, average='weighted')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred1))

print("Precision:",metrics.precision_score(y_test, y_pred1, average='weighted'))

print("Recall:",metrics.recall_score(y_test, y_pred1, average='weighted'))

print('\nClassification Report\n')

print(classification_report(y_test, y_pred1, target_names=['0', '1']))

clf2.fit(X_train, y_train)

y_pred2=clf2.predict(X_test)

acc2=metrics.accuracy_score(y_test, y_pred2)

prec2=metrics.precision_score(y_test, y_pred2, average='weighted')

recall2=metrics.recall_score(y_test, y_pred2, average='weighted')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred2))

print("Precision:",metrics.precision_score(y_test, y_pred2, average='weighted'))

print("Recall:",metrics.recall_score(y_test, y_pred2, average='weighted'))

print('\nClassification Report\n')

print(classification_report(y_test, y_pred2, target_names=['0', '1']))
from mlxtend.evaluate import confusion_matrix

from mlxtend.plotting import plot_confusion_matrix



y_preds1=pd.DataFrame(y_pred1)

cm = confusion_matrix(y_target=y_test, 

                      y_predicted=y_preds1, 

                      binary=False)

cm





import matplotlib.pyplot as plt

from mlxtend.evaluate import confusion_matrix

print("Decision Tree")

fig, ax = plot_confusion_matrix(conf_mat=cm)

plt.show()
from mlxtend.evaluate import confusion_matrix

from mlxtend.plotting import plot_confusion_matrix



y_preds2=pd.DataFrame(y_pred2)



cm = confusion_matrix(y_target=y_test, 

                      y_predicted=y_preds2, 

                      binary=False)

cm





import matplotlib.pyplot as plt

from mlxtend.evaluate import confusion_matrix

print("KNN Classifier")

fig, ax = plot_confusion_matrix(conf_mat=cm)

plt.show()
rp={'Model': ['Decision Tree', 'KNN' ],'Accuracy':[acc1,acc2], 'Precision': [prec1,prec2], 'Recall': [recall1, recall2]}
report=pd.DataFrame(rp, columns=['Model','Accuracy', 'Precision', 'Recall'])
report
X_train, X_test, y_train, y_test = train_test_split(X2, Y, test_size=0.2, random_state=10)
clf1.fit(X_train, y_train)

y_pred1=clf1.predict(X_test)

acc1=metrics.accuracy_score(y_test, y_pred1)

prec1=metrics.precision_score(y_test, y_pred1, average='weighted')

recall1=metrics.recall_score(y_test, y_pred1, average='weighted')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred1))

print("Precision:",metrics.precision_score(y_test, y_pred1, average='weighted'))

print("Recall:",metrics.recall_score(y_test, y_pred1, average='weighted'))

print('\nClassification Report\n')

print(classification_report(y_test, y_pred1, target_names=['0', '1']))

clf2.fit(X_train, y_train)

y_pred2=clf2.predict(X_test)

acc2=metrics.accuracy_score(y_test, y_pred2)

prec2=metrics.precision_score(y_test, y_pred2, average='weighted')

recall2=metrics.recall_score(y_test, y_pred2, average='weighted')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred2))

print("Precision:",metrics.precision_score(y_test, y_pred2, average='weighted'))

print("Recall:",metrics.recall_score(y_test, y_pred2, average='weighted'))

print('\nClassification Report\n')

print(classification_report(y_test, y_pred2, target_names=['0', '1']))
from mlxtend.evaluate import confusion_matrix

from mlxtend.plotting import plot_confusion_matrix



y_preds1=pd.DataFrame(y_pred1)

cm = confusion_matrix(y_target=y_test, 

                      y_predicted=y_preds1, 

                      binary=False)

cm





import matplotlib.pyplot as plt

from mlxtend.evaluate import confusion_matrix

print("Decision Tree")

fig, ax = plot_confusion_matrix(conf_mat=cm)

plt.show()
from mlxtend.evaluate import confusion_matrix

from mlxtend.plotting import plot_confusion_matrix



y_preds2=pd.DataFrame(y_pred2)



cm = confusion_matrix(y_target=y_test, 

                      y_predicted=y_preds2, 

                      binary=False)

cm





import matplotlib.pyplot as plt

from mlxtend.evaluate import confusion_matrix

print("KNN Classifier")

fig, ax = plot_confusion_matrix(conf_mat=cm)

plt.show()
rp={'Model': ['Decision Tree', 'KNN' ],'Accuracy':[acc1,acc2], 'Precision': [prec1,prec2], 'Recall': [recall1, recall2]}
report=pd.DataFrame(rp, columns=['Model','Accuracy', 'Precision', 'Recall'])
report