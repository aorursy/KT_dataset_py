import numpy as np

import pylab as pl

import pandas as pd

import matplotlib.pyplot as plt 

%matplotlib inline

import seaborn as sns

from sklearn.utils import shuffle

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.model_selection import cross_val_score, GridSearchCV

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/creditcardfraud/creditcard.csv")

data.info()

data[0:10]
print("Any missing sample in test set:",data.isnull().values.any(), "\n")
#Frequency distribution of classes"

train_outcome = pd.crosstab(index=data["Class"],  # Make a crosstab

                              columns="count")      # Name the count column



train_outcome
cnt_pro = data['Class'].value_counts()

plt.figure(figsize=(6,4))

sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)

plt.ylabel('Number of cp_type', fontsize=12)

plt.xlabel('cp_type', fontsize=12)

plt.xticks(rotation=80)

plt.show();
#Top 10 credit card fraud losses by time

top_fraud = data.sort_values(by='Amount', ascending=False)[:10]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=top_fraud.Time, x=top_fraud.Amount)

plt.xticks()

plt.xlabel('Amount')

plt.ylabel('Time')

plt.title('Credit card fraud losses')

plt.show()
fig, ax = plt.subplots(1, 2, figsize=(18,4))



amount_val = data['Amount'].values

time_val = data['Time'].values



sns.distplot(amount_val, ax=ax[0], color='r')

ax[0].set_title('Distribution of Transaction Amount', fontsize=14)

ax[0].set_xlim([min(amount_val), max(amount_val)])



sns.distplot(time_val, ax=ax[1], color='b')

ax[1].set_title('Distribution of Transaction Time', fontsize=14)

ax[1].set_xlim([min(time_val), max(time_val)])







plt.show()
data = data[['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10'

               ,'V11','V12','V13','V14','V15','V16','V17','V18', 'V19','V20'

               ,'V21','V22','V23','V24','V25','V26','V27','V28','Amount','Class']] #Subsetting the data

cor = data.corr() #Calculate the correlation of the above variables

sns.heatmap(cor, square = True) #Plot the correlation as heat map
from sklearn.model_selection import train_test_split

Y = data['Class']

X = data.drop(columns=['Class'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=9)
print('X train shape: ', X_train.shape)

print('Y train shape: ', Y_train.shape)

print('X test shape: ', X_test.shape)

print('Y test shape: ', Y_test.shape)
from sklearn.naive_bayes import GaussianNB



# We define the model

nbcla = GaussianNB()



# We train model

nbcla.fit(X_train, Y_train)



# We predict target values

Y_predict3 = nbcla.predict(X_test)
test_acc_nbcla  = round(nbcla .fit(X_train,Y_train).score(X_test, Y_test)* 100, 2)

train_acc_nbcla  = round(nbcla .fit(X_train, Y_train).score(X_train, Y_train)* 100, 2)
# The confusion matrix

nbcla_cm = confusion_matrix(Y_test, Y_predict3)

f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(nbcla_cm, annot=True, linewidth=0.7, linecolor='black', fmt='g', ax=ax, cmap="BuPu")

plt.title('Naive Bayes Classification Confusion Matrix')

plt.xlabel('Y predict')

plt.ylabel('Y test')

plt.show()
model1 = pd.DataFrame({

    'Model': ['Naive Bayes'],

    'Train Score': [train_acc_nbcla],

    'Test Score': [test_acc_nbcla]

})

model1.sort_values(by='Test Score', ascending=False)
from sklearn.metrics import average_precision_score

average_precision = average_precision_score(Y_test, Y_predict3)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))


from sklearn.metrics import precision_recall_curve

from sklearn.metrics import plot_precision_recall_curve

import matplotlib.pyplot as plt



disp = plot_precision_recall_curve(nbcla,X_train, Y_train)

disp.ax_.set_title('2-class Precision-Recall curve: '

                   'AP={0:0.2f}'.format(average_precision))
from sklearn.metrics import roc_curve





# Naive Bayes Classification

Y_predict3_proba = nbcla.predict_proba(X_test)

Y_predict3_proba = Y_predict3_proba[:, 1]

fpr, tpr, thresholds = roc_curve(Y_test, Y_predict3_proba)

plt.subplot(332)

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr, label='ANN')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('ROC Curve Naive Bayes')

plt.grid(True)

plt.subplots_adjust(top=2, bottom=0.08, left=0.10, right=1.4, hspace=0.45, wspace=0.45)

plt.show()