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
import scipy.stats as st

import statsmodels.api as sm

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix

import matplotlib.mlab as mlab
df = pd.read_csv(r'/kaggle/input/framingham-heart-study-dataset/framingham.csv')
df.head()
df.rename(columns = {'male' : 'sex_is_male'}, inplace = True)
df.info()
df.isnull().sum()
df.dropna(axis = 0,inplace = True)
def draw_histograms(dataframe,features,rows,cols):

    fig = plt.figure(figsize = (20,20))

    for i, feature in enumerate(features):

        a = fig.add_subplot(rows,cols,i+1)

        dataframe[feature].hist(bins = 20,ax=a,facecolor = 'green')

        a.set_title(feature + " Distribution",color = 'black')

    fig.tight_layout()

    plt.show()

draw_histograms(df,df.columns,6,3)
#sns.pairplot(df)
df.describe()
corr_matrix = df.corr()

f, ax = plt.subplots(figsize =(9, 8)) 

sns.heatmap(corr_matrix, ax = ax, cmap ="YlGnBu", linewidths = 0.1) 
df.head()
df['sex_is_male'].unique()
df['age'].unique()
df['prevalentStroke'].unique()
df['diabetes'].unique()
df['sysBP'].unique()
import sklearn
x = df[['sex_is_male','age','prevalentStroke','diabetes','sysBP']]

#x = df[['sex_is_male','age','prevalentStroke','sysBP']]

y = df[['TenYearCHD']]



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3)
from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression()

lgr.fit(x_test,y_test)

y_pred = lgr.predict(x_test)
sklearn.metrics.accuracy_score(y_test,y_pred)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize = (8,5))

sns.heatmap(conf_matrix,annot = True,fmt = 'd',cmap = 'YlGnBu')
TN=cm[0,0]

TP=cm[1,1]

FN=cm[1,0]

FP=cm[0,1]

sensitivity=TP/float(TP+FN)

specificity=TN/float(TN+FP)
print('The accuracy of the model = TP+TN/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n',



'The Missclassification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',



'Sensitivity = ',TP/float(TP+FN),'\n',



'Specificity = ',TN/float(TN+FP),'\n',



'Positive Predictive value = TP/(TP+FP) = ',TP/float(TP+FP),'\n',



'Negative predictive Value = TN/(TN+FN) = ',TN/float(TN+FN),'\n',



'Positive Likelihood Ratio = ',sensitivity/(1-specificity),'\n',



'Negative likelihood Ratio = ',(1-sensitivity)/specificity)
from sklearn.metrics import roc_curve

y_pred_prob_yes = lgr.predict_proba(x_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_yes[:,1])

plt.plot(fpr,tpr,color='r')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.title('ROC curve for Heart disease classifier')

plt.xlabel('False positive rate (1-Specificity)')

plt.ylabel('True positive rate (Sensitivity)')

plt.grid(True)

plt.show()
sklearn.metrics.roc_auc_score(y_test,y_pred_prob_yes[:,1])