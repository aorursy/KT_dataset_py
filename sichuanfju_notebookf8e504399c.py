import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_recall_curve
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.tail()
test.tail()
train.info()
test.info()
train.shape
test.shape
#Age
fig=plt.figure(figsize=(10,6))

train[train['Survived']==1]['Age'].value_counts().plot(kind='kde',label='Survivied',alpha=0.6)
train[train['Survived']==0]['Age'].value_counts().plot(kind='kde',color='#FA2379',label='Died',alpha=0.6)
plt.xlabel('Age')
plt.legend(loc="best")
plt.grid()