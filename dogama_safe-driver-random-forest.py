# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd

import sklearn

# visualization

import seaborn as sns



import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

#Алгоритм машинного обучения

from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import train_test_split

from sklearn.externals import joblib



%matplotlib inline



# for seaborn issue:

import warnings

warnings.filterwarnings("ignore")



import pandas as pd

import numpy as np

from scipy import stats

import sklearn as sk

import itertools

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns

from statsmodels.graphics.mosaicplot import mosaic



from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Perceptron

from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import VotingClassifier

from sklearn import svm





from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV



sns.set(style='white', context='notebook', palette='deep')
data = pd.read_csv('../input/train.csv')

data.head()
colormap = plt.cm.inferno

plt.figure(figsize=(16,12))

plt.title('correlation ', y=1.05, size=40)

sns.heatmap(data.corr(),linewidths=0.1,vmax=4.0, square=True, cmap=colormap, linecolor='white', annot=True)
x_train = pd.read_csv('../input/train.csv', sep = ',')

x_train.head()

y_train = x_train['target'].to_frame()

y_train.head()
del x_train['target']

x_train.head()
if len(y_train) == len(x_train):

        print('Равны')
inputs_train, inputs_test, expected_output_train, expected_output_test = train_test_split(x_train, y_train, test_size = 0.33, random_state = 42)
rf = RandomForestClassifier(n_estimators=100)

rf.fit(inputs_train, expected_output_train)
#rf = RandomForestClassifier(n_estimators=100)

#rf.fit(inputs_train, expected_output_train)

accuracy1 = rf.score(inputs_test, expected_output_test)

accuracy2 = rf.score(inputs_train, expected_output_train)

    

print("Accuracy_test = {}%".format(accuracy1 * 100))

print("Accuracy_train = {}%".format(accuracy2 * 100))
joblib.dump(rf, "driver", compress=9)
x_test = pd.read_csv('../input/test.csv', sep = ',')

x_test.head()
pred = rf.predict(x_test)# поместите сюда ваши переменные данные

pred
import pandas as pd 

df = pd.DataFrame(pred)

df.plot()
