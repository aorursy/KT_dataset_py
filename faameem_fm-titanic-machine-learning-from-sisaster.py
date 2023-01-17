import seaborn

import numpy

import sys



from pandas import read_csv

from pandas import set_option

from matplotlib import pyplot



from sklearn.preprocessing import StandardScaler



from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV



from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



from sklearn.pipeline import Pipeline



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
_df_gender_class_model = read_csv('../input/genderclassmodel.csv')
_df_gender_class_model.head()
_df_gender_model = read_csv('../input/gendermodel.csv')
_df_gender_model.head()
_df_test = read_csv('../input/test.csv')
_df_test.head()
_df_train = read_csv('../input/train.csv')
_df_train.head()
_file = open('../input/gendermodel.py').read()
print(_file)
_file = open('../input/myfirstforest.py').read()
print(_file)
_df_train.shape
_df_test.shape
_df_train.dtypes
_df_test.dtypes
set_option('precision',2)
_df_train.describe()
_df_train.describe(include='all')
_df_train.corr(method='pearson')
_df_train.groupby('Survived').size()
_df_train.plot(kind='density', subplots=True, layout=(4,3), figsize=(12,20), sharex=False, sharey=False)

pyplot.show()