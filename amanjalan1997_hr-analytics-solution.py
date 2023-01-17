# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neural_network import MLPClassifier

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import OneClassSVM



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df= pd.read_csv('../input/HR_comma_sep.csv')

df_copy = pd.read_csv('../input/HR_comma_sep.csv')

df.head()
df[['sales','left']].groupby(['sales']).count()

number = LabelEncoder()

df['sales'] = number.fit_transform(df['sales'])

df['salary'] = number.fit_transform(df['salary'])

df[['salary','left']].groupby(['salary']).count()
features = ['satisfaction_level', 'last_evaluation', 'number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','sales', 'salary']
#forest = RandomForestClassifier()

#forest.fit(df[features], df['left'])

#pred = forest.predict(df[features])



#gb = GradientBoostingClassifier()

#gb.fit(df[features], df['left'])

#pred = gb.predict(df[features])



#lg = LogisticRegression()

#lg.fit(df[features], df['left'])

#pred = lg.predict(df[features])



dt = DecisionTreeClassifier()

dt.fit(df[features], df['left'])

pred = dt.predict(df[features])

df['pred'] = pred

acc = df['left'] - df['pred']

acc=acc.abs()

acc.sum()