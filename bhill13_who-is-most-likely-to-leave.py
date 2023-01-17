# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

% matplotlib inline

import seaborn as sb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# read in data, get idea of dimension and column values

humans = pd.read_csv('../input/HR_comma_sep.csv')

humans.sample(n = 15)
# some sample statistics

humans.describe()
# scatterplot matrix

sb.pairplot(humans, hue = 'left', palette='husl')
group_by_project = humans.groupby(['left','number_project'])['satisfaction_level'].aggregate(np.mean)

group_by_project
sb.factorplot('number_project', # x-axis, number of projects

               col='salary', # one plot for each salary type

              data=humans, # data to analyze

              kind = 'count', # kind of plot to be made

              size = 3, aspect = 0.7, # sizing parameters

              hue = 'left') # color code by left/stayed
sb.factorplot('left', # x axis, number of boxplots per graph

              'average_montly_hours', # column to create boxplots from

              data = humans, 

              kind = 'box', # create boxplots

              col='salary', # number of graphs to make

              size = 3, # sizing parameters 

              aspect = 0.8)
sb.factorplot('left',

             'satisfaction_level', 

             data = humans, 

             kind = 'box',

             col = 'salary',

             size = 3, 

             aspect = 0.8)
sb.boxplot(x='left', y='satisfaction_level', data=humans)
# encode labels for the sales and salary variables

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

humans["sales"] = le.fit_transform(humans["sales"])

humans["salary"] = le.fit_transform(humans["salary"])
sales_df = pd.get_dummies(humans['sales'], prefix = 'sales')

salary_df = pd.get_dummies(humans['salary'], prefix='sal')

sales_df.head()
# concatenate the frames with the original dataframe 

humans = pd.concat([humans,sales_df,salary_df], axis = 1)

humans.sample(n = 15)
# create training and test sets

from sklearn import tree 

from sklearn.cross_validation import train_test_split



humans.drop('sales', inplace=True,axis = 1)

humans.drop('salary', inplace = True, axis = 1)

y = humans.pop('left')

X_train, X_test, Y_train, Y_test = train_test_split(humans, y, test_size = 0.1, random_state = 0)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 100)

forest = rfc.fit(X_train, Y_train)

forest.score(X_train, Y_train)
from sklearn import metrics

y_pred = forest.predict(X_test)

acc_test = metrics.accuracy_score(Y_test, y_pred)

acc_test
# place importances in a data frame

importances = pd.DataFrame({'feature':X_train.columns,

             'importance':np.round(forest.feature_importances_,3)})

# sort them by importance

importances = importances.sort_values('importance', ascending = False).set_index('feature')

importances

importances.plot.bar()