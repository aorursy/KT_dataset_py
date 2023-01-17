import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import seaborn as sns

import statsmodels.api as sm

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix, f1_score, auc
def adjusted_r2score(model, X, y):

    return 1 - (1-model.score(X, y))*(len(y)-1)/(len(y)-X.shape[1]-1)
insurance = pd.read_csv('../input/insurance.csv')

insurance.head()
sns.pairplot(insurance, hue='smoker')
insurance.boxplot('expenses', by='smoker')
insurance.boxplot('expenses', by='region')
insurance_dummy = pd.get_dummies(insurance, drop_first=True)

X = insurance_dummy.drop('expenses', axis=1)

y = insurance['expenses']
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=100)

scaler = StandardScaler()

scaler.fit(train_x)

train_x = pd.DataFrame(scaler.transform(train_x), columns=train_x.columns, index=train_x.index)

test_x = pd.DataFrame(scaler.transform(test_x), columns=test_x.columns, index=test_x.index)
model = LinearRegression()

model.fit(train_x, train_y)

test_pred = model.predict(test_x)
print(r2_score(test_y, test_pred))

print(adjusted_r2score(model, train_x, train_y))
X2 = sm.add_constant(train_x)

est = sm.OLS(train_y, X2)

est2 = est.fit()

print(est2.summary())
def draw_tree(model, columns):

    import pydotplus

    from sklearn.externals.six import StringIO

    from IPython.display import Image

    import os

    from sklearn import tree

    

    #graphviz_path = 'C:\Program Files (x86)\Graphviz2.38/bin/'

    #os.environ["PATH"] += os.pathsep + graphviz_path



    dot_data = StringIO()

    tree.export_graphviz(model,

                         out_file=dot_data,

                         feature_names=columns)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

    return Image(graph.create_png())