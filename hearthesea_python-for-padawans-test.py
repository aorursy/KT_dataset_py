%matplotlib inline

import os

import pandas as pd

from matplotlib import pyplot as plt

import numpy as np

import math
data = pd.read_csv('../input/loan.csv', low_memory=False)

data.drop(['id', 'member_id', 'emp_title'], axis=1, inplace=True)



data.replace('n/a', np.nan,inplace=True)

data.emp_length.fillna(value=0,inplace=True)



data['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)

data['emp_length'] = data['emp_length'].astype(int)



data['term'] = data['term'].apply(lambda x: x.lstrip())
import seaborn as sns

import matplotlib



s = pd.value_counts(data['emp_length']).to_frame().reset_index()

s.columns = ['type', 'count']



def emp_dur_graph(graph_title):



    sns.set_style("whitegrid")

    ax = sns.barplot(y = "count", x = 'type', data=s)

    ax.set(xlabel = '', ylabel = '', title = graph_title)

    ax.get_yaxis().set_major_formatter(

    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    

emp_dur_graph('Distribution of employment length for issued loans')
import seaborn as sns

import matplotlib



print (plt.style.available)
import seaborn as sns

import matplotlib



plt.style.use('fivethirtyeight')

ax = emp_dur_graph('Fivethirty eight style')
plt.style.use('seaborn-notebook')

ax = emp_dur_graph('Seaborn-notebook style')
plt.style.use('ggplot')

ax = emp_dur_graph('ggplot style')
plt.style.use('classic')

ax = emp_dur_graph('classic style')
import datetime



data.issue_d.fillna(value=np.nan,inplace=True)

issue_d_todate = pd.to_datetime(data.issue_d)

data.issue_d = pd.Series(data.issue_d).str.replace('-2015', '')

data.emp_length.fillna(value=np.nan,inplace=True)



data.drop(['loan_status'],1, inplace=True)



data.drop(['pymnt_plan','url','desc','title' ],1, inplace=True)



data.earliest_cr_line = pd.to_datetime(data.earliest_cr_line)

import datetime as dt

data['earliest_cr_line_year'] = data['earliest_cr_line'].dt.year
import seaborn as sns

import matplotlib.pyplot as plt



s = pd.value_counts(data['earliest_cr_line']).to_frame().reset_index()

s.columns = ['date', 'count']



s['year'] = s['date'].dt.year

s['month'] = s['date'].dt.month



d = s[s['year'] > 2008]



plt.rcParams.update(plt.rcParamsDefault)

sns.set_style("whitegrid")



g = sns.FacetGrid(d, col="year")

g = g.map(sns.pointplot, "month", "count")

g.set(xlabel = 'Month', ylabel = '')

axes = plt.gca()

_ = axes.set_ylim([0, d.year.max()])

plt.tight_layout()
mths = [s for s in data.columns.values if "mths" in s]

mths



data.drop(mths, axis=1, inplace=True)
group = data.groupby('grade').agg([np.mean])

loan_amt_mean = group['loan_amnt'].reset_index()



import seaborn as sns

import matplotlib



plt.style.use('fivethirtyeight')



sns.set_style("whitegrid")

ax = sns.barplot(y = "mean", x = 'grade', data=loan_amt_mean)

ax.set(xlabel = '', ylabel = '', title = 'Average amount loaned, by loan grade')

ax.get_yaxis().set_major_formatter(

matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
filtered  = data[data['earliest_cr_line_year'] > 2008]

group = filtered.groupby(['grade', 'earliest_cr_line_year']).agg([np.mean])



graph_df = group['int_rate'].reset_index()



import seaborn as sns

import matplotlib



plt.style.use('fivethirtyeight')

plt.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')



sns.set_style("whitegrid")

g = sns.FacetGrid(graph_df, col="grade", col_wrap = 2)

g = g.map(sns.pointplot, "earliest_cr_line_year", "mean")

g.set(xlabel = 'Year', ylabel = '')

axes = plt.gca()

axes.set_ylim([0, graph_df['mean'].max()])

_ = plt.tight_layout()
#data['emp_length'].fillna(data['emp_length'].mean())

#data['emp_length'].fillna(data['emp_length'].median())

#data['emp_length'].fillna(data['earliest_cr_line_year'].median())



from sklearn.ensemble import RandomForestClassifier

rf =  RandomForestClassifier(max_depth=5, n_estimators=100, max_features=1)



data['emp_length'].replace(to_replace=0, value=np.nan, inplace=True, regex=True)



cat_variables = ['term', 'purpose', 'grade']

columns = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'grade', 'purpose', 'term']



def impute_missing_algo(df, target, cat_vars, cols, algo):



    y = pd.DataFrame(df[target])

    X = df[cols].copy()

    X.drop(cat_vars, axis=1, inplace=True)



    cat_vars = pd.get_dummies(df[cat_vars])



    X = pd.concat([X, cat_vars], axis = 1)



    y['null'] = y[target].isnull()

    y['null'] = y.loc[:, target].isnull()

    X['null'] = y[target].isnull()



    y_missing = y[y['null'] == True]

    y_notmissing = y[y['null'] == False]

    X_missing = X[X['null'] == True]

    X_notmissing = X[X['null'] == False]



    y_missing.loc[:, target] = ''



    dfs = [y_missing, y_notmissing, X_missing, X_notmissing]

    

    for df in dfs:

        df.drop('null', inplace = True, axis = 1)



    y_missing = y_missing.values.ravel(order='C')

    y_notmissing = y_notmissing.values.ravel(order='C')

    X_missing = X_missing.as_matrix()

    X_notmissing = X_notmissing.as_matrix()

    

    algo.fit(X_notmissing, y_notmissing)

    y_missing = algo.predict(X_missing)



    y.loc[(y['null'] == True), target] = y_missing

    y.loc[(y['null'] == False), target] = y_notmissing

    

    return(y[target])



data['emp_length'] = impute_missing_algo(data, 'emp_length', cat_variables, columns, rf)

data['earliest_cr_line_year'] = impute_missing_algo(data, 'earliest_cr_line_year', cat_variables, columns, rf)
y = data.term



cols = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'grade', 'emp_length', 'purpose', 'earliest_cr_line_year']

X = pd.get_dummies(data[cols])



from sklearn import preprocessing



y = y.apply(lambda x: x.lstrip())



le = preprocessing.LabelEncoder()

le.fit(y)



y = le.transform(y)

X = X.as_matrix()



from sklearn import linear_model



logistic = linear_model.LogisticRegression()



logistic.fit(X, y)
from sklearn import linear_model, decomposition

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.grid_search import GridSearchCV



rf = RandomForestClassifier(max_depth=5, max_features=1)



pca = decomposition.PCA()

pipe = Pipeline(steps=[('pca', pca), ('rf', rf)])



n_comp = [3, 5]

n_est = [10, 20]



estimator = GridSearchCV(pipe,

                         dict(pca__n_components=n_comp,

                              rf__n_estimators=n_est))



estimator.fit(X, y)