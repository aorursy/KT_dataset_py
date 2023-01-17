# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from kaggle.competitions import twosigmanews

# You can only call make_env() once, so don't lose it!

env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



from wordcloud import WordCloud

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))



import datetime

import lightgbm as lgb

from scipy import stats

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split

from nltk.util import ngrams

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler
# fill with 0s, but knowing it won't be good

# and still having outliers

market_train_df['returnsClosePrevMktres1'].fillna(0, inplace=True)



market_train_df.head()
sns.boxplot(x=market_train_df['returnsOpenNextMktres10'])
Q1 = market_train_df.quantile(0.25)

Q3 = market_train_df.quantile(0.75)

IQR = Q3 - Q1

lowerBound = Q1 - 1.5 * IQR

upperBound = Q3 + 1.5 * IQR

print(IQR)
IQR_df = market_train_df.loc[lambda df: (df['returnsOpenNextMktres10'] < lowerBound['returnsOpenNextMktres10']) |  (df['returnsOpenNextMktres10'] > upperBound['returnsOpenNextMktres10'])]
outliers = market_train_df[(market_train_df['returnsOpenNextMktres10'] < lowerBound['returnsOpenNextMktres10']) | (market_train_df['returnsOpenNextMktres10'] > upperBound['returnsOpenNextMktres10'])]

print('Identified outliers: %d' % len(outliers))
# to have a list

outliers_list = [x for x in market_train_df['returnsOpenNextMktres10'] if x < lowerBound['returnsOpenNextMktres10'] or x > upperBound['returnsOpenNextMktres10']]

print('Identified outliers: %d' % len(outliers))
# removing outliers, that is not wanted HERE IS A DF WITHOUT OUTLIERS

cond = market_train_df['returnsOpenNextMktres10'].isin(outliers['returnsOpenNextMktres10']) == True # compares

market_train_df_reduced = market_train_df.drop(market_train_df[cond].index, inplace = True) # drops outliers
market_train_df.sort_values('returnsOpenNextMktres10').head()
outliers_removed = [x for x in market_train_df['returnsOpenNextMktres10'] if x >= lowerBound['returnsOpenNextMktres10'] or x <= upperBound['returnsOpenNextMktres10']]

print('Non-outlier observations: %d' % len(outliers_removed))
data = [go.Histogram(x=outliers_removed[:10000])]

layout = dict(title = "returnsOpenNextMktres10 (random 10.000 sample; without outliers)")

py.iplot(dict(data=data, layout=layout), filename='basic-non-outliers')
sns.boxplot(x=market_train_df['returnsOpenNextMktres10'])
# take all continuous; maybe remove those returns not Savitzky-Golay-ed, or use it on them too.

X = market_train_df[['volume', 'close', 'open', 'returnsClosePrevRaw1','returnsOpenPrevRaw1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10']]

y = market_train_df['returnsOpenNextMktres10']



Z = market_train_df[['close', 'open', 'volume', 'returnsOpenPrevMktres10']] # for correlation
X = X.dropna(axis=0)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
from sklearn.linear_model import LinearRegression



slr = LinearRegression()

slr.fit(X_train, y_train)

print('Intercept: %.3f' % slr.intercept_)

print('Beta 1:  %.3f' % slr.coef_[0])

print('Beta 2:  %.3f' % slr.coef_[1])

print('Beta 3:  %.3f' % slr.coef_[2])



y_pred = slr.predict(X_test)

print(y_pred[:5])



from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test, y_pred))
from sklearn.ensemble import RandomForestRegressor



forest = RandomForestRegressor(n_estimators=100, 

                               random_state=123,

                               max_depth=7,

                               n_jobs=-1)

forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)



from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, y_pred)
# Calculate the absolute errors

errors = abs(y_pred - y_test)

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 2), '% of return?')
# Calculate mean absolute percentage error (MAPE)

mape = 100 * (errors / y_test)

# Calculate and display accuracy

accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')
# Tree picture

from sklearn.tree import export_graphviz

import pydot

# Pull out one tree from the forest

tree = forest.estimators_[5]

# Import tools needed for visualization



from sklearn.tree import export_graphviz

import pydot



# Pull out one tree from the forest

tree = forest.estimators_[5]



# Export the image to a dot file

export_graphviz(tree, out_file = 'tree.dot', 

                feature_names = list(X_train.columns), 

                rounded = True, 

                precision = 5)



# Use dot file to create a graph

(graph, ) = pydot.graph_from_dot_file('tree.dot')



# Write graph to a png file

graph.write_png('tree.png')
from IPython.display import Image

Image(filename = 'tree.png')
import xgboost as xgb



#Fitting XGB regressor 

model = xgb.XGBRegressor(n_estimators=100, 

                         n_jobs=-1,

                         max_depth=3,

                         random_state=123)

model.fit(X_train, y_train)

preds = model.predict(X_test)



from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, preds)
model.score(X_test, y_test) # explained variance?
from sklearn.metrics import explained_variance_score # why

print(explained_variance_score(y_test, preds))
plt.scatter(y_test, preds)

plt.show()