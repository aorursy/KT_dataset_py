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
ml_df = pd.read_csv('/kaggle/input/stackindex/MLTollsStackOverflow.csv')

ml_df['month']
ml_df.columns
ml_df[['nltk', 'spacy', 'stanford-nlp', 'python', 'r', 'numpy',

       'scipy', 'matlab', 'machine-learning', 'pandas', 'pytorch', 'keras',

       'nlp', 'apache-spark', 'hadoop', 'pyspark', 'python-3.x', 'tensorflow',

       'deep-learning', 'neural-network', 'lstm', 'time-series', 'pillow',

       'rasa', 'opencv', 'pipenv', 'seaborn', 'Dask', 'jupyter', 'AllenNLP',

       'Theano', 'plotly', 'scikit-learn', 'BeautifulSoup', 'scrapy', 'Gensim',

       'FastText', 'Pydot', 'Pybrain', 'Pytil', 'Pygame', 'Colab', 'Shogun',

       'KNIME', 'Apache', 'Gunicorn', 'Pygtk', 'Weka', 'Conda', 'Ray',

       'matlab.1', 'accord.net', 'regression', 'classification', 'correlation',

       'cluster-analysis', 'H2o', 'Mallet', 'Numba', 'Tableau', 'Trifacta',

       'PyArrow', 'Rasterio', 'Orange3', 'PyMC3', 'Opennn', 'Oryx', 'Istio',

       'Venes', 'Plotnine', 'Gluon', 'Plato', 'Sympy', 'Flair',

       'stanford-nlp.1', 'pyqt', 'Nolearn', 'Lasagne', 'OCR',

       'Apache-spark-mlib', 'azure-virtual-machine']] += 1

ml_df[['nltk', 'spacy', 'stanford-nlp', 'python', 'r', 'numpy',

       'scipy', 'matlab', 'machine-learning', 'pandas', 'pytorch', 'keras',

       'nlp', 'apache-spark', 'hadoop', 'pyspark', 'python-3.x', 'tensorflow',

       'deep-learning', 'neural-network', 'lstm', 'time-series', 'pillow',

       'rasa', 'opencv', 'pipenv', 'seaborn', 'Dask', 'jupyter', 'AllenNLP',

       'Theano', 'plotly', 'scikit-learn', 'BeautifulSoup', 'scrapy', 'Gensim',

       'FastText', 'Pydot', 'Pybrain', 'Pytil', 'Pygame', 'Colab', 'Shogun',

       'KNIME', 'Apache', 'Gunicorn', 'Pygtk', 'Weka', 'Conda', 'Ray',

       'matlab.1', 'accord.net', 'regression', 'classification', 'correlation',

       'cluster-analysis', 'H2o', 'Mallet', 'Numba', 'Tableau', 'Trifacta',

       'PyArrow', 'Rasterio', 'Orange3', 'PyMC3', 'Opennn', 'Oryx', 'Istio',

       'Venes', 'Plotnine', 'Gluon', 'Plato', 'Sympy', 'Flair',

       'stanford-nlp.1', 'pyqt', 'Nolearn', 'Lasagne', 'OCR',

       'Apache-spark-mlib', 'azure-virtual-machine']] = np.log(ml_df[['nltk', 'spacy', 'stanford-nlp', 'python', 'r', 'numpy',

       'scipy', 'matlab', 'machine-learning', 'pandas', 'pytorch', 'keras',

       'nlp', 'apache-spark', 'hadoop', 'pyspark', 'python-3.x', 'tensorflow',

       'deep-learning', 'neural-network', 'lstm', 'time-series', 'pillow',

       'rasa', 'opencv', 'pipenv', 'seaborn', 'Dask', 'jupyter', 'AllenNLP',

       'Theano', 'plotly', 'scikit-learn', 'BeautifulSoup', 'scrapy', 'Gensim',

       'FastText', 'Pydot', 'Pybrain', 'Pytil', 'Pygame', 'Colab', 'Shogun',

       'KNIME', 'Apache', 'Gunicorn', 'Pygtk', 'Weka', 'Conda', 'Ray',

       'matlab.1', 'accord.net', 'regression', 'classification', 'correlation',

       'cluster-analysis', 'H2o', 'Mallet', 'Numba', 'Tableau', 'Trifacta',

       'PyArrow', 'Rasterio', 'Orange3', 'PyMC3', 'Opennn', 'Oryx', 'Istio',

       'Venes', 'Plotnine', 'Gluon', 'Plato', 'Sympy', 'Flair',

       'stanford-nlp.1', 'pyqt', 'Nolearn', 'Lasagne', 'OCR',

       'Apache-spark-mlib', 'azure-virtual-machine']])
ml_df.plot(x='month', figsize=(20,10))
ax = ml_df.plot.line(x='month',y=['Apache'], figsize=(20,10))

ax = ml_df.plot.line(x='month', y = ['OCR'],figsize=(20,10))
from sklearn.tree import DecisionTreeRegressor

from datetime import datetime

#predicting the future of OCR

regr_1 = DecisionTreeRegressor(max_depth=4)

get_year = lambda x: int(x[:2])

get_month = lambda x: int(str(datetime.strptime(x[3:], "%b"))[5:7])

X_full = pd.DataFrame()



y = ml_df['python']

X_full['year'] = ml_df['month'].apply(get_year)

X_full['month'] = ml_df['month'].apply(get_month)



y_train, X_train = y[:-10], X_full[:-10]

y_test, X_test = y[-10:], X_full[-10:]

regr_1.fit(X_train, y_train)
y_pred = pd.Series(regr_1.predict(X_test))
from sklearn.metrics import mean_absolute_error as MAE

print('MAE is {:.1f}%'.format(100*MAE(y_test, y_pred)/y_test.mean()))
import matplotlib.pyplot as plt

plt.plot(X_test['month'],y_pred)

plt.plot(X_test['month'],y_test)
from sklearn.linear_model import LinearRegression as OUR_MODEL

from datetime import datetime



class Keyword_model:

    def __init__(self, keyword, dataset='/kaggle/input/stackindex/MLTollsStackOverflow.csv'):

        

        self.df = pd.read_csv('/kaggle/input/stackindex/MLTollsStackOverflow.csv')

        self.keyword = keyword

        self.df[keyword] += 1

        self.df[keyword] = self.df[keyword].apply(np.log)

        self.df = self.df[[keyword, 'month']]

        

    def fit(self):

        self.lreg = OUR_MODEL()

        

        get_year = lambda x: int(x[:2])

        get_month = lambda x: int(str(datetime.strptime(x[3:], "%b"))[5:7])

        fix_null = lambda x: 0 if(x != x) else x

        

        y = self.df[self.keyword].apply(fix_null)

        X_full['year'] = self.df['month'].apply(get_year)

        X_full['month'] = self.df['month'].apply(get_month)



        self.lreg.fit(X_full, y)

        

    def predict(self):

        month = [x for x in range(1, 13)]+[x for x in range(1, 13)]

        year = [20 for _ in range(12)]+[21 for _ in range(12)]

        X_future = pd.DataFrame(list(zip(year, month)), 

               columns =['year', 'month'])

        

        return self.lreg.predict(X_future)

        

    def ploting(self):

        ax = self.df.plot.line(x='month', y = [self.keyword],figsize=(10,5))

        return ax

    

    def get_future(self):

        month = [x for x in range(1, 13)]+[x for x in range(1, 13)]

        year = [20 for _ in range(12)]+[21 for _ in range(12)]

        X_future = pd.DataFrame(list(zip(year, month)), 

               columns =['year', 'month'])

        months = {

          1: 'Jan', 

          2: 'Feb', 

          3: 'Mar', 

          4: 'Apr', 

          5: 'May', 

          6: 'Jun', 

          7: 'Jul', 

          8: 'Aug', 

          9: 'Sep', 

          10: 'Oct', 

          11: 'Nov', 

          12: 'Dec'}

        X_future['dd'] = X_future['month'].apply(lambda x: months[x]) + '-' + X_future['year'].apply(lambda x: str(x))

        

        pred = pd.DataFrame(zip(X_future['dd'], pd.Series(self.predict())), columns=['month', self.keyword])

        

        ax = self.df.append(pred)

        return ax



print('done')
obj1 = Keyword_model('python')

obj1.fit()

obj1.get_future().plot(x='month')
new = {}

for col in ml_df.columns[1:]:

    print(col, end=',')

    obj1 = Keyword_model(col)

    obj1.fit()

    new[col] = obj1.get_future()
new_df = pd.DataFrame()

for dic in new:

    new_df[dic] = new[dic][dic]

    

new_df['month'] = new[dic]['month']
import plotly.express as px

cols = new_df.columns[:-1]

px.line(new_df, x="month", y=cols, title='SO words')