# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/workout-measurements/8094130048356.csv', encoding='ISO-8859-2')

df.head()
df.isnull().sum()
corr = df.corr()

corr.style.background_gradient(cmap = 'cubehelix')
import statsmodels.formula.api as smf



from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, auc

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.preprocessing import LabelEncoder, StandardScaler 



from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.neural_network import MLPClassifier



from warnings import filterwarnings

filterwarnings('ignore')
x = df.drop(['intensity', 'calories'], axis=1)

x.fillna(999999, inplace=True)

y = df['calories']
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(x, y)
dt_feat = pd.DataFrame(dt.feature_importances_, index=x.columns, columns=['feat_importance'])

dt_feat.sort_values('feat_importance').tail(8).plot.barh()

plt.show()
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from graphviz import Source
from IPython.display import SVG

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'



graph = Source(export_graphviz(dt, out_file=None, feature_names=x.columns, filled = True))

display(SVG(graph.pipe(format='svg')))
