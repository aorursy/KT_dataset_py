!pip install dexplot
!pip install chart_studio
!pip install pandas-profiling
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
import dexplot as dxp
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

import seaborn as sns
from pandas_profiling import ProfileReport
import pandas_profiling
import plotly.express as px
df = pd.read_csv('../input/hranalysis/train.csv')
df.describe()
pandas_profiling.ProfileReport(df)
import numpy as np # linear algebra
from scipy import stats # statistic library
import pandas as pd # To table manipulations
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('../input/hranalysis/train.csv')
df.head()
#To check
df.groupby("department").avg_training_score.mean().sort_values(ascending=False)[:5].plot.bar()
df['previous_year_rating'].value_counts().sort_index().plot.bar()
df['length_of_service'].plot.hist()
sns.pairplot(df)
g = sns.FacetGrid(df, col='department')
g = g.map(sns.kdeplot, 'length_of_service')
df1 = df[(df['previous_year_rating']>=1) & (df['avg_training_score']>80)]
sns.boxplot('avg_training_score', 'previous_year_rating', data=df1)

sns.distplot(df['avg_training_score'], bins=10, kde=True)
#A plot to get a idea on training score range and skewness.
corr = df.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
df.isnull().sum()

#to predict rating
X = df[['avg_training_score', 'no_of_trainings','awards_won?','KPIs_met >80%']]
y = df['is_promoted']
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X, y)
predictedpromotion = regr.predict([[20, 1, 0, 1]])
print(predictedpromotion)