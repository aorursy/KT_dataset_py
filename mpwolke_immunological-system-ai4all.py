#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/png;base64,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',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



import matplotlib.pyplot as plt

import statsmodels.formula.api as smf



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

from sklearn.linear_model import Ridge

from yellowbrick.regressor import PredictionError, ResidualsPlot
df = pd.read_csv('../input/ai4all-project/results/deconvolution/CIBERSORTx_Results_Krasnow_facs_droplet.csv', encoding='ISO-8859-2')

df.head()
(sns.FacetGrid(df, hue = 'B cell',

             height = 6,

             xlim = (0,500))

    .map(sns.kdeplot, 'Unnamed: 0', shade = True)

    .add_legend());
# categorical features

categorical_feat = [feature for feature in df.columns if df[feature].dtypes=='O']

print('Total categorical features: ', len(categorical_feat))

print('\n',categorical_feat)
from sklearn.preprocessing import LabelEncoder

categorical_col = ('czb_id', 'viral_load')

        

        

for col in categorical_col:

    label = LabelEncoder() 

    label.fit(list(df[col].values)) 

    df[col] = label.transform(list(df[col].values))



print('Shape all_data: {}'.format(df.shape))
df = df.rename(columns={'B cell':'Bcell', 'T cell': 'Tcell'})
import matplotlib.gridspec as gridspec



columns=df.columns[:8]

plt.figure(figsize=(12,28*4))

gs = gridspec.GridSpec(28, 1)

for i, cn in enumerate(df[columns]):

    ax = plt.subplot(gs[i])

    sns.distplot(df[cn][df.Bcell == 1], bins=50)

    sns.distplot(df[cn][df.Bcell == 0], bins=50)

    ax.set_xlabel('B Cell')

    plt.legend(df["Bcell"])

    ax.set_title('B cell : ' + str(cn))

plt.show()
import seaborn as sns

sns.set_style('white')



f, axes = plt.subplots(ncols=4, figsize=(16,4))



# Lot Area: In Square Feet

sns.distplot(df['Bcell'], kde=False, color="#DF3A01", ax=axes[0]).set_title("Distribution of B cell")

#axes[0].set_ylabel("Square Ft")

axes[0].set_xlabel("B cell")



# MoSold: Year of the Month sold

sns.distplot(df['Tcell'], kde=False, color="#045FB4", ax=axes[1]).set_title("T cell Distribution")

#axes[1].set_ylabel("Amount of Houses Sold")

axes[1].set_xlabel("T cell")



# House Value

sns.distplot(df['Monocytes/macrophages'], kde=False, color="#088A4B", ax=axes[2]).set_title("Monocytes/macrophages Distribution")

#axes[2].set_ylabel("Number of Houses ")

axes[2].set_xlabel("Monocytes/macrophages")



# YrSold: Year the house was sold.

sns.distplot(df['Dendritic'], kde=False, color="#FE2E64", ax=axes[3]).set_title("Dendritic")

#axes[3].set_ylabel("Number of Houses ")

axes[3].set_xlabel("Dendritic")



plt.show()
# Maybe we can try this with plotly.

plt.figure(figsize=(10,6))

sns.distplot(df['Bcell'], color='r')

plt.title('Distribution of B cell', fontsize=18)



plt.show()
sns.set(style="whitegrid")

plt.figure(figsize=(12,8))

sns.countplot(y="Neutrophil", hue="Bcell", data=df)

plt.show()
corr = df.corr()



g = sns.heatmap(corr,annot=True,cmap='coolwarm',linewidths=0.2,annot_kws={'size':20})

g.set_xticklabels(g.get_xticklabels(), rotation = 90, fontsize = 8)

fig=plt.gcf()

fig.set_size_inches(14,10)

plt.title(" Immunological System Correlation", fontsize=18)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.show()
#Now let's see the influence of Research on Chance of Admit

sns.swarmplot(x = df['Bcell'],y = df['Monocytes/macrophages'])
import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('seaborn-white')



fig, ax = plt.subplots(figsize=(14,8))

palette = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71", "#FF8000", "#AEB404", "#FE2EF7", "#64FE2E"]



sns.swarmplot(x="Monocytes/macrophages", y="Bcell", data=df, ax=ax, palette=palette, linewidth=1)

plt.title('Correlation between Monocytes/macrophages and B Cell', fontsize=18)

plt.ylabel('B cell', fontsize=14)

plt.show()
#with sns.plotting_context("notebook",font_scale=1):

 #   g = sns.factorplot(x="Tcell", y="Bcell", hue="Monocytes/macrophages",

  #                 col="Basal", data=df, kind="bar", size=5, aspect=.75, sharex=False, col_wrap=3,

   #                   palette="YlOrRd");

    

#plt.show()
from scipy.stats import norm



# norm = a normal continous variable.



log_style = np.log(df['Bcell'])  # log of salesprice



fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(14,10))

plt.suptitle('Probability Plots', fontsize=18)

ax1 = sns.distplot(df['Bcell'], color="#FA5858", ax=ax1, fit=norm)

ax1.set_title("Distribution of B cell with Positive Skewness", fontsize=14)

#ax2 = sns.distplot(log_style, color="#58FA82",ax=ax2, fit=norm)

#ax2.set_title("Normal Distribution with Log Transformations", fontsize=14)

#ax3 = stats.probplot(df['Bcell'], plot=ax3)

#ax4 = stats.probplot(log_style, plot=ax4)



plt.show()
print('Skewness for Normal D.: %f'% df['Bcell'].skew())

print('Skewness for Log D.: %f'% log_style.skew())

print('Kurtosis for Normal D.: %f' % df['Bcell'].kurt())

print('Kurtosis for Log D.: %f' % log_style.kurt())