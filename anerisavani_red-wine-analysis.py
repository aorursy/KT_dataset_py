import pandas as pd
from pandas import DataFrame
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from matplotlib import rcParams
import plotly.graph_objects as go
import plotly.express as px
from plotly.colors import n_colors
import numpy as np
import seaborn as sns
import pandas_profiling
%matplotlib inline
from matplotlib import rc
import scipy.stats
from scipy.stats import skew
from scipy.stats.stats import pearsonr
#reading the file
red_wine = pd.read_csv("../input/wine-quality-selection/winequality-red.csv",delimiter=',')
red_wine
red_wine.columns
red_wine.info()
red_wine.describe()
sns.set(rc={'figure.figsize':(8,8)})
sns.countplot(red_wine['quality'])
sns.pairplot(red_wine)
sns.heatmap(red_wine.corr(), annot =True, fmt ='.2f', linewidths = 2)
sns.distplot(red_wine['alcohol'])
skew(red_wine['alcohol'])
np.mean(red_wine['alcohol'])
np.median(red_wine['alcohol'])
sns.boxplot(x='quality',y='alcohol', data = red_wine, showfliers = False)
joint_plt = sns.jointplot(x='alcohol', y ='pH', data = red_wine, kind = 'reg')
def get_corr(var1,var2,df):
    pearson_coefficient, p_value = pearsonr(df[var1], df[var2])
    print('Pearsonr correlation between {} and {} is {}'.format(var1,var2,pearson_coefficient))
    print("P value of this correlation is {}".format(p_value))
get_corr('alcohol','pH',red_wine)
joint_plt = sns.jointplot(x='alcohol', y ='density', data = red_wine, kind = 'reg')
get_corr('alcohol','density',red_wine)
a = sns.FacetGrid(red_wine, col = 'quality')
a = a.map(sns.regplot,'density', 'alcohol')
sns.boxplot(x='quality',y='sulphates', data = red_wine, showfliers = False)
sns.boxplot(x='quality',y='total sulfur dioxide', data = red_wine, showfliers = False)
sns.boxplot(x='quality',y='free sulfur dioxide', data = red_wine, showfliers = False)
sns.boxplot(x='quality',y='fixed acidity', data = red_wine, showfliers = False)
sns.boxplot(x='quality',y='citric acid', data = red_wine, showfliers = False)
sns.boxplot(x='quality',y='volatile acidity', data = red_wine, showfliers = False)
red_wine['total acidity'] = (red_wine['fixed acidity']+red_wine['citric acid']+red_wine['volatile acidity'])
sns.boxplot(x='quality',y='total acidity', data = red_wine, showfliers = False)
sns.regplot(x='pH', y= 'total acidity', data = red_wine)
b  = sns.FacetGrid(red_wine, col = 'quality')
b = b.map(sns.regplot,'total acidity', 'pH')
get_corr('total acidity','pH',red_wine)