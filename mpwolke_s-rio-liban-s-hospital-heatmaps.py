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
df = pd.read_excel('/kaggle/input/covid19/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx')

df.head()
df.corr()

plt.figure(figsize=(10,4))

sns.heatmap(df.corr(),annot=True,cmap='YlOrRd_r')

plt.show()
corr = df.corr(method='pearson')

sns.heatmap(corr)
corr=df[df.columns.sort_values()].corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



fig = go.Figure(data=go.Heatmap(z=corr.mask(mask),

                                x=corr.columns.values,

                                y=corr.columns.values,

                                xgap=1, ygap=1,

                                colorscale="Rainbow",

                                colorbar_thickness=20,

                                colorbar_ticklen=3,

                                zmid=0),

                layout = go.Layout(title_text='Correlation Matrix', template='plotly_dark',

                height=900,

                xaxis_showgrid=False,

                yaxis_showgrid=False,

                yaxis_autorange='reversed'))

fig.show()
def plot_correlation_map( df ):

    corr = df.corr()

    _ , ax = plt.subplots( figsize =( 8 , 6 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    _ = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

    )



plot_correlation_map(df) 
#STRONG POSITIVELY CORRELATED

corr_mat = df.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corr_mat[corr_mat > 0.7], vmax=.8, annot = True, square=True);
#STRONG NEGATIVELY CORRELATED

corr_mat = df.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corr_mat[corr_mat < -0.3], vmax=.8, annot = True, square=True);

# sns.heatmap(corr_mat, mask = corr_mat < -0.4, vmax=.8, annot = True, square=True);
dfcorr=df.corr()

dfcorr

plt.figure(figsize=(10,4))

sns.heatmap(df.corr(),annot=False,cmap='summer')

plt.show()
#scatterplot

sns.set(palette = 'deep')

cols = ['HEMATOCRITE_MAX', 'LEUKOCYTES_MAX', 'HEMOGLOBIN_MAX', 'NEUTROPHILES_MAX', 'TGO_MAX', 'TGP_MAX', 'RESPIRATORY_RATE_DIFF_REL', 'TEMPERATURE_DIFF_REL']

sns.pairplot(df[cols], height = 2.5)

plt.show();
#bivariate analysis saleprice/grlivarea

sns.jointplot(x = 'NEUTROPHILES_MAX', y = 'HEMATOCRITE_MAX', data = df, kind = 'reg');
from scipy.stats import norm

#histogram

sns.distplot(df['HEMATOCRITE_MAX'], fit = norm);
#skewness and kurtosis

print("Skewness: %f" % df['HEMATOCRITE_MAX'].skew())

print("Kurtosis: %f" % df['HEMATOCRITE_MAX'].kurt())
from scipy import stats

#Normal probability plot

fig = plt.figure()

res = stats.probplot(df['HEMATOCRITE_MAX'], plot=plt)
#histogram and normal probability plot

sns.distplot(df['LEUKOCYTES_MAX'], fit=norm);

fig = plt.figure()

res = stats.probplot(df['LEUKOCYTES_MAX'], plot=plt)
#scatter plot grlivarea/saleprice

var = 'RESPIRATORY_RATE_DIFF_REL'

sns.lmplot(x=var, y='HEMATOCRITE_MAX', markers = 'x', data = df)
from sklearn.preprocessing import StandardScaler

#standardizing data

saleprice_scaled = StandardScaler().fit_transform(df['HEMATOCRITE_MAX'][:,np.newaxis]);
#applying log transformation

df['HEMATOCRITE_MAX_Log'] = np.log(df['HEMATOCRITE_MAX'])
sns.distplot(df['HEMATOCRITE_MAX_Log'], fit = norm);

fig = plt.figure()

res = stats.probplot(df['HEMATOCRITE_MAX_Log'], plot=plt)
#standardizing data

totalBsmtSF_scaled = StandardScaler().fit_transform(df['LEUKOCYTES_MAX'][:,np.newaxis]);
#transform data

df.loc[df['LEUKOCYTES_DIFF']==1,'LEUKOCYTES_MAX'] = np.log(df['LEUKOCYTES_MAX'])
sns.distplot(df['LEUKOCYTES_MAX'], fit = norm);

fig = plt.figure()

res = stats.probplot(df['LEUKOCYTES_MAX'], plot=plt)