import pandas as pd #Analysis 
import matplotlib.pyplot as plt #Visulization
import seaborn as sns #Visulization
import numpy as np #Analysis 
from scipy.stats import norm #Analysis 
from sklearn.preprocessing import StandardScaler #Analysis 
from scipy import stats #Analysis 
import warnings 
warnings.filterwarnings('ignore')
%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
df_train = pd.read_csv('../input/train_V2.csv')
df_test  = pd.read_csv('../input/test_V2.csv')
print("Train : ",df_train.shape)
print("Test : ",df_test.shape)
df_train.head()
#histogram
f, ax = plt.subplots(figsize=(18, 8))
sns.distplot(df_train['matchDuration'])
plt.figure(figsize = (18, 8))

sns.kdeplot(df_train.loc[df_train['matchType'] == 'squad-fpp', 'matchDuration'] , label = 'squad-fpp')

sns.kdeplot(df_train.loc[df_train['matchType'] == 'duo-fpp', 'matchDuration'] , label = 'duo-fpp')

sns.kdeplot(df_train.loc[df_train['matchType'] == 'squad', 'matchDuration'] , label = 'squad')

sns.kdeplot(df_train.loc[df_train['matchType'] == 'solo-fpp', 'matchDuration'] , label = 'solo-fpp')

sns.kdeplot(df_train.loc[df_train['matchType'] == 'duo', 'matchDuration'] , label = 'duo')

sns.kdeplot(df_train.loc[df_train['matchType'] == 'solo', 'matchDuration'] , label = 'solo')

# Labeling of plot
plt.xlabel('matchDuration'); plt.ylabel('Density'); plt.title('Distribution of matchDuration with Basic mode');
plt.figure(figsize = (18, 8))

sns.kdeplot(df_train.loc[df_train['matchType'] == 'crashfpp', 'matchDuration'] , label = 'crashfpp')

sns.kdeplot(df_train.loc[df_train['matchType'] == 'crashtpp', 'matchDuration'] , label = 'crashtpp')

sns.kdeplot(df_train.loc[df_train['matchType'] == 'flarefpp', 'matchDuration'] , label = 'flarefpp')

sns.kdeplot(df_train.loc[df_train['matchType'] == 'flaretpp', 'matchDuration'] , label = 'flaretpp')


# Labeling of plot
plt.xlabel('matchDuration'); plt.ylabel('Density'); plt.title('Distribution of matchDuration with Event mode');
f, ax = plt.subplots(figsize=(8, 6))
df_train['matchType'].value_counts().sort_values(ascending=False).plot.bar()
plt.show()
print("Unique value of matchType :",len(df_train['matchType'].unique()))
#histogram
f, ax = plt.subplots(figsize=(18, 8))
sns.distplot(df_train['rankPoints'])
print("rankPoints has -1 :", len(df_train[df_train['rankPoints'] == -1])/df_train.shape[0])
print("rankPoints has 0 :", len(df_train[df_train['rankPoints'] == 0])/df_train.shape[0])
print("rankPoints has others :", len(df_train[(df_train['rankPoints'] != 0) & (df_train['rankPoints'] != -1)])/df_train.shape[0])
f, ax = plt.subplots(figsize=(18, 8))
df_train[df_train['rankPoints'] == -1]['matchType'].value_counts().sort_values(ascending=False).plot.bar()
plt.title('Count of matchType with rankPoints == -1');
plt.show()
f, ax = plt.subplots(figsize=(18, 8))
df_train[(df_train['rankPoints'] != 0) & (df_train['rankPoints'] != -1)]['matchType'].value_counts().sort_values(ascending=False).plot.bar()
plt.title('Count of matchType with rankPoints != -1 & != 0');
plt.show()
f, ax = plt.subplots(figsize=(18, 8))
df_train[df_train['rankPoints'] == 0]['matchType'].value_counts().sort_values(ascending=False).plot.bar()
plt.title('Count of matchType with rankPoints == 0');
plt.show()