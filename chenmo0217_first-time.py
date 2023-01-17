#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
df_train = pd.read_csv('../input/pubg-finish-placement-prediction/train.csv')
df_test = pd.read_csv('../input/pubg-finish-placement-prediction/test.csv')
df_train.columns
#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
X_train = df_train.drop(["Id", "groupId","matchId","winPlacePerc"], axis=1)
X_train.head()
y_train = np.log1p(df_train["winPlacePerc"].values)
y_train
X_test = df_test.drop(["Id", "groupId","matchId"], axis=1)
X_test.head()


