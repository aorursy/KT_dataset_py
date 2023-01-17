# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, mean_absolute_error

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/dollar-prices-and-infos/database_15min.csv', header=None, names=['time','opPrice_min_candle', 'maxPrice_min_candle', 'minPrice_min_candle', 'cloPrice_min_candle','volume', 'financial_information','negotiation', 'ma_last13', 'ma_last72', 'avg_last15_high', 
                                                                                                'avg_last15_low','diffMACD','deaMACD','MACDlh','difflh','dealh','opPrice_future15'])
df.head()
sns.heatmap(df.corr())
ax = sns.distplot(df['opPrice_future15'])
df['target'] = df['opPrice_future15'] > df['cloPrice_min_candle']
df.head()
feat = pd.DataFrame()
#feat['neg_diffMACD'] = df['negotiation']*df['diffMACD']
feat['volume'] = df['volume']*10
feat['avg_canPrice'] = df['financial_information']/feat['volume']
feat['delta_price'] = df['cloPrice_min_candle']-df['opPrice_min_candle']
feat['maxmin'] = df['maxPrice_min_candle']-df['minPrice_min_candle']
#feat['delta_maxmin'] = feat['maxmin'].diff()
feat['movAvg13_closePrice'] = df['ma_last13'].diff()
feat['movAvg72_closePrice'] = df['ma_last72'].diff()
#feat['delta_avgtops'] = df['avg_last15_high'].diff()
#feat['delta_avgbottoms'] = df['avg_last15_low'].diff()
feat['delta_tops'] = df['maxPrice_min_candle'].diff()
feat['delta_bottoms'] = df['minPrice_min_candle'].diff()
feat[['diffMACD','deaMACD','MACDlh','difflh','dealh']] = df[['diffMACD','deaMACD','MACDlh','difflh','dealh']]
#feat[['difflh','dealh']] = df[['difflh','dealh']]
feat.fillna(0, inplace=True)
feat.head()
sns.heatmap(feat.corr())
feat.drop(columns=['difflh','dealh'], inplace=True)
sns.heatmap(feat.corr())
feat_correl = feat.copy()
feat_correl['target'] = df['opPrice_future15']
sns.heatmap(feat_correl.corr())
feat_org = df[['opPrice_min_candle','maxPrice_min_candle', 'minPrice_min_candle', 'cloPrice_min_candle','ma_last13', 'ma_last72', 'avg_last15_high', 'avg_last15_low']]
feat_correl = feat_org.copy()
feat_correl['target'] = df['opPrice_future15']


sns.heatmap(feat_org.corr())              
from sklearn.preprocessing import StandardScaler

y_linear = df['opPrice_future15']
y_boolean = df['target']

scaler = StandardScaler()
scaler.fit(feat)
X = scaler.transform(feat)
scaler.fit(feat_org)
Xorg = scaler.transform(feat_org)

Xlin_train, Xlin_test, ylin_train, ylin_test = train_test_split(X, y_linear, random_state = 0)
Xorglin_train, Xorglin_test, yorglin_train, yorglin_test = train_test_split(Xorg, y_linear, random_state = 0)

X_train, X_test, y_train, y_test = train_test_split(X, y_boolean, random_state = 0)
Xorg_train, Xorg_test, yorg_train, yorg_test = train_test_split(Xorg, y_boolean, random_state = 0)
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV

hummerGBC = GradientBoostingClassifier(random_state=0,n_estimators=1000, n_iter_no_change=3, learning_rate=1)
hummerGBC.fit(X_train, y_train)
print('Hummer features GBC score:', hummerGBC.score(X_test, y_test))

orgGBC = GradientBoostingClassifier(random_state=0,n_estimators=1000, n_iter_no_change=3, learning_rate=1)
orgGBC.fit(Xorg_train, yorg_train)
print('Original features GBC score:', orgGBC.score(Xorg_test, yorg_test))

hummerSVC = LinearSVC(random_state=0, dual=False)
hummerSVC.fit(X_train, y_train)
print('Hummer features SVC score:', hummerSVC.score(X_test, y_test))

orgSVC = LinearSVC(random_state=0, dual=False)
orgSVC.fit(Xorg_train, yorg_train)
print('Original features SVC score:', orgSVC.score(Xorg_test, yorg_test))

hummerAda = AdaBoostClassifier(n_estimators=100, random_state=0)
hummerAda.fit(X_train,y_train)
print('Hummer features ADA score:',hummerAda.score(X_test, y_test))

orgAda = AdaBoostClassifier(n_estimators=100, random_state=0)
orgAda.fit(Xorg_train,yorg_train)
print('Original features ADA score:',orgAda.score(Xorg_test, yorg_test))

hummerLog = LogisticRegressionCV(cv=5, random_state=0)
hummerLog.fit(X_train, y_train)
print('Hummer features Log score:', hummerLog.score(X_test, y_test))

orgLog = LogisticRegressionCV(cv=5, random_state=0)
orgLog.fit(Xorg_train, yorg_train)
print('Original features Log score:', orgLog.score(Xorg_test, yorg_test))
import eli5
from eli5.sklearn import PermutationImportance

Xorg_test_df = pd.DataFrame(Xorg_test, columns=feat_org.columns, index = None)
perm = PermutationImportance(orgGBC, random_state=0).fit(Xorg_test, yorg_test)
eli5.show_weights(perm, feature_names = Xorg_test_df.columns.tolist())
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVR

hummerGBR = GradientBoostingRegressor(random_state=0)
hummerGBR.fit(Xlin_train, ylin_train)
print('Hummer feature GBR score:',hummerGBR.score(Xlin_test, ylin_test))

orgGBR = GradientBoostingRegressor(random_state=0)
orgGBR.fit(Xorglin_train, yorglin_train)
print('Original feature GBR score:',orgGBR.score(Xorglin_test, yorglin_test))
Xorg_test_df = pd.DataFrame(Xorg_test, columns=feat_org.columns, index = None)
perm = PermutationImportance(orgGBR, random_state=0).fit(Xorg_test, yorg_test)
eli5.show_weights(perm, feature_names = Xorg_test_df.columns.tolist())