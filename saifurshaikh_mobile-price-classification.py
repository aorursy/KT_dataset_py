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
df_train = pd.read_csv('../input/mobile-price-classification/train.csv')
df_test = pd.read_csv('../input/mobile-price-classification/test.csv')
pd.set_option('display.max_columns', None)
print(df_train.shape)
df_train.head()
df_train.describe()
print(df_train.columns.values)
df_train.rename(columns={'battery_power':'Battery_mAh', 'blue':'Bluetooth', 'clock_speed':'Processor_Speed', 
                        'dual_sim':'Dual_Sim', 'fc':'Front_Camera_MP', 'four_g':'4G', 
                         'int_memory':'Internal_Memory', 'mobile_wt':'Weight','n_cores':'Num_Cores', 
                         'px_height':'Pixel_Hgt', 'm_dep':'Thickness_cm', 'pc':'PrimaryCam_Megapixel',
                        'px_width':'Pixel_Width', 'ram':'RAM', 'sc_h':'Screen_Hgt', 'sc_w':'Screen_Width', 
                        'talk_time':'Talktime', 'three_g':'3G', 'touch_screen':'TouchScreen', 
                        'wifi':'Wi-Fi', 'price_range':'Price_Range'}, inplace=True)
df_train.head()
df_train.rename(columns={'blue':'Bluetooth','pc':'PrimaryCam_Megapixel','m_dep':'Thickness_cm'}, inplace=True)
df_train.head()
df_train.columns
df_train.info()
import seaborn as sns
import matplotlib.pyplot as plt
corr = df_train.corr()
plt.figure(figsize=(20,15))
sns.heatmap(corr, annot=True,square=True, linewidths=.5, cmap='YlGnBu')
plt.show()
# RAM is highlt correlated with the price. 
# Whereas, Battery, Pixel Heigth and Pixel Width have very small correlation with price. 
# We also see that, Front Camera and Primary Camera, Pixel Height and Pixel Width, and 3G and 4G have multicollinearity. 
# For now, we will not drop any column based on correlation or multicollineartiy
# Lets preprocess the data for now and we will drop columns later while feature selection
sns.barplot(x=df_train.Price_Range, y=df_train.RAM, data=df_train)
df_train.groupby(['Price_Range'])['RAM'].mean()
sns.barplot(x=df_train.Price_Range, y=df_train.Screen_Hgt, data=df_train)
sns.barplot(x=df_train.Price_Range, y=df_train.Screen_Width, data=df_train)
sns.countplot(x=df_train.Price_Range, data=df_train, hue=df_train['4G'])
sns.countplot(x=df_train.Price_Range, data=df_train, hue=df_train['3G'])
X = df_train.drop('Price_Range', axis=1)
y = df_train.Price_Range
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import chi2, SelectKBest, SelectFromModel
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled
kbest = SelectKBest(score_func=chi2, k=10)
scores = kbest.fit(X_scaled,y)
feat_imp = pd.Series(scores.scores_, index=X.columns)
feat_imp.nlargest(10).index.values
X_new = X[['RAM', 'Battery_mAh', 'Pixel_Width', 'Pixel_Hgt', 'TouchScreen',
       'Weight', 'Num_Cores', 'Internal_Memory', '4G', 'Screen_Hgt']]
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.2, stratify=y)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=2)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_pred
print(metrics.accuracy_score(y_pred, y_test))
print('Confusion Matrix: ')
print('\t')
print(metrics.confusion_matrix(y_pred, y_test))
print('\t')
print('\t\t\t Classification Report: ')
print(metrics.classification_report(y_pred, y_test))
print('\t')
print('Training Score: ', rf.score(X_train, y_train))
print('Validation Score: ', rf.score(X_train, y_train))
print('\t')
print('Accuracy Score: ', metrics.accuracy_score(y_pred, y_test))
from sklearn.svm import SVC
svc = SVC(kernel='linear', C=0.001)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
y_pred
print('Confusion Matrix: ')
print('\t')
print(metrics.confusion_matrix(y_pred, y_test))
print('\t')
print('\t\t\t Classification Report: ')
print(metrics.classification_report(y_pred, y_test))
print('\t')
print('Training Score: ', svc.score(X_train, y_train))
print('Validation Score: ', svc.score(X_train, y_train))
print('\t')
print('Accuracy Score: ', metrics.accuracy_score(y_pred, y_test))
