import numpy as np

import pandas as pd
dataset = {'isim' : ['Mert','Nilay','Dogancan','Omer','Merve','Onur'],

           'soyad': ['Cobanov','Mertel','Mavideniz','Cengiz','Noyan','Sahil'],

           'yas'  : [24,22,24,23,'bilinmiyor','23'],

           'sehir': ['Bursa','Ankara','Istanbul',np.nan,'Izmir','Istanbul'],

           'ulke' : ['Turkiye','Turkiye','Turkiye','Turkiye','Turkiye','Turkiye'],

           'GANO' : [np.nan,np.nan,np.nan,np.nan,3.90,np.nan]}
df = pd.DataFrame(dataset)
df_2 = df.drop(labels=['GANO', 'ulke'], axis=1)
df_2
df_2['yas'].replace('bilinmiyor', np.nan, inplace=True)
df_2 = df_2.astype({"yas": float})
df_2.yas
df_2.yas.fillna(value=df_2.yas.mean(), inplace=True)
df_2.yas
from sklearn.impute import SimpleImputer
imp_freq=SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df_2.yas=imp_freq.fit_transform(df_2[['yas']])
df_2
s = pd.Series([0, 1, np.nan, 3])

print(s)
from sklearn.impute import KNNImputer
X = [[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]]

pd.DataFrame(X)
imputer = KNNImputer(n_neighbors=2, weights='uniform')

X = imputer.fit_transform(X)
pd.DataFrame(X)
df_2
df_2['sehir'] = df['sehir'].replace(np.nan, 'diger')
df_2
from sklearn.preprocessing import StandardScaler
df_ss = df_2.copy()
df_ss['yas_scaled'] = StandardScaler().fit_transform(df_ss[['yas']])
df_ss
print(df_ss['yas_scaled'].mean(axis=0))

print(df_ss['yas_scaled'].std(axis=0))
from sklearn.preprocessing import MinMaxScaler
df_mm = df_2.copy()
df_mm['yas_scaled'] = MinMaxScaler().fit_transform(df_mm[['yas']])
df_mm
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_le = df_2.copy()
le.fit(df_le['sehir'])
list(le.classes_)
df_le['sehir'] = le.transform(df_le['sehir'])
df_le
le.inverse_transform([2,1,0,1])
pd.get_dummies(df_2['sehir'])
X = np.array([[-3., 5., 15],

              [ 0., 6., 14],

              [ 6., 3., 11]])
from sklearn import preprocessing
preprocessing.KBinsDiscretizer(n_bins=[3,2,2], encode='ordinal').fit_transform(X)
binarizer=preprocessing.Binarizer(threshold=5.1)
binarizer.transform(X)
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split
import os

import io
pwd
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/mushroom-csv-file/mushrooms.csv")
data.head()
url = 'https://raw.githubusercontent.com/cobanov/ML_Days/master/mushrooms.csv'

data2 = pd.read_csv(url, error_bad_lines=False)
data2.head()
X = data.drop(['class'], axis=1)

y = data['class']
X_encoded = pd.get_dummies(X, prefix_sep='_')
y_encoded = LabelEncoder().fit_transform(y)
X_scaled = StandardScaler().fit_transform(X_encoded)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=101)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix

import time
start = time.process_time()

model = RandomForestClassifier(n_estimators=700).fit(X_train,y_train)

print(time.process_time()-start)
preds=model.predict(X_test)
print(confusion_matrix(y_test,preds))
import matplotlib.pyplot as plt
feature_imp = pd.Series(model.feature_importances_, index=X_encoded.columns)
feature_imp
feature_imp.nlargest(10).plot(kind='barh')
best_feat=feature_imp.nlargest(4).index.to_list()
X_reduced = X_encoded[best_feat]
X_reduced
Xr_scaled=StandardScaler().fit_transform(X_reduced)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr_scaled, y, test_size=0.30, random_state=101)
start=time.process_time()

rmodel=RandomForestClassifier(n_estimators=700).fit(Xr_train, yr_train)

print(time.process_time()-start)

#2.875 vs. 1.640 sec diff
rpred = rmodel.predict(Xr_test)

print(confusion_matrix(yr_test, rpred))

print(classification_report(yr_test,rpred))

#gain in time vs. loss in accuracy
import seaborn as sns
X = data.drop(['class'], axis=1)

y = data['class']

X_encoded = pd.get_dummies(X, prefix_sep='_')

y_encoded = LabelEncoder().fit_transform(y)

X_encoded['Class'] = y_encoded
sns.heatmap(X_encoded.iloc[:,-7:].corr(), annot=True)
X_reduced_col_names = X_encoded.corr().abs()['Class'].nlargest(10).index
plt.figure(figsize=(9,9), dpi=100)

sns.heatmap(X_encoded[X_reduced_col_names].corr().abs(), annot=True)