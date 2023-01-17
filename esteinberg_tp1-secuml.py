import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

%matplotlib inline
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
db = pd.read_csv('/kaggle/input/unsw-nb15/UNSW_NB15_training-set.csv')

db.head().T

db = db.drop(columns='id')
db.label = db.label.astype('object')

db.head()
#pr = ProfileReport(db)
def descr(bdd):
	for i in [np.number, ['object', 'bool', 'category']]:
		with pd.option_context("display.precision", 2, 'display.float_format', lambda x: '%.3f' % x):
			print(bdd.describe(include=i).T)
			print('\n')
	for i in bdd:
		if bdd[i].isna().sum()>0:
			print("pour {}\t\t il y a {} \t valeurs manquantes".format(i, str(len(bdd[i]) - bdd[i].count())))

descr(db)
print (len(db.columns),'\n', db.columns,'\n', db.dtypes,'\n', len(db))
db.select_dtypes('object').columns
db.select_dtypes('object').describe()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    for i in db.select_dtypes('object').columns:
        print ('for', i)
        print(db[i].value_counts())
        print('#'*30)

db.select_dtypes(['int', 'float']).describe()
len(db[((db.label == 0) & (db.attack_cat != 'Normal'))]), len(db[((db.label != 0) & (db.attack_cat == 'Normal'))])
db.proto.hist(by=db.label)
db.service.hist(by=db.label)
db.proto[db.label==0].value_counts()
db.proto[db.label==1].value_counts()
db.service[db.label==0].value_counts()
db.service[db.label==1].value_counts()
db.service.value_counts().plot.bar()
fig = plt.figure(figsize = (35,10))
ax = fig.gca()
db.proto.value_counts().plot.bar()
# essayons avec les protocoles les plus utilisés

print (float(db.proto.isin(db.proto.value_counts()[db.proto.value_counts()>100].index ).sum()) / len(db))
db[db.proto.isin(db.proto.value_counts()[db.proto.value_counts()>100].index )].proto.value_counts().plot.bar()
db.attack_cat.value_counts().plot.bar()
plt.matshow(db[db.label==0].drop(columns='label').corr())
plt.matshow(db[db.label==1].drop(columns='label').corr())
r=db[db.label==0].drop(columns='label').corr()
s = db.corr().abs().unstack()
so = s.sort_values(kind="quicksort")
so[-59:-39]
so[-59:-39:2]
X = db.drop(columns=['label', 'attack_cat'])
y = db.label
ytype = db.attack_cat
X = pd.get_dummies(X)
y = y.to_numpy().astype('int')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

X_train, X_test, y_train, y_test = train_test_split(X,y)

model = GradientBoostingClassifier(n_estimators=200, )
model.fit(X_train, y_train)
from sklearn.metrics import plot_precision_recall_curve
plot_precision_recall_curve(model, X_test, y_test)
R=pd.DataFrame (zip(X.columns, model.feature_importances_)).sort_values(1, ascending=False)
R[R[1]>0].plot.bar(x=0, figsize=(35, 5), fontsize=20)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import confusion_matrix
pca = PCA(0.999999)
stds = StandardScaler()
xt = stds.fit_transform(X_train)
pca.fit(xt)
lr = LinearRegression()
lr.fit(pca.transform(stds.transform(X_train)), y_train)
# Observons les contributions absolues :
sorted(zip(X_train.columns , lr.coef_), key=lambda x:x[1]**2, reverse=True)