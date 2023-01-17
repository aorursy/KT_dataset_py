

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_excel('/kaggle/input/covid19/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx')
print(df.shape)

df.head()

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
df.columns = [i for i in range(len(df.columns))]
corr = df.corr()
f = plt.figure(figsize=(19, 15))
plt.matshow(corr, fignum=f.number)
plt.xticks(range(df.shape[1]), df.columns, fontsize=5, rotation=45)
plt.yticks(range(df.shape[1]), df.columns, fontsize=5)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)

import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
print('Libraries Imported')

import numpy
for i in df.columns:
    if type(df[i].iloc[0]) == str:
        factor = pd.factorize(df[i])
        df[i] = factor[0]
        definitions = factor[1]


X = df[list(df.columns)[:-1]].values
y = df[df.columns[-1]].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(np.nan_to_num(X_train))
X_test = scaler.transform(np.nan_to_num(X_test))
        
model = RandomForestClassifier(n_jobs=64,n_estimators=200,criterion='entropy',oob_score=True)
model.fit(X_train, y_train)
print('Training is done')
y_pred = model.predict(X_test)
print('Testing is done')
acc =  metrics.accuracy_score(y_test, y_pred)
print('accuracy ' +str(acc))
#print('average auc ' +str(roc_auc["average"]))
prfs = precision_recall_fscore_support(y_test, y_pred, labels = [0,1])
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print('precision:',prfs[0] )
print('recall', prfs[1])
print('fscore', prfs[2])
print('support',prfs[3])
print('auc',roc_auc)
