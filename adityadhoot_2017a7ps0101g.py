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
df = pd.read_csv("/kaggle/input/minor-project-2020/train.csv")
df.drop('id',axis=1,inplace=True)
df.head()
olni1sdf = df[df['target'] == 1]
#olni1sdf.info()
olni0sdf = df[df['target'] == 0]
#olni0sdf.info()
df_new_1 = olni1sdf
for i in range (12):
    df_new_1 = pd.concat([df_new_1, olni1sdf])

#df_new_1.info()
np.random.seed(1)

remove_n = 700000
drop_indices = np.random.choice(olni0sdf.index, remove_n, replace=False)
df_new_0 = olni0sdf.drop(drop_indices)
#df_new_0.info()
df_new = pd.DataFrame()
df_new = pd.concat([df_new_0, df_new_1])
#df_new.info()
correlationdf = df_new.corr()
zzkk = abs(correlationdf).tail(1)
dfObj = zzkk.sort_values(by ='target', axis=1)
#print(dfObj)
thelistofcorrcols = list(dfObj.columns)
#print(thelistofcorrcols)
top10corrcols = thelistofcorrcols[78:-1]
top20corrcols = thelistofcorrcols[68:-1]
top40corrcols = thelistofcorrcols[48:-1]
top5corrcols = thelistofcorrcols[83:-1]
allcorrs = thelistofcorrcols[:-1]
print(allcorrs)
X=df_new[allcorrs]
y=df_new[['target']]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(len(X_train), len(X_test))
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scaled_X_train = scalar.fit_transform(X_train)
scaled_X_test = scalar.transform(X_test)
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
dt = DecisionTreeClassifier()
dt.fit(scaled_X_train, y_train)
y_pred = dt.predict(scaled_X_test)
print("Accuracy is : {}".format(dt.score(scaled_X_test, y_test)))
column_names = X_train.columns
feature_importances = pd.DataFrame(dt.feature_importances_,
                                   index = column_names,
                                    columns=['importance'])
feature_importances.sort_values(by='importance', ascending=False).head(20)
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns 
print("Confusion Matrix: ")

print(confusion_matrix(y_test, y_pred))
plot_confusion_matrix(dt, scaled_X_test, y_test, cmap = plt.cm.Blues)
print("Classification Report: ")
print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_curve, auc
plt.style.use('seaborn-pastel')

FPR, TPR, _ = roc_curve(y_test, y_pred)
ROC_AUC = auc(FPR, TPR)
print (ROC_AUC)
testerdataorig = pd.read_csv("/kaggle/input/minor-project-2020/test.csv")
testerdataorig.head()
X_findf=testerdataorig[allcorrs]
scaled_X_findf = scalar.transform(X_findf)
y_findf=dt.predict(scaled_X_findf)

print(y_findf)
submission = pd.DataFrame({'id':testerdataorig['id'],'target':y_findf})
print(submission)
filename = 'MinorLab Predictions 3dt_all.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)