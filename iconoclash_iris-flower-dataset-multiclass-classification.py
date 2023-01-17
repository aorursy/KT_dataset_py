# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
iris = pd.read_table('../input/IRIS.csv',sep=",")
type(iris)
iris.head()
listItem = []
for col in iris.columns :
    listItem.append([col,iris[col].dtype,
                     
                     iris[col].isna().sum(),
                     
                     round((iris[col].isna().sum()/len(iris[col])) * 100,2),
                     
                     iris[col].nunique(),
                     
                     list(iris[col].sample(5).drop_duplicates().values)]);

dfDesc = pd.DataFrame(columns=['dataFeatures', 'dataType', 'null', 'nullPct', 'unique', 'uniqueSample'],
                     data=listItem)
dfDesc
iris.groupby(["species"]).count()
iris.groupby(["species"]).mean()
iris.groupby(["species"]).mean().sort_values(by="petal_length",ascending =True)["petal_length"]
plt.figure(figsize=(20, 20))
sns.pairplot(iris,hue="species", markers="o",plot_kws={"s": 50,'alpha':0.5})
ciris = iris.copy()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ciris['species'] = le.fit_transform(ciris["species"])
plt.figure(figsize=(20, 20))
sns.pairplot(ciris,hue="species", markers="o",plot_kws={"s": 50,'alpha':0.5})
feature=ciris.drop('species',axis=1)
target=ciris['species']

ciris_corr = feature.join(target).corr()

mask = np.zeros((5,5))
mask[:4,:]=1

plt.figure(figsize=(10, 10))
with sns.axes_style("white"):
    sns.heatmap(ciris_corr, annot=True,square=True,mask=mask)
sns.boxplot(x="species",y="sepal_length",data=iris)
sns.boxplot(x="species",y="sepal_width",data=iris)
sns.boxplot(x="species",y="petal_length",data=iris)
sns.boxplot(x="species",y="petal_width",data=iris)
iris = pd.read_table('../input/IRIS.csv',sep=",")

feature = iris.drop(["species"],axis=1)
target = iris["species"]
feature["sepalarea"] = feature["sepal_length"]*feature["sepal_width"]
feature["petalare"] = feature["petal_length"]*feature["petal_width"]
# from sklearn.preprocessing import MinMaxScaler,StandardScaler,robust_scale
# scaler = StandardScaler();



# colscal=["sepal_length","petal_length","petal_width]

# scaler.fit(feature[colscal])
# scaled_features = pd.DataFrame(scaler.transform(feature[colscal]),columns=colscal)

# feature =feature.drop(colscal,axis=1)
# feature = scaled_features.join(feature)
from sklearn.model_selection import train_test_split,cross_val_score
X_train, X_test, y_train, y_test = train_test_split(feature,target,
                                                    test_size=0.20,random_state=101)
from xgboost import XGBClassifier

model = XGBClassifier(random_state=101)
model.fit(X_train, y_train)
from sklearn.metrics import classification_report, matthews_corrcoef, roc_auc_score, confusion_matrix, accuracy_score,recall_score

predict = model.predict(X_train)
predictProb = model.predict_proba(X_train)


print("confusion_matrix :\n",confusion_matrix(y_train, predict))
print("\nclassification_report :\n",classification_report(y_train, predict))
print('Recall Score',recall_score(y_train, predict,average=None))
print('Accuracy :',accuracy_score(y_train, predict))
print('Matthews Corr_coef :',matthews_corrcoef(y_train, predict))
from sklearn.metrics import classification_report, matthews_corrcoef, roc_auc_score, confusion_matrix, accuracy_score,recall_score

predict = model.predict(X_test)
predictProb = model.predict_proba(X_test)


print("confusion_matrix :\n",confusion_matrix(y_test, predict))
print("\nclassification_report :\n",classification_report(y_test, predict))
print('Recall Score',recall_score(y_test, predict,average=None))
print('Accuracy :',accuracy_score(y_test, predict))
print('Matthews Corr_coef :',matthews_corrcoef(y_test, predict))
coef1 = pd.Series(model.feature_importances_,feature.columns).sort_values(ascending=False)

pd.DataFrame(coef1,columns=["Features"]).transpose().plot(kind="bar",title="Feature Importances") #for the legends

coef1.plot(kind="bar",title="Feature Importances")
