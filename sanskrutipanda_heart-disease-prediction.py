import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("../input/heart-attack-prediction/data.csv",na_values=['?'], low_memory = False)
df
df.shape
df.columns
df['age'].unique()
df['sex'].unique()
df['fbs'].unique()
df['chol'].unique()
df['fbs'].unique()
df['thalach'].unique()
df['cp'].unique()
df['restecg'].unique()
df['slope'].unique()
df['ca'].unique()
df['thal'].unique()
df['exang'].unique()
df.head()
df.tail()
df.info()
df.shape
df.drop(columns=['ca'],inplace=True)
df.shape
df.isnull().sum()
df.skew()
df['thalach'].fillna(df['thalach'].mean(),inplace=True)
df.fillna(df.median(),inplace=True)
df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.shape
cor=df.corr()
plt.figure(figsize=(20,12))
sns.heatmap(cor,annot=True,cmap='coolwarm')
plt.show()
df.columns
df=df.rename(columns={"num       ":"num"})
df.columns
plt.figure(figsize=(12,5))
plt.title("AGE VRS NUM")
sns.distplot(df.age[df.num==0],color="darkblue")
sns.distplot(df.age[df.num==1],color="cyan")
plt.legend(['0','1'])
plt.show()
df.hist()
plt.figure(figsize=(20,12))
plt.show()
plt.figure(figsize=(12,5))
plt.title("TRESTBPS VRS NUM")
sns.distplot(df.trestbps[df.num==0],color="darkblue")
sns.distplot(df.trestbps[df.num==1],color="cyan")
plt.legend(['0','1'])
plt.show()
plt.figure(figsize=(12,5))
plt.title("CHO VRS NUM")
sns.distplot(df.chol[df.num==0],color="darkblue")
sns.distplot(df.chol[df.num==1],color="cyan")
plt.legend(['0','1'])
plt.show()
plt.figure(figsize=(12,5))
plt.title("THALACH VRS NUM")
sns.distplot(df.thalach[df.num==0],color="darkblue")
sns.distplot(df.thalach[df.num==1],color="cyan")
plt.legend(['0','1'])
plt.show()
plt.figure(figsize=(6,3))
plt.title("SEX VRS NUM")
sns.countplot(df.sex)
plt.show()
sns.countplot(df.sex[df.num==1])
plt.show()
plt.figure(figsize=(6,3))
plt.title("CP VRS NUM")
sns.countplot(df.cp)
plt.show()
sns.countplot(df.cp[df.num==1])
plt.show()
plt.figure(figsize=(6,3))
plt.title("FBS VRS NUM")
sns.countplot(df.fbs)
plt.show()
sns.countplot(df.fbs[df.num==1])
plt.show()
plt.figure(figsize=(6,3))
plt.title("RESTECG VRS NUM")
sns.countplot(df.restecg)
plt.show()
sns.countplot(df.restecg[df.num==1])
plt.show()
plt.figure(figsize=(6,3))
plt.title("EXANG VRS NUM")
sns.countplot(df.exang)
plt.show()
sns.countplot(df.exang[df.num==1])
plt.show()
plt.figure(figsize=(6,3))
plt.title("SLOPE VRS NUM")
sns.countplot(df.slope)
plt.show()
sns.countplot(df.slope[df.num==1])
plt.show()
plt.figure(figsize=(6,3))
plt.title("THAL VRS NUM")
sns.countplot(df.thal)
plt.show()
sns.countplot(df.thal[df.num==1])
plt.show()
plt.figure(figsize=(12,5))
plt.title("SEX VRS AGE VRS NUM")
sns.pointplot(x = 'sex',y='age',hue='num',data=df)
plt.show()
plt.figure(figsize=(12,5))
plt.title("CHOL VRS AGE VRS NUM")
sns.scatterplot(x='chol',y='age',hue='num',data=df)
plt.show()
x=df[['age', 'sex', 'cp','trestbps', 'chol', 'fbs','restecg', 'thalach', 'exang', 'oldpeak', 'slope','thal']]
y=df['num']
print(x)
print(y)
# split data into train and test
from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts = train_test_split(x,y,test_size=0.2)
# we have to split the data into 80% as train and 20% as test so we have specified test_size as 0.2
print(x.shape)
print(xtr.shape)
print(xts.shape)
print(y.shape)
print(ytr.shape)
print(yts.shape)
from sklearn.decomposition import PCA
pca = PCA(n_components = 1)
xtr = pca.fit_transform(xtr)
xts = pca.transform(xts)
explained_variance = pca.explained_variance_ratio_
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
svc_model=SVC()
svc_model.fit(xtr,ytr)
y_pred=svc_model.predict(xts)
cm=confusion_matrix(yts,y_pred)
sns.heatmap(cm,annot=True)
min_train = xtr.min()
min_train
range_train = (xtr - min_train).max()
range_train
X_train_scaled = (xtr - min_train)/range_train
X_train_scaled
min_test = xts.min()
range_test = (xts - min_test).max()
X_test_scaled = (xts - min_test)/range_test
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train_scaled, ytr)
y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(yts, y_predict)

sns.heatmap(cm,annot=True,fmt="d")
print(classification_report(yts,y_predict))
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
grid.fit(X_train_scaled,ytr)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test_scaled)
cm = confusion_matrix(yts, grid_predictions)
sns.heatmap(cm, annot=True)
print(classification_report(yts,grid_predictions))
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = xtr, y = ytr, cv = 10)
accuracies
print(accuracies.mean())
print(accuracies.std())
