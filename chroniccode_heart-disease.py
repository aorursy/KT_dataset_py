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
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling as pp
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
df.shape
df.head()
df.describe()
pp.ProfileReport(df)
df.target.value_counts()
df.isnull().sum()
for column in df.columns:
    print(column,df[column].nunique())
plt.rcParams['figure.figsize'] = (16, 14)
# plt.style.use('ggplot')
sns.heatmap(df.corr(), annot = True, cmap = 'PiYG')
plt.title('Heatmap of Data', fontsize = 20)
plt.show()
f,ax=plt.subplots(3,2,figsize=(12,12))
f.delaxes(ax[2,1])

for i,feature in enumerate(['age','thalach','chol','trestbps','oldpeak']):
    sns.distplot(df[feature], ax=ax[i//2,i%2], hist=True, color= 'y' )
f,ax=plt.subplots(4,2,figsize=(10,8))

for i,feature in enumerate(['sex','cp','fbs','restecg','exang','slope','ca','thal']):
    sns.countplot(x=feature,data=df,ax=ax[i//2,i%2], alpha=0.8, edgecolor=('white'), linewidth=2)
    plt.tight_layout()
plt.rcParams['figure.figsize'] = (8, 6)
sns.violinplot(df['target'], df['age'], palette = 'colorblind')
plt.title('Age vs Target', fontsize = 20, fontweight = 30)
plt.show()
plt.rcParams['figure.figsize'] = (8, 6)
sns.violinplot(df['target'], df['thalach'], palette = 'colorblind')
plt.title('thalach vs Target', fontsize = 20, fontweight = 30)
plt.show()
plt.rcParams['figure.figsize'] = (8, 6)
sns.violinplot(df['target'], df['chol'], palette = 'colorblind')
plt.title('chol vs Target', fontsize = 20, fontweight = 30)
plt.show()
plt.rcParams['figure.figsize'] = (8, 6)
sns.violinplot(df['target'], df['trestbps'], palette = 'colorblind')
plt.title('trestbps vs Target', fontsize = 20, fontweight = 30)
plt.show()
plt.rcParams['figure.figsize'] = (8, 6)
sns.violinplot(df['target'], df['oldpeak'], palette = 'colorblind')
plt.title('oldpeak vs Target', fontsize = 20, fontweight = 30)
plt.show()
plt.rcParams['figure.figsize'] = (8, 6)
dat = pd.crosstab(df['target'], df['restecg']) 
dat.div(dat.sum(1).astype(float), axis = 0).plot(kind = 'bar')
plt.title('Relation of ECG measurement with Target', fontsize = 20, fontweight = 30)
plt.show()
plt.rcParams['figure.figsize'] = (8, 6)
dat = pd.crosstab(df['target'], df['fbs']) 
dat.div(dat.sum(1).astype(float), axis = 0).plot(kind = 'bar')
plt.title('Relation of blood sugar with Target', fontsize = 20, fontweight = 30)
plt.show()
plt.rcParams['figure.figsize'] = (8, 6)
dat = pd.crosstab(df['target'], df['sex'])
dat.div(dat.sum(1).astype(float), axis = 0).plot(kind = 'bar')
plt.title('Relation of Gender with Target', fontsize = 20, fontweight = 30)
plt.show()
plt.rcParams['figure.figsize'] = (8, 6)
dat = pd.crosstab(df['target'], df['cp']) 
dat.div(dat.sum(1).astype(float), axis = 0).plot(kind = 'bar')
plt.title('Relation of  chest pain with Target', fontsize = 20, fontweight = 30)
plt.show()
plt.rcParams['figure.figsize'] = (8, 6)
dat = pd.crosstab(df['target'], df['exang']) 
dat.div(dat.sum(1).astype(float), axis = 0).plot(kind = 'bar')
plt.title('Relation of Exercise induced angina with Target', fontsize = 20, fontweight = 30)
plt.show()
plt.rcParams['figure.figsize'] = (8, 6)
dat = pd.crosstab(df['target'], df['slope']) 
dat.div(dat.sum(1).astype(float), axis = 0).plot(kind = 'bar')
plt.title('Relation of slope with Target', fontsize = 20, fontweight = 30)
plt.show()
plt.rcParams['figure.figsize'] = (8, 6)
dat = pd.crosstab(df['target'], df['ca']) 
dat.div(dat.sum(1).astype(float), axis = 0).plot(kind = 'bar')
plt.title('Relation of major vessels with Target', fontsize = 20, fontweight = 30)
plt.show()
plt.rcParams['figure.figsize'] = (8, 6)
dat = pd.crosstab(df['target'], df['thal']) 
dat.div(dat.sum(1).astype(float), axis = 0).plot(kind = 'bar')
plt.title('Relation thalassemia with Target', fontsize = 20, fontweight = 30)
plt.show()
categorical_cols = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
for i in categorical_cols:
    print(i,'\n', df[i].value_counts())
multi_label_cols = [i for i in categorical_cols if df[i].nunique()>2]
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
df[numeric_cols] = std.fit_transform(df[numeric_cols])
df.shape
df = pd.get_dummies(data = df,columns = multi_label_cols)
x = df.drop(['target'],axis=1)
y = df['target']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
print("Shape of x_train :", x_train.shape)
print("Shape of x_test :", x_test.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of y_test :", y_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
lr = LogisticRegression()
svm = SVC(probability=True)
rf = RandomForestClassifier(n_estimators=100, max_depth=5)
xg = xgb.XGBClassifier()
models = ['lr','svm','rf','xg']
for model in models:
    clf = eval(model)
    clf.fit(x_train, y_train)
    y_pred_prob = clf.predict_proba(x_test)[:, 1]
    y_pred = clf.predict(x_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)
    # evaluating the model
    print(f"Training Accuracy for model {model} is: ", clf.score(x_train, y_train))
    print(f"Testing Accuracy for model {model} is:", clf.score(x_test, y_test))
    print(f"AUC Score for model {model} is: {auc}")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot = True)
    print(classification_report(y_test, y_pred))
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f'ROC for model {model}')
from sklearn.model_selection import GridSearchCV
grid={"C":np.logspace(-3,3,10), "penalty":["l1","l2"]}
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(x,y)
print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)
