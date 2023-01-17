import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
training_set = pd.read_csv('/kaggle/input/unsw-nb15/UNSW_NB15_training-set.csv')
training_set.info()
training_set.head()
print(training_set['label'].unique())
print(training_set['attack_cat'].unique())
mask = (training_set.dtypes == np.object) #离散型变量的类型
print(training_set.loc[:,mask].head())
list_cat = training_set.loc[:,mask].columns.tolist()
print(list_cat)
print(training_set.loc[:,mask].values)

mask = (training_set.dtypes != np.object)
print(training_set.loc[:,mask].head())
list_cat = training_set.loc[:,mask].columns.tolist()
print(list_cat)
training_set.loc[:,mask].describe()
# number of occurrences for each attack category
training_set.attack_cat.value_counts()
mask = (training_set.label == 1)
print(training_set.loc[mask,:].service.value_counts())
print(training_set.loc[mask,:].proto.value_counts())
mask = (training_set.label == 0)
print(training_set.loc[mask,:].service.value_counts())
print(training_set.loc[mask,:].proto.value_counts())
#将attack_cat多类转为数值
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
num_cat = le.fit_transform(training_set.attack_cat)
print(le.classes_)
print(np.unique(num_cat))
Y=num_cat.tolist()
X = training_set.drop(columns=['id','attack_cat','label']) #去除无关变量
mask = (X.dtypes == np.object)
list_cat = X.loc[:,mask].columns.tolist()
list_cat
X = pd.get_dummies(X, columns=list_cat)
X.head()
Y[:10]
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
import xgboost as xgb
from sklearn import svm
from sklearn.metrics import classification_report,roc_auc_score,average_precision_score,confusion_matrix

params = {
    'max_depth': 10,
    'objective': 'multi:softmax',  # error evaluation for multiclass training
    'num_class': 10,                # Number of classes 
    'n_gpus': 1
}

xg_clf = xgb.XGBClassifier(**params)
pred = xg_clf.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))
clf = svm.SVC(gamma='auto')
predsvm=clf.fit(X_train,y_train).predict(X_test)
print(classification_report(y_test, predsvm))
print(confusion_matrix(y_test, predsvm))