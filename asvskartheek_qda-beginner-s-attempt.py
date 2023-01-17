# STD Libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# Load Dataset
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
# print("Shape: ",train.shape)
# train.head()
df_output = pd.DataFrame()
test_ids = []
predictions = []
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.ensemble import BaggingClassifier as Bagging
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.svm import SVC

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler as SS
from sklearn.feature_selection import VarianceThreshold as VT

for i in range(0,512):
    df_temp = pd.DataFrame()
    
    df_sub = train[train['wheezy-copper-turtle-magic']==i]
    
    y_train = df_sub['target']
    X_train = df_sub.drop(['id','target'],axis=1)
    
    X_test = test[test['wheezy-copper-turtle-magic']==i]
    test_sub_ids = X_test['id']
    X_test = X_test.drop(['id'],axis=1)
    
    vt_selector = VT(threshold=2)
    X_train = vt_selector.fit_transform(X_train)
    X_test = vt_selector.transform(X_test)
    
    scaler = SS()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
  
    model_qda = QDA(0.5)
    model_qda.fit(X_train,y_train)
  
  
    predictions_sub = model_qda.predict(X_test)
    for sample_t_id in test_sub_ids:
        test_ids.append(sample_t_id)
    for sample_pred in predictions_sub:
        predictions.append(sample_pred)

# print(test_ids)
    
df_output['id'] = test_ids
df_output['target'] = predictions
df_output.describe()
print(df_output['id'])
df_output.to_csv("submission_QDA_asd.csv",mode="w",index=False)