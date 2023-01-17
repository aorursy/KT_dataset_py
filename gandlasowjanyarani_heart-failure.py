import pandas as pd
import numpy as np
file=pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
file.head(15)
file.isna().sum()
file.isnull().sum()
file.describe()
file.shape
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
X = file.iloc[:,0:12] 
Y = file.iloc[:,-1]    
model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(12).plot(kind='bar')
plt.show()
file.info()
file.columns
X=file[['age','creatinine_phosphokinase','platelets',
       'ejection_fraction',
       'serum_creatinine', 'serum_sodium',  'time']]
Y=file[['DEATH_EVENT']]
from sklearn.model_selection import train_test_split
X_trainset,X_testset,Y_trainset,Y_testset=train_test_split(X,Y,test_size=0.3)
from sklearn.ensemble import RandomForestClassifier

rfm=RandomForestClassifier(n_estimators=40,oob_score=True,n_jobs=-1,random_state=40,max_features=None,min_samples_leaf=20)
rfm.fit(X_trainset,Y_trainset)
pre=rfm.predict(X_testset)
pre[0:5]
Y_testset[0:5]
from sklearn import metrics
print("Test set Accuracy: ", metrics.accuracy_score(Y_testset, pre))
