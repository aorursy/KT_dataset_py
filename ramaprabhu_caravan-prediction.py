import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
df= pd.read_csv('../input/crvtest/crv.csv')
df.head()
df['CARAVAN'].value_counts()
#Module for resamplingPython
from sklearn.utils import resample
# Separate majority and minority classes
df_majority = df[df.CARAVAN==0]
df_minority = df[df.CARAVAN==1]
 

# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=9236,    # to match majority class
                                 random_state=2) # reproducible results
 

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled.CARAVAN.value_counts()

X_train, X_test, y_train, y_test = train_test_split(df_upsampled.drop(['ORIGIN','CARAVAN'],axis='columns'),df_upsampled.CARAVAN, test_size=0.2)
X_train.nunique() 
coltypes = (X_train.nunique() < 5)  
coltypes   
cat_cols = coltypes[coltypes==True].index.tolist()
num_cols = coltypes[coltypes==False].index.tolist()
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
ct= ColumnTransformer([('abc', StandardScaler(),num_cols),('cde', OneHotEncoder(handle_unknown='ignore'),cat_cols) ], remainder = 'passthrough')
ct.fit(X_train,y_train)
from sklearn.ensemble import RandomForestClassifier as rf
pipe = Pipeline([ ('ct',ct), ('rf', rf() )])
# Train model
pipe.fit(X_train,y_train)
# Predict on training set
pred_y1 = pipe.predict(X_test)
# Is our model still predicting just one class?
print( np.unique( pred_y1 ) )
# [0 1]
# How's our accuracy?
print( accuracy_score(y_test, pred_y1) )
prob_y1 = pipe.predict_proba(X)
prob_y1 = [p[1] for p in prob_y1]
print( roc_auc_score(y, prob_y1) )
np.sum(pred_y1 == y_test)/len(y_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,pred_y1)
cm
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
# Separate majority and minority classes
df_majority = df[df.CARAVAN==0]
df_minority = df[df.CARAVAN==1]
 
# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=586,     # to match minority class
                                 random_state=400) # reproducible results
# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])
# Display new class counts
df_downsampled.CARAVAN.value_counts()
X_train, X_test, y_train, y_test = train_test_split(df_downsampled.drop(['ORIGIN','CARAVAN'],axis='columns'),df_downsampled.CARAVAN, test_size=0.2)
ct= ColumnTransformer([('abc', StandardScaler(),num_cols),('cde', OneHotEncoder(handle_unknown='ignore'),cat_cols) ], remainder = 'passthrough')
# Train model
ct.fit(X_train,y_train)
pipe = Pipeline([ ('ct',ct), ('rf', rf() )])
# Train model
pipe.fit(X_train,y_train)
# Predict on training set
pred_y2 = pipe.predict(X_test)
# How's our accuracy?
print( accuracy_score(y_test, pred_y2) )
cm=confusion_matrix(y_test,pred_y2)
cm
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
