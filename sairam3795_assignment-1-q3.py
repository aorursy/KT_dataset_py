import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import os
print(os.listdir("../input"))

train=pd.read_csv('../input/train.csv')
train.head()

#test_fin=pd.read_csv('../input/test.csv')


# Any results you write to the current directory are saved as output.

train1, Validate = train_test_split(train,
                               test_size = 0.3,
                               random_state=100)

train1_y = train1['label']
Validate_y = Validate['label']

train1_x = train1.drop('label', axis=1)
Validate_x = Validate.drop('label', axis=1)


# Creating/Fitting a model - Decision Tree
model_dt = DecisionTreeClassifier()
model_dt.fit(train1_x, train1_y)

# predicting on validate data
test_pred_dt = model_dt.predict(Validate_x)

from sklearn.metrics import accuracy_score, classification_report

Accuracy_dt=accuracy_score(Validate_y,test_pred_dt)

Accuracy_dt
# Creating/Fitting a model - KNN
model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(train1_x, train1_y)

# predicting on validate data
test_pred_knn = model_knn.predict(Validate_x)
#print(len(test_pred_knn))
#test_pred_knn

df_pred_knn = pd.DataFrame({'actual': Validate_y,
                         'predicted': test_pred_knn})


Accuracy_knn=accuracy_score(Validate_y,test_pred_knn)

Accuracy_knn
from sklearn.ensemble import RandomForestClassifier

# Creating/Fitting a model - Random Forest
model_rf = RandomForestClassifier(random_state=100)
model_rf.fit(train1_x, train1_y)

# predicting on validate data
test_pred_rf = model_rf.predict(Validate_x)
#print(len(test_pred_rf))
#test_pred_rf

df_pred_rf = pd.DataFrame({'actual': Validate_y,
                         'predicted': test_pred_rf})
Accuracy_rf=accuracy_score(Validate_y,test_pred_rf)
Accuracy_rf
from sklearn.ensemble import AdaBoostClassifier
# Creating/Fitting a model - AdaBoost
model_ada = AdaBoostClassifier(random_state=100)
model_ada.fit(train1_x, train1_y)

# predicting on validate data
test_pred_ada = model_ada.predict(Validate_x)
#print(len(test_pred_dt))
test_pred_ada

df_pred_ada = pd.DataFrame({'actual': Validate_y,
                         'predicted': test_pred_ada})
Accuracy_ada=accuracy_score(Validate_y,test_pred_ada)
Accuracy_ada
# Final prediction on test data using KNN

test_fin=pd.read_csv('../input/test.csv')

train_x=train.drop('label',axis=1)
train_y=train['label']
model_final=KNeighborsClassifier(n_neighbors=3)
model_final.fit(train_x,train_y)
final_prediction=model_final.predict(test_fin)
df=pd.DataFrame({'Label':final_prediction})
df['ImageId']=test_fin.index+1
df[['ImageId','Label']].to_csv('Final_Prediction.csv',index=False)