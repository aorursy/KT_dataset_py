import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import os
print(os.listdir("../input"))
train_0 = pd.read_csv('../input/train.csv')
test_0 = pd.read_csv('../input/test.csv')
sample_submission_0 = pd.read_csv('../input/sample_submission.csv')
df_train_0, df_validate_0 = train_test_split(train_0, test_size=0.3,random_state=100)
train_x_0 = df_train_0.drop('label', axis=1)
train_y_0 = df_train_0['label']
validate_x_0 = df_validate_0.drop('label', axis=1)
validate_y_0 = df_validate_0['label']
model_dt_0 = DecisionTreeClassifier(max_depth=5)
model_dt_0.fit(train_x_0, train_y_0)
validate_pred_0 = model_dt_0.predict(validate_x_0)
accuracy_score(validate_y_0, validate_pred_0)*100
model_dt_1 = RandomForestClassifier(random_state=100)
model_dt_1.fit(train_x_0,train_y_0)
validate_pred_1 = model_dt_1.predict(validate_x_0)
accuracy_score(validate_y_0, validate_pred_1)*100
model_dt_2 = AdaBoostClassifier(random_state=100)
model_dt_2.fit(train_x_0,train_y_0)
validate_pred_2 = model_dt_2.predict(validate_x_0)
accuracy_score(validate_y_0, validate_pred_2)*100
model_dt_3 = RandomForestClassifier()
model_dt_3.fit(train_x_0, train_y_0)
pred = model_dt_3.predict(test_0)
index = test_0.index + 1
df = pd.DataFrame({'ImageId':index, 'Label':pred})
import pandas as pd
df.to_csv("prediction.csv", index = False)
