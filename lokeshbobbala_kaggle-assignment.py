# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
data= pd.read_csv("../input/sample_submission.csv")
data.head()
# Any results you write to the current directory are saved as output.
test.head()
import matplotlib.pyplot as plt
first_digit = train.iloc[3].drop('label').values.reshape(28,28)
plt.imshow(first_digit)
import pandas as pd
import numpy as np
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
df_train, df_validate = train_test_split(train, test_size = 0.3, random_state=100)
train_x = df_train.drop('label',axis = 1)
train_y = df_train['label']
validate_x = df_validate.drop('label',axis=1)
validate_y = df_validate['label']
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(train_x,train_y)
validate_predict = model.predict(validate_x)
print(accuracy_score(validate_y,validate_predict))
test_pred = model.predict(test)
df_test_predict = pd.DataFrame(test_pred,columns = ['Label'])
df_test_predict['ImageId'] = test.index+1
df_test_predict[['ImageId','Label']].to_csv('submission_1.csv',index = False)
rf_model = RandomForestClassifier(random_state=100)
rf_model.fit(train_x,train_y)
validate_pred_rf1 = rf_model.predict(test)
#submission = pd.DataFrame(validate_pred_rf1, columns=['Label'])
#submission['ImageId'] = test.index + 1
#submission[['ImageId', 'Label']].to_csv('submission.csv', index=False)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=100)
model
train_y = train['label']
test.shape

train_x = train.drop('label',axis = 1)
#test_x = test.drop('label',axis = 1)
model.fit(train_x,train_y)
test_pred1 = model.predict(test)
#submission = pd.DataFrame(test_pred1, columns=['Label'])
#submission['ImageId'] = test.index + 1
#submission[['ImageId', 'Label']].to_csv('submission.csv', index=False)
dt_predictions = pd.DataFrame({'actual': data['Label'],'predicted': test_pred1})
dt_predictions.head()
dt_predictions['Pred_Status'] = dt_predictions['actual']==dt_predictions['predicted']
dt_predictions['Pred_Status'].value_counts()
DT_Accuracy = dt_predictions[dt_predictions['Pred_Status'] == True].shape[0] /dt_predictions.shape[0] * 100
DT_Accuracy
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(random_state=100)




model_rf.fit(train_x,train_y)

test_pred2 = model_rf.predict(test)
df_pred = pd.DataFrame({'actual': data['Label'],'predicted': test_pred2})
df_pred['pred_status'] = df_pred['actual'] == df_pred['predicted']
#submission = pd.DataFrame(test_pred2, columns=['Label'])
#submission['ImageId'] = test.index + 1
#submission[['ImageId', 'Label']].to_csv('submission.csv', index=False)
RF_Accuracy = df_pred['pred_status'].sum() / df_pred.shape[0] * 100
RF_Accuracy
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(random_state=100)
model.fit(train_x,train_y)

test_pred3 = model.predict(test)
#submission = pd.DataFrame(test_pred3, columns=['Label'])
#submission['ImageId'] = test.index + 1
#submission[['ImageId', 'Label']].to_csv('submission.csv', index=False)
df_pred1 = pd.DataFrame({'actual': data['Label'],'predicted': test_pred3})
df_pred1['pred_status'] = df_pred1['actual'] == df_pred1['predicted']
AB_Accuracy = df_pred1['pred_status'].sum() / df_pred1.shape[0] * 100
AB_Accuracy