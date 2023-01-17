import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
train_file = pd.read_csv('../input/train.csv')
test_f = pd.read_csv('../input/test.csv')
test_file = test_f.iloc[:,0:784]
print(test_file.shape)
print(train_file.shape)
df_x=train_file.iloc[:,1:]
df_y=train_file.iloc[:,0]
import seaborn as sns
sns.countplot(df_y)
import time
start_time = time.time()
clf=RandomForestClassifier(n_estimators=500,warm_start = True, oob_score =True, random_state = 42,
                           max_features="sqrt")
RF = clf.fit(df_x,df_y)
RF

error = 1 - clf.oob_score_
accuracy = 1 - error

print("Accuracy Percentage   : ",(round(accuracy, 4) *100), "%")
print("Error Percentage      : ",(round(error, 4) *100), "%")
prediction_test = clf.predict(test_file)
display(prediction_test)
results_data = pd.DataFrame({'ImageId': range(1, len(prediction_test)+1), 'Label': prediction_test})
results_data.to_csv('results.csv', sep=',', index=False)