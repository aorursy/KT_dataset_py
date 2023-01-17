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
#%%  Read both csv files
## Reading csv files and adding them
filepath1 = os.path.join(dirname, filenames[0])
filepath2 = os.path.join(dirname, filenames[1])
df1 = pd.read_csv(filepath1)
df2 = pd.read_csv(filepath1)
print(df1.shape)
print(df2.shape)
df = pd.concat([df1, df2], ignore_index=True)
print(df.info())
del(df1)
del(df2)
seed = 1001
column1 = ['DAY_OF_MONTH', 'DAY_OF_WEEK','OP_CARRIER_AIRLINE_ID', 'TAIL_NUM','OP_CARRIER_FL_NUM',
 'DEST','DEP_TIME','DEP_DEL15', 'ARR_TIME', 'ARR_DEL15', 
 'CANCELLED','DIVERTED', 'DISTANCE']


df = df[column1]

# Drops rows with na values from dataframe.
df.dropna(inplace=True)
## Data set has a class imbalance.  There are 921482 flights that weren't delayed, however 
## there were only 210444 flights that were delayed.  I have tried to add more balance by downsampling the 
## number of not delayed classes.

not_delay = df[df['ARR_DEL15']==0]
delay = df[df['ARR_DEL15']==1]

print(len(not_delay))
print(len(delay))
df2 = df[['TAIL_NUM', 'ARR_DEL15']]
df2 = df2.groupby(['TAIL_NUM']).sum()
df2 = df2.sort_values(by=['ARR_DEL15'], ascending=False)
df2.columns = ['tail_delay']
import matplotlib.pyplot as plt
plt.hist(df2['tail_delay'], 8, facecolor='green', alpha=0.75)
plt.xlabel('Number of Flight Delays')
plt.ylabel('Counts')
plt.title(r'Histogram of flight delays a plane has')
## Converts TAIL_NUM into a category from 1-6 based on the amount of delays the plane has
df2['tail_delay'] = np.floor(df2['tail_delay']/20)
df = pd.merge(df, df2, on='TAIL_NUM')
del(df2)
df.drop(columns = ['TAIL_NUM'], inplace=True)

## rounds departure time down to the nearest hour. Example 6:15 or 6:45 would be both be rounded to 6.
df['DEP_TIME'] = round((df['DEP_TIME']/100), 0)
df['DEP_TIME']  = df['DEP_TIME'].astype(int)
df['ARR_TIME'] = round((df['ARR_TIME']/100), 0)
df['ARR_TIME']  = df['ARR_TIME'].astype(int)  
import matplotlib.pyplot as plt
column = 'DEST'
df2 = df[[column, 'ARR_DEL15']]
df2 = df2.groupby(column).sum()
df2 = df2.sort_values(by=['ARR_DEL15'], ascending=False)
df2.columns = ['city_delay']

##plt.boxplot(df2['delay_counts'], notch=True)
plt.hist(df2['city_delay'], 8, facecolor='green', alpha=0.75)
plt.xlabel('Number of Flight Delays')
plt.ylabel('Counts')
plt.title(r'Histogram of Flight Delays based on Destination Airport')
df2['city_delay'] = np.floor(df2['city_delay']/1000)
df = pd.merge(df, df2, on=column)
del(df2)
df.drop(columns = column, inplace=True)
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

column2 = ['OP_CARRIER_AIRLINE_ID', 'tail_delay', 'city_delay']

for col in column2:
    enc = OneHotEncoder(sparse = False, handle_unknown='ignore')
    encoded_frame = enc.fit_transform(np.array(df[col]).reshape(-1,1))
    column_name = enc.get_feature_names()
    column_name = col + column_name
    one_hot_encoded_frame =  pd.DataFrame(encoded_frame, columns= column_name)
    
    df = pd.concat([df, one_hot_encoded_frame], axis=1)
    del(encoded_frame)
    del(one_hot_encoded_frame)
    df.drop(columns = col, inplace=True)

#%%    splitting data into training and test sets.
## 80/20 split for training and evaluation
split   = 0.8 
x_train = df.sample(frac=split, random_state=seed) 
x_test  = df.drop(x_train.index)
print(x_train.shape, x_test.shape)

# Extract and remove the label (to be predicted) set
y_train = x_train.pop('ARR_DEL15')
y_test  = x_test.pop('ARR_DEL15')
print(y_train.shape, y_test.shape)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state = 0)
clf.fit(x_train, y_train) 
!pip install pydotplus
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn.externals.six import StringIO  
import pydotplus
from IPython.display import Image 
dot_data = StringIO()
from sklearn import tree

tree.export_graphviz(clf, max_depth = 3, feature_names=x_test.columns, out_file= dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
graph.write_png('tree.png')
Image(graph.create_png())
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
  
y_predict = clf.predict(x_test)
cmatrix = confusion_matrix(y_test, y_predict) 
accuracy = accuracy_score(y_test, y_predict) 
classification_report = classification_report(y_test, y_predict) 

print(cmatrix)
print(classification_report)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

x_test2 = df
y_test2 = x_test2.pop('ARR_DEL15')
scoring='f1_macro'
scores = cross_validate(clf, x_test2, y_test2, cv=5, scoring = ['accuracy', 'f1'])
results = list()
results.append([scores['test_accuracy'].mean(), scores['test_f1'].mean()])
print('The average accuracy is %f and the average F1 score is %f' %((results[0][0]),(results[0][1])))
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

probs = clf.predict_proba(x_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
import matplotlib.pyplot as plt

plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()