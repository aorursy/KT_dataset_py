
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import warnings                        # to hide error messages(if any)
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))

df = pd.read_csv('../input/Dataset_spine.csv')

#Renaming the columns
df = df.rename(columns = {'Col1':'pelvic_incidence',
                          'Col2':'pelvic tilt',
                            'Col3':'lumbar_lordosis_angle',
                            'Col4':'sacral_slope',
                            'Col5':'pelvic_radius',
                            'Col6':'degree_spondylolisthesis',
                            'Col7':'pelvic_slope',
                            'Col8':'Direct_tilt',
                            'Col9':'thoracic_slope',
                            'Col10':'cervical_tilt',
                            'Col11':'sacrum_angle',
                            'Col12':'scoliosis_slope',
                            'Class_att':'label'})

#Removing the unnecessary colum('Unnamed: 13')
df = df.drop('Unnamed: 13', axis = 1)
df.head()
def label_values(label):
    if label == 'Abnormal':
        return 1
    elif label == 'Normal':
        return 0


df['label_value'] = df['label'].apply(label_values)
df.head()
df.isnull().sum()
df.shape
df.dtypes
df.describe()
#Generating heatmap
plt.figure(figsize = (20,12))
sns.heatmap(df.corr(), annot = True, cmap = 'Paired')
plt.show()
sns.pairplot(df, hue = 'label')
plt.show()
#Count of the attribures
sns.countplot(x = 'label', data = df)
plt.show()
# importing the sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
#Listing features to be used for prediction
features = ['pelvic_incidence','pelvic tilt',
'lumbar_lordosis_angle','sacral_slope',
'pelvic_radius',
'degree_spondylolisthesis',
'pelvic_slope',
'Direct_tilt',
'thoracic_slope',
'cervical_tilt',
'sacrum_angle',
'scoliosis_slope']
#Drawing box plots for the various featues in X
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10, 7))
for idx, feat in enumerate(features):
    ax = axes[int(idx / 4), idx % 4]
    sns.boxplot(x='label', y=feat, data=df, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel(feat)
fig.tight_layout();
X = df[features]
y =df['label_value']
#Splitting the data set into training and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
# Storing the predicted values in y_pred for X_test
y_pred = logreg.predict(X_test)
#Generating the cofusion matrix
cnf_matrix = metrics.confusion_matrix(y_test,y_pred)
cnf_matrix
class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)

#create a heat map
sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'YlGnBu',
           fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix', y = 1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred))
print('Precision Score: ',metrics.precision_score(y_test,y_pred))
print('Recall Score: ',metrics.recall_score(y_test,y_pred))