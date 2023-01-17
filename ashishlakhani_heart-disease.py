# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
data.head()
#converting to categorical
data.replace({'sex': {1: "Male", 2: "Female"}},inplace=True)
data.replace({'exang': {1: "exc_ind_ang", 0: "no_exc_ind_ang"}},inplace=True)
data.replace({'fbs': {1: "fbs_present", 0: "no_fbs"}},inplace=True)
data.replace({'ca': {0: "ca_0", 1: "ca_1",2: "ca_2",3: "ca_3",4: "ca_4"}},inplace=True)
data.replace({'slope': {0: "slope_0", 1: "slope_1",2: "slope_2"}},inplace=True)
data.replace({'thal': {0: "thal_0", 1: "thal_1",2: "thal_2",3: "thal_3"}},inplace=True)
data.replace({'cp': {0: "cp_0", 1: "cp_1",2: "cp_2",3: "cp_3"}},inplace=True)
data.replace({'restecg': {0: "restecg_0", 1: "restecg_1",2: "restecg_2"}},inplace=True)

data.dtypes

positive_heart_issues = data[data['target'] == 1]
negative_heart_issues = data[data['target'] == 0]
#some data exploration
#Affect of age on heart problems
import matplotlib.pyplot as plt



plt.hist(positive_heart_issues['age'],bins=10)

plt.hist(negative_heart_issues['age'],bins=10)
#plt.hist(positive_heart_issues['sex'])
plt.hist(positive_heart_issues['chol'],bins=10)
plt.hist(negative_heart_issues['chol'],bins=10)
bins = [0,45, 50, 55,60,65,100]
labels = ['<45','45-50','50-55','55-60','60-65','>65']
data['age_bin'] = pd.cut(data['age'], bins=bins, labels=labels)

bins = [0,150,175,200,225,250,275,300,325,500]
labels = ['<150','150-175','175-200','200-225','225-250','250-275','275-300','300-325','325-500']
data['chol_bin'] = pd.cut(data['chol'], bins=bins, labels=labels)
data['age_bin'].value_counts()
data['chol_bin'].value_counts()
data.drop('age',axis=1,inplace=True)
data.drop('chol',axis=1,inplace=True)
y=data['target']
data.drop('target',axis=1,inplace=True)
X=pd.get_dummies(data)
X
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
clf = RandomForestClassifier(verbose=1,n_estimators=50,max_depth=5, random_state=56)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(X_train, y_train)
predicted_labels = clf.predict(X_test)
predicted_quant = clf.predict_proba(X_test)[:, 1]
score=accuracy_score(y_test, predicted_labels)
print ("FINISHED classifying accuracy score : ",score)

predicted_labels
predicted_quant
from sklearn.metrics import confusion_matrix #for model evaluation
confusion_matrix = confusion_matrix(y_test, predicted_labels)
confusion_matrix
total=sum(sum(confusion_matrix))

sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])
print('Sensitivity : ', sensitivity )

specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
print('Specificity : ', specificity)
from sklearn.metrics import roc_curve,auc
fpr, tpr, thresholds = roc_curve(y_test, predicted_quant)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
auc(fpr, tpr)
from sklearn.metrics import recall_score
recall_score(y_test, predicted_labels)
feature_importances = pd.DataFrame(clf.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',                                                                 ascending=False)
feature_importances
estimator = clf.estimators_[1]
feature_names = [i for i in X_train.columns]

y_train_str = y_train.astype('str')
y_train_str[y_train_str == '0'] = 'no disease'
y_train_str[y_train_str == '1'] = 'disease'
y_train_str = y_train_str.values
from sklearn.tree import export_graphviz #plot tree
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = feature_names,
                class_names = y_train_str,
                rounded = True, proportion = True, 
                label='root',
                precision = 2, filled = True)

from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

from IPython.display import Image
Image(filename = 'tree.png')