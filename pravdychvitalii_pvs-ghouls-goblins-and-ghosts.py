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
        
import zipfile
with zipfile.ZipFile('../input/ghouls-goblins-and-ghosts-boo/sample_submission.csv.zip', 'r') as zip_obj:
   # Extract all the contents of zip file in current directory
   zip_obj.extractall('/kaggle/working/')
with zipfile.ZipFile('../input/ghouls-goblins-and-ghosts-boo/test.csv.zip', 'r') as zip_obj:
   # Extract all the contents of zip file in current directory
   zip_obj.extractall('/kaggle/working/')
with zipfile.ZipFile('../input/ghouls-goblins-and-ghosts-boo/train.csv.zip', 'r') as zip_obj:
   # Extract all the contents of zip file in current directory
   zip_obj.extractall('/kaggle/working/')
    
print('After zip extraction:')
print(os.listdir("/kaggle/working/"))
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train = pd.read_csv('./train.csv', index_col = 'id')
train
test = pd.read_csv('./test.csv', index_col = 'id')
test
ds = train.merge(test, how='outer')
ds
ds.describe()
ds.info()
ds['type'] = ds['type'].replace(['Ghoul', 'Goblin', 'Ghost'], [1, 2, 3])
ds['type'] = ds['type'].fillna(0)
ds['color'] = ds['color'].replace(['clear', 'green', 'black', 'white', 'blue', 'blood'], [1, 2, 3, 4, 5, 6])
ds.corr()['type'].sort_values()
plt.plot(ds['has_soul'])
plt.hist(ds['has_soul'])
plt.plot(ds['hair_length'])
plt.hist(ds['hair_length'])
plt.plot(ds['bone_length'])
ds['bone_length'] = np.log(ds['bone_length']+1)
plt.hist(ds['bone_length'])
plt.plot(ds['color'])
plt.hist(ds['color'])
ds = pd.get_dummies(ds, columns=['color'])
ds
plt.plot(ds['rotting_flesh'])
plt.hist(ds['rotting_flesh'])
ds
test = ds[ds['type'] == 0]
train = ds[ds['type'] != 0]
print(test.shape, train.shape)
y = train['type']
X = train.drop(['type'], axis = 1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()
]

log_cols = ["Classifier", "Accuracy"]
log = pd.DataFrame(columns=log_cols)

acc_dict = {}

for clf in classifiers:
    name = clf.__class__.__name__
    clf.fit(X_train, y_train)
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
        
    if name in acc_dict:
        acc_dict[name] += acc
    else:
        acc_dict[name] = acc

for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf]
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns = log_cols)
    log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x = 'Accuracy', y = 'Classifier', data = log, color = "b")
acc_dict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

print('score=',lda.score(X_test, y_test))
y_pred = lda.predict(X_test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
X_pred = test.drop(['type'], axis = 1)
pred = lda.predict(X_pred)
sample_submission = pd.read_csv('./sample_submission.csv', index_col = 'id')
sample_submission['type'] = pred
sample_submission.head()
sample_submission['type'] = sample_submission['type'].replace([1.0, 2.0, 3.0],['Ghoul', 'Goblin', 'Ghost'])
sample_submission.head()
sample_submission.to_csv('out.csv') 