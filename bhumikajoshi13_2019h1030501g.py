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
import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats



%matplotlib inline



# Optimization module in scipy

from scipy import optimize



from collections import Counter

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

from sklearn.metrics import roc_curve, auc, roc_auc_score

from csv import writer

from csv import reader

from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import RandomOverSampler

from sklearn.datasets import make_classification

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression
file_path = '/kaggle/input/minor-project-2020/train.csv'

df = pd.read_csv(file_path) 
df.head()
df.isnull()
df.info()

#tells that no non-null i.e. NAN is present in any col
y = df[['target']]

X = df.drop(['target', 'id'], axis=1)
print(type(y))

li = []

for i in range(len(y)):

    li.append(y._get_value(i,'target'))

# summarize class distribution

counter = Counter(li)

print(counter)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
print(len(X_train), len(X_test))

print(len(y_train), len(y_test))

print(len(X_train.columns), len(X_test.columns))

print(len(y_train.columns), len(y_test.columns))
# see count of 0 and 1 in splitted train data

dct = y_train.to_dict()

temp=dct['target']

dict_train = temp.copy()

print(type(y_train))

li = []

for key in dict_train:

    li.append(dict_train[key])

# summarize class distribution

counter = Counter(li)

print(counter)
# see count of 0 and 1 in splitted test data

dct = y_test.to_dict()

temp=dct['target']

dict_test = temp.copy()

print(type(y_test))

li = []

for key in dict_test:

    li.append(dict_test[key])

# summarize class distribution

counter = Counter(li)

print(counter)
file_path = '/kaggle/input/minor-project-2020/test.csv'

df_test = pd.read_csv(file_path) 

df_test.columns



y_test_file = df_test.iloc[:, 1].values

X_test_file = df_test.drop(['id'], axis=1)
# Create an undersampler object

rus = RandomOverSampler(sampling_strategy='minority', random_state=10)
# Resample the features for training data and the target

X_resampled, y_resampled = rus.fit_sample(X_train, y_train)
print(len(X_resampled), len(y_resampled))
# Revert resampeled data into a dataframe

X_resampled = pd.DataFrame(X_resampled)

y_resampled = pd.DataFrame(y_resampled)

X_train = X_resampled

y_train = y_resampled
# see count of 0 and 1 in sampled splitted train data

dct = y_train.to_dict()

temp=dct['target']

dict_train = temp.copy()

print(type(y_train))

li = []

for key in dict_train:

    li.append(dict_train[key])

# summarize class distribution

counter = Counter(li)

print(counter)
# see count of 0 and 1 in sampled splitted test data

dct = y_test.to_dict()

temp=dct['target']

dict_test = temp.copy()

print(type(y_test))

li = []

for key in dict_test:

    li.append(dict_test[key])

# summarize class distribution

counter = Counter(li)

print(counter)
# performed scaling but dont have much effect of it as have used RandomOverSampler but still kept it

scalar = StandardScaler()

scaled_X_train = scalar.fit_transform(X_train)

scaled_X_test = scalar.transform(X_test)
# C = range(0, 120)

# parameters = {'C': C}

# lr_cv = LogisticRegression(random_state=10)

# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)

# clf = GridSearchCV(lr_cv, parameters, verbose=1, cv=kfold, n_jobs=-1)

# clf.fit(scaled_X_train, y_train)

# clf_best = clf.best_params_.get('C')

# print(clf.best_params_)



# Have not executed StratifiedKFold for k=5 and Grid Search again but have selected the best value of parameter C 

# obtained i.e. 113 directly which was obtained when I ran it.

# You can uncomment the above code and comment the below code in this cell itself if you want to rerun the above

# StratifiedKFold and Grid Search.

# Also I have verified my score by directly putting the C value.



clf_best = 113

print(clf_best)

clf = LogisticRegression(C=clf_best, random_state=10)

clf.fit(scaled_X_train, y_train)
y_pred = clf.predict_proba(scaled_X_test)[:,1]
# for probabilities prediction

plt.style.use('seaborn-pastel')

FPR, TPR, _ = roc_curve(y_test, y_pred)

ROC_AUC = auc(FPR, TPR)

print (ROC_AUC)



# roc_auc_score(y_test, y_pred)#same as above
plt.figure(figsize =[11,9])

plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)

plt.plot([0,1],[0,1], 'k--', linewidth = 4)

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.05])

plt.xlabel('False Positive Rate', fontsize = 18)

plt.ylabel('True Positive Rate', fontsize = 18)

plt.title('ROC for Given Dataset', fontsize= 18)

plt.show()
# scaled_X_train = scalar.fit_transform(X)

scaled_X_test_file = scalar.transform(X_test_file)



clf_file = LogisticRegression(C=clf_best, random_state=10)

clf_file.fit(scaled_X_train, y_train)
y_pred_file = clf_file.predict_proba(scaled_X_test_file)[:,1]
id_arr=df_test["id"]

id_list = id_arr.tolist()

target_list = y_pred_file.tolist()

with open('sample_submission.csv', 'w', newline='') as write_obj:

    csv_writer = writer(write_obj)

    fields = ['id','target']

    csv_writer.writerow(fields)

    for i in range(len(id_list)):

        # Append the default text in the row / list

        row = []

        row.append(id_list[i])

        row.append(target_list[i])

        # Add the updated row / list to the output file

        csv_writer.writerow(row)