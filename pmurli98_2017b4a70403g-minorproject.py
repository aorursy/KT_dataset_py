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
# import important libraries here
import numpy as np
import pandas as pd
import seaborn as sns
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
!ls
# read csv file
input_csv = '/kaggle/input/minor-project-2020/train.csv'
df = pd.read_csv(input_csv)
df.head()
df.describe()



df.drop('id', axis=1, inplace=True)
df.head()
df.shape

#try dropping duplicate rows and then train.

column_names = list(df)

#print(column_names)

newdf = df.drop_duplicates(column_names, 'first')

print(newdf.shape)
print(df.shape)
#split into X and y

X = df.drop('target', axis=1, inplace=False)
y = df['target']

X.head()
y.head()

print(X.shape)
print(y.shape)

#plot correlations. refer lab3 material later

mask = np.zeros_like(newdf.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.set_style('whitegrid')
plt.subplots(figsize = (100,100))
sns.heatmap(df.corr(),
            annot=True,
            mask = mask,
            cmap = 'RdBu', ## in order to reverse the bar replace "RdBu" with "RdBu_r"
            linewidths=.9, 
            linecolor='white',
            fmt='.2g',
            center = 0,
            square=True)
#check variances of columns

col_variances = newdf.var()

print(min(col_variances))
print(max(col_variances))

#print(col_variances)
#remove features that are constant or very less variance and then try training


constant_filter = VarianceThreshold(threshold=1)

constant_filter.fit(X)


print(len(X.columns[constant_filter.get_support()]))

print(X.columns[constant_filter.get_support()])

X_new = newdf[X.columns[constant_filter.get_support()]]

X_new.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)
# normalise
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train.shape
y_train.shape
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
#Check dataset.

print("Ones : ", (y_train!=0).sum())
print("Zeros : ", (y_train==0).sum())

#Giga imbalanced 
# resample X and y
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

smote = SMOTE()

# undersamp = RandomUnderSampler(sampling_strategy=0.4) #not working for binary classification. see later

# steps = [('s', smote), ('u', undersamp)]
# pipeline = Pipeline(steps=steps)

#X_train, y_train = pipeline.fit_resample(X_train, y_train)

X_train, y_train = smote.fit_resample(X_train,y_train)

print("Ones : ", (y_train!=0).sum())
print("Zeros : ", (y_train==0).sum())

print(X_train.shape)
print(y_train.shape)

#now same number of 0s and 1s
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

#param_grid={"C":np.logspace(-5,5,10), "penalty":["l2"],"solver":["newton-cg"]}

#just trial and error here 
param_grid={"C":[100, 1000, 1500, 2000, 2500, 5000], "penalty":["l2"],"solver":["newton-cg"]}


model = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='roc_auc',verbose=20,n_jobs=-1)
model.fit(X_train, y_train)

print(model.best_score_)
print(model.best_params_)
pred=model.predict_proba(X_test)
from sklearn.metrics import roc_auc_score

print(roc_auc_score(y_test,pred[:,1]))
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV


xgbmodel = XGBClassifier()

weights = [100, 500, 1000]
param_grid = dict(scale_pos_weight=weights)

xgbmodel = GridSearchCV(XGBClassifier(), param_grid, cv=5, scoring='roc_auc', verbose=10,n_jobs=-1)
xgbmodel.fit(X_train, y_train)

print(xgbmodel.best_score_)

pred=xgbmodel.predict(X_test)
print(roc_auc_score(y_test,pred))
# read csv file
test_csv = '/kaggle/input/minor-project-2020/test.csv'
testdf = pd.read_csv(test_csv)
#print(testdf.describe())
#print(testdf.head())
X_preds = testdf.drop('id',axis=1, inplace=False)
scaled_X_preds = scaler.transform(X_preds)

y_preds = model.predict_proba(scaled_X_preds)[:,1]

print(model)
print(y_preds.shape)

print(testdf.shape)
print(X_preds.shape)


ids = list(testdf['id'])
y_predlist = list(y_preds)
submission_dict = {'id':ids,'target':y_predlist}



submission = pd.DataFrame(sub_dict)
print(submission.head())
submission.to_csv('submission_4_C.csv',index=False)
