# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
%matplotlib inline
pd.set_option("max_rows", 10)
np.set_printoptions(suppress=True)

from seaborn import set_style
set_style("darkgrid")
import seaborn as sns
import matplotlib.pyplot as plt
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
train.info()
train.describe()
train.columns.equals(test.columns)
train.columns.difference(test.columns)
train[["Id", "Response"]]

type(train[["Response"]])
type(train["Response"])
ax = train.groupby("Response").size().plot(kind="barh", figsize=(8, 8))

# ax.set_xticklabels([])  # turn off x tick labels

# resize y label
ylabel = ax.yaxis.get_label()
ylabel.set_fontsize(24)

# resize x tick labels
labels = ax.yaxis.get_ticklabels()
[label.set_fontsize(20) for label in labels];

# resize y tick labels
labels = ax.xaxis.get_ticklabels()
[label.set_fontsize(20) for label in labels]
[label.set_rotation(-45) for label in labels];
g = sns.factorplot("Ins_Age", "BMI", hue="Response", col="Product_Info_2", data=train)
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

# combine train and test
all_data = train.append(test)

# Found at https://www.kaggle.com/marcellonegro/prudential-life-insurance-assessment/xgb-offset0501/run/137585/code
# create any new variables    
all_data['Product_Info_2_char'] = all_data.Product_Info_2.str[0]
all_data['Product_Info_2_num'] = all_data.Product_Info_2.str[1]

# factorize categorical variables
all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]
all_data['Product_Info_2_char'] = pd.factorize(all_data['Product_Info_2_char'])[0]
all_data['Product_Info_2_num'] = pd.factorize(all_data['Product_Info_2_num'])[0]

all_data['BMI_Age'] = all_data['BMI'] * all_data['Ins_Age']

med_keyword_columns = all_data.columns[all_data.columns.str.startswith('Medical_Keyword_')]
all_data['Med_Keywords_Count'] = all_data[med_keyword_columns].sum(axis=1)

all_data.fillna(-1, inplace=True)

# fix the dtype on the label column
all_data['Response'] = all_data['Response'].astype(int)

# split train and test
train = all_data[all_data['Response']>0].copy()
test = all_data[all_data['Response']<1].copy()

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, train['Response'], test_size=0.20, random_state=1)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe_lr = Pipeline([('scl', StandardScaler()),
            ('clf', LogisticRegression(random_state=1))])

pipe_lr.fit(X_train, y_train)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))
y_pred = pipe_lr.predict(X_test)
submission = pd.DataFrame({
        "Id": X_test["Id"],
        "Response": y_pred
    })
submission.to_csv('prudential_logisticRegression.csv', index=False)
import numpy as np
from sklearn.cross_validation import StratifiedKFold

kfold = StratifiedKFold(y=y_train, 
                        n_folds=10,
                        random_state=1)


from sklearn.preprocessing import Imputer
from sklearn.cross_validation import StratifiedKFold

imp = Imputer(missing_values='NaN', strategy='median', axis=1) 
imp.fit(y_train)

y_train = imp.fit_transform(y_train)
kfold = StratifiedKFold(y=y_train, 
                        n_folds=10,
                        random_state=1)
scores = []
for k, (traink, testk) in enumerate(kfold):
    a = X_train.iloc[traink]

    b = y_train[traink]
    c = X_train.iloc[testk]
    d = y_train[testk]
    pipe_lr.fit(a,b)
    score = pipe_lr.score(c,d)
    scores.append(score)
    print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1, np.bincount(y_train[train]), score))
    
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))