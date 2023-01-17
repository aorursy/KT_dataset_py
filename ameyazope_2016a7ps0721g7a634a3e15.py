import numpy as np # linear algebra

import pandas as pd 
np.random.seed(0)
train = pd.read_csv("/kaggle/input/eval-lab-2-f464/train.csv")

test = pd.read_csv("/kaggle/input/eval-lab-2-f464/test.csv")
pd.set_option('display.max_rows', 5400)

# "The maximum width in characters of a column"
train.head(120)
train.info()
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from scipy import stats

import matplotlib.pyplot as plt
df_tr = train

#TODO

from sklearn.preprocessing import MinMaxScaler

m = MinMaxScaler()

#m.fit_transform(df_tr)

X = df_tr.drop(['class'],axis=1)

y = train['class'].tolist()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#y=train['class']

#x=train.drop('class',axis=1)

##Oversampling

#from imblearn.over_sampling import SMOTE

#from imblearn.over_sampling import ADASYN

#import random



#oversampler=SMOTE(kind='regular',k_neighbors=3,random_state=random.randint(1,100000))



#x_resampled, y_resampled = oversampler.fit_resample(x, y)
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from datetime import datetime
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from datetime import datetime
param_test5={'booster':['gbtree'],

'eta':[0,0.02,0.04,0.06],

'gamma':np.arange(0,0.6,0.1),

'learning_rate':np.arange(0.01,0.07,0.01),

'n_estimators':[120,140,160,180],

'max_depth':range(3,7,1),

'min_child_weight':range(1,6,2),

'objective':['multi:softprob','multi:softmax'] }



gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier(), 

 param_grid = param_test5, cv = 3, n_jobs = -1,verbose=2)

gsearch1.fit(X,y)

gsearch1.best_params_
from xgboost import XGBClassifier

# fit model no training data

model = xgb.XGBClassifier(booster='gbtree',eta=0,gamma=0.4,learning_rate =0.03, max_depth=4,

 min_child_weight=3,n_estimators=140,objective='multi:softprob')

model.fit(X, y)

# make predictions for test data

y_pred = model.predict(X_test)

print(y_pred)

predictions = [round(value) for value in y_pred]

print(predictions)
y_pred = model.predict(test)

y_ans = [round(value) for value in y_pred]
import pandas as pd

c1=pd.DataFrame(test, columns=['id'])

c2 = pd.DataFrame(y_ans, columns = ['class'])

c2 = c2.astype(int)

data_frame = pd.merge(c1, c2, right_index=True, left_index=True)

data_frame
y_ans
data_frame.to_csv('submission.csv',columns=['id','class'],index=False)
import numpy as np

import pandas as pd
pd.set_option('display.max_rows', 5400)

# "The maximum width in characters of a column",to ensure we can see all data via .head() function

train = pd.read_csv("/kaggle/input/eval-lab-2-f464/train.csv")

test = pd.read_csv("/kaggle/input/eval-lab-2-f464/test.csv")
train.isnull().values.any()
test.isnull().values.any()
train.info()
test.info()
train = train.drop(['id'], axis=1)

tester = test.drop(['id'],axis=1)
#y=train['class']

#x=train.drop('class',axis=1)

##Oversampling

#from imblearn.over_sampling import SMOTE

#from imblearn.over_sampling import ADASYN

#import random



#oversampler=SMOTE(kind='regular',k_neighbors=3,random_state=random.randint(1,100000))



#x_resampled, y_resampled = oversampler.fit_resample(x, y)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()

X_lda = lda.fit_transform(train.drop(['class'],axis=1), train['class'])
afterLDATransform = pd.DataFrame(lda.transform(tester))

tester['lda_0'] = afterLDATransform[0]

tester['lda_1'] = afterLDATransform[1]

tester['lda_2'] = afterLDATransform[2]

tester['lda_3'] = afterLDATransform[3]

tester['lda_4'] = afterLDATransform[4]
tester.head()
new_features = pd.DataFrame(lda.transform(train.drop(['class'],axis=1)))

train['lda_0'] = new_features[0]

train['lda_1'] = new_features[1]

train['lda_2'] = new_features[2]

train['lda_3'] = new_features[3]

train['lda_4'] = new_features[4]
train.head()
train['factor'] = (train['chem_3']*train['attribute']*train['chem_4'])

tester['factor'] = (tester['chem_3']*tester['attribute']*tester['chem_4'])
X = train.drop(['class'],axis=1)

y = train['class']
import matplotlib.pyplot as plt

import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))

corr = train.corr()

sns.heatmap(corr, center=0)
import matplotlib.pyplot as plt

import seaborn as sns

# Compute the correlation matrix

corr = train.corr(method="kendall")



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0.5,

            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)



plt.show()
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()

X_lda = lda.fit_transform(X, y)
#Constructing a scatter plot of the data

plt.scatter(X_lda[:,0],X_lda[:,1],c=y,cmap='rainbow',alpha=0.7,edgecolors='g')
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

trainScaled = train.copy()

testScaled = tester.copy()

columns = tester.columns

trainScaled[columns]=scaler.fit_transform(train[columns])

testScaled[columns]=scaler.transform(tester[columns])
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()

X_lda = lda.fit_transform(X, y)

plt.xlabel('row0')

plt.ylabel('row1')

plt.scatter(X_lda[:,0],X_lda[:,1],c=y,cmap='rainbow',alpha=0.7,edgecolors='g')
trainScaled.head()
testScaled.head()
from sklearn.decomposition import PCA



model=PCA(n_components=2)

fittedData = model.fit(trainScaled.drop('class',axis=1)).transform(trainScaled.drop('class',axis=1))
plt.figure(figsize=(10,8))

plt.xlabel('row0')

plt.ylabel('row1')

plt.legend()

plt.scatter(fittedData[:,0],fittedData[:,1],label = train['class'],c=train['class'])

plt.show()
x = trainScaled.drop(['class'],axis=1)

y = trainScaled['class']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import accuracy_score



clf = ExtraTreesClassifier(n_estimators=1000, random_state=42, class_weight="balanced")

clf.fit(x_train,y_train)

y_ans = clf.predict(x_test)

accuracy_score(y_test, y_ans)
clf = ExtraTreesClassifier(n_estimators=1000, random_state=42, class_weight="balanced")

clf.fit(x,y)

test['class'] = clf.predict(testScaled)

data_frame = test[['id','class']]

data_frame.info()
data_frame.head(120)
data_frame.to_csv('submission.csv',columns=['id','class'],index=False)