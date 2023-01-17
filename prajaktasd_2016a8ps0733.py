import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

data1 = pd.read_csv("train_lab3.csv")
data2 = pd.read_csv("test_data.csv")
data2.info()
data1.head()
data1.describe()
data1.info()
data1.isnull().values.any()
data1.duplicated().sum()
data1['castleTowerDestroys'].unique()
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))

corr = data1.corr()

sns.heatmap(corr, center=0)
data = data1.drop(data1.columns[[0]], axis=1)

data_test = data2.drop(data2.columns[[0]], axis=1)
data_test.describe()
data.describe()
data.info()
data_test.info()
X=data.drop('bestSoldierPerc',axis=1)



y=data['bestSoldierPerc']

X.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
#             Decision Tree Model

#Finding optimum depth using Elbow Method

from sklearn.tree import DecisionTreeClassifier

error_rate_train = []

for i in range(1,50):

    dTree = DecisionTreeClassifier(max_depth=i)

    dTree.fit(X_train,y_train)

    pred_i = dTree.predict(X_train)

    error_rate_train.append(np.mean(pred_i != y_train))

    error_rate_test = []

for i in range(1,50):

    dTree = DecisionTreeClassifier(max_depth=i)

    dTree.fit(X_train,y_train)

    pred_i = dTree.predict(X_test)

    error_rate_test.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

train_score,=plt.plot(range(1,50),error_rate_train,color='blue', linestyle='dashed')

test_score,=plt.plot(range(1,50),error_rate_test,color='red',linestyle='dashed')

plt.legend( [train_score,test_score],["Train Error","Test Error"])

plt.title('Error Rate vs. max_depth')

plt.xlabel('max_depth')

plt.ylabel('Error Rate')
# Fitting Decision Tree model to the Training set with max_depth 8

dTree = DecisionTreeClassifier(max_depth=100)

dTree.fit(X_train,y_train)
#   Mean Accuracy

dTree.score(X_test,y_test)
# Predicting the Test Results

y_pred=dTree.predict(X_test)

# Predicting the Test Results

y_dt=dTree.predict(data_test)

y_dt
soldierId = data_test['soldierId']

a = np.array(soldierId)

a = a.astype(str)
p = [a,y_dt]

p = p

df = pd.DataFrame(p)

df = df.transpose()

df.columns = ['soldierId','bestSoldierPerc']

df.astype(str)

df.info()
df
df.to_csv("solution111.csv", index = False)
import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings("ignore")
train=pd.read_csv('train_lab3.csv')

test=pd.read_csv('test_data.csv')
train.head(10)
test.head()
train['knockedOutSoldiers'] = train['knockedOutSoldiers'].fillna(0)
train['respectEarned'] = train['respectEarned'].fillna(np.mean(train['respectEarned']))
train['respectEarned'] = round(train['respectEarned'])
train_new.head()
train.shape
test.shape
train.dtypes



train.shape



train_new = train



train_new.shape



train.shape



test.head()



test_new = test



test_new.shape



test_new.head()



test_new.shape
cols = ['soldierId', 'bestSoldierPerc']

cols_test = ['soldierId']

random_rows=list(np.random.random_integers(0, train.shape[0], 50000))

train_new = train.iloc[random_rows,:]

train_new.shape
X = train_new.drop(cols,axis=1)

X.shape



X_t = test.drop(cols_test,axis=1)

X_t.shape
X.head()

y = train_new['bestSoldierPerc']
y.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train.shape
import xgboost as xgb

from sklearn.metrics import make_scorer

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error

clsf = xgb.XGBRegressor(n_estimators=100, learning_rate=0.09, gamma = 0, subsample=1,

                         colsample_bytree=1, max_depth=15, min_child_weight = 1).fit(X_train, y_train, sample_weight=None)
predictions = clsf.predict(X_test)
np.mean(abs(predictions-y_test))
feature_importance=clsf.feature_importances_

len(feature_importance)

#list_nouse = []

#cols_of_value = []

for i in range(X.shape[1]):

   # if i < 25 :

    #    continue

    #if feature_importance[i] < 0.02:

     #   list_nouse.append(X.columns[i])

    #else:

     #   cols_of_value.append(X.columns[i])

    print(X.columns[i],':\t',feature_importance[i])
X_t.shape



test_new.shape
predtest = clsf.predict(X_t)

predtest


b=test.soldierId



sol=pd.DataFrame()



b.astype(float)
sol['soldierId']=b.astype(float)



pred= pd.DataFrame(predtest)


pred.head()
sol['bestSoldierPerc']=round(pred.iloc[:,0]).astype(int)

sol.head()
sol.shape



test.shape

#sol['soldierId']=sol['soldierId'].astype(str)
sol.info()
sol.to_csv('submission_file909.csv',index=False)