import xgboost
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import cross_validation, metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
train = pd.read_excel("../input/research_student (1).xlsx")


train.head()
train.info
train.describe()
train = train.drop([0,221,222])
train.head()
train.Branch.value_counts()
train = train.fillna(0)

train[['Branch','Gender']]
train.fillna(0)
train.columns
scale_list = [ 'Marks[10th]', 'Marks[12th]',
       'GPA 1', 'Rank', 'Normalized Rank', 'CGPA',
       'Current Back', 'Ever Back', 'GPA 2', 'GPA 3', 'GPA 4', 'GPA 5',
       'GPA 6', 'Olympiads Qualified', 'Technical Projects', 'Tech Quiz',
       'Engg. Coaching', 'NTSE Scholarships', 'Miscellany Tech Events']
sc = train[scale_list]
      
sc.head()
sc.tail()
sc=sc.fillna(0)
scaler = StandardScaler()
sc = scaler.fit_transform(sc)
train[scale_list] = sc
train[scale_list].head()
train.head()

train.info()

encoding_list = ['Branch','Gender','Board[10th]','Board[12th]','Category']
train[encoding_list] = train[encoding_list].apply(LabelEncoder().fit_transform)
train.head()
train.info()
train.head()
y = train['CGPA']
x = train.drop('CGPA', axis=1)
x.info()
y.info
X_train, X_test, y_train, y_test = train_test_split(x, y ,test_size=0.3)
X_train.shape
X_test.shape
logreg=LinearRegression()
logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)
y_test
print(metrics.mean_squared_error(y_test, y_pred))
xgb = xgboost.XGBRegressor(n_estimators=2500, learning_rate=0.06, gamma=0, subsample=0.6,
                           colsample_bytree=0.7, min_child_weight=4, max_depth=3)
xgb.fit(X_train,y_train)

X_train.head()
train.head()
train.info()
train.plot(kind="scatter",x="Marks[12th]",y="CGPA")
train.plot(kind="scatter",x="Rank",y="CGPA")
train.head()
train.plot(kind="scatter",x="GPA 2",y="CGPA")
train.plot(kind="scatter",x="GPA 4",y="CGPA")
train.plot(kind="scatter",x="Marks[10th]",y="Marks[12th]")
train
a = np.random.random((16, 16))
plt.imshow(a, cmap="Marks[10th]", interpolation="CGPA")
plt.show()
plt.hist("Marks[10th]",bins=56,histtype="bar",rwidth=0.8)



