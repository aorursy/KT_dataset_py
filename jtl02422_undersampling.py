import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from numpy import mean
plt.style.use('ggplot')
data = pd.read_csv('../input/creditcardfraud/creditcard.csv')
print(data)
data.isnull().sum()
sb.countplot(x='Class', data=data)
data.Class[data.Class == 1].count()
data.Class[data.Class == 0].count()
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1,1))
data['Time'] = scaler.fit_transform(data['Time'].values.reshape(-1,1))
data.Time.describe()
train=data.sample(frac=0.8,random_state=200)
test=data.drop(train.index)
positives = train[train.Class == 1]
negatives = train[train.Class == 0]
negativeSample = negatives.sample(positives.Class.count())
df = pd.concat([negativeSample, positives], axis=0)
sb.countplot(x='Class', data=df)

X = df.drop('Class', axis=1)
y = df['Class']
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X, y)
X = test.drop('Class', axis=1)
y = test['Class']
score = clf.score(X, y)
print(score)