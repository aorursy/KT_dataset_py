import numpy as np 

import pandas as pd 

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.preprocessing import Normalizer

from sklearn.model_selection import train_test_split
train = pd.read_csv('../input/Train.csv')

test = pd.read_csv('../input/Test.csv')
train.describe()
test.describe()
train.isnull().sum()
test.isnull().sum()
#plotting correlations

num_feat=train.columns[train.dtypes!=object]

num_feat=num_feat[1:-1] 

labels = []

values = []

for col in num_feat:

    labels.append(col)

    values.append(np.corrcoef(train[col].values, train.target.values)[0,1])



ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots(figsize=(12,8))

rects = ax.barh(ind, np.array(values), color='red')

ax.set_yticks(ind+((width)/2.))

ax.set_yticklabels(labels, rotation='horizontal')

ax.set_xlabel("Correlation coefficient")

ax.set_title("Correlation Coefficients w.r.t target");
#Heatmap

corrMatrix=train[num_feat].corr()

sns.set(font_scale=1.10)

plt.figure(figsize=(10, 10))

sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,

            square=True,annot=True,cmap='viridis',linecolor="white")

plt.title('Correlation between features');
plt.hist((train.target),bins=152)

plt.show()
# Let's see how the numeric data is distributed.



train.hist(bins=10, figsize=(20,15), color='#E14906')

plt.show()
combined = train.append(test)

combined.reset_index(inplace=True)

combined.describe()
y = combined['target']

x = combined.drop('target', axis=1)
y.describe()
x.describe()
X_train, X_test, y_train, y_test = train_test_split(x, y ,test_size=0.2)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
# Implimenting Linear Regression from inbuilt function of sklearn

from sklearn.linear_model import LinearRegression

reg=LinearRegression(fit_intercept=True)

model = reg.fit(X_train,y_train)

predict = model.predict(X_test)
#Predicting Score

from sklearn.metrics import r2_score

print(r2_score(y_test,predict))
print("Training Score %.4f"%reg.score(X_train,y_train))

print("Testing Score %.4f"%reg.score(X_test,y_test))
from sklearn.model_selection import cross_val_score
scores=cross_val_score(reg,X_train,y_train,cv=10,scoring='r2')
print(scores)
print(scores.mean())
print(scores.std())
scores=cross_val_score(reg,X_train,y_train,cv=10,scoring='neg_mean_squared_error')
print(scores)
#Average loss

print(scores.mean())
print(scores.std())
y_pred = reg.predict(X_test)

plt.figure(figsize=(10, 5))

plt.scatter(y_test, y_pred, s=20)

plt.title('Predicted vs. Actual')

plt.xlabel('Actual target')

plt.ylabel('Predicted target')



plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],'r')

plt.tight_layout()
y_pred.shape
print("predicted values for 400 test data are", y_pred)
df = pd.DataFrame(y_pred)

df.index.names = ["index"]
df.columns = ["Predicted values"]
df.head()
df.to_csv('predictedres.csv')