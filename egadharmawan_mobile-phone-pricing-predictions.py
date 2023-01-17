import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
train = pd.read_csv('../input/mobile-price-classification/train.csv')
test = pd.read_csv('../input/mobile-price-classification/test.csv')
train.head()
train.info()
test.head()
test.info()
# 0 for low cost
# 1 for medium cost
# 2 for high cost
# 3 for very high cost

train.price_range.value_counts()
## EDA
# Distribution

train.hist(bins=30, figsize=(15, 15))
# Most important feature

Corr = train.corr()

IF = Corr['price_range'].sort_values(ascending=False).head(10).to_frame()
IF.head(5)
f = plt.figure(figsize=(15,12))

# corr with ram
ax = f.add_subplot(221)
ax = sns.scatterplot(x="price_range", y="ram", color='b', data=train)
ax.set_title('Corr with RAM')

# corr with Battery
ax = f.add_subplot(222)
ax = sns.scatterplot(x="price_range", y="battery_power", color='c', data=train)
ax.set_title('Corr with battery')

# corr with px_width
ax = f.add_subplot(223)
ax = sns.scatterplot(x="price_range", y="px_width", color='r', data=train)
ax.set_title('Corr with px width')

# corr with height
ax = f.add_subplot(224)
ax = sns.scatterplot(x="price_range", y="px_height", color='g', data=train)
ax.set_title('Corr with px height')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Split Data

X = train.drop('price_range', axis=1)
y = train['price_range']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
print('X_train : ' + str(X_train.shape))
print('X_test : ' + str(X_test.shape))
print('y_train : ' + str(y_train.shape))
print('y_test : ' + str(y_test.shape))
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 12)
classifier.fit(X_train, y_train)

# predict
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
import xgboost as xgb

# Instantiate the XGBClassifier: xg_cl
xg_cl = xgb.XGBClassifier(objective='multi:softmax', num_class=3, n_estimators=150, seed=123)
xg_cl.fit(X_train, y_train)
preds = xg_cl.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svm = SVC()

parameters = {'C':[0.1, 1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
model = GridSearchCV(svm, param_grid=parameters)
model.fit(X_train, y_train)

# Best parameters
print("Best CV params", model.best_params_)

# accuracy
print("Test accuracy :", model.score(X_test, y_test))
## Assign the value

predicted_value = model.predict(X_test)
actual_value = y
## COmparison distribution in Train data

sns.distplot(actual_value, hist=False, label="Actual Values")
sns.distplot(predicted_value, hist=False, label="Predicted Values")
plt.title('Distribution Comaprison with SVM')
plt.show()
test.head()
X2 = test.drop('id', axis=1)
## Perform predictions

predicted_test_value = model.predict(X2)
pd.value_counts(predicted_test_value)
# Here is how distribution look like in test data price range

sns.distplot(predicted_value, hist=False, label="Predicted Values")
plt.show()