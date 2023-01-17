import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras import Sequential
data = pd.read_csv('../input/creditcardfraud/creditcard.csv')
data.head(3)
data.info()
data.describe().transpose()
data.shape
data.isnull().sum()
data.var()
plt.figure(figsize = (10,8))
sns.countplot(data=data, x='Class')
plt.figure(figsize = (10,6))
sns.boxplot(x = 'Class', y = 'Amount', data = data, hue = 'Class')
corr = data.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr)
sns.lmplot(x = 'Amount', y = 'V2', data = data, hue = 'Class', palette = 'Set1', fit_reg=True, scatter_kws={'edgecolor':'white', 'alpha':0.6, 'linewidths':1})
sns.lmplot(x = 'Amount', y = 'V5', data = data, hue = 'Class', palette = 'magma', fit_reg=True, scatter_kws={'edgecolor':'white', 'alpha':0.6, 'linewidths':1})
sns.lmplot(x = 'Amount', y = 'V7', data = data, hue = 'Class', palette = 'Dark2', fit_reg=True, scatter_kws={'edgecolor':'white', 'alpha':0.6, 'linewidths':1})
sns.lmplot(x = 'Amount', y = 'V20', data = data, hue = 'Class', palette = 'Dark2_r', fit_reg=True, scatter_kws={'edgecolor':'white', 'alpha':0.6, 'linewidths':1})
amount = data['Amount'].values
time = data['Time'].values
fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize=(18, 4))
sns.distplot(amount, ax = ax1, color = 'r', hist_kws={'edgecolor':'black'})
ax1.set_title('Amount')
sns.distplot(time, ax = ax2, color = 'b', hist_kws={'edgecolor':'black'})
ax2.set_title('Time')
data['Class'].value_counts()
data = data.sample(frac=1)
fraud = data.loc[data['Class'] == 1]
not_fraud = data.loc[data['Class'] == 0][ : 492]
data = pd.concat([fraud, not_fraud])
data = data.sample(frac=1, random_state=123)
data.head(3)
data.shape
plt.figure(figsize = (10,8))
sns.countplot(data=data, x='Class')
X = data.drop(['Class'], axis = 1).values
y = data['Class'].values
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state = 123)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print('Classification Report\n', classification_report(y_test, y_pred))
print('Confusion Matrix\n', confusion_matrix(y_test, y_pred))
print('Training Score : ', lr.score(X_train, y_train))
print('Test Score : ', lr.score(X_test, y_test))
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print('Classification Report\n', classification_report(y_test, y_pred))
print('Confusion Matrix\n', confusion_matrix(y_test, y_pred))
print('Training Score : ', svc.score(X_train, y_train))
print('Test Score : ', svc.score(X_test, y_test))
dt = DecisionTreeClassifier(max_depth= 4, min_samples_leaf= 4, random_state=123 )
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print('Classification Report\n', classification_report(y_test, y_pred))
print('Confusion Matrix\n', confusion_matrix(y_test, y_pred))
print('Training Score : ', dt.score(X_train, y_train))
print('Test Score : ', dt.score(X_test, y_test))
rfc = RandomForestClassifier(max_depth= 6, min_samples_leaf= 6, random_state=123)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print('Classification Report\n', classification_report(y_test, y_pred))
print('Confusion Matrix\n', confusion_matrix(y_test, y_pred))
print('Training Score : ', rfc.score(X_train, y_train))
print('Test Score : ', rfc.score(X_test, y_test))
adb = AdaBoostClassifier(base_estimator= rfc, n_estimators=50, random_state=123)
adb.fit(X_train, y_train)
y_pred = adb.predict(X_test)
print('Classification Report\n', classification_report(y_test, y_pred))
print('Confusion Matrix\n', confusion_matrix(y_test, y_pred))
print('Training Score : ', adb.score(X_train, y_train))
print('Test Score : ', adb.score(X_test, y_test))
gdb = GradientBoostingClassifier(n_estimators= 50, max_depth= 2, min_samples_leaf= 10, random_state=123)
gdb.fit(X_train, y_train)
y_pred = gdb.predict(X_test)
print('Classification Report\n', classification_report(y_test, y_pred))
print('Confusion Matrix\n', confusion_matrix(y_test, y_pred))
print('Training Score : ', gdb.score(X_train, y_train))
print('Test Score : ', gdb.score(X_test, y_test))
model = Sequential()
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
early_stop = EarlyStopping(monitor='val_loss', patience= 2, verbose= 0, mode = 'min')
model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs= 20, validation_data=(X_test, y_test), callbacks= [early_stop])
loss = pd.DataFrame(model.history.history)
loss.plot()