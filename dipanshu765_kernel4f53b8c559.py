import numpy as np
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from xgboost import XGBClassifier
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df.head()
plt.figure(figsize=(16,6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
df.info()
df.describe()
fraud_num = len(df[df['Class']==1])
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr())
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
sns.boxplot(df['Time'], ax=axes[2])
sns.countplot(x='Class', data=df, ax=axes[0])
sns.distplot(df['Amount'], ax=axes[1], color='r')
plt.tight_layout()
sns.set_style('darkgrid')
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Time', y='Amount', data=df[df['Class']==1])                                                           #Fraudulent points
sns.set_style('darkgrid')
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Time', y='Amount', data=df[df['Class']==0])                                                          #Genuine Points
sns.lmplot(y='Amount',x='Time',data=df,hue='Class',
           palette='Set1')
def equalsplit(df1, m):
    fraud_df = df1[df1['Class']==1] 
    genuine_df = df1[df1['Class']==0]
    fraud_train = fraud_df.sample(n=int(len(fraud_df)*m))
    genuine_train = genuine_df.sample(n=int(len(genuine_df)*m)) 
    fraud_test = fraud_df.drop(fraud_train.index, axis=0)
    genuine_test = genuine_df.drop(genuine_train.index, axis=0) 
    train = pd.concat([fraud_train, genuine_train], axis=0)
    test = pd.concat([fraud_test, genuine_test], axis=0)
    return [train, test]
[train, test] = equalsplit(df, 0.6)
[test, cv] = equalsplit(test, 0.5)
X_Train = train.drop('Class', axis=1)
y_train = train['Class']
X_Test = test.drop('Class', axis=1)
y_test = test['Class']
X_CV = cv.drop('Class', axis=1)
y_cv = cv['Class']
scaler = StandardScaler()
scaler.fit(X_Train)
X_Train = pd.DataFrame(scaler.transform(X_Train), columns=train.drop('Class', axis=1).columns)
scaler.fit(X_Test)
X_Test = pd.DataFrame(scaler.transform(X_Test), columns=test.drop('Class', axis=1).columns)
scaler.fit(X_CV)
X_CV = pd.DataFrame(scaler.transform(X_CV), columns=cv.drop('Class', axis=1).columns)
smote = SMOTE(sampling_strategy=0.25)
X_sm, y_sm = smote.fit_sample(X_Train, y_train)
gs_lr = GridSearchCV(
            estimator=LogisticRegression(max_iter=150),
            param_grid={'C': (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 1.0)}
        )
pipeline_lr = make_pipeline(SMOTE(sampling_strategy=0.25), gs_lr)
model_lr = pipeline_lr.fit(X_Train, y_train)
best_lr = gs_lr.best_estimator_

prediction = best_lr.predict(X_Train)
print(classification_report(y_train, prediction))
def bool_to_binary(Y):
    for i in range(len(Y)):
        if Y[i]:
            Y[i]=1
        else:
            Y[i]=0
    return Y
pred_prob_lr = best_lr.predict_proba(X_CV)
for threshold in np.linspace(0.1,1,30):
    print('\n\n\n')
    print(threshold)
    print(classification_report(y_cv, bool_to_binary(list(pred_prob_lr[:, 1]>threshold))))
    print('\n\n\n')
    print('-'*50)
threshold_lr = 0.968
pred_prob_lr = best_lr.predict_proba(X_Test)
print(classification_report(y_test, bool_to_binary(list(pred_prob_lr[:, 1]>threshold_lr))))
import xgboost
#gs_xgb = GridSearchCV(XGBClassifier(num_class=2, objective='multi:softmax'), 
#                     param_grid={
#                         'max-depth': (2,4,6,8),
#                         'eta': (0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003),
#                     },
#                     verbose=3)


#pipeline_xgb = make_pipeline(SMOTE(sampling_strategy=0.25), gs_xgb)
#model_xgb = pipeline_xgb.fit(X_Train, y_train)
eta = 0.3
max_depth = 6
best_xgb = XGBClassifier(eta=eta, max_depth=max_depth)
best_xgb.fit(X_sm, y_sm)
print(classification_report(y_sm, best_xgb.predict(X_sm)))
print(classification_report(y_test, best_xgb.predict(X_Test)))
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_sm, y_sm)
print(classification_report(y_cv, rfc.predict(X_CV)))
model = Sequential()
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
epochs=500 
early_stop=EarlyStopping(monitor='val_loss', mode='min', patience=20)
model.fit(x=np.array(X_sm), 
          y=np.array(y_sm), 
          epochs=epochs,
          batch_size=256,
          validation_data=(X_CV, y_cv),
         callbacks=[early_stop])
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
print(classification_report(y_cv, model.predict_classes(X_CV)))
