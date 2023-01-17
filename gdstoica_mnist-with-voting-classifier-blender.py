

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import validation_curve

from sklearn.ensemble import VotingClassifier

from sklearn.calibration import CalibratedClassifierCV

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_val_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")



X = train_val_data.iloc[:, 1:]

y = train_val_data.iloc[:, 0]



print(X.shape)

print(y.shape)

print(X.describe())

print(train_val_data.describe())

print(train_val_data['label'].value_counts())

print(y.value_counts())
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)
from sklearn.decomposition import PCA

import numpy as np



pca = PCA(random_state=15)

pca.fit(X)

cumsum = np.cumsum(pca.explained_variance_ratio_)

d = np.argmax(cumsum >= 0.95)+1

print(d)

pca = PCA(n_components=d, random_state=15)

X_reduced = pca.fit_transform(X)
X_train, X_1, y_train, y_1 = train_test_split(X_reduced, y, test_size=0.3, random_state=15)

print(X_1.shape)

print(y_1.shape)
print(type(X_1))
X_val_test = np.split(X_1, 2)

y_val_test = np.split(y_1, 2)

X_val = X_val_test[0]

X_test = X_val_test[1]

y_val = y_val_test[0]

y_test = y_val_test[1]



print(X_val.shape)

print(y_val.shape)

print(X_test.shape)

print(y_test.shape)
model_svc = LinearSVC(random_state=15)

cclf = CalibratedClassifierCV(model_svc)  #added in order to have predict_proba() for LinearSVC

model_rf = RandomForestClassifier(n_estimators=250, random_state=15)

model_et = ExtraTreesClassifier(n_estimators=250, random_state=15)



from sklearn.metrics import accuracy_score



# voting_clf = VotingClassifier(

#     estimators=[('linearsvc', model_svc), ('rf', model_rf), ('et', model_et)],

#     voting='hard')



# voting_clf = VotingClassifier(

#     estimators=[('calibratedclf', cclf), ('rf', model_rf), ('et', model_et)],

#     voting='soft')



# voting_clf.fit(X_train, y_train)

# y_pred = voting_clf.predict(X_val)

# print(accuracy_score(y_val, y_pred))
# for clf in (model_svc, model_rf, model_et, voting_clf):

#     clf.fit(X_train, y_train)

#     y_pred = clf.predict(X_val)

#     print(clf.__class__.__name__, accuracy_score(y_val, y_pred))
for clf in (model_svc, model_rf, model_et):

    clf.fit(X_train, y_train)



y_pred_val_svc = model_svc.predict(X_val)

y_pred_val_rf = model_rf.predict(X_val)

y_pred_val_et = model_et.predict(X_val)



print(y_pred_val_svc.shape)

print(y_pred_val_rf.shape)

print(y_pred_val_et.shape)

print(y_val.shape)

columns=['SVC', 'RF', 'ET']

data = np.column_stack((y_pred_val_svc.T, y_pred_val_rf.T,y_pred_val_et.T))

df = pd.DataFrame(data,columns=columns )

print(df.shape)
blender = RandomForestClassifier(n_estimators=1000,oob_score=True, random_state=15)

blender.fit(df, y_val)
print('Score: ', blender.score(df, y_val))
y_test_svc = model_svc.predict(X_test)

y_test_rf = model_rf.predict(X_test)

y_test_et = model_et.predict(X_test)



columns=['SVC', 'RF', 'ET']

test_data = np.column_stack((y_test_svc.T, y_test_rf.T,y_test_et.T))

test_df = pd.DataFrame(test_data,columns=columns )

y_pred = blender.predict(test_df)

print(classification_report(y_test, y_pred))
