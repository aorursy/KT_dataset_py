

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head(2)
train_df.info()
train_df.dropna(axis=0, subset=['Embarked'], inplace=True)
y = train_df['Survived']
#parch - ???

np.unique(train_df['Parch'])
real_features = ['Age', 'Fare']

categorial_features = ['Sex', 'Pclass', 'SibSp', 'Parch', 'Cabin', 'Embarked']
X_num = train_df[real_features]

X_test_num = test_df[real_features]
age_mean = np.round(((X_num['Age'].mean() + X_test_num['Age'].mean())/2), 3)

print(age_mean)
# filling empty 'age' with mean value:

X_num.fillna(age_mean, inplace=True)

X_test_num.fillna(age_mean, inplace=True)
X_cat = train_df[categorial_features]

X_for_test_cat = test_df[categorial_features]
X_cat.info(), X_for_test_cat.info()
X_cat.fillna('Z', inplace=True)

X_for_test_cat.fillna('Z', inplace=True)
X_cat.head(4)
#took only first symbol of cabin number:

cabin_id_train = [x[0] for x in X_cat['Cabin']]

cabin_id_test = [x[0] for x in X_for_test_cat['Cabin']]
print(len(cabin_id_train), len(cabin_id_test))
X_cat['Cabin_id'] = cabin_id_train

X_for_test_cat['Cabin_id'] = cabin_id_test
X_cat.drop(['Cabin'], axis=1, inplace=True)

X_for_test_cat.drop(['Cabin'], axis=1, inplace=True)
from sklearn.feature_extraction import DictVectorizer as DV

encoder = DV(sparse=False)
X_cat_encoded = encoder.fit_transform(X_cat.T.to_dict().values())

X_test_cat_encoded = encoder.fit_transform(X_for_test_cat.T.to_dict().values())
print(X_cat.shape, X_cat_encoded.shape)

print(X_for_test_cat.shape, X_test_cat_encoded.shape)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_num, y)

scaler.fit(X_test_num)

X_num_scaled = scaler.transform(X_num)

X_test_num_scaled = scaler.transform(X_test_num)
from sklearn.model_selection import train_test_split
(X_train_num, X_test_num, 

 y_train, y_test) = train_test_split(X_num_scaled, y, test_size=0.25, random_state=123)



(X_train_cat, X_test_cat) = train_test_split(X_cat_encoded, test_size=0.25, random_state=123)
X_train = np.hstack((X_train_num, X_train_cat))

X_test = np.hstack((X_test_num, X_test_cat))



# and for full submission:

X_full_test = np.hstack((X_test_num_scaled, X_test_cat_encoded))
from sklearn.ensemble import ExtraTreesClassifier

model_ETC = ExtraTreesClassifier()
model_ETC.fit(X_train, y_train)

print(model_ETC.feature_importances_)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

model_LR = LogisticRegression()
from sklearn.neighbors import KNeighborsClassifier

model_KN = KNeighborsClassifier()
new_cat_features = ['Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked'] 

X_cat_2 = X_cat[new_cat_features]

X_for_test_cat_2 = X_for_test_cat[new_cat_features]
X_cat_2_encoded = encoder.fit_transform(X_cat_2.T.to_dict().values())

X_test_cat_encoded_2 = encoder.fit_transform(X_for_test_cat_2.T.to_dict().values())
(X_train_cat, X_test_cat) = train_test_split(X_cat_2_encoded, test_size=0.25, random_state=123)
X_train = np.hstack((X_train_num, X_train_cat))

X_test = np.hstack((X_test_num, X_test_cat))



# and for full submission:

X_full_test = np.hstack((X_test_num_scaled, X_test_cat_encoded_2))
model_LR.fit(X_train, y_train)

prediction_LR = model_LR.predict(X_test)

print(metrics.accuracy_score(y_test, prediction_LR))
model_KN.fit(X_train, y_train)

prediction_KN = model_KN.predict(X_test)

print(metrics.accuracy_score(y_test, prediction_KN))
prediction_full = model_KN.predict(X_full_test)

answer = pd.DataFrame({'PassengerId':test_df['PassengerId'], 'Survived':prediction_full})

answer.to_csv('prediction_titanic_knn1.csv', index=False)

answer.head()
model_ETC.fit(X_train, y_train)

print(model_ETC.feature_importances_)
model_LR.fit(X_train_num, y_train)

prediction_LR = model_LR.predict(X_test_num)

print(metrics.accuracy_score(y_test, prediction_LR))
model_KN.fit(X_train_num, y_train)

prediction_KN = model_KN.predict(X_test_num)

print(metrics.accuracy_score(y_test, prediction_KN))
cat_features_3 = ['Pclass', 'SibSp', 'Parch']
X_cat_3 = X_cat[cat_features_3]

X_for_test_cat_3 = X_for_test_cat[cat_features_3]



X_cat_3_encoded = encoder.fit_transform(X_cat_3.T.to_dict().values())

X_test_cat_encoded_3 = encoder.fit_transform(X_for_test_cat_3.T.to_dict().values())



(X_train_cat, X_test_cat) = train_test_split(X_cat_3_encoded, test_size=0.25, random_state=123)
X_train = np.hstack((X_train_num, X_train_cat))

X_test = np.hstack((X_test_num, X_test_cat))



# and for full submission:

X_full_test = np.hstack((X_test_num_scaled, X_test_cat_encoded_3))
model_LR_3 = LogisticRegression()

model_LR_3.fit(X_train, y_train)

prediction_LR_3 = model_LR_3.predict(X_test)

print (metrics.accuracy_score(y_test, prediction_LR_3))