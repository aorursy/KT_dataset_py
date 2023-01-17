import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.model_selection import train_test_split,cross_val_score



wine = pd.read_csv('../input/winequality-red.csv')



bins = (2,4,6,8)

group_names = ['average','bad' ,'good']

wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)



label_quality = LabelEncoder()



wine['quality'] = label_quality.fit_transform(wine['quality'])



wine['quality'].value_counts()



X = wine.drop('quality', axis = 1)

y = wine['quality']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



sc = StandardScaler()



X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)



rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(X_train, y_train)

pred_rfc = rfc.predict(X_test)



print(classification_report(y_test, pred_rfc))



rfc_eval = cross_val_score(estimator = rfc, X = X_train, y = y_train, cv = 20)

rfc_eval.mean()