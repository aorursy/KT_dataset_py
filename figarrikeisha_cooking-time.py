import numpy as np
import pandas as pd 
df1 = pd.read_json('../input/train.json')
df2 = pd.read_json('../input/test.json')
df1['ingredients1'] = df1['ingredients'].apply(','.join)
df2['ingredients2'] = df2['ingredients'].apply(','.join)
from sklearn.feature_extraction.text import TfidfVectorizer
vector = TfidfVectorizer(binary=True).fit(df1['ingredients1'].values)

X_train = vector.transform(df1['ingredients1'].values)
X_train = X_train.astype('float')
X_test = vector.transform(df2['ingredients2'].values)
X_test = X_test.astype('float')
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
y_train = lb.fit_transform(df1.cuisine)
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
xx = X_train.todense()
dtc.fit(X_train, y_train)
#from sklearn.linear_model import LogisticRegression
#clf_lr = LogisticRegression(C=5.0, max_iter=50)
#clf_lr.fit(X_train, y_train)
#from sklearn.model_selection import cross_val_score
#res = cross_val_score(clf_lr, X_train, y_train, cv=10, scoring='accuracy')
#print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
#print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
#from sklearn.model_selection import GridSearchCV

#params = {'C':(0.1, 1.0, 5.0, 10.0), 
 #         'max_iter':(50, 100, 200)} 
#clf_lr_grid = GridSearchCV(clf_lr, params, n_jobs=-1,
#                            cv=3, verbose=1, scoring='accuracy')
#clf_lr_grid.fit(X_train, y_train)
#clf_lr_grid.best_estimator_.get_params
xx_test = X_test.todense()
y_pred = dtc.predict(X_test)
y_pred_ = lb.inverse_transform(y_pred)
submission = pd.DataFrame({"id": df2["id"], "cuisine": y_pred_})
submission.to_csv('submission.csv', index = False)
