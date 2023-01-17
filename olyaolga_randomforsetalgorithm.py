from sklearn.datasets import load_digits
digits = load_digits()
#print(digits.images.shape)

import matplotlib.pyplot as plt 
plt.gray() 
plt.matshow(digits.images[800]) 
plt.show() 
import numpy as np
transformed_imgs = []
for img in digits.images:
    transformed_imgs.append(np.reshape(img, -1))
import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( np.array(transformed_imgs), digits.target, test_size=0.2, random_state=20)
print("X_train shape: {0}, X_test shape: {1}, y_train shape: {2}, y_test shape: {3}".format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))
from sklearn.ensemble import RandomForestClassifier
rnd_clf=RandomForestClassifier(n_estimators=500,n_jobs=-1, max_depth=6) 
rnd_clf.fit(X_train,y_train)
y_pred_rf_test=rnd_clf.predict(X_test)
y_pred_rf_tr = rnd_clf.predict(X_train)
print("train accuracy: {0}".format(1 - np.count_nonzero(y_pred_rf_tr-y_train)/y_train.shape[0]))
print("test accuracy: {0}".format(1 - np.count_nonzero(y_pred_rf_test-y_test)/y_test.shape[0]))
from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(max_depth=6)
tree_clf.fit(X_train,y_train)
y_single_tree_pred_test = tree_clf.predict(X_test)
y_single_tree_pred_train = tree_clf.predict(X_train)
print("train accuracy: {0}".format(1 - np.count_nonzero(y_single_tree_pred_train-y_train)/y_train.shape[0]))

print("test accuracy: {0}".format(1 - np.count_nonzero(y_single_tree_pred_test-y_test)/y_test.shape[0]))
