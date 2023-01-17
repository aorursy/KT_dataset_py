import pandas as pd
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
Y = train['label']
X = train.drop(labels=["label"], axis=1)
X = X / 255.0
test = test / 255.0
import seaborn as sns
g = sns.countplot(Y)
Y.value_counts()
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
Xshow = X.values.reshape(-1,28,28,1)
g = plt.imshow(Xshow[0][:,:,0])
from sklearn.model_selection import train_test_split
random_seed=0
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=random_seed)
#from sklearn.linear_model import SGDClassifier
#sgd_clf = SGDClassifier(max_iter=5, random_state=42)
# Train the model using splited train dataset
#sgd_clf.fit(X_train, Y_train)
# Check the score using splited test dataset
#sgd_clf.score(X_val, Y_val)
from sklearn.ensemble import RandomForestClassifier
rforest_model = RandomForestClassifier(n_estimators=200)
# Train the model using splited train dataset
rforest_model.fit(X_train, Y_train)
# Check the score using splited test dataset
rforest_model.score(X_val, Y_val)
#from sklearn.neighbors import KNeighborsClassifier
#knn = KNeighborsClassifier(n_neighbors=3)
# Train the model using splited train dataset
#knn.fit(X_train, Y_train)
# Check the score using splited test dataset
#knn.score(X_val, Y_val)
#from sklearn.linear_model import LogisticRegression
#lr = LogisticRegression(C=1)
# Train the model using splited train dataset
#lr.fit(X_train, Y_train)
# Check the score using splited test dataset
#lr.score(X_val, Y_val)
#from sklearn.svm import LinearSVC
#svc = LinearSVC(C=0.1)
#svc.fit(X_train, Y_train)
#svc.score(X_val, Y_val)
#from sklearn.tree import DecisionTreeClassifier
#tree = DecisionTreeClassifier(max_depth=20, random_state=0)
#tree.fit(X_train, Y_train)
#tree.score(X_val, Y_val)
#from sklearn.ensemble import GradientBoostingClassifier
#gbrt = GradientBoostingClassifier(random_state=0)
#gbrt.fit(X_train, Y_train)
#gbrt.score(X_val, Y_val)
#from sklearn.svm import SVC
#svm = SVC(kernel='rbf', C=10, gamma=0.1)
#svm.fit(X_train, Y_train)
#svm.score(X_val, Y_val)
#from sklearn.neural_network import MLPClassifier
# 필요하면 activiation, alpha
#mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[20])
#mlp.fit(X_train, Y_train)
#mlp.score(X_val, Y_val)
rforest_model.fit(X, Y)
results = rforest_model.predict(test)
submission = pd.read_csv('../input/sample_submission.csv')
submission["Label"] = results
submission.head()
submission.to_csv('./simpleMNIST.csv', index=False)