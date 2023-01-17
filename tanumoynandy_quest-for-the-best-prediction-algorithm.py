import numpy as np
data = np.loadtxt(open("../input/diabetes.csv", "rb"), delimiter=",", skiprows=1)
X = data[:,:8]
Y = data [:,8]
print(X[0])
print(Y[0])
#scale data
#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler(feature_range=(0, 1))
#rescaledX = scaler.fit_transform(X)
#X = rescaledX
#print(X[0:5,:])
#reduced accuracy from 75 to 72 for logistic regression
#Normalize data
#from sklearn.preprocessing import Normalizer
#scaler = Normalizer().fit(X)
#normalizedX = scaler.transform(X)
#X = normalizedX
#print(normalizedX[0:5,:])
#reduced accuracy to 64 for logistic regression
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
print(X_train[0])
print(Y_train[0])
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, Y_train)
#Predict labels for new data (new images)
# Returns a NumPy Array
# Predict for One Observation (image)
print(logisticRegr.predict(X_test[0].reshape(1,-1)))
Y_test[0]
predictions = logisticRegr.predict(X_test)
# Use score method to get accuracy of model
score = logisticRegr.score(X_test, Y_test)
print(score)
from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, Y_train) 
print (svclassifier.predict(X_test[0].reshape(1,-1)) )
Y_test[0]
predictions = svclassifier.predict(X_test)
# Use score method to get accuracy of model
score = svclassifier.score(X_test, Y_test)
print(score)
# Bagged Decision Trees for Classification
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
print(results.mean())
# Random Forest Classification
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
seed = 7
num_trees = 100
max_features = 3
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
print(results.mean())
#bagging - trying to improve accuracy - Extra Trees Classification
from sklearn import model_selection
from sklearn.ensemble import ExtraTreesClassifier
seed = 7
num_trees = 100
max_features = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
print(results.mean())
# AdaBoost Classification
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
seed = 7
num_trees = 30
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
print(results.mean())
# Stochastic Gradient Boosting Classification
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier
seed = 7
num_trees = 100
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
