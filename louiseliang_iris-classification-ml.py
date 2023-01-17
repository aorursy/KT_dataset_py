import pandas as pd

from sklearn import datasets

import seaborn as sns
data = datasets.load_iris()

data
Iris = pd.DataFrame(data['data'], columns=data['feature_names'])

Iris['target'] = data['target']

Iris.head()
Iris.describe()
sns.pairplot(hue="target", data=Iris.iloc[:, :])

#Import SVC

from sklearn.svm import SVC



X = Iris.iloc[:, 0:4].values

y = Iris["target"].values



model = SVC(kernel='linear',random_state=555)



#Trainig

model.fit(X=X, y=y)

#Mean Accuracy

model.score(X=X, y=y)
predict_y = model.predict(X=X)



predict_y
# Confusion Matrix



from sklearn import metrics



confmat = metrics.confusion_matrix(y, predict_y)

confmat



sns.heatmap(confmat, cmap="YlGnBu_r", annot=True, fmt="d")



# Classifiction Report



report = metrics.classification_report(y, predict_y)

print(report)



# Import SVC

from sklearn.svm import SVC



X = Iris.iloc[:, 0:4].values

y = Iris["target"].values



# Trainig

model = SVC(kernel='rbf',random_state=555)

model.fit(X=X, y=y)
# Mean Accuracy



model.score(X=X, y=y)
# Prediction



predict_y = model.predict(X=X)



predict_y



metrics.accuracy_score(y, predict_y)
# Confusion Matrix



from sklearn import metrics



confmat = metrics.confusion_matrix(y, predict_y)

confmat



sns.heatmap(confmat, cmap="YlGnBu_r", annot=True, fmt="d")



# Classifiction Report



report = metrics.classification_report(y, predict_y)

print(report)
# Assign kernal= 'poly'

model3 = SVC(kernel='poly',degree=3, random_state=555)

model4 = SVC(kernel='poly',degree=4, random_state=555)

model5 = SVC(kernel='poly',degree=5, random_state=555)

model6 = SVC(kernel='poly',degree=6, random_state=555)



# Trainig

model3.fit(X=X, y=y)

model4.fit(X=X, y=y)

model5.fit(X=X, y=y)

model6.fit(X=X, y=y)
# mean accuracy



print('degree=3,  mean accuracy=', model3.score(X=X, y=y))

print('degree=4,  mean accuracy=', model4.score(X=X, y=y))

print('degree=5,  mean accuracy=', model5.score(X=X, y=y))

print('degree=6,  mean accuracy=', model6.score(X=X, y=y))

predict_y3 = model3.predict(X=X)

predict_y4 = model4.predict(X=X)

predict_y5 = model5.predict(X=X)

predict_y6 = model6.predict(X=X)



print(predict_y3)

print(predict_y4)

print(predict_y5)

print(predict_y6)
# Confusion Matrix



from sklearn import metrics



confmat3 = metrics.confusion_matrix(y, predict_y3)

confmat4 = metrics.confusion_matrix(y, predict_y4)

confmat5 = metrics.confusion_matrix(y, predict_y5)

confmat6 = metrics.confusion_matrix(y, predict_y6)



print(confmat3)

print(confmat4)

print(confmat5)

print(confmat6)
sns.heatmap(confmat3, cmap="YlGnBu_r", annot=True, fmt="d")

report3 = metrics.classification_report(y, predict_y3)

print(report3)
sns.heatmap(confmat4, cmap="YlGnBu_r", annot=True, fmt="d")

report4 = metrics.classification_report(y, predict_y3)

print(report4)
sns.heatmap(confmat5, cmap="YlGnBu_r", annot=True, fmt="d")

report5 = metrics.classification_report(y, predict_y3)

print(report5)
sns.heatmap(confmat6, cmap="YlGnBu_r", annot=True, fmt="d")

report6 = metrics.classification_report(y, predict_y3)

print(report6)
#Accuracy Score



print('degree=3, accuracy score=', metrics.accuracy_score(y, predict_y3))

print('degree=4, accuracy score=', metrics.accuracy_score(y, predict_y4))

print('degree=5, accuracy score=', metrics.accuracy_score(y, predict_y5))

print('degree=6, accuracy score=', metrics.accuracy_score(y, predict_y6))
#Precision Score



print('degree=3, precision score=', metrics.precision_score(y, predict_y3,average='weighted'))

print('degree=4, precision score=', metrics.precision_score(y, predict_y4,average='weighted'))

print('degree=5, precision score=', metrics.precision_score(y, predict_y5,average='weighted'))

print('degree=6, precision score=', metrics.precision_score(y, predict_y6,average='weighted'))
#Recall Score



print('degree=3, recall score=', metrics.recall_score(y, predict_y3,average='weighted'))

print('degree=4, recall score=', metrics.recall_score(y, predict_y4,average='weighted'))

print('degree=5, recall score=', metrics.recall_score(y, predict_y5,average='weighted'))

print('degree=6, recall score=', metrics.recall_score(y, predict_y6,average='weighted'))