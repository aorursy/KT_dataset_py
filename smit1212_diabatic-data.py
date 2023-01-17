# I cleaning data that is right way or not?
# Clustering using k-means
# Any one can help me out how to predict the LogisticRegression() model.

# Import Libraries 
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# read in all our data
diabet_data = pd.read_csv("../input/diabetic_data.csv")

#fetch requred data
df = diabet_data[['gender', 'readmitted']]
df.head()

# Convert string to numbers
af = df.replace(['Female','Male','NO','>30','<30'],[0,1,2,1,0])
af.head()
# Total no of Readmitted patient's in class wise 
print(af.groupby('readmitted').size())

# show histograms of readmitted
da = diabet_data.readmitted.hist()
plt.show()
# Split-out validation dataset
array = af.values
X = array[:,0:1]
Y = array[:,1]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# Make predictions on validation dataset

Lr = LogisticRegression()
Lr.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
