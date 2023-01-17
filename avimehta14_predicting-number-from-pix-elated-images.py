import matplotlib.pyplot as plt

import pandas as pd



%matplotlib inline



from sklearn.datasets import load_digits
digits = load_digits()

digits
dir(digits)
digits.data[0]
plt.gray()

for i in range(3):

    plt.matshow(digits.images[i])
digits.target[0:5]
# USING DATA AND TARGET ATTRIBUTES TO TRAIN MODEL 
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test=train_test_split(digits.data, digits.target,test_size=0.2)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
model.score(X_test,y_test)
plt.matshow(digits.images[67])
model.predict([digits.data[67]])
# TO CHECK WHERE THE MODEL FAILS , WE USE A CONFUSION MATRIX
y_predicted = model.predict(X_test)
from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test,y_predicted)
cm
import seaborn as sn



plt.figure(figsize =(10,7))

sn.heatmap(cm,annot=True)

plt.xlabel("PREDICTED")

plt.ylabel("True Value")