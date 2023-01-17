import cv2

face_cascade = cv2.CascadeClassifier('../input/test11/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../input/test11/haarcascade_eye.xml')

img = cv2.imread('../input/test11/community.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imwrite('out.jpg', img)
# Install the imageai python module
!pip install imageai
import imageai
from imageai.Prediction import ImagePrediction

prediction = ImagePrediction()
prediction.setModelTypeAsSqueezeNet() 
prediction.setModelPath("../input/test11/squeezenet_weights_tf_dim_ordering_tf_kernels.h5")
prediction.loadModel()

!cp ../input/test11/snake.jpg snake.jpg
predictions, probabilities = prediction.predictImage("../input/test11/snake.jpg", result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)
import requests
r = requests.get('https://media.wired.com/photos/5d09594a62bcb0c9752779d9/master/w_2560%2Cc_limit/Transpo_G70_TA-518126.jpg')
with open('image.jpg', 'wb') as f:
    f.write(r.content)

predictions, probabilities = prediction.predictImage("image.jpg", result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/heart-disease-uci/heart.csv')
df.head()

df.target.value_counts()

# target=1 has heart disease
df.sex.value_counts()

# sex=1 is male
df.describe()
pd.crosstab(df.target, df.sex)

# Out of the people having heart disease, 72 are female, 93 are male
df.age.plot.hist();
# Correlation matrix of the data
df.corr()
# Correlation matrix in heatmap format
plt.subplots(figsize=(15,10))
sns.heatmap(df.corr(), annot=True, cmap='BuGn');
# Separate target column from the data

X = df.drop('target', axis=1)  # data

y = df.target   # target vector to be predicted

X
y
# Split to data - 80% to train the model, 20% to test the model

from sklearn.model_selection import train_test_split
np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# 80% of the data will be used to train the model (learn)
X_train

# 20% of the data will be used to test the model (measure the quality of the learning)
X_test
# Train using KNeighborsClassifier model
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)
model = KNeighborsClassifier()

#  try with different number of neighbors to optimize (Can create a for loop)
model.set_params(n_neighbors=11)    
model.fit(X_train, y_train)

# Evaluate accuracy using test data
model.score(X_test, y_test)


# Predictions of KNeighborsClassifier
y_preds = model.predict(X_test)

y_preds
# Compare predictions of the model to the actual heart disease in test data
# Confusion matrix 
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_preds))

# 9 False positive
# 6 False negative
# Train using RandomForestClassifier model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate accuracy using test data
model.score(X_test, y_test)
# Predictions of RandomForestClassifier model
y_preds = model.predict(X_test)

# Compare predictions of the model to the actual heart disease in test data
# Confusion matrix 
print(confusion_matrix(y_test, y_preds))

# 5 False positive
# 5 False negative
# Optimize RandomForestClassifier by passing in parameters
# Scikit learn library provides ways to optimize all 4 parameters at once without creating for loops

model = RandomForestClassifier(n_estimators=210, 
                               min_samples_split=4, 
                               min_samples_leaf=19,
                               max_depth=3)
model.fit(X_train, y_train)
model.score(X_test, y_test)
y_preds = model.predict(X_test)

print(confusion_matrix(y_test, y_preds))

# 5 False positive
# 3 False negative  <---- came down after optimization of the model
# Feature importances for the RandomForestClassifier
# How much weight was given to input parameters to make the predictions?

model.feature_importances_
feature_dict = dict(zip(df.columns, model.feature_importances_))

feature_dict

# RandomForestClassifier gives the most weight to 
# 19% ca: number of major vessels (0-3) colored by flourosopy
# 15% thal:  3 = normal; 6 = fixed defect; 7 = reversable defect
# 15% cp: chest pain type

# Different models give different weights
# Compare feature_dict to the correlation matrix
plt.subplots(figsize=(15,10))
sns.heatmap(df.corr(), annot=True, cmap='BuGn');
