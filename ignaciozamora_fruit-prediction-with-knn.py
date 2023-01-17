import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
# load data from file into pandas dataframe
data=pd.read_table('/kaggle/input/fruits-with-colors-dataset/fruit_data_with_colors.txt')

# show top 5
data.head()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# scale mass variable
scaler.fit(np.array(data.mass).reshape(-1,1))
data.mass = scaler.transform(np.array(data.mass).reshape(-1,1))

# scale width and height
scaler.fit(np.array(data.width).reshape(-1,1))
data.width = scaler.transform(np.array(data.width).reshape(-1,1))

# scale height
scaler.fit(np.array(data.height).reshape(-1,1))
data.height = scaler.transform(np.array(data.height).reshape(-1,1))

data.head()
# lets isolate the response variable
y = data["fruit_label"]

# lets frop name and subtype
data = data.drop(['fruit_name', 'fruit_subtype', "fruit_label"], axis=1)

# lets call X all other variables
X = data
# lets train KNN with different values of K and lets find out which K works best
K = np.arange(1,30)

scores = []

# train model ans get test set accuracy metrics
for k in K:
    model = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
    score = cross_val_score(model, X, y, cv=5).mean()
    scores.append(score)

print("model trained")
plt.plot(K,scores, c = "red")
plt.title("KNN 5-Fold Cross Validation")
plt.ylabel("Accuracy")
plt.xlabel("K")
plt.legend()
plt.show()
best_k = K[np.argmax(scores)]
best_s = scores[np.argmax(scores)] 
print("With K =",best_k, "the model was able to predict with",best_s,"accuracy")
# Test with K = 4
model = KNeighborsClassifier(n_neighbors=4).fit(X_train,y_train)
score = cross_val_score(model, X, y, cv=5).mean()
print("Model score with k=4, is",score)