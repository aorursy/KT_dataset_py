#The data sets are given by kaggle --> https://www.kaggle.com/c/digit-recognizer/data



#Load the data



import pandas as pd

import numpy as np



train_data = pd.read_csv('../input/digit-recognizer/train.csv', dtype=np.int)
train_data.head()
#checking missing data

train_data.isnull().sum()
#Split data

X_train = train_data.drop("label",axis = 1)

y_train = train_data["label"]
X_train.head()
y_train.head()
X_train.iloc[0,210:220]
X_train = X_train / 255
X_train.iloc[0,210:220]
import matplotlib.pyplot as plt



plt.imshow(np.array(X_train.iloc[3]).reshape(28,28), cmap = "gray")

plt.title(str(y_train[3]))

plt.show()
test_data = pd.read_csv('../input/digit-recognizer/test.csv')
test_data.isnull().sum()
test_data = test_data / 255
test_data.head()
from sklearn.neural_network import MLPClassifier



model = MLPClassifier(solver="adam", activation="relu", hidden_layer_sizes=(64,64))
model.fit(X_train, y_train)
prediction = model.predict(test_data)
for i in range(3):

    plt.imshow(np.array(test_data.iloc[i]).reshape(28,28), cmap = "gray")

    plt.title(str(prediction[i]))

    plt.show()  
ImageId = [i for i in range(1,28001)]

submission = pd.DataFrame({"ImageId": ImageId,"Label": prediction})
submission.head()
submission.to_csv("submission.csv", index=False)