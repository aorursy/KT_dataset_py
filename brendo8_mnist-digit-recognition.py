import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
print("Libraries Imported")
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
print("Data imported")
print("Number of images: %d" % len(train_data))
train_data.head()
image1 = train_data.loc[0, train_data.columns != "label"]
plt.imshow(np.array(image1).reshape((28, 28)), cmap="gray")
plt.show()
plt.hist(image1)
plt.xlabel("Pixel Intensity")
plt.ylabel("Counts")
plt.show()
#clean and split data
train_images = train_data.loc[:, train_data.columns != "label"] / 255
train_labels = train_data.label
test_data = test_data.loc[:, :] / 255
x_train, x_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.25, random_state=1)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

print("Data cleaned and split")
sample_size = len(x_train)
x_train_sample = x_train.iloc[0:sample_size, :]
y_train_sample = y_train[0:sample_size]
x_test_sample = x_test.iloc[0:sample_size, :]
y_test_sample = y_test[0:sample_size]

print("Data samples created")
#SVC classifier
model = SVC()
model.fit(x_train_sample, y_train_sample)
print("Model trained")
#training metrics
train_predicts = model.predict(x_train_sample)
train_acc = round(accuracy_score(y_train_sample, train_predicts) * 100)
print("Training Accuracy: %d%%" %train_acc)

#test metrics
test_predicts = model.predict(x_test_sample)
test_acc = round(accuracy_score(y_test_sample, test_predicts) * 100)
print("Training Accuracy: %d%%" %test_acc)
#submission predictions
predictions = model.predict(test_data)
print("Finished submission predictions")

#export submission data
submission = pd.DataFrame(predictions)
submission.index.name = "ImageId"
submission.index += 1
submission.columns = ["Label"]
submission.to_csv("digit_submissions.csv", header=True)

print("Exported submission predictions")