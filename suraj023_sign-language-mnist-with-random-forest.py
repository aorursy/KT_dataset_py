import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



Train_data = pd.read_csv("../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv")
Test_data = pd.read_csv("../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv")

a=Train_data.iloc[0,1:].values
a=a.reshape(28,28).astype('uint8')
plt.imshow(a)

X = Train_data.iloc[:,1:]
Y = Train_data[['label']]
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=42)
RFC=RandomForestClassifier(n_estimators=100)

RFC.fit(X_train,y_train)

X = Test_data.iloc[:,1:]
Y = Test_data.iloc[:,0]
RFC.score(X_test,y_test)

res=RFC.predict(X)
total_correct_labels = np.sum(np.squeeze(Y) == res)
test_acc = total_correct_labels / Y.shape[0]
print("Test Accuracy: {} %".format(test_acc * 100))
imageid = np.arange(res.shape[0]) + 1

df = pd.DataFrame({'ImageId': imageid, 'Label': res})
df.head(20)