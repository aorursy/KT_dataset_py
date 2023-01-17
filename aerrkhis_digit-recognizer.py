import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
#load data
data = pd.read_csv('../input/digit-recognizer/train.csv')
data.head()
#split data to labels and features
labels = data.iloc[:30000,0]
train = data.iloc[:30000,1:]
#SVM
x_train,x_test,y_train,y_test = train_test_split(train,labels,random_state=4,test_size=0.2)
model = svm.SVC(C=1,kernel="linear",degree=3,gamma="auto")
model.fit(x_train,y_train)
print("SVM accuracy :",model.score(x_test,y_test)*100,"%")
#Random Forest
model_ = RandomForestClassifier(n_estimators=100)
model_.fit(x_train,y_train)
print("Random Forest accuracy :",model_.score(x_test,y_test)*100,"%")
#testing data
test_data = pd.read_csv('../input/digit-recognizer/test.csv')
image_ = test_data.iloc[4327,:].values
#reshape image test
image = image_.reshape(28,28)
image.shape
#show image
plt.imshow(image)
plt.show()
#SVM prediction
print("predicted number is :",model.predict(image.reshape(1,-1))[0])
#Random Forest prediction
print("predicted number is :",model_.predict(image.reshape(1,-1))[0])