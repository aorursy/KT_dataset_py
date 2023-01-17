import numpy as np # 数组常用库

import pandas as pd # 读入csv常用库

import matplotlib.pyplot as plt # 画图常用库

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score  
train=np.array(pd.read_csv("../input/train.csv"))

test=np.array(pd.read_csv("../input/test.csv"))

print("Train shape:",train.shape,"Test shape:",test.shape)
example_pic=np.reshape(train[2121,1:],(28,28))

plt.imshow(example_pic)

plt.show()
train_data=train[:,1:]

train_label=train[:,0]

X_train, X_test, Y_train, Y_test = train_test_split(train_data, train_label, test_size=0.2, random_state=0) 



print("Train set:",X_train.shape, Y_train.shape)

print("Test set:",X_test.shape, Y_test.shape)

score=np.zeros(20)

for k in range(1,21):

    knn = KNeighborsClassifier(n_neighbors=k) 

    knn.fit(X_train, Y_train)

    score[k-1]=accuracy_score(Y_test, knn.predict(X_test))
k=np.arange(1,21)

plt.plot(k,score)

plt.show()

    
#So we use K=1
knn = KNeighborsClassifier(n_neighbors=1) 

knn.fit(X_train, Y_train)

Test_result=knn.predict(test)
print("Predit result:",Test_result[12121])

print("The original picture:")

example_pic=np.reshape(test[12121],(28,28))

plt.imshow(example_pic)

plt.show()
df = pd.DataFrame(Test_result)

df.to_csv("../Digit_Recogniser_Result.csv",index=False)