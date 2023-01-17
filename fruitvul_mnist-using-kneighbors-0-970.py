import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import os

import matplotlib

import matplotlib.pyplot as plt



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Load data set:

train = pd.read_csv("../input/train.csv")



test = pd.read_csv("../input/test.csv")
X,y = train.drop("label",axis=1),train["label"]
sns.countplot(y)
print(X.shape,y.shape)


example_digit = X.iloc[np.random.randint(0,42000)]

example_digit_image  = example_digit.values.reshape(28,28)

plt.imshow(example_digit_image,cmap=matplotlib.cm.binary,interpolation="nearest")

plt.axis("off")



from sklearn.model_selection import train_test_split



X_train, X_test, y_train,y_test = train_test_split(X[0:]/255.0,y[0:],test_size = 0.30,random_state =42)

from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier(n_neighbors = 4,weights="distance")

knc.fit(X_train,y_train)

predictions = knc.predict(X_test[0:])
from sklearn.metrics import confusion_matrix
con_mat = confusion_matrix(y_test,predictions)
sns.heatmap(con_mat)
row_sums = con_mat.sum(axis=1, keepdims=True)
sns.heatmap(con_mat/row_sums)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,predictions))
X.iloc[2]= X.iloc[2] 

example_digit = X.iloc[2]

example_digit_image  = example_digit.values.reshape(28,28)

plt.imshow(example_digit_image,cmap=matplotlib.cm.binary,interpolation="nearest")

plt.axis("on")
example2 = np.roll(example_digit_image,2,axis=1)

plt.imshow(example2,cmap=matplotlib.cm.binary,interpolation="nearest")

plt.axis("on")
print("X: Number of images before:"+ str(len(X)))

print("Y: Number of images before:"+ str(len(X)))

append_count = 1000

for i in range(0,append_count):

    byby = X.iloc[i].values.reshape(28,28)

    redf= np.roll(byby,np.random.randint(-3,3),axis=1) # Move random int to right

    redf=np.roll(byby,np.random.randint(-3,3),axis =0) # Move random int to top

    redf = redf.reshape(784,1)  

    X = X.append(pd.DataFrame(redf.transpose(),columns=X.columns,index=[len(X)+1]))

    y[len(y)+1] = y[i]

print("X: Number of images after:"+ str(len(X)))

print("Y: Number of images after:"+ str(len(X)))

example_digit = X.iloc[np.random.randint(0,43000)]

example_digit_image  = example_digit.values.reshape(28,28)

plt.imshow(example_digit_image,cmap=matplotlib.cm.binary,interpolation="nearest")

plt.axis("on")
X_train, X_test, y_train,y_test = train_test_split(X[0:],y[0:],test_size = 0.30,random_state =42)

kncc = KNeighborsClassifier(n_neighbors = 5,weights="distance")

kncc.fit(X_train,y_train)

predictions = kncc.predict(X_test[0:])
print(accuracy_score(y_test,predictions))
con_mat = confusion_matrix(y_test,predictions)

row_sums = con_mat.sum(axis=1, keepdims=True)

sns.heatmap(con_mat/row_sums)
con_mat
len(predictions)
y_test[12000]
import matplotlib.pyplot as plt

miss_arr = []

plt.figure(figsize=(25,40))

x=0

for i,y in enumerate(y_test[0:8000]):

    

    tests = y-predictions[i]

    if tests != 0 and x < 17*17:

        x +=1

        plt.subplot(17,17,x)

        plt.axis("off")

        grid_data = X_test.iloc[i].values.reshape(28,28)  # reshape from 1d to 2d pixel array

        plt.imshow(grid_data, interpolation = "none", cmap = "bone_r")

        plt.title("Predicted as "+str(predictions[i]))

        plt.xticks([])

        plt.yticks([]) 

        
#predict for submission



predictions = kncc.predict(test)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("DR.csv", index=False, header=True)
test