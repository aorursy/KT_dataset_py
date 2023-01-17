import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input/"))
from collections import Counter
data=pd.read_csv("../input/train.csv")[["Survived","Age","Fare"]]
data=data.fillna(data.mean())
data.head()

passenger_id=pd.read_csv("../input/test.csv")["PassengerId"]
test_data=pd.read_csv("../input/test.csv")[["Age","Fare"]]
test_data=test_data.fillna(test_data.mean())
test_data.head()
test_data=test_data.values
import matplotlib.pyplot as plt
plt.scatter( x=[7.25],y=[22])
plt.show()
import matplotlib.pyplot as plt
plt.scatter( x=[7.25, 19],y=[22,30])
plt.show()
import matplotlib.pyplot as plt
col=data["Survived"]
plt.scatter( x=data["Fare"],y=data["Age"], c=col)
plt.show()
def compute_distances_one_loop(new_points, X_train):
    #X_Train have all of our training Samples, 
    # new_points is our new points which we want to predict
    num_test = len(new_points)
    num_train =X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
        
        difference = new_points[i] - X_train
        difference = np.square(difference)
        sum1 = np.sum(difference, axis=1)
        dists[i] = np.sqrt(sum1)
    return dists
#Example:
X_train=np.array([[1,2],[2,3],[4,5], [6,7]])
Y_train=np.array([1,0,0,1])
col=Y_train
plt.scatter( x=X_train[:,0],y=X_train[:,1], c=col)
plt.show()
#OUR TESTING DATA
new_point=[[3,4]]
dists=compute_distances_one_loop(new_point, X_train)
print (dists)
print ("ARRAY DIMENSIONS IS :")
print (dists.shape)
# Now that we have distance from each point, we will simple find out the distance which is
#least
def predict(dists, training_labels, k=3):
    closest_y = []
    rank = list(np.argsort(dists))
    for x in range(0, k):
        closest_y.append(training_labels[rank[x]])
    closest_y = np.asarray(closest_y)
    c=Counter(closest_y)
    return (c.most_common()[0][0])
result=predict(dists[0], Y_train)
print (result)
compute_distances_one_loop([[1,2]],data[["Fare","Age"]]).shape
dists=compute_distances_one_loop(test_data,data[["Fare","Age"]])
Results=[]
for x in dists:
    Results.append(predict(x, data["Survived"]))
f=open("result.csv","w")
f.write("PassengerId,Survived")
for i in range(0, len(Results)):
    f.write("\n")
    f.write(str(passenger_id[i])+","+ str(Results[i]))
    
f.close()
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(data[["Fare","Age"]], data["Survived"])
scikit_result=neigh.predict(test_data)


count=0
notr=[]
for i in range(0, len(scikit_result)):
    if scikit_result[i]==Results[i]:
        count+=1
    else:
        notr.append(i)
print ("TOTAL INSTANCES")
print (len(scikit_result))
print ("SIMILAR RESULT B/W KNN AND SCIKIT LEARN INBUILT KNN")
print (count)
print ("")

print ("INDEXES OF WRONG RESULT")
print (notr)
Results[55]
scikit_result[55]
data.iloc[55]