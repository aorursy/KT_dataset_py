import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#for clustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.stats import itemfreq
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import math
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Absenteeism_at_work.csv', delimiter=',')
df.head()
df1 = pd.DataFrame(df)
split = np.random.rand(len(df)) < 0.67
train_data = df[split]
test_data = df[~split]
train_data.head()
test_data.head()
import math
#Calculating Euclidean diastance
def Edistance(inst1, inst2, dimensions):
    distance = 0
    for i in range(dimensions):
        distance += pow(int(inst1[i] - inst2[i]), 2)
    return math.sqrt(distance)
import operator
#Returning K nearest neighbours based on distance
def Neighbors(train, test, k):
    distances = []
    length = len(test)-1    #length of 1 test data
    for x in range(len(train)):
        dist = Edistance(test, train[x], length)
        distances.append((train[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors
#Choosing the classes obtained from Neighbours
def Response(neighbors):
    Choices = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in Choices:
            Choices[response] += 1
        else:
            Choices[response] = 1
    sortedChoices = sorted(Choices.items(), key=operator.itemgetter(1), reverse=True)
    return sortedChoices[0][0]
#Calculating Accuracy
def Accuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
#Combining all functions to get KNN
trainingSet = train_data.values.tolist()
test = test_data.values.tolist()
print("Train set size: ",len(trainingSet))
print("Test set size: ",len(test))
predictions=[]
k = 19  #no. of neighbors
count=0
for x in range(len(test)):
    count+=1
    neighbors = Neighbors(trainingSet, test[x], k)
    result = Response(neighbors)
    predictions.append(result)
accuracy = Accuracy(test, predictions)
print("Accuracy: ",accuracy)
#the train data has outliers 
#kmeans method will vary greatly for outliers
#finding outlier and removing it

Q1 = train_data['Absenteeism time in hours'].quantile(0.25)
Q3 = train_data['Absenteeism time in hours'].quantile(0.75)
IQR = Q3 - Q1
print(IQR,Q1,Q3)
mask = train_data["Absenteeism time in hours"].between(Q1-IQR,Q3+IQR)
filtered_train=train_data[mask]

# printing unique values in filtered train data
print(np.unique(filtered_train["Absenteeism time in hours"]))
def myFunc(data):
    resu=data.iloc[:, [14]]
    arr=resu.values
    l=[]
    for i in arr:
        if i in range(0,5):
            l.append(0)
        if(i in range(5,10)):
            l.append(1)
        if(i in range(10,20)):
            l.append(2)
        if(i in range(20,30)):
            l.append(3)
        if(i in range(30,50)):
            l.append(4)
        if(i in range(50,100)):
            l.append(5)
        if(i in range(100,200)):
            l.append(6)
    print(len(data))
    se=pd.Series(l)
    data['Cluster']=se.values
myFunc(filtered_train)
myFunc(train_data)
myFunc(test_data)
# print(list(df.columns.values))
sns.set_style('whitegrid')
sns.lmplot('Distance from Residence to Work','Reason for absence',data=df, hue='Absenteeism time in hours',palette="coolwarm",size=6,aspect=1,fit_reg=False)

sns.set_style('whitegrid')
sns.lmplot('Distance from Residence to Work','Reason for absence',data=train_data, hue='Cluster',palette="coolwarm",size=6,aspect=1,fit_reg=False)
sns.set_style('whitegrid')
sns.lmplot('Reason for absence','Work load Average/day ',data=train_data, hue='Cluster',palette='coolwarm',size=6,aspect=1,fit_reg=False)
def root_mean_squared_error(y_actual,y_predicted):
    errorsquare=(y_actual-y_predicted)**2
    sum_of_error_square=np.mean(errorsquare)
    rmse=np.sqrt(sum_of_error_square)
    return(rmse)
    
# rmse = root_mean_squared_error(data_train, avg_rating)
# rmse
X=filtered_train.drop(["Absenteeism time in hours","ID","Cluster"],1)
wcss = []
for i in range(1,15):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,15),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('elbow.png')
plt.show()
# print(X)
# X=train.drop(["Absenteeism time in hours"],1)
k_means = KMeans(n_clusters=7, max_iter=600, algorithm = 'auto')
k_means.fit(X)
predict=k_means.predict(test_data.iloc[:,1:14])
type(predict[1])
a=test_data.iloc[:,15]
type(a)
a=test_data["Cluster"].values
print(a[1])
# print(a.columns.values)
err=root_mean_squared_error(a,predict)
print(err)
# print((kmeans.labels_))
print(np.unique(train_data["Cluster"]))

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(filtered_train["Cluster"],k_means.labels_))
print(classification_report(filtered_train["Cluster"],k_means.labels_))