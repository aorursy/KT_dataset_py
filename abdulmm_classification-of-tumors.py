import pandas as pd              #Data structures and data analysis library.
import numpy as np               #Vectors, matrices and linear algebra in general.

import seaborn as sns            #Statistical visualization library.
import matplotlib.pyplot as plt  #Plotting library.
import graphviz                  #Graph vizualisation library.

#Machine learning.
from sklearn.preprocessing import StandardScaler       #Standardize data.
from sklearn.model_selection import train_test_split   #Splitting the data.
from sklearn import linear_model                       #Linear and logistic regression.
from sklearn.metrics import confusion_matrix           #Evaluating machine learning models.
from sklearn import tree                               #Decision tree.
from sklearn.ensemble import RandomForestClassifier    #Random Forests.
from sklearn.cluster import KMeans                     #k-Means clustering.
from sklearn.metrics import jaccard_similarity_score   #Measure for comparing ratio of similar objects.

#Deep Learning.
from keras.models import Sequential                    #Sequential layering.
from keras.layers import Dense                         #Core of a layer.
data = pd.read_csv("../input/data.csv")
data.sample(5)
data.info()
data.drop(columns=['id', 'Unnamed: 32'], inplace=True)
data.replace(["B","M"],[0,1], inplace=True)
Y = data['diagnosis']
X = data.drop('diagnosis', axis=1).values
X = StandardScaler().fit_transform(X)
#Distribution of the diagnosis.
Y.value_counts()
#The first row of data after standardizing.
X[:1]
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.20, random_state = 42)
log_reg = linear_model.LogisticRegression(C=150, random_state=11, solver='lbfgs', max_iter=5000)
log_reg.fit(xTrain, yTrain)
yPred = log_reg.predict(xTest)
def correctPred(yTest,yPred):
    return sum(yTest == yPred)

def printPred(model_name, yTest, yPred):
    correct = correctPred(yTest,yPred)
    total = len(yTest)
    return "The "+model_name+" correctly predicted "+str(correct)+" out of "+str(total)+" predictions.\nRatio of correct predictions: "+str(correct/total)
print(printPred("logistic model", yTest, yPred))
def confusionMatrix(yTest,yPred):
    df_cm = pd.DataFrame(confusion_matrix(yTest, yPred), index = [i for i in "BM"],
                      columns = [i for i in "BM"])
    plt.figure(figsize = (5,3))
    sns.set(font_scale=2)
    sns.heatmap(df_cm, annot=True)
    plt.show()
    
confusionMatrix(yTest,yPred)
yTest[yTest != yPred]
tre = tree.DecisionTreeClassifier(max_depth=3)
tre.fit(xTrain, yTrain)
yPred = tre.predict(xTest)
dot_data = tree.export_graphviz(tre, out_file=None, max_depth=2, feature_names=list(data.drop('diagnosis', axis=1).columns.values), filled=True, rounded=True)
desTree = graphviz.Source(dot_data) 
desTree
print(printPred("decision tree", yTest, yPred))
confusionMatrix(yTest,yPred)
#We compute SSE for each k and plot the results
kValues = []
SSE = []
for k in range(1,14):
    kMean = KMeans(n_clusters=k)
    kMean = kMean.fit(X)
    SSE.append(kMean.inertia_)
    kValues.append(k)

sns.set(font_scale=1)
plt.plot(kValues, SSE, 'bx-')
plt.xlabel('k, number of clusters')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.show()
#y denotes the distance between each plotted point and the secant line.
y = []
end0 = np.array([kValues[0],SSE[0]])
end1 = np.array([kValues[len(SSE)-1],SSE[len(SSE)-1]])
for i in range(1,13):
    point = np.array([kValues[i],SSE[i]])
    distancePointToSecant = np.linalg.norm(np.cross(end1-end0, end0-point))/np.linalg.norm(end1-end0)
    y.append(distancePointToSecant)

print("The optimal k is k="+str(np.array(y).argmax()+1))

kMean = KMeans(n_clusters=2, init="k-means++", n_init=10)
kMeanPredict = kMean.fit_predict(X)
clusterCenters = kMean.cluster_centers_
plt.scatter(X[:,0], X[:,1], c=kMeanPredict, cmap=plt.cm.Paired)
plt.title("2-Means Clustering")
plt.scatter(clusterCenters[:, 0], clusterCenters[:, 1], c='green', s=200, alpha=0.5);
plt.show()
plt.scatter(X[:,0], X[:,1], c=data["diagnosis"], cmap=plt.cm.Paired)
plt.title("Actual diagnosis")
plt.scatter(clusterCenters[:, 0], clusterCenters[:, 1], c='green', s=200, alpha=0.5);
plt.show()
j = jaccard_similarity_score(data["diagnosis"], kMeanPredict)
print("Jaccard Index: ",j)
confusion_matrix(data["diagnosis"], kMeanPredict)
nn_model = Sequential()
print("Number of variables in input layer:",X[:1].shape[1])
nn_model.add(Dense(16, input_dim=30, activation='relu'))
nn_model.add(Dense(8, activation='relu'))
nn_model.add(Dense(1, activation='sigmoid'))
nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
nn_model.fit(xTrain, yTrain, epochs=150, batch_size=10)
scores = nn_model.evaluate(xTest, yTest)
print(nn_model.metrics_names[1], scores[1]*100)
