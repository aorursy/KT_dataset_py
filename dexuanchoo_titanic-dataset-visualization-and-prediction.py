import numpy as np # linear algebra
from numpy import *
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visulization
%matplotlib inline

import seaborn as sns # data visulization
import missingno as msno # missing data visualization
import math # Calcuation 
from math import log

import operator # Operation
import sys
#import treePlotter # Visualization tool for decision tree
from time import time # time info

from sklearn.cross_validation import train_test_split # dataset split

from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.ensemble import BaggingClassifier # Bagging
from sklearn.tree import DecisionTreeClassifier # Decision Tree
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn import linear_model 
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.linear_model import Perceptron # Perceptron
from sklearn.linear_model import SGDClassifier # Stochastic Gradient Descent 
from sklearn.svm import SVC, LinearSVC # Support Vector Machine (Normal, linear)
from sklearn.naive_bayes import GaussianNB # Naive Bayes

from sklearn.cluster import KMeans # K-means

from sklearn.neural_network import MLPClassifier # Multiple Layers Perceptron

from sklearn.metrics import accuracy_score # Accuracy Calculation
from sklearn.metrics import precision_score, recall_score # calculate precision and recall
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import tensorflow as tf # Deep Learning Library
from keras.models import Sequential
from keras.layers import Dense, Activation

np.random.seed(2)

import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#print (train)
#print (test)
#print (train.head())
#train.describe(include="all")
#train.isnull().any()
#train.isnull().sum()
msno.matrix(train,figsize=(12,5))
survived = train["Survived"]
total = survived.shape[0]
result_survived = pd.value_counts(survived)
print (result_survived)

labels_survived = 'Survived', 'Dead'
size_survived = [result_survived[1]/total, result_survived[0]/total]
explode_survived = [0.1, 0]

plt.figure(figsize = (5,5))
plt.pie(size_survived, explode = explode_survived, labels = labels_survived, center = (0, 0), labeldistance=1.1, autopct='%1.2f%%', pctdistance=0.5, shadow=True)
plt.title("Survived")

plt.show()
passenger_class = train["Pclass"]
result_class = pd.value_counts(passenger_class)

labels_class = 'Class 1', 'Class 2', 'Class 3'
size_class = [result_class[1]/total, result_class[2]/total, result_class[3]/total]
explode_class = [0.1, 0.1, 0.1]

plt.figure(figsize = (5,4.5))
plt.pie(size_class, explode = explode_class, labels = labels_class, center = (0, 0), labeldistance=1.1, autopct='%1.2f%%', pctdistance=0.5, shadow=True)
plt.title("Passenger class")

plt.show()
train[["Pclass", "Survived"]].groupby(["Pclass"]).mean().plot.bar()
sns.countplot("Pclass", hue = "Survived", data = train)

plt.show()
passenger_sex = train["Sex"]
result_sex = pd.value_counts(passenger_sex)
   
labels_sex = 'Male', 'Female'
size_sex = [result_sex['male']/total, result_sex['female']/total]
explode_sex = [0.1, 0]

plt.figure(figsize = (5,4.5))
plt.pie(size_sex, explode = explode_sex, labels = labels_sex, center = (0, 0), labeldistance=1.1, autopct='%1.2f%%', pctdistance=0.5, shadow=True)
plt.title("Sex")

plt.show()
train[["Sex", "Survived"]].groupby(["Sex"]).mean().plot.bar()
sns.countplot("Sex", hue = "Survived", data = train)

plt.show()
sns.catplot(x = "Pclass", y = "Survived", hue = "Sex", data = train, height = 5, kind = "bar")

plt.show()
age = train["Age"]
result_age = pd.value_counts(age)
x = np.arange(0,90,0.1)

#age.isnull().sum()
age = age.dropna(axis = 0, how = "any") # Delete "nan" recoards
#print (age)

plt.bar(x,result_age[x])
plt.show
sns.violinplot(x = "Pclass", y = "Age", hue = "Survived", split = True, inner = "quart",data = train)

plt.show()
sns.violinplot(x = "Sex", y = "Age", hue = "Survived", split = True, inner = "quart",data = train)

plt.show()
sibsp = train["SibSp"]
result_sibsp = pd.value_counts(sibsp)
x_1 = np.arange(0,10,1)

parch = train["Parch"]
result_parch = pd.value_counts(parch)
x_2 = np.arange(0,10,1)

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

sns.barplot(x_1, result_sibsp[x_1], ax = ax1)
sns.barplot(x_2, result_parch[x_2], ax = ax2)
plt.show
sibsp_survived = pd.crosstab([train.SibSp],train.Survived)
print (sibsp_survived)

parch_survived = pd.crosstab([train.Parch],train.Survived)
print (parch_survived)
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), )

sns.barplot('SibSp','Survived', data=train, ax = ax1)
sns.barplot('Parch','Survived', data=train, ax = ax2)
class_fare = pd.crosstab([train.Pclass],train.Fare)
print (class_fare)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), )
sns.boxenplot(x = "Pclass", y = "Fare", color = "blue", scale = "linear", data = train, ax = ax1)
sns.violinplot(x = "Pclass", y = "Fare", hue = "Survived", split = True, inner = "quart",data = train, ax = ax2)
plt.show()
train = train.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Cabin'])

train.isnull().sum()
train.fillna({"Embarked":"S"},inplace=True)
train.isnull().sum()
ports = {"S": 0, "C": 1, "Q": 2}
train['Embarked'] = train['Embarked'].map(ports)

train.Embarked.describe()
genders = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(genders)

train.Sex.describe()
age_med = train.groupby(["Pclass","Sex"]).Age.median()
train.set_index(["Pclass","Sex"],inplace = True)
train.Age.fillna(age_med, inplace = True)
train.reset_index(inplace = True)

train.Age.describe()
train_test, train_eval = train_test_split(train, test_size = 0.2)

print (train_test)
print (train_eval)
train_test_learning = train_test.drop("Survived", axis = 1)
train_test_results = train_test["Survived"] # generate the results list

train_eval_learning = train_eval.drop("Survived", axis = 1)
train_eval_results = train_eval["Survived"] # generate the results list

print (train_test_learning, train_test_results)
# Eculidean distance calculation
def euclideanDistance(instance1,instance2,length):
    distance = 0
    for x in range(length):
        distance = pow((instance1[x] - instance2[x]),2)
    return math.sqrt(distance)
 
# Return K nearest distance
def getNeighbors(trainingSet,testInstance,k):
    distances = []
    length = len(testInstance) -1
    # Calculate test record to each train records
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x],dist))
    # Sort of all distance
    distances.sort(key = operator.itemgetter(1))
    neighbors = []
    # Return K nearest value
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors
 
# Merge all KNN and find the largest value
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    # Sort of the KNN
    sortedVotes = sorted(classVotes.items(),key = operator.itemgetter(1),reverse =True)
    return sortedVotes[0][0]
 
# Evaluate the model
def getAccuracy(testSet,predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct+=1
    return (correct/float(len(testSet))) * 100.0

# Convert the dataframe to array
trainingSet = pd.concat([train_test_learning,train_test_results],axis=1).values
testSet = pd.concat([train_eval_learning,train_eval_results],axis=1).values

# Generate the prediction list
predictions = []

# Define K value
k = 5

#print (trainingSet)

# Main Part
for x in range(len(testSet)):
    neighbors = getNeighbors(trainingSet, testSet[x], k)
    result = getResponse(neighbors)
    predictions.append(result)
    print (">predicted = " + repr(result) + ",actual = " + repr(testSet[x][-1]))
accuracy = getAccuracy(testSet, predictions)
print ("Accuracy:" + repr(accuracy) + "%")
knn = KNeighborsClassifier(n_neighbors = 20, weights = 'uniform', algorithm = 'auto', leaf_size = 30, p = 2, metric = 'minkowski', metric_params = None, n_jobs = 1)
knn.fit(train_test_learning, train_test_results)  
eval_pred_knn = knn.predict(train_eval_learning)  
acc_knn = round(knn.score(train_test_learning, train_test_results) * 100, 2)

#print (eval_pred)
#print (train_eval_results)
print (acc_knn)

accuracy_score(train_eval_results, eval_pred_knn)
class Bagging(object):
    # Initialization
    def __init__(self,n_estimators,estimator,rate=1.0):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.rate = rate

    def Voting(self,data):          # Define voting method
        term = np.transpose(data)   
        result = list()            

        def Vote(df):               # vote for each raw or each simple model output
            store = defaultdict()
            for kw in df:
                store.setdefault(kw, 0)
                store[kw] += 1
            return max(store,key = store.get)

        result = map(Vote,term)      # Generate results
        return result

    # Define Under-Sampling
    def UnderSampling(self,data):
        #np.random.seed(np.random.randint(0,1000))
        data = np.array(data)
        np.random.shuffle(data)    # Personally think shuffle is important          
        newdata = data[0:int(data.shape[0] * self.rate),:]   # Define the number of elements in new set
        return newdata   

    def TrainPredict(self,train,test):          # Build simple model
        clf = self.estimator.fit(train[:,0:-1],train[:,-1])
        result = clf.predict(test[:,0:-1])
        return result

    # General sampling method
    def RepetitionRandomSampling(self,data,number):     
        sample = []
        for i in range(int(self.rate * number)):
             sample.append(data[random.randint(0,len(data)-1)])
        return sample

    def Metrics(self,predict_data,test):        # Evaluation
        score = predict_data
        recall = recall_score(test[:,-1], score, average = None)    # Recall
        precision = precision_score(test[:,-1], score, average = None)  # Precision
        return recall,precision


    def MutModel_clf(self,train,test,sample_type = "RepetitionRandomSampling"):
        print ("self.Bagging Mul_basemodel")
        result = list()
        num_estimators = len(self.estimator)   

        if sample_type == "RepetitionRandomSampling":
            print ("Sample Method：",sample_type)
            sample_function = self.RepetitionRandomSampling
        elif sample_type == "UnderSampling":
            print ("Sample Method：",sample_type)
            sample_function = self.UnderSampling 
            print ("Sampling Rate",self.rate)
        elif sample_type == "IF_SubSample":
            print ("Sample Method：",sample_type)
            sample_function = self.IF_SubSample 
            print ("Sampling Rate",(1.0-self.rate))

        for estimator in self.estimator:
            print (estimator)
            for i in range(int(self.n_estimators/num_estimators)):
                sample = np.array(sample_function(train,len(train)))       
                clf = estimator.fit(sample[:,0:-1],sample[:,-1])
                result.append(clf.predict(test[:,0:-1]))      # Summerize simple model output

        score = self.Voting(result)
        recall,precosoion = self.Metrics(score,test)
        return recall,precosoion  

train_r = Bagging(trainingSet,100,10)

print (train_r)
bagging = BaggingClassifier(base_estimator = None, n_estimators = 10, max_samples = 1.0, max_features = 1.0, bootstrap = True, bootstrap_features = False, oob_score = False, warm_start = False, n_jobs = 1, random_state = None, verbose = 0)
bagging.fit(train_test_learning, train_test_results)

eval_pred_bg = bagging.predict(train_eval_learning)
acc_bg = round(bagging.score(train_test_learning, train_test_results) * 100, 2)

#print (eval_pred_bg)
print (acc_bg)

accuracy_score(train_eval_results, eval_pred_bg)
decision_tree = DecisionTreeClassifier(criterion = 'gini', splitter = 'best', max_depth = None, min_samples_split = 2, min_samples_leaf = 1, min_weight_fraction_leaf = 0.0, max_features = None, random_state = None, max_leaf_nodes = None, min_impurity_decrease = 0.0, min_impurity_split = None, class_weight = None, presort = False)
decision_tree.fit(train_test_learning, train_test_results)

eval_pred_dt = decision_tree.predict(train_eval_learning)
acc_dt = round(decision_tree.score(train_test_learning, train_test_results) * 100, 2)

#print (eval_pred_dt)
print (acc_dt)

accuracy_score(train_eval_results, eval_pred_dt)
random_forest = RandomForestClassifier(n_estimators = 100, criterion = 'gini', max_depth = None, min_samples_split = 2, min_samples_leaf = 1, min_weight_fraction_leaf = 0.0, max_features = 'auto', max_leaf_nodes = None, min_impurity_decrease = 0.0, min_impurity_split = None, bootstrap = True, oob_score = True, n_jobs = 1, random_state = None, verbose = 0, warm_start = False, class_weight = None)
random_forest.fit(train_test_learning, train_test_results)

eval_pred_rf = random_forest.predict(train_eval_learning)

acc_rf = round(random_forest.score(train_test_learning, train_test_results) * 100, 2)

#print (eval_pred_rf)
print (acc_rf)

accuracy_score(train_eval_results, eval_pred_rf)
logistic_regression = LogisticRegression(penalty = 'l2', dual = False, tol = 0.0001, C = 1.0, fit_intercept = True, intercept_scaling = 1, class_weight = None, random_state = None, solver = 'liblinear', max_iter = 100, multi_class = 'ovr', verbose = 0, warm_start = False, n_jobs = 1)
logistic_regression.fit(train_test_learning, train_test_results)


eval_pred_lr = logistic_regression.predict(train_eval_learning)

acc_lr = round(logistic_regression.score(train_test_learning, train_test_results) * 100, 2)

#print (eval_pred_lr)
print (acc_lr)

accuracy_score(train_eval_results, eval_pred_lr)
def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    if test_data:
        n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size] 
                for k in xrange(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print ("Epoch {0}: {1}/{2}".format(j, self.evaluate(test_data),n_test))
            else:
                print ("Epoch {0} complete".format(j))
sgd = SGDClassifier(loss = 'hinge', penalty = 'l2', alpha = 0.0001, l1_ratio = 0.15, fit_intercept = True, max_iter = None, tol = None, shuffle = True, verbose = 0, epsilon = 0.1, n_jobs = 1, random_state = None, learning_rate = 'optimal', eta0 = 0.0, power_t = 0.5, class_weight = None, warm_start = False, average = False, n_iter = None)
sgd.fit(train_test_learning, train_test_results)

eval_pred_sgd = sgd.predict(train_eval_learning)

acc_sgd = round(sgd.score(train_test_learning, train_test_results) * 100, 2)

print (acc_sgd)

accuracy_score(train_eval_results, eval_pred_sgd)
perceptron = Perceptron(penalty = None, alpha = 0.0001, fit_intercept = True, max_iter = None, tol = None, shuffle = True, verbose = 0, eta0 = 1.0, n_jobs = 1, random_state = 0, class_weight = None, warm_start = False, n_iter = None)
perceptron.fit(train_test_learning, train_test_results)


eval_pred_pp = perceptron.predict(train_eval_learning)

acc_pp = round(perceptron.score(train_test_learning, train_test_results) * 100, 2)

#print (eval_pred_pp)
print (acc_pp)

accuracy_score(train_eval_results, eval_pred_pp)
linear_svc = LinearSVC()
linear_svc.fit(train_test_learning, train_test_results)

eval_pred_liner_svc = linear_svc.predict(train_eval_learning)

acc_linear_svc = round(linear_svc.score(train_test_learning, train_test_results) * 100, 2)

print (acc_linear_svc)

accuracy_score(train_eval_results, eval_pred_liner_svc)
svc = SVC(C = 1.0, kernel = 'rbf', degree = 3, gamma = 'auto', coef0 = 0.0, shrinking = True, probability = False, tol = 0.001, cache_size = 200, class_weight = None, verbose = False, max_iter = -1, decision_function_shape = 'ovr', random_state = None)
svc.fit(train_test_learning, train_test_results)

eval_pred_svc = svc.predict(train_eval_learning)

acc_svc = round(svc.score(train_test_learning, train_test_results) * 100, 2)

print (acc_svc)

accuracy_score(train_eval_results, eval_pred_svc)
gaussian_naive_bayes = GaussianNB(priors = None)
gaussian_naive_bayes.fit(train_test_learning, train_test_results)

eval_pred_gnb = gaussian_naive_bayes.predict(train_eval_learning)

acc_gnb = round(gaussian_naive_bayes.score(train_test_learning, train_test_results) * 100, 2)

print (acc_gnb)

accuracy_score(train_eval_results, eval_pred_gnb)
def kmeans(data,k=2):
    def _distance(p1,p2):
        """
        Return Eclud distance between two points.
        p1 = np.array([0,0]), p2 = np.array([1,1]) => 1.414
        """
        tmp = np.sum((p1-p2)**2)
        return np.sqrt(tmp)
    def _rand_center(data,k):
        """Generate k center within the range of data set."""
        n = data.shape[1] # features
        centroids = np.zeros((k,n)) # init with (0,0)....
        for i in range(n):
            dmin, dmax = np.min(data[:,i]), np.max(data[:,i])
            centroids[:,i] = dmin + (dmax - dmin) * np.random.rand(k)
        return centroids
    
    def _converged(centroids1, centroids2):
        
        # if centroids not changed, we say 'converged'
         set1 = set([tuple(c) for c in centroids1])
         set2 = set([tuple(c) for c in centroids2])
         return (set1 == set2)
        
    
    n = data.shape[0] # number of entries
    centroids = _rand_center(data,k)
    label = np.zeros(n,dtype=np.int) # track the nearest centroid
    assement = np.zeros(n) # for the assement of our model
    converged = False
    
    while not converged:
        old_centroids = np.copy(centroids)
        for i in range(n):
            # determine the nearest centroid and track it with label
            min_dist, min_index = np.inf, -1
            for j in range(k):
                dist = _distance(data[i],centroids[j])
                if dist < min_dist:
                    min_dist, min_index = dist, j
                    label[i] = j
            assement[i] = _distance(data[i],centroids[label[i]])**2
        
        # update centroid
        for m in range(k):
            centroids[m] = np.mean(data[label==m],axis=0)
        converged = _converged(old_centroids,centroids)    
    return centroids, label, np.sum(assement)
#Trainset = train_test_learning.values

#print (Trainset)

kmeans = KMeans(n_clusters = 2, init = 'k-means++', n_init = 10, max_iter = 300, tol = 0.0001, precompute_distances = 'auto', verbose = 0, random_state = None, copy_x = True, n_jobs = 1, algorithm = 'auto')
kmeans.fit_predict(train_test_learning)
label_pred = kmeans.labels_
centroids = kmeans.cluster_centers_ # Clustering center
inertia = kmeans.inertia_ # Clustering inertia summary

#print (label_pred)
print (centroids)
print (inertia)

acc_k = accuracy_score(train_test_results, label_pred)*100

print (acc_k)
mlp = MLPClassifier(hidden_layer_sizes = (50, ), activation = 'relu', solver = 'adam', alpha = 0.0001, batch_size = 'auto', learning_rate = 'constant', learning_rate_init = 0.001, power_t = 0.5, max_iter = 200, shuffle = True, random_state = None, tol = 0.0001, verbose = False, warm_start = False, momentum = 0.9, nesterovs_momentum = True, early_stopping = False, validation_fraction = 0.1, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)
mlp.fit(train_test_learning, train_test_results)

eval_pred_mlp = mlp.predict(train_eval_learning)

acc_mlp = round(mlp.score(train_test_learning, train_test_results) * 100, 2)

print (acc_mlp)

accuracy_score(train_eval_results, eval_pred_mlp)
start = time() # use "time" function to calculate the model process time

model = Sequential() # very improtent, it defines the model is built one layer by one layer
model.add(Dense(input_dim=7, output_dim=1)) # .add means add a layer into model; dense is the layer I added, dense layer is fully connected layer
model.add(Activation("relu")) # add activation function, I choose "relu" for classification
# this is a single layer model, it is a simple one. If need, you can add more layers by using code .add
# take care, before you built the model, it is better to have a whole model topology 
# after this, we define the model topology

# next, we need activate model by using code .complie
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# we define the optimizer, loss function
# optimize can be custimize first, than add into the complie function
# metrics is used for evaluate the model, can put accuracy, score or cost into it

# train the model by using .fit
model.fit(train_test_learning, train_test_results)
# train data, train label
# can add the number of epoch and batch size as well.

loss, accuracy = model.evaluate(train_test_learning, train_test_results)
acc_nn = 100*accuracy

print (loss, accuracy)
print ('\ntime taken %s seconds' % str(time() - start))

# predic the test data
dp1_pred = model.predict_classes(train_eval_learning)
#print (dp1_pred)
#print (train_eval_results)
print ("\n\naccuracy", np.sum(dp1_pred == train_eval_results.values) / float(len(train_eval_results.values)))
comparesion = pd.DataFrame({
    'Model': ['KNN', 'bagging', 'Decision Tree', 'Random Forest', 'Logistic Regression', 
              'Stochastic Gradient Decent', 'Perceptron', 'Linear Support Vector Machines', 
              'Support Vector Machines', 'Naive Bayes', 'K-Means', 'N.N(sklearn)', 'N.N(keras)'],
    'Score': [acc_knn, acc_bg, acc_dt, acc_rf, acc_lr, acc_sgd, acc_pp, acc_linear_svc, acc_svc,
              acc_gnb, acc_k, acc_mlp, acc_nn
              ]})
comparesion_df = comparesion.sort_values(by='Score', ascending=False)
comparesion_df = comparesion_df.set_index('Score')
comparesion_df.head(14)
predictions = cross_val_predict(random_forest, train_test_learning, train_test_results, cv=3)
print (precision_score(train_test_results, predictions), recall_score(train_test_results, predictions))
y_scores = random_forest.predict_proba(train_test_learning)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(train_test_results, y_scores)

def plot_precision_vs_recall(precision, recall):
    plt.plot(recall, precision, "g--", linewidth=2.5)
    plt.ylabel("recall", fontsize=19)
    plt.xlabel("precision", fontsize=19)
    plt.axis([0, 1, 0, 1])

plt.figure(figsize=(10, 5))
plot_precision_vs_recall(precision, recall)
plt.show()
false_positive_rate, true_positive_rate, thresholds = roc_curve(train_test_results, y_scores)
# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(10, 5))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()
final = roc_auc_score(train_test_results, y_scores)

print (final)
test_mod = test.drop(columns = ['Name', 'Ticket', 'Cabin'])

age_med_test = test_mod.groupby(["Pclass","Sex"]).Age.median()
test_mod.set_index(["Pclass","Sex"],inplace = True)
test_mod.Age.fillna(age_med_test, inplace = True)
test_mod.reset_index(inplace = True)

fare_med_test = test_mod.groupby(["Pclass"]).Fare.median()
test_mod.set_index(["Pclass"],inplace = True)
test_mod.Fare.fillna(fare_med_test, inplace = True)
test_mod.reset_index(inplace = True)

test_mod['Embarked'] = test_mod['Embarked'].map(ports)
test_mod['Sex'] = test_mod['Sex'].map(genders)

test_mod.isnull().sum()

test_mod_pred = test_mod.drop("PassengerId", axis = 1)
test_mod_id = test_mod["PassengerId"] # generate the results list

pred = random_forest.predict(test_mod_pred)

print (pred)

submission = pd.DataFrame({
        "PassengerId": test_mod_id,
        "Survived": pred
    })

submission.to_csv("submission.csv",index=False)