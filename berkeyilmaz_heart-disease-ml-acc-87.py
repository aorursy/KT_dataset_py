# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data= pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
data.head()
data.tail()
data.isnull().any() # firstly check any data has nulled or missing value
data['target'].value_counts() 
y = data.target.values

y
x_data=data.drop('target',axis=1)

x_data
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

# (x - min(x))/(max(x)-min(x))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42) 

#We seperated data here %80 train %20 test. We dont touch test for matching our predict data



x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T

# we must that values transpose for matching arrays For example x_train:  (242, 13) matris we need 13,242 1,242 arrays not important. if 2>= important.

print("x_train: ",x_train.shape)

print("x_test: ",x_test.shape)

print("y_train: ",y_train.shape)

print("y_test: ",y_test.shape)
def initialize_weights_and_bias(dimension):

    

    w = np.full((dimension,1),0.01)

    b = 0.0

    return w,b





# w,b = initialize_weights_and_bias(30)



def sigmoid(z):

    

    y_head = 1/(1+ np.exp(-z)) 

    return y_head

# print(sigmoid(0))sigmoid rules, give 0 return "0.5" you can check that way your formules are good/bad.
def forward_backward_propagation(w,b,x_train,y_train):

    # forward propagation

    z = np.dot(w.T,x_train) + b  #b0+b1.x

    y_head = sigmoid(z) # normalise value 0-1 because some values are 400 some of 5 that way we see equal values

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling ( /242)normalise that value wşth division

    

    # backward propagation

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling

    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

    

    return cost,gradients
def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):

    cost_list = []

    cost_list2 = []

    index = []

    

    # updating(learning) parameters is number_of_iterarion times

    for i in range(number_of_iterarion):

        # make forward and backward propagation and find cost and gradients

        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        # lets update

        w = w - learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"]

        if i % 10 == 0:

            cost_list2.append(cost)

            index.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))

            

    # we update(learn) parameters weights and bias

    parameters = {"weight": w,"bias": b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list
def predict(w,b,x_test):

    # x_test is a input for forward propagation

    z = sigmoid(np.dot(w.T,x_test)+b)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    # if z is bigger than 0.5, our prediction is sign one (y_head=1),

    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),

    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):

    # initialize

    dimension =  x_train.shape[0]  # that is 30

    w,b = initialize_weights_and_bias(dimension)

    # do not change learning rate

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)



    # Print test Errors

    

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 210)    

print(x_train)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train.T,y_train.T)

print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))

color_list = ['red' if i=='0' else 'green' for i in data.loc[:,'target']]

pd.plotting.scatter_matrix(data.loc[:, data.columns != 'target'],

                                       c=color_list,

                                       figsize= [15,15],

                                       diagonal='hist',

                                       alpha=0.5,

                                       s = 200,

                                       marker = '*',

                                       edgecolor= "black")

plt.show()
sns.countplot(x="target", data=data)

data.loc[:,'target'].value_counts()
# KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

x,y = data.loc[:,data.columns != 'target'], data.loc[:,'target']

knn.fit(x,y)

prediction = knn.predict(x)

print('Prediction: {}'.format(prediction))
# train test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)

knn = KNeighborsClassifier(n_neighbors = 3)

x,y = data.loc[:,data.columns != 'target'], data.loc[:,'target']

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

#print('Prediction: {}'.format(prediction))

print('With KNN (K=3) accuracy is: ',knn.score(x_test,y_test)) # accuracy
# Model complexity

neig = np.arange(1, 25)

train_accuracy = []

test_accuracy = []

# Loop over different values of k

for i, k in enumerate(neig):

    # k from 1 to 25(exclude)

    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit with knn

    knn.fit(x_train,y_train)

    #train accuracy

    train_accuracy.append(knn.score(x_train, y_train))

    # test accuracy

    test_accuracy.append(knn.score(x_test, y_test))



# Plot

plt.figure(figsize=[13,8])

plt.plot(neig, test_accuracy, label = 'Testing Accuracy')

plt.plot(neig, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.title('-value VS Accuracy')

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.xticks(neig)

plt.savefig('graph.png')

plt.show()

print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
# create data1 that includes pelvic_incidence that is feature and sacral_slope that is target variable

data1 = data[data['target'] == 1]

x = np.array(data1.loc[:,'cp']).reshape(-1,1)

y = np.array(data1.loc[:,'trestbps']).reshape(-1,1)

# Scatter

plt.figure(figsize=[10,10])

plt.scatter(x=x,y=y)

plt.xlabel('pelvic_incidence')

plt.ylabel('sacral_slope')

plt.show()
# LinearRegression

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

# Predict space

predict_space = np.linspace(min(x), max(x)).reshape(-1,1)

# Fit

reg.fit(x,y)

# Predict

predicted = reg.predict(predict_space)

# R^2 

print('R^2 score: ',reg.score(x, y))

# Plot regression line and scatter

plt.plot(predict_space, predicted, color='black', linewidth=3)

plt.scatter(x=x,y=y)

plt.xlabel('pelvic_incidence')

plt.ylabel('sacral_slope')

plt.show()
from sklearn.model_selection import cross_val_score

reg = LinearRegression()

k = 5

cv_result = cross_val_score(reg,x,y,cv=k) # uses R^2 as score 

print('CV Scores: ',cv_result)

print('CV scores average: ',np.sum(cv_result)/k)
# Ridge

from sklearn.linear_model import Ridge

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 2, test_size = 0.3)

ridge = Ridge(alpha = 0.1, normalize = True)

ridge.fit(x_train,y_train)

ridge_predict = ridge.predict(x_test)

print('Ridge score: ',ridge.score(x_test,y_test))
data1

# Confusion matrix with random forest

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

x,y = data.loc[:,data.columns != 'target'], data.loc[:,'target']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)

rf = RandomForestClassifier(random_state = 4)

rf.fit(x_train,y_train)

y_pred = rf.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

print('Confusion matrix: \n',cm)

print('Classification report: \n',classification_report(y_test,y_pred))
# visualize with seaborn library

sns.heatmap(cm,annot=True,fmt="d") 

plt.show()
# ROC Curve with logistic regression

from sklearn.metrics import roc_curve

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report

# abnormal = 1 and normal = 0

data['target_binary'] = [1 if i == 1 else 0 for i in data.loc[:,'target']]

x,y = data.loc[:,(data.columns != 'target') & (data.columns != 'target_binary')], data.loc[:,'target_binary']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=42)

logreg = LogisticRegression()

logreg.fit(x_train,y_train)

y_pred_prob = logreg.predict_proba(x_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC')

plt.show()

# grid search cross validation with 1 hyperparameter

from sklearn.model_selection import GridSearchCV

grid = {'n_neighbors': np.arange(1,50)}

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid, cv=3) # GridSearchCV

knn_cv.fit(x,y)# Fit



# Print hyperparameter

print("Tuned hyperparameter k: {}".format(knn_cv.best_params_)) 

print("Best score: {}".format(knn_cv.best_score_))
# As you can see there is no labels in data

plt.scatter(data['trestbps'],data['chol'])

plt.xlabel('trestbps')

plt.ylabel('chol')

plt.show()
# KMeans Clustering

data2 = data.loc[:,['trestbps','chol']]

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 2)

kmeans.fit(data2)

labels = kmeans.predict(data2)

plt.scatter(data['trestbps'],data['chol'],c = labels)

plt.xlabel('trestbps')

plt.xlabel('chol')

plt.show()
# cross tabulation table

df = pd.DataFrame({'labels':labels,"target":data['target']})

ct = pd.crosstab(df['target'],df['target'])

print(ct)
# inertia

inertia_list = np.empty(8)

for i in range(1,8):

    kmeans = KMeans(n_clusters=i)

    kmeans.fit(data2)

    inertia_list[i] = kmeans.inertia_

plt.plot(range(0,8),inertia_list,'-o')

plt.xlabel('Number of cluster')

plt.ylabel('Inertia')

plt.show()
data3 = data.drop('target',axis = 1)
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

scalar = StandardScaler()

kmeans = KMeans(n_clusters = 2)

pipe = make_pipeline(scalar,kmeans)

pipe.fit(data3)

labels = pipe.predict(data3)

df = pd.DataFrame({'labels':labels,"target":data['target']})

ct = pd.crosstab(df['labels'],df['target'])

print(ct)
from scipy.cluster.hierarchy import linkage,dendrogram



merg = linkage(data3.iloc[200:220,:],method = 'single')

dendrogram(merg, leaf_rotation = 90, leaf_font_size = 6)

plt.show()

from sklearn.manifold import TSNE

model = TSNE(learning_rate=100)

transformed = model.fit_transform(data2)

x = transformed[:,0]

y = transformed[:,1]

plt.scatter(x,y,c = color_list )

plt.xlabel('trestbps')

plt.xlabel('chol')

plt.show()
# PCA

from sklearn.decomposition import PCA

model = PCA()

model.fit(data3)

transformed = model.transform(data3)

print('Principle components: ',model.components_)
# PCA variance

scaler = StandardScaler()

pca = PCA()

pipeline = make_pipeline(scaler,pca)

pipeline.fit(data3)



plt.bar(range(pca.n_components_), pca.explained_variance_)

plt.xlabel('PCA feature')

plt.ylabel('variance')

plt.show()
data3
# apply PCA

pca = PCA(n_components = 2)

pca.fit(data3)

transformed = pca.transform(data3)

x = transformed[:,0]

y = transformed[:,1]

plt.scatter(x,y,c = color_list)

plt.show()
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
random_state = 42

classifier = [DecisionTreeClassifier(random_state = random_state),

             SVC(random_state = random_state),

             RandomForestClassifier(random_state = random_state),

             LogisticRegression(random_state = random_state),

             KNeighborsClassifier()]



dt_param_grid = {"min_samples_split" : range(10,500,20),

                "max_depth": range(1,20,2)}



svc_param_grid = {"kernel" : ["rbf"],

                 "gamma": [0.001, 0.01, 0.1, 1],

                 "C": [1,10,50,100,200,300,1000]}



rf_param_grid = {"max_features": [1,3,10],

                "min_samples_split":[2,3,10],

                "min_samples_leaf":[1,3,10],

                "bootstrap":[False],

                "n_estimators":[100,300],

                "criterion":["gini"]}



logreg_param_grid = {"C":np.logspace(-3,3,7),

                    "penalty": ["l1","l2"]}



knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),

                 "weights": ["uniform","distance"],

                 "metric":["euclidean","manhattan"]}

classifier_param = [dt_param_grid,

                   svc_param_grid,

                   rf_param_grid,

                   logreg_param_grid,

                   knn_param_grid]
cv_result = []

best_estimators = []

for i in range(len(classifier)):

    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)

    clf.fit(x_train,y_train)

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])
cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",

             "LogisticRegression",

             "KNeighborsClassifier"]})



g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Scores")
votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),

                                        ("rfc",best_estimators[2]),

                                        ("lr",best_estimators[3])],

                                        voting = "hard", n_jobs = -1)

votingC = votingC.fit(x_train, y_train)

print(accuracy_score(votingC.predict(x_test),y_test))
model = RandomForestClassifier(max_depth=5)

model.fit(x_train, y_train)
estimator = model.estimators_[1]

feature_names = [i for i in x_train.columns]



y_train_str = y_train.astype('str')

y_train_str[y_train_str == '0'] = 'no disease'

y_train_str[y_train_str == '1'] = 'disease'

y_train_str = y_train_str.values
from sklearn.tree import export_graphviz #plot tree
#code from https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c



export_graphviz(estimator, out_file='tree.dot', 

                feature_names = feature_names,

                class_names = y_train_str,

                rounded = True, proportion = True, 

                label='root',

                precision = 2, filled = True)



from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



from IPython.display import Image

Image(filename = 'tree.png')