# Load libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn import metrics, neighbors, linear_model
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
# Creating dataframes
col = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
train_df = pd.read_csv('../input/training-data.txt', sep=',', header=None, names=col)
test_df = pd.read_csv('../input/testing-data.txt', sep=',', header=None, names=col)
# Splitting the dataframes
X_train = train_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
train_label_df = train_df[['species']].replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
train_label_array = train_label_df.as_matrix()
Y_train = np.ravel(train_label_array)


X_test = test_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
test_label_df = test_df[['species']].replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
test_label_array = test_label_df.as_matrix()
Y_test = np.ravel(test_label_array)
# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=90, learning_rate=0.2)

# Train Adaboost Classifer
model = abc.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = model.predict(X_test)
Y_pred = pd.DataFrame(Y_pred, columns=['species'])

print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
# source:
# http://www.insightsbot.com/blog/C8Fm4/perceptron-algorithm-part-2-python-code-machine-learning-101

import numpy as np
import pandas as pd

class Perceptron(object):
    #The constructor of our class.
    def __init__(self, learningRate=0.2, n_iter=50, random_state=1):
        self.learningRate = learningRate
        self.n_iter = n_iter
        self.random_state = random_state
        self.errors_ = []
        
    def fit(self, X, y):
        #for reproducing the same results
        random_generator = np.random.RandomState(self.random_state)
        
        #Step 0 = Get the shape of the input vector X
        #We are adding 1 to the columns for the Bias Term
        x_rows, x_columns = X.shape
        x_columns = x_columns+1
        
        #Step 1 - Initialize all weights to 0 or a small random number  
        #weight[0] = the weight of the Bias Term
        self.weights = random_generator.normal(loc=0.0, scale=0.001, size=x_columns) 
        
        #for how many number of training iterrations where defined
        for _ in range(self.n_iter):
            errors = 0
            for xi, y_actual in zip(X, y):
                #create a prediction for the given sample xi
                y_predicted = self.predict(xi)
                #print(y_actual, y_predicted)
                #calculte the delta
                delta = self.learningRate*(y_actual - y_predicted)
                #update all the weights but the bias
                self.weights[1:] += delta * xi
                #for the bias delta*1 = delta
                self.weights[0] += delta

                #if there is an error. Add to the error count for the batch
                errors += int(delta != 0.0)

            #add the error count of the batch to the errors variable
            self.errors_.append(errors)           
        
        #print(self.errors_)
            
    def Errors(self):
        return self.errors_
    
    def z(self, X):
        #np.dot(X, self.w_[1:]) + self.w_[0]
        z = np.dot(X, self.weights[1:]) + self.weights[0]
        return z
        
    def predict(self, X):
        #Heaviside function. Returns 1 or 0 
        return np.where(self.z(X) >= 0.0, 1, 0)
    
ppn = Perceptron(learningRate=0.2, n_iter=15)
X_train_array = X_train.as_matrix()
ppn.fit(X_train_array, Y_train)  
print(ppn.errors_)
print(metrics.accuracy_score(ppn.errors_, Y_test))
# dt=DecisionTreeClassifier(max_depth=1,min_samples_leaf=int(0.5*len(X_train)))
percep = linear_model.Perceptron(tol=1e-3, random_state=0)
boosted_dt=AdaBoostClassifier(percep,algorithm='SAMME',n_estimators=90,learning_rate=0.2)
boosted_dt.fit(X_train,Y_train)
Y_predicted=boosted_dt.predict(X_test)
print(metrics.accuracy_score(Y_test, Y_predicted))
