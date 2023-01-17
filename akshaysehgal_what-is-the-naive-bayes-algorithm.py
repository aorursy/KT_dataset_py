# Loading the dependencies for the model
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import math
# Loading the iris data into X, y. Each dataset will have its own way of doing this.
iris = load_iris()
X = iris.data
y = iris.target
y_labels = iris.target_names
X_labels = iris.feature_names
#4 features, 150 samples, 3 target values to predict
print("Column names - ",X_labels)
print("Shape of X - ",X.shape)
print("Target values - ",y_labels)
#Separating test and train data for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
#Instantiate the classifier and fit it to training data
nb = GaussianNB()
nb.fit(X_train, y_train)
#Evaluation of the model using test data
nb.score(X_test, y_test)
# Model's prediction when sepal_len = 5.8, sepal_wid = 2.2, petal_len = 4.2, petal_wid = 0.4
nb.predict([[5.8,2.2,4.2,0.4]])
#Putting all the data into a dataframe
df = pd.DataFrame(X, columns=X_labels)
df['class'] = y
df.head()
#Calculating class priors (simply the probability of class labels)
priors = np.unique(y, return_counts=True)[1] / len(y)
priors
#Class wise mean for each feature
classwise_means = df.groupby(['class']).mean().reset_index()
classwise_means
#Class wise mean for each feature
classwise_std = df.groupby(['class']).std().reset_index()
classwise_std
# Gaussian probability function vectorized to work on multidimensional arrays with broadcasting
def pdf(x, mean, sd):
    return (1 / np.sqrt(2*np.pi*sd**2)) * np.exp(-1*((x-mean)**2/(2*sd**2)))

vectorized_pdf = np.vectorize(pdf)
#Converting everything to numpy arrays
mean_array = np.array(classwise_means.iloc[:,1:])
std_array = np.array(classwise_std.iloc[:,1:])

#Sample is the feature vector, whose label needs to be predicted
sample = [5.1, 3.5, 1.4, 0.2]
#Element wise application of the pdf(x,mean,std) function on broadcasted arrays
elementwise_pdf = vectorized_pdf(sample, mean_array, std_array)
elementwise_pdf
#Rowwise product of the above matrix
rowwise_product = np.product(elementwise_pdf,axis=1)
rowwise_product
#Elementwise multiplication with corresponding class probabilities (priors)
multiply_priors = np.multiply(priors,rowwise_product)
multiply_priors
#Argmax to predict class label
prediction = np.argmax(multiply_priors)
prediction
from sklearn.feature_extraction.text import CountVectorizer
#Defining the training data
documents = ['quite happy with it', 
             'bad device', 
             'great job with the features', 
             'bad experience',
             'horrible device',
             'very happy with the product']

classes = ['positive','negative','positive','negative','negative','positive']
#Fitting the count vectorizer to get word representation for each sentence
cnt = CountVectorizer()
cnt_matrix = cnt.fit_transform(documents).todense()
cnt_matrix.shape
#Creating a dataframe to calculate summary statistics later
training_data = pd.DataFrame(cnt_matrix,
                                index = documents, 
                                columns=cnt.get_feature_names())
training_data['Class'] = classes
training_data
#Calculating label probability (priors)
label_proba = np.unique(classes, return_counts=True)[1]/len(classes)
label_proba
#Calculating word frequency for each word in training data (label-wise)

## 0 = negative, 1 = positive
Nc = training_data.groupby(['Class']).sum().reset_index('Class').drop('Class',axis=1)
Nc
# Calculating N which is the total number of word occurances associated with each label
N = np.sum(np.array(Nc),axis=1)
N
#Calculating length of vocabulary
d = len(cnt.get_feature_names())
d
#Setting value of the laplace smoothing factor alpha
a = 1
#Getting theta for a single input word
def get_thetas(word):
    try:
        return (np.array(Nc[word]) + a) / (N+d)
    except:
        return (np.array([0]*len(Nc)) + a) / (N+d)
    
get_thetas('happy')
#The sentence for which label needs to be predicted (word tokenized from start)
sample = ['happy', 'with', 'the', 'product']
# Calculating product of the probabilities for each word (label-wise)
probability_product = np.product(np.array([list(get_thetas(i)) for i in sample]), axis=0)
probability_product
#Prediction of label using argmax after multiplying the thetas with label probabilities (priors)
np.argmax(np.multiply(label_proba, probability_product))