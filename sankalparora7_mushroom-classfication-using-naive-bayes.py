# I have used the naive bayes classifier to get the highrst score
## Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
df=pd.read_csv("../input/mushroom-classification/mushrooms.csv")
df.head()
print(df.shape)
le=LabelEncoder()
#apply the tranform to each columns
ds=df.apply(le.fit_transform)
ds.head()
ds.shape
#convert into numpy arrays
data=ds.values
data.shape
print(data[:5,:])
#break the data into x an y
y=data[:,0]
x=data[:,1:]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(x_test)
print(y_test)
np.unique(y_train)
def prior_prob(y_train,label):
    total_examples=y_train.shape[0]
    class_examples=np.sum(y_train==label)
    return class_examples/float(total_examples)
    
#prior_prob(y,1)
def cond_prob(x_train,y_train,feature_col,feature_val,label):
    x_filtered=x_train[y_train==label]
    numerator=np.sum(x_filtered[:,feature_col]==feature_val)
    denominator=np.sum(y_train==label)
    return numerator/float(denominator)

def predict(x_train,y_train,xtest):
    """Xtest is a single testing point, n features"""
    
    classes = np.unique(y_train)
    n_features = x_train.shape[1]
    post_probs = [] # List of prob for all classes and given a single testing point
    #Compute Posterior for each class
    for label in classes:
        
        #Post_c = likelihood*prior
        likelihood = 1.0
        for f in range(n_features):
            cond = cond_prob(x_train,y_train,f,xtest[f],label)
            likelihood *= cond 
            
        prior = prior_prob(y_train,label)
        post = likelihood*prior
        post_probs.append(post)
        
    pred = np.argmax(post_probs)
    return pred
output = predict(x_train,y_train,x_test[1])
print(output)
print(y_test[1])

def score(x_train,y_train,x_test,y_test):

    pred = []
    for i in range(x_test.shape[0]):
        pred_label = predict(x_train,y_train,x_test[i])
        pred.append(pred_label) # <===Correction
    
    pred = np.array(pred)
    
    accuracy = np.sum(pred==y_test)/y_test.shape[0]
    return accuracy
print(score(x_train,y_train,x_test,y_test))


