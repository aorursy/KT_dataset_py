# Accuracy and Confusion matrices

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# A simple function to display the confusion matrix (accuracy, precision, recall ...)
def my_confusion_matrix(y_test, y_pred):
    
    #Create Confusion Matrix
    # y_pred = DecTreeClf.predict(x_test)


    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# A simple function to display just Accuracy
def my_accuracy(y_test,y_pred):
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    print('Acc: {:.4f}'.format(acc))
## Function to split features and classes into train and test split
def split_data(features,classes,test_size = 0.25):
    from sklearn.model_selection import train_test_split
    import pandas as pd
    # Split features in 75%,25% proportion for the model
    x_train, x_test, y_train, y_test =  train_test_split(all_features,all_classes, test_size = 0.25, random_state = 0)
    return x_train, x_test, y_train, y_test
### DECISION TREE CLASSIFIER
def fun_decisiontree(x_train,y_train):
    from sklearn import tree
    # from sklearn.ensemble import RandomForestClassifier
    import graphviz

    DecTreeClf = tree.DecisionTreeClassifier()
    DecTreeClf = DecTreeClf.fit(x_train,y_train)
    return DecTreeClf

### RANDOM FOREST CLASSIFIER
def fun_randomforest(x_train,y_train):
    from sklearn.ensemble import RandomForestClassifier

    # Create the random Forest Classifier and the corresponding K-Fold score
    RandForestClf = RandomForestClassifier()
    RandForestClf.fit(x_train,y_train)
    return RandForestClf

### SVM.SVC  CLASSIFIER    
def fun_SVM(x_train, y_train,C=1.0,kernel='linear'):    
    from sklearn import svm, datasets

    # Build the model and the corresponding predictions
    C = 1.0
    svc = svm.SVC(kernel='kernel',C=C).fit(x_train, y_train)
    return svc
# Read the mammographic csv file and put it into a panda
def fun_read_csv_data():
    import pandas as pd
    columns_list = ["BI-RADS","Age","Shape","Margin","Density","Severity"]
    masses = pd.read_csv("../input/mammograph-data-set/mammographic_masses.data.txt",na_values = "?", names = columns_list)
    # masses = pd.read_csv("https://data.world/uci/mammographic-mass/file/mammographic_masses.data.csv",na_values = "?", names = columns_list)
    return masses
    

masses = fun_read_csv_data()
    
masses.head()
masses.describe() 
# import pandas as pd
# columns_list = ["BI-RADS","Age","Shape","Margin","Density","Severity"]
# masses = pd.read_csv("../input/mammograph-data-set/mammographic_masses.data.txt",na_values = "?", names = columns_list)
# # masses = pd.read_csv("https://data.world/uci/mammographic-mass/file/mammographic_masses.data.csv",na_values = "?", names = columns_list)

# masses.head()
# masses.describe()
# Check the Panda to see what's inside
# And check out we've opened it properly

from matplotlib import pyplot as plt
# print (masses.describe())
masses.describe()
print ("count : " + str(masses.count()))



# Create a panda that will hold "Non Nana" data (if we drop nan, outliers etc ...)
# This will allow us to first check how many rows would be dropped
masses_nonan = masses.dropna()
masses_nonan.describe()

# In this section we will create histograms plot for each paramaters
# And we will put 2 bars for each
# One with all the rows and one with all NAN values dropped
# This will allow us to check whether NAN value are randomly distributed
# -> i.e.: that simply dropping NaN will not create artificial bias

# Plot age distribution
plt.hist(masses["Age"],color="blue",label="Full file")
plt.hist(masses_nonan["Age"],color="red",label = "with nan dropped")
plt.title("Age distribution")
plt.legend()
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Plot Shape distribution
plt.hist(masses["Shape"],color="blue",label="Full file")
plt.hist(masses_nonan["Shape"],color="red",label = "with nan dropped")
plt.title("Shape distribution")
plt.legend()
plt.xlabel("Shape")
plt.ylabel("Count")
plt.show()

# Plot Margin distribution
plt.hist(masses["Margin"],color="blue",label="Full file")
plt.hist(masses_nonan["Margin"],color="red",label = "with nan dropped")
plt.title("Margin distribution")
plt.legend()
plt.xlabel("Margin")
plt.ylabel("Count")
plt.show()

# Plot Density distribution
plt.hist(masses["Density"],color="blue",label="Full file")
plt.hist(masses_nonan["Density"],color="red",label = "with nan dropped")
plt.title("Density")
plt.legend()
plt.xlabel("Density")
plt.ylabel("Count")
plt.show()

# Plot Shape distribution
plt.hist(masses["Severity"],color="blue",label="Full file")
plt.hist(masses_nonan["Severity"],color="red",label = "with nan dropped")
plt.title("Severity distribution")
# plt.legend()
plt.xlabel("Severity")
plt.ylabel("Count")
plt.show()
# Yep! It looks like the NaN are randomly distributed. It should be safe to drop them

# Create a panda that will hold "cleaned" data (if we drop nan, outliers etc ...)
masses = masses.dropna()
masses.describe()
def fun_convert_numpy(masses):

    import numpy as np

# # Convert the pandas into numpy for later on
# # Unlike what it is being said, we'll keep it into a single numpy for now (we'll split features and labels later)
    feature_names = ["Age","Shape","Margin","Density"]

    all_features = masses[feature_names].values
    all_classes = masses['Severity'].values
    return all_features,all_classes

all_features, all_classes = fun_convert_numpy(masses)
# print (masses)
print (all_features)
print (all_classes)
def fun_get_scaled_features(masses):
    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    
    all_features,all_classes = fun_convert_numpy(masses)
    
    # Scale features data using StandardScaler 
    # No need to scale the labels in all_classes as they are just 0 and 1's
    scaler = StandardScaler()
    all_features = scaler.fit_transform(all_features)
    return all_features,all_classes

all_features,all_classes = fun_get_scaled_features(masses)
print (all_features)
print (all_classes)

def split_data(all_features,all_classes):
    from sklearn.model_selection import train_test_split
    import pandas as pd


    # Split features in 75%,25% proportion for the model
    x_train, x_test, y_train, y_test =  train_test_split(all_features,all_classes, test_size = 0.25, random_state = 0)
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = split_data(all_features,all_classes)
from sklearn import tree
# from sklearn.ensemble import RandomForestClassifier
import graphviz

DecTreeClf = tree.DecisionTreeClassifier(random_state=1)
DecTreeClf = DecTreeClf.fit(x_train,y_train)
import graphviz

# Save tree as dot file
dot_data = tree.export_graphviz(DecTreeClf, out_file=None) 
graph = graphviz.Source(dot_data)  
graph 

# Call the prediction function
y_pred = DecTreeClf.predict(x_test)

# Display the confusion matrix to check result (accuracy and so on)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import svm

# Create K-Fold scores
scores = cross_val_score(DecTreeClf, all_features, all_classes, cv=10)

# Print the accuracy for each fold:
print("***SCORES :*** ")
print(scores)
print("*************\n")

# And the mean accuracy of all 5 folds:
print("***MEAN:***")
print(scores.mean())
print("*************\n")
from sklearn.ensemble import RandomForestClassifier

# Create the random Forest Classifier and the corresponding K-Fold score
RandForestClf = RandomForestClassifier()
scores = cross_val_score(RandForestClf, all_features, all_classes, cv=10)

# Print the accuracy for each fold:

print("***SCORES :*** ")
print(scores)
print("*************\n")

# And the mean accuracy of all 5 folds:
print("***MEAN:***")
print(scores.mean())
print("*************\n")

from sklearn import svm, datasets

# Build the model and the corresponding predictions
C = 1.0
svc = svm.SVC(kernel='linear', C=C,random_state=1).fit(x_train, y_train)
y_pred = svc.predict(x_test)


y_pred = svc.predict(x_test)

#Create Confusion Matrix

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
def my_confusion_matrix(y_test, y_pred):
    
    #Create Confusion Matrix

    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def my_accuracy(y_test,y_pred):
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    print('Acc: {:.4f}'.format(acc))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def KNN_loop(x_train,y_train,x_test,MAX):
    
    # Loop MAX times
    for i in range(MAX):
        # Create the corresponding KNN Classifier 
        KNeighClf = KNeighborsClassifier(n_neighbors=i+1)
        
        # Fit and predict the model
        KNN = KNeighClf.fit(x_train, y_train)
        y_pred = KNN.predict(x_test)
        
        # Display the results
        # my_confusion_matrix(y_test, y_pred)
        print ("Neighbors = " + str(i+1))
        my_accuracy(y_test,y_pred)

KNN_loop(x_train,y_train,x_test,50)
# KNN with K=14 seems the best fit
# Run it again with full results this time
KNeighClf = KNeighborsClassifier(14)
KNN = KNeighClf.fit(x_train, y_train)

y_pred = KNN.predict(x_test)

my_confusion_matrix(y_test, y_pred)
my_accuracy(y_test,y_pred)
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler

#Define Classifier
MNomNBClf = MultinomialNB()

# define min max scaler
scaler = MinMaxScaler()
# transform data
scaled_train = scaler.fit_transform(x_train)
scaled_test = scaler.fit_transform(x_test)

# Train using the Classifier and training data
MNomNBClf.fit(scaled_train, y_train)
y_pred = MNomNBClf.predict(scaled_test)

#Print Results
my_confusion_matrix(y_test, y_pred)
my_accuracy(y_test,y_pred)

# print(scaled)
# print(x_train)
from sklearn import svm, datasets
#USE SVM RBF
# Build the model and the corresponding predictions
C = 1.0
svc = svm.SVC(kernel='rbf', C=C).fit(x_train, y_train)
y_pred = svc.predict(x_test)

# Print Results
my_confusion_matrix(y_test, y_pred)
my_accuracy(y_test,y_pred)
from sklearn import svm, datasets
#USE SVM Sigmoid
# Build the model and the corresponding predictions
C = 1.0
svc = svm.SVC(kernel='sigmoid', C=C).fit(x_train, y_train)
y_pred = svc.predict(x_test)

# Print Results
my_confusion_matrix(y_test, y_pred)
my_accuracy(y_test,y_pred)
from sklearn import svm, datasets
#USE SVM Poly
# Build the model and the corresponding predictions
C = 1.0
svc = svm.SVC(kernel='poly', C=C).fit(x_train, y_train)
y_pred = svc.predict(x_test)

# Print Results
my_confusion_matrix(y_test, y_pred)
my_accuracy(y_test,y_pred)
import statsmodels.api as sm

# This is a simple fit using SM.ols
# use summary for results
est = sm.OLS(y_train, scaled_train).fit()
est.summary()
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
cv_scores = cross_val_score(clf, all_features, all_classes, cv=10)
cv_scores.mean()
# The data will need to be rescaled for the Neural Network!
# Create the scaler, change type to float32 and apply the scaler to the features
scaler = MinMaxScaler()
all_features.astype('float32')
all_features = scaler.fit_transform(all_features)

# As the labels are just 0 or 1 (for benignant or malignant tumors) there is no need to rescale them
all_classes.astype('float32')
all_classes = all_classes.reshape(-1,1)
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import cross_val_score

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras


# This section will contain the hyperparameters
# hyper_batch_size=1 # in case we want to try the SGD method
hyper_epochs=100
hyper_drop_out=0.25
hyper_learning_rate = 0.025
hyper_optimizer = keras.optimizers.Adam(learning_rate=hyper_learning_rate)
hyper_verbose = 0

#### This Function creates the topography of the model
def create_model():
    model = Sequential()
    
    model.add(Dense(12,input_dim=4, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(hyper_drop_out))
    
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(hyper_drop_out))
    
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(hyper_drop_out))
    
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    opt = keras.optimizers.Adam(learning_rate=hyper_learning_rate)    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

estimator = KerasClassifier(build_fn=create_model,
#                             batch_size=hyper_batch_size,
                            epochs=hyper_epochs,
                            verbose=hyper_verbose)



# Now we can use scikit_learn's cross_val_score to evaluate this model identically to the others
cv_scores = cross_val_score(estimator, all_features, all_classes, cv=10)
cv_scores.mean()
# Let's rebuild our data entirely from scratch
# Much better for this since we'll be using one-hot-encoding for the first time
# And it has been a long while since we've done this

# Import section
import numpy as np 
import pandas as pd
# importing one hot encoder from sklearn 
from sklearn.preprocessing import OneHotEncoder 


# Let's start with fresh data
feature_names = ["Age","Shape","Margin","Density"]
all_features = masses[feature_names].values
all_classes = masses['Severity'].values


# The data will need to be rescaled for the Neural Network!
# Create the scaler, change type to float32 and apply the scaler to the features
scaler = MinMaxScaler()
all_features.astype('float32')
all_features = scaler.fit_transform(all_features)

# As the labels are just 0 or 1 (for benignant or malignant tumors) there is no need to rescale them
all_classes.astype('float32')
all_classes = all_classes.reshape(-1,1)



# creating one hot encoder object by default 
# We'll one hot encode Shape, Margin and Density
OH_columns = ["Shape","Margin","Density"]

# Create the one hot encoder object
onehotencoder = OneHotEncoder() 

OH_all_features = np.delete(all_features, np.s_[:1], axis=1) 
age_feature = np.delete(all_features,np.s_[1:],axis=1)

# print(type(OH_all_features))
# print(type(age_feature))
# print(OH_all_features.shape)


OH_all_features = OneHotEncoder().fit_transform(OH_all_features).toarray()
all_features = np.append(age_feature, OH_all_features, axis=1)
print(all_features.shape)


from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import cross_val_score

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras


# This section will contain the hyperparameters
# hyper_batch_size=1 # in case we want to try the SGD method
hyper_epochs=50
hyper_drop_out=0.25
hyper_learning_rate = 0.025
hyper_optimizer = keras.optimizers.Adam(learning_rate=hyper_learning_rate)
hyper_verbose = 0

#### This Function creates the topography of the model
def create_model_OHEC():
    model2 = Sequential()
    model2.add(Dense(12,input_dim=14, kernel_initializer='normal', activation='relu'))
    model2.add(Dropout(hyper_drop_out))
    model2.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model2.add(Dropout(hyper_drop_out))
    
#     model2.add(Dense(16, kernel_initializer='normal', activation='relu'))
#     model2.add(Dropout(hyper_drop_out))
#     model.add(Dense(16, kernel_initializer='normal', activation='relu'))
#     model.add(Dropout(hyper_drop_out))
    model2.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    opt = keras.optimizers.Adam(learning_rate=hyper_learning_rate)    
    model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model2



from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

#Wrap our Keras model in an estimator compatible with scikit_learn
estimator_OHEC = KerasClassifier(build_fn=create_model_OHEC,
#                             batch_size=hyper_batch_size,
                            epochs=hyper_epochs,
                            verbose=hyper_verbose)



# Now we can use scikit_learn's cross_val_score to evaluate this model identically to the others
cv_scores = cross_val_score(estimator_OHEC, all_features, all_classes, cv=10)
cv_scores.mean()
from xgboost import XGBClassifier
xgb = XGBClassifier(random_state =1)
cv_score = cross_val_score(xgb,all_features,all_classes,cv=5)
cv_score.mean()
# Now, let's try this with Voting Classifier using sklearn methods
from sklearn.ensemble import VotingClassifier


voting_clf = VotingClassifier(estimators = [('RandForestClf',RandForestClf),('svc',svc),('xgb',xgb) ], voting = 'soft') 
voting_clf.fit(all_features,all_classes)
cv = cross_val_score(voting_clf,all_features,all_classes,cv=5)
print(cv)
print(cv.mean())
voting_clf.fit(X_train_scaled,y_train)
y_hat_base_vc = voting_clf.predict(X_test_scaled).astype(int)