# importing necessary libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
class DataFram:

    df = pd.read_csv(r'E:\BHAVYA\KAGGLE\mushroom-classification\mushrooms.csv')

    datalist = [df]

    

    # encoding our classification data

    def encode(self, colname):

        size = len(set(self.df[colname]))

     

        keys = list(set(self.df[colname]))

       

        values = [i for i in range(size)]

       

        mapping = dict(zip(keys,values))

       

        for row in self.datalist:

            row[colname] = row[colname].map(mapping)

          

            

    def checkMissingValues(self):

        attributes = list(self.df.columns)

        for name in attributes:

            print(name,' ',set(self.df[name]))

            print()

            for values in set(self.df[name]):

                if str(values).isalnum()==False:

                    print('----- ',name,' has missing values = ',values,' -------')

                    print()
obj = DataFram()

'''

obj.df=obj.df.groupby(['stalk-root','class']).size()

obj.df=obj.df.unstack()

obj.df.plot(kind='bar')

'''
obj.checkMissingValues()



attributes = list(obj.df.columns)

for name in attributes:

    obj.encode(name)

obj.df.head(10)

df = obj.df









X = df.iloc[:,1:23]



y = df.iloc[:,0]

print(y)

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = pd.DataFrame(sc.fit_transform(X_train))

X_test = pd.DataFrame(sc.fit_transform(X_test))









#modelling



from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

#svm



classifier = SVC(kernel = 'linear', random_state = 0) # kernel is straight line so many incorrect predictions

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

print(accuracy_score(y_pred,y_test)*100)
#knn



classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

classifier.fit(X_train, y_train)





# Predicting the Test set results

y_pred = classifier.predict(X_test)

print(accuracy_score(y_pred,y_test)*100)
#naive bayes



classifier = GaussianNB()

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)

print(accuracy_score(y_pred,y_test)*100)
#logistic regression



classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)

print(accuracy_score(y_pred,y_test)*100)

# decision tree



classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)

print(accuracy_score(y_pred,y_test)*100)
# random forest



classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)

print(accuracy_score(y_pred,y_test)*100)
# Ann



# Importing the Keras libraries and packages

import tensorflow

from tensorflow.keras.models import Sequential # keras will build NN based on Tf background

from tensorflow.keras.layers import Dense # seq model used for initializing and dense for layers in NN



classifier = Sequential()





# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 12, kernel_initializer='glorot_uniform' , activation = 'relu', input_dim = 22)) # outputdim = noo .of nodes in hidden layer is 22+1/2=6

# uniform func initialises weights close to 0

# AF = rectifier for hidden & input nodes = 11 variables in 1st row here



# Adding the second hidden layer (no need of input_dim as its same here)

classifier.add(Dense(units= 12, kernel_initializer='glorot_uniform' , activation = 'relu'))



# so here we are using 2 hidden layers



# Adding the output layer

classifier.add(Dense(units = 1, kernel_initializer='glorot_uniform', activation = 'sigmoid')) # 1 node for output, Af = sigmoid for 2 output categories

# softMax for more than 2 o/p categories



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# optimizer = algo for finding best weights for powerful NN (here adam: stochastic gradient descent algo type)

# loss = loss within adam algo (like sum of sqred differences for cost function but here log type(binary_classentropy) for 2 categories)

# if 3 or more o/p then loss = "categorical_crossentropy"

# mertrics = here accuracy to evaluate our model





# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)



# Predicting the Test set results

y_pred = classifier.predict(X_test) # a col of all probabilities obtained!!

y_pred = (y_pred > 0.5)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_pred,y_test)*100)
