import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
data = pd.read_csv('../input/Iris.csv')

# First and last five observations or row
data.head(5)
data.tail(5)
# Our target is 'species' So need to check how many of them
print("Species")
print(data['Species'].unique())
data.describe()
import seaborn as sns
sns.FacetGrid(data, hue="Species", size=6) \
   .map(plt.scatter, "SepalLengthCm", "SepalWidthCm") \
   .add_legend()

plt.show()


# Preprocessing
# Let Separate Features and Target for machine Learning
# Step1 


features = list(data.columns[1:5])            # SepalLengthCm	SepalWidthCm	PetalLengthCm	PetalWidthCm	
target = data.columns[5]                      # Species

print('Features:',features)
print('Target:',target)

# store feature matrix in "X"
X = data.iloc[:,1:5]                          # slicing: all rows and 1 to 4 cols

# store response vector in "y"
y = data.iloc[:,5]                            # slicing: all rows and 5th col


print(y.shape)
print(X.shape)
# Read more: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y)
y=le.transform(y)

# new col
data['EncodedSpecies'] = y

print('Classes:',le.classes_)
print('Response variable after encoding:',y)
data.tail(10)
#Step2: Model

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)

#Step3: Prediction for new observation
value = knn.predict([[3, 5, 4, 2]])
print('prediction value:',value)

print('Predicted Class' , data.loc[data['EncodedSpecies'] == 2, 'Species'].values[0])
#data.loc[data['EncodedSpecies'] == 2, 'Species'].values[0]
# more predictions for other rows

X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]                        # Consider them as two new rows of input features in X
knn.predict(X_new)
# Different value of K 
# instantiate the model (using the value K=5)

knn = KNeighborsClassifier(n_neighbors=5)

# fit the model with data
knn.fit(X, y)

# predict the response for new observations
print(knn.predict(X_new))

kypred = knn.predict(X)
# For an optimal value of K for KNN

from sklearn import metrics
v=[]




k_range = list(range(1, 50))
for i in k_range:
    knn = KNeighborsClassifier(n_neighbors=i)
    # fit the model with data
    knn.fit(X, y)
    k_pred = knn.predict(X)
    v.append( metrics.accuracy_score(y, k_pred))


import matplotlib.pyplot as plt
plt.plot(k_range,v,c='Orange',)
plt.show()

# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X, y)

# predict the response values for the observations in column: [3, 5, 4, 2]
logreg.predict([[3, 5, 4, 2]]) # Col vector # See previous result


y_pred = logreg.predict(X)

print(y_pred)
# 1 KNN ACCURACY

# compute classification accuracy for the KNN model
from sklearn import metrics
print(metrics.accuracy_score(y, kypred))
# 2 
# compute classification accuracy for the logistic regression model
from sklearn import metrics
print(metrics.accuracy_score(y, y_pred))
# STEP 1: split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)

print(X_train.shape)
print(y_train.shape)
logres = LogisticRegression()
logres.fit(X_train,y_train) # train data


# predict from test
log_pred = logres.predict(X_test)

# check accuracy
import sklearn.metrics as mt
mt.accuracy_score(log_pred,y_test)

from sklearn import metrics
v=[]


k_range = list(range(1, 50))
for i in k_range:
    knn = KNeighborsClassifier(n_neighbors=i)
    # fit the model with data
    knn.fit(X_train, y_train)
    k_pred = knn.predict(X_test)
    v.append( metrics.accuracy_score(y_test, k_pred))



import matplotlib.pyplot as plt
plt.plot(k_range,v)

plt.show()

print('from above the best value is near:',10)
knn = KNeighborsClassifier(n_neighbors=12)
# fit the model with data
knn.fit(X_train, y_train)
k_pred = knn.predict(X_test)

metrics.accuracy_score(y_test, k_pred)
