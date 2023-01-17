# Different model accuracy scores on Cancer Dataset
# in this part we have used
# 1:Logistic Regression
# 2:K nearest neighbors
# 3:SVM using linear kernel
# 4:Decision Tree
# 5:Naive Bayes
# 6:Random Forest
# 7:Gradient boosting Classifier


#initializing the libraries that we need
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# Getting out data in out cancer Dataframe
# as well as cleaning the data

Cancer_dataset = pd.read_csv('../input/data.csv')
Cancer_dataset.dropna()
Cancer_dataset.replace('?',-99999,inplace=True)
# printing the first five instances of our dataframe

Cancer_dataset.head()
# Getting our Features that is our x data

X_cancer = Cancer_dataset[['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
                           'compactness_mean','concavity_mean','concave points_mean','texture_worst',
                           'perimeter_worst','area_worst','smoothness_worst','compactness_worst',
                           'concavity_worst','concave points_worst','symmetry_worst',
                           'fractal_dimension_worst']]
# Getting our outcome that is our y data

Y_cancer = Cancer_dataset['diagnosis']
# Splitting data into training and testing

X_train,x_test,Y_train,y_test = train_test_split(X_cancer,Y_cancer)
# Normalizing the data for creating different features
# normalizing is used when all the features are relatively close to the space 
# We are using MinMax Scaling here another one is polynomial


scaler = MinMaxScaler().fit(X_train)
new_x_train = scaler.transform(X_train)
new_x_test = scaler.transform(x_test)
print('This is the normalized training data:',new_x_train[1:8])
print('This is the normalized testing data:',new_x_test[1:8])
# 1 : Logistic Regression 


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(new_x_train,Y_train)
print('The training accuracy of Logistic Regression is :',clf.score(new_x_train,Y_train))
print('The testing accuracy of Logistic Regression is :',clf.score(new_x_test,y_test))
# 2 : KNN(K nearest neighbors)
from sklearn.neighbors import KNeighborsClassifier
clf2 = KNeighborsClassifier(n_neighbors = 10)
clf2.fit(new_x_train,Y_train)
print('The training accuracy of KNN is :',clf2.score(new_x_train,Y_train))
print('The testing accuracy of KNN is :',clf2.score(new_x_test,y_test))
# 3 : SVM using kernel as linear
from sklearn.svm import SVC
clf3 = SVC(kernel='linear',gamma=5)
clf3.fit(new_x_train,Y_train)
print('The training accuracy of SVM is :',clf3.score(new_x_train,Y_train))
print('The testing accuracy of SVM is :',clf3.score(new_x_test,y_test))
# 4 : Decision trees
 
    
from sklearn.tree import DecisionTreeClassifier
clf4 = DecisionTreeClassifier(max_depth=12)
clf4.fit(new_x_train,Y_train)
print('The training accuracy of Decision Tree is :',clf4.score(new_x_train,Y_train))
print('The testing accuracy of Decision Tree is :',clf4.score(new_x_test,y_test))
# 5 : Naive Bayes


from sklearn.naive_bayes import GaussianNB
clf5 = GaussianNB()
clf5.fit(new_x_train,Y_train)
print('The training accuracy of Gaussian Naive Bayes is :',clf5.score(new_x_train,Y_train))
print('The testing accuracy of Gaussian Naive Bayes is :',clf5.score(new_x_test,y_test))
# 6 : Random Forests


from sklearn.ensemble import RandomForestClassifier
clf6 = RandomForestClassifier(n_estimators = 15)
clf6.fit(new_x_train,Y_train)
print('The training accuracy of Random Forest is :',clf6.score(new_x_train,Y_train))
print('The testing accuracy of Random Forest is :',clf6.score(new_x_test,y_test))
# 7 : Gradient boosted Decision Tree


from sklearn.ensemble import GradientBoostingClassifier
clf7 = GradientBoostingClassifier(n_estimators = 10,learning_rate = 0.1)
clf7.fit(new_x_train,Y_train)
print('The training accuracy of Gradient Boosted Classifier is :',clf7.score(new_x_train,Y_train))
print('The testing accuracy of Gradient Boosted Classifier is :',clf7.score(new_x_test,y_test))
