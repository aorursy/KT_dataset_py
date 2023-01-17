#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions
import seaborn as sns
#import libraries for pandas profiling
# Detailing for pandas profiling: https://github.com/pandas-profiling/pandas-profiling
from pathlib import Path
from pandas_profiling import ProfileReport
from pandas_profiling.utils.cache import cache_file
diabetes = pd.read_csv("../input/diabetes/diabetes_data.csv")
diabetes.head()
#Profiling: we can see our dataset details,simply.
profile = ProfileReport(diabetes, title="diabetes")
# The Notebook Widgets Interface
profile.to_widgets()
# gives information about the data types,columns, null value counts, memory usage etc
diabetes.info()
diabetes.shape
# basic statistic details about the data (note only numerical columns would be displayed here unless parameter include="all")
diabetes.describe()
diabetes.describe().T
#dpf = DiabetesPedigreeFunction
diabetes_copy = diabetes.copy(deep=True)
diabetes_copy[['glucose','diastolic','triceps','insulin','bmi']] = diabetes_copy[['glucose','diastolic','triceps','insulin','bmi']].replace(0,np.NaN)
# showing the count of Nans
print(diabetes_copy.isnull().sum())
his = diabetes.hist(figsize=(20,20))
#'glucose','diastolic','triceps','insulin','bmi'
diabetes_copy["glucose"].fillna(diabetes_copy["glucose"].mean(), inplace=True)
diabetes_copy["diastolic"].fillna(diabetes_copy["diastolic"].mean(), inplace=True)
diabetes_copy["triceps"].fillna(diabetes_copy["triceps"].median(), inplace=True)
diabetes_copy["insulin"].fillna(diabetes_copy["insulin"].median(), inplace=True)
diabetes_copy["bmi"].fillna(diabetes_copy["bmi"].median(), inplace=True)
his = diabetes_copy.hist(figsize=(20,20))
from pandas.plotting import scatter_matrix
p = scatter_matrix(diabetes,figsize=(20,20))
#from pandas.plotting import scatter_matrix
#p = scatter_matrix(diabetes_copy,figsize=(20,20))
p = sns.pairplot(diabetes_copy, hue="diabetes")
plt.figure(figsize=(12,10))  
p=sns.heatmap(diabetes.corr(), annot=True,cmap ='RdYlGn')
plt.figure(figsize=(12,10)) 
p=sns.heatmap(diabetes_copy.corr(), annot=True,cmap ='RdYlGn') 
from sklearn.preprocessing import StandardScaler
St_sc = StandardScaler()
X = pd.DataFrame(St_sc.fit_transform(diabetes_copy.drop(["diabetes"],axis=1)),columns=["pregnancies","glucose",
                                                                                      "diastolic","triceps","insulin","bmi","dpf","age"])
y = diabetes_copy.diabetes

X.head()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42, stratify=y)
#Train Test Split data==> 70% of data set for Train, 30% of data set for Test
# Create a k-NN classifier
knn = KNeighborsClassifier(n_neighbors = 6, metric= "manhattan")
# Fit the classifier to the training data
knn.fit(X_train,y_train)
#PREDICTION
print("Prediction of features (test set): ",knn.predict(X_test))
print("Actual label variables: (test set)",y_test)
#ACCURACY
# Print the accuracy
print(knn.score(X_test, y_test))

# Setup arrays to store train and test accuracies
neighbors = np.arange(1,15)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors = k, metric= "manhattan")
    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

max_train_score = max(train_accuracy)
train_scores_ind = [i for i, v in enumerate(train_accuracy) if v == max_train_score]
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))
max_test_score = max(test_accuracy)
test_scores_ind = [i for i, v in enumerate(test_accuracy) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))
# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
#The best result is captured at k = 12
knn = KNeighborsClassifier(12, metric= "manhattan")

knn.fit(X_train,y_train)
knn.score(X_test,y_test)
#import confusion matrix
from sklearn.metrics import confusion_matrix
#let us get the predictions using the classifier we had fit above
y_predict = knn.predict(X_test)
confusion_matrix(y_test,y_predict)
pd.crosstab(y_test,y_predict,rownames=["True"],colnames=["Predicted"],margins=True)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_predict)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict))
from sklearn.metrics import roc_curve
y_predict_prob = knn.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test,y_predict_prob)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Knn(n_neighbors=11) ROC curve')
plt.show()
#Area under ROC curve
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_predict_prob)