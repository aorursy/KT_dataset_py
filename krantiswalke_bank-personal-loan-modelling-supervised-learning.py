# To enable plotting graphs in Jupyter notebook
%matplotlib inline

# Importing libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression

# importing ploting libraries
import matplotlib.pyplot as plt   

#importing seaborn for statistical plots
import seaborn as sns

#Let us break the X and y dataframes into training set and test set. For this we will use
#Sklearn package's data splitting function which is based on random function

from sklearn.model_selection import train_test_split

import numpy as np
#import os,sys
from scipy import stats

# calculate accuracy measures and confusion matrix
from sklearn import metrics

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
datapath = '../input'
my_data = pd.read_csv(datapath+'/Bank_Personal_Loan_Modelling.csv')
my_data.columns = ["ID","Age","Experience","Income","ZIPCode","Family","CCAvg","Education","Mortgage","Personal_Loan","SecuritiesAccount","CDAccount","Online","CreditCard"]

my_data.head(10)
my_data.shape
my_data.columns
my_data.dtypes
#null values
my_data.isnull().values.any()
val=my_data.isnull().values.any()

if val==True:
    print("Missing values present : ", my_data.isnull().values.sum())
    my_data=my_data.dropna()
else:
    print("No missing values present")
my_data.describe().T
my_data.info()
my_data.apply(lambda x: len(x.unique()))
#Find Shape
my_data.shape
#Find Mean
my_data.mean()
#Find Median
my_data.median()
#Find Standard Deviation
my_data.std()
my_data.hist(figsize=(10,10),color="blueviolet",grid=False)
plt.show()
sns.pairplot(my_data.iloc[:,1:])
my_data[my_data['Experience'] < 0]['Experience'].count()
my_dataExp = my_data.loc[my_data['Experience'] >0]
negExp = my_data.Experience < 0
column_name = 'Experience'
my_data_list = my_data.loc[negExp]['ID'].tolist()
negExp.value_counts()
for id in my_data_list:
    age = my_data.loc[np.where(my_data['ID']==id)]["Age"].tolist()[0]
    education = my_data.loc[np.where(my_data['ID']==id)]["Education"].tolist()[0]
    df_filtered = my_dataExp[(my_dataExp.Age == age) & (my_dataExp.Education == education)]
    exp = df_filtered['Experience'].median()
    my_data.loc[my_data.loc[np.where(my_data['ID']==id)].index, 'Experience'] = exp
    
#The records with the ID, get the values of Age and Education columns.
#Then apply filter for the records matching the criteria from the dataframe 
#which has records with positive experience and take the median.
#Apply the median again to the location(records) which had negative experience.   
my_data[my_data['Experience'] < 0]['Experience'].count()
my_data.describe().T
my_data.skew(axis = 0, skipna = True) 
sns.boxplot(x=my_data["Age"])
sns.boxplot(x=my_data["Experience"])
sns.boxplot(x=my_data["Income"])
import matplotlib.pylab as plt

my_data.boxplot(by = 'Personal_Loan',  layout=(4,4), figsize=(20, 20))
print(my_data.boxplot('Age'))
print(my_data.boxplot('Income'))
print(my_data.boxplot('Education'))

my_data['Personal_Loan'].hist(bins=10)
sns.boxplot(x='Education',y='Income',hue='Personal_Loan',data=my_data)
sns.boxplot(x="Education", y='Mortgage', hue="Personal_Loan", data=my_data)
sns.boxplot(x="Family",y="Income",hue="Personal_Loan",data=my_data)
sns.countplot(x='Family',data=my_data,hue='Personal_Loan')
sns.countplot(x="SecuritiesAccount", data=my_data,hue="Personal_Loan")
sns.countplot(x='CDAccount',data=my_data,hue='Personal_Loan')
sns.countplot(x='Online',data=my_data,hue='Personal_Loan')
sns.countplot(x='CreditCard',data=my_data,hue='Personal_Loan')
plt.figure(figsize = (10,8))
sns.scatterplot(x = "Experience", y = "Age",data =my_data, hue = "Education")
plt.xlabel("Experience")
plt.ylabel("Age")
plt.title("Distribution of Education by Age and Experience")
sns.distplot( my_data[my_data.Personal_Loan == 0]['CCAvg'])
sns.distplot( my_data[my_data.Personal_Loan == 1]['CCAvg'])
#Credit card spending of Non-Loan customers
my_data[my_data.Personal_Loan == 0]['CCAvg'].median()*1000
#Credit card spending of Loan customers
my_data[my_data.Personal_Loan == 1]['CCAvg'].median()*1000
cor=my_data.corr()
cor
plt.subplots(figsize=(10,8))
sns.heatmap(cor,annot=True)

data=my_data.drop(['ID','ZIPCode','Experience'], axis =1 )
data.head(10)
data.info()
data1=data[['Age','Income','Family','CCAvg','Education','Mortgage','SecuritiesAccount','CDAccount','Online','CreditCard','Personal_Loan']]
data1.head(10)
data1.shape
data1["Personal_Loan"].value_counts(normalize=True)
array = data1.values
X = array[:,0:9] # select all rows and first 10 columns which are the attributes
Y = array[:,10]   # select all rows and the 10th column which is the classification "0", "1"
test_size = 0.30 # taking 70:30 training and test set
seed = 15  # Random numbmer seeding for reapeatability of the code
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed) # To set the random state
type(X_train)
# Fit the model on 30%
model = LogisticRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
model_score = model.score(X_test, y_test)
print('Accuracy:',model_score)
print('confusion_matrix:')
print(metrics.confusion_matrix(y_test, y_predict))
A=model_score  # Accuracy of Logistic regression model
#from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
X = data1.values[:,0:9]  ## Features
Y = data1.values[:,10]  ## Target.values[:,10]  ## Target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 7)
clf = GaussianNB()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
B=accuracy_score(Y_test, Y_pred, normalize = True) #Accuracy of Naive Bayes' Model
print('Accuracy_score:',B)
from sklearn.metrics import recall_score
print(recall_score(Y_test, Y_pred))
print('Confusion_matrix:')
print(metrics.confusion_matrix(Y_test,Y_pred))
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
X_std = pd.DataFrame(StandardScaler().fit_transform(data1))
X_std.columns = data1.columns
#split the dataset into training and test datasets
import numpy as np
from sklearn.model_selection import train_test_split

# Transform data into features and target
X = np.array(data1.iloc[:,1:11]) 
y = np.array(data1['Personal_Loan'])

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
print(X_train.shape)
print(y_train.shape)
print(X_train.shape)
print(y_train.shape)
# loading library
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

# instantiate learning model (k = 1)
knn = KNeighborsClassifier(n_neighbors = 1)

# fitting the model
knn.fit(X_train, y_train)

# predict the response
y_pred = knn.predict(X_test)

# evaluate accuracy
print(accuracy_score(y_test, y_pred))

# instantiate learning model (k = 5)
knn = KNeighborsClassifier(n_neighbors=5)

# fitting the model
knn.fit(X_train, y_train)

# predict the response
y_pred = knn.predict(X_test)

# evaluate accuracy
print(accuracy_score(y_test, y_pred))

# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# fitting the model
knn.fit(X_train, y_train)

# predict the response
y_pred = knn.predict(X_test)

# evaluate accuracy
print(accuracy_score(y_test, y_pred))
# instantiate learning model (k = 7)
knn = KNeighborsClassifier(n_neighbors=7)

# fitting the model
knn.fit(X_train, y_train)

# predict the response
y_pred = knn.predict(X_test)

# evaluate accuracy
print(accuracy_score(y_test, y_pred))
myList = list(range(1,20))

# subsetting just the odd ones
neighbors = list(filter(lambda x: x % 2 != 0, myList))
ac_scores = []

# perform accuracy metrics for values from 1,3,5....19
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    # predict the response
    y_pred = knn.predict(X_test)
    # evaluate accuracy
    scores = accuracy_score(y_test, y_pred)
    ac_scores.append(scores)

# changing to misclassification error
MSE = [1 - x for x in ac_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
#Plot misclassification error vs k (with k value on X-axis) using matplotlib.
import matplotlib.pyplot as plt
# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
#Use k=1 as the final model for prediction
knn = KNeighborsClassifier(n_neighbors = 1)

# fitting the model
knn.fit(X_train, y_train)

# predict the response
y_pred = knn.predict(X_test)

# evaluate accuracy
C=accuracy_score(y_test, y_pred)   #Accuracy of KNN model
print('Accuracy_score:',C)    
print(recall_score(y_test, y_pred))

print('Confusion_matrix:')
print(metrics.confusion_matrix(y_test, y_pred))
from sklearn.model_selection import train_test_split

# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score, confusion_matrix

target = my_data["Personal_Loan"]
features=my_data.drop(['ID','ZIPCode','Experience'], axis =1 )
X_train, X_test, y_train, y_test = train_test_split(features,target, test_size = 0.30, random_state = 10)
from sklearn.svm import SVC

# Building a Support Vector Machine on train data
svc_model= SVC(kernel='linear')
svc_model.fit(X_train, y_train)

prediction = svc_model.predict(X_test)

# check the accuracy on the training set
print(svc_model.score(X_train, y_train))
print(svc_model.score(X_test, y_test))
print("Confusion Matrix:\n",confusion_matrix(prediction,y_test))
#Store the accuracy results for each kernel in a dataframe for final comparison
resultsDf = pd.DataFrame({'Kernel':['Linear'], 'Accuracy': svc_model.score(X_train, y_train)})
resultsDf = resultsDf[['Kernel', 'Accuracy']]
resultsDf
# Building a Support Vector Machine on train data
svc_model = SVC(kernel='rbf')
svc_model.fit(X_train, y_train)
print(svc_model.score(X_train, y_train))
print(svc_model.score(X_test, y_test))
#Store the accuracy results for each kernel in a dataframe for final comparison
tempResultsDf = pd.DataFrame({'Kernel':['RBF'], 'Accuracy': svc_model.score(X_train, y_train)})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf = resultsDf[['Kernel', 'Accuracy']]
resultsDf
#Building a Support Vector Machine on train data(changing the kernel)
svc_model  = SVC(kernel='poly')
svc_model.fit(X_train, y_train)

prediction = svc_model.predict(X_test)

print(svc_model.score(X_train, y_train))
print(svc_model.score(X_test, y_test))
#Store the accuracy results for each kernel in a dataframe for final comparison
tempResultsDf = pd.DataFrame({'Kernel':['Poly'], 'Accuracy': svc_model.score(X_train, y_train)})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf = resultsDf[['Kernel', 'Accuracy']]
resultsDf
svc_model = SVC(kernel='sigmoid')
svc_model.fit(X_train, y_train)

prediction = svc_model.predict(X_test)

##print(svc_model.score(X_train, y_train))
print(svc_model.score(X_test, y_test))
#Store the accuracy results for each kernel in a dataframe for final comparison
tempResultsDf = pd.DataFrame({'Kernel':['Sigmoid'], 'Accuracy': svc_model.score(X_train, y_train)})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf = resultsDf[['Kernel', 'Accuracy']]
resultsDf
print(A) #Accuracy of Logistic regression model
print(B) #Accuracy of Naive Bayes' Model
print(C)  #Accuracy of KNN Model
resultsDf #Accuracy of SVM Model
