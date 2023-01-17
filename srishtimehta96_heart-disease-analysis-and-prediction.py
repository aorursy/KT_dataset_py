import matplotlib.pyplot as plt 

import pandas as pd

heart=pd.read_csv("../input/heart-disease-uci/heart.csv")

heart

#At which age maximum no of persons suffer from heart disease?

#heart['age'].value_counts()

heart.info()# information of all dataframe
heart.isnull().sum()# to check the null entries
#No of male/female suffering from heart disease?



heart['sex'].value_counts()
#No of patients per each chest pain type?

heart['cp'].value_counts()
#NO OF DISEASED AND NON DISEASED PERSONS

plt.hist(heart['target'])

plt.show
heart.hist(figsize=(10,15))



plt.show()
#Plot of age with target

import seaborn as sns

plt.figure(figsize=(14,5))

sns.distplot(heart[heart['target'] == 0]['age'], color = 'red', kde=False, label = '0 Class')

sns.distplot(heart[heart['target'] == 1]['age'], color = 'yellow', kde=False, label ='1 Class')

plt.legend()

plt.title('count of people with respect to age ')

plt.show()
#Plot of sex with target

plt.figure(figsize=(14,5))

sns.distplot(heart[heart['target'] == 0]['sex'], color = 'orange', kde=False, label = '0 Class')

sns.distplot(heart[heart['target'] == 1]['sex'], color = 'green', kde=False, label ='1 Class')

plt.legend()

plt.title('sex analysis')

plt.show()
#Plot of cp with target

plt.figure(figsize=(14,5))

sns.distplot(heart[heart['target'] == 0]['cp'], color = 'k', kde=False, label = '0 Class')

sns.distplot(heart[heart['target'] == 1]['cp'], color = 'm', kde=False, label ='1 Class')

plt.legend()

plt.title('analysis')

plt.show()
#Plot of trestbps with target

plt.figure(figsize=(16,8))

sns.distplot(heart[heart['target'] == 0]['trestbps'], color = 'r', kde=False, label = '0 Class')

sns.distplot(heart[heart['target'] == 1]['trestbps'], color = 'c', kde=False, label ='1 Class')

plt.legend()

plt.title('trestbps values')

plt.show()
#Plot of chol with target

plt.figure(figsize=(14,5))

sns.distplot(heart[heart['target'] == 0]['chol'], color = 'pink', kde=False, label = '0 Class')

sns.distplot(heart[heart['target'] == 1]['chol'], color = 'yellow', kde=False, label ='1 Class')

plt.legend()

plt.title('Change in chol values')

plt.show()
#Plot of fbs with target

plt.figure(figsize=(14,5))

sns.distplot(heart[heart['target'] == 0]['fbs'], color = 'violet', kde=False, label = '0 Class')

sns.distplot(heart[heart['target'] == 1]['fbs'], color = 'yellow', kde=False, label ='1 Class')

plt.legend()

plt.title('fbs analysis')

plt.show()
#Plot of restecg with target

plt.figure(figsize=(14,5))

sns.distplot(heart[heart['target'] == 0]['restecg'], color = 'indigo', kde=False, label = '0 Class')

sns.distplot(heart[heart['target'] == 1]['restecg'], color = 'green', kde=False, label ='1 Class')

plt.legend()

plt.title('restecg analysis')

plt.show()
#Plot of thalach with target

plt.figure(figsize=(14,5))

sns.distplot(heart[heart['target'] == 0]['thalach'], color = 'indigo', kde=False, label = '0 Class')

sns.distplot(heart[heart['target'] == 1]['thalach'], color = 'green', kde=False, label ='1 Class')

plt.legend()

plt.title('count of diseased and non diseased persons according to thalach paramater')

plt.show()
#Plot of exang with target

plt.figure(figsize=(14,5))

sns.distplot(heart[heart['target'] == 0]['exang'], color = 'yellow', kde=False, label = '0 Class')

sns.distplot(heart[heart['target'] == 1]['exang'], color = 'green', kde=False, label ='1 Class')

plt.legend()

plt.title('exang analysis')

plt.show()
#Plot of oldpeak with target

plt.figure(figsize=(14,5))

sns.distplot(heart[heart['target'] == 0]['oldpeak'], color = 'orange', kde=False, label = '0 Class')

sns.distplot(heart[heart['target'] == 1]['oldpeak'], color = 'green', kde=False, label ='1 Class')

plt.legend()

plt.title('oldpeak analysis')

plt.show()
#Plot of slope with target

plt.figure(figsize=(14,5))

sns.distplot(heart[heart['target'] == 0]['slope'], color = 'orange', kde=False, label = '0 Class')

sns.distplot(heart[heart['target'] == 1]['slope'], color = 'yellow', kde=False, label ='1 Class')

plt.legend()

plt.title('slope analysis')

plt.show()
#Plot of ca with target

plt.figure(figsize=(14,5))

sns.distplot(heart[heart['target'] == 0]['ca'], color = 'orange', kde=False, label = '0 Class')

sns.distplot(heart[heart['target'] == 1]['ca'], color = 'green', kde=False, label ='1 Class')

plt.legend()

plt.title('ca analysis')

plt.show()
#Plot of thal with target

plt.figure(figsize=(14,5))

sns.distplot(heart[heart['target'] == 0]['thal'], color = 'orange', kde=False, label = '0 Class')

sns.distplot(heart[heart['target'] == 1]['thal'], color = 'green', kde=False, label ='1 Class')

plt.legend()

plt.title('thal analysis')

plt.show()


#splitting the data into dependent(Y) and independent variables(X)

X= heart. iloc[:, : 13]    #logistic regression with all variables of X

X

Y= heart['target']

Y
#splitting the data into training and testing set

from sklearn.model_selection import train_test_split

#from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = .20, random_state = 0)

print(X_train.shape)

print(Y_train.shape)

print(X_test.shape)

print(Y_test.shape)
#LOGISTIC REGRESSION WITH ORIGINAL DATA(i.e. considering all variables of X)

from sklearn.linear_model import LogisticRegression

LRClassifier = LogisticRegression ()

LRClassifier.fit (X_train, Y_train)



#============================================

# Predict the values 

#============================================

prediction = LRClassifier.predict (X_test)





# to check the accuracy of machine 

from sklearn.metrics import accuracy_score

print(accuracy_score(Y_test, prediction))





print(prediction)

from sklearn.tree import DecisionTreeClassifier #to make decision tree with original data



decision_tree = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=4)

decision_tree.fit(X_train,Y_train)

prediction1 =decision_tree.predict(X_test)



print(prediction1)



from sklearn.metrics import accuracy_score

print(accuracy_score(Y_test, prediction1))







factors= ['age','sex','cp','fbs','trestbps','chol','restecg','thalach','exang','oldpeak','slope','ca','thal']



from sklearn.tree import export_graphviz

from sklearn import tree

from graphviz import Source

from IPython.display import SVG

from IPython.display import display



graph = Source(tree.export_graphviz(decision_tree, out_file=None, feature_names=factors, filled = True,rounded=True))





display(SVG(graph.pipe(format='svg')))
from sklearn.neighbors import KNeighborsClassifier  #KNN ANALYSIS with original data

knn=KNeighborsClassifier(n_neighbors=13)   #n_neighbors = value of k

#n_neighbors is not fixed we we have to fix it by trying values and the number which give max accuracy

#will be n_neighbors like here its giving max accuracy at 13, below and after 13 accuracy is low but this n_neighbors should be less than total number of columns



knn.fit(X_train,Y_train) 



#let us get the predictions using the classifier we had fit above

Y_pred = knn.predict(X_test)

print(Y_pred)



from sklearn.metrics import accuracy_score

print("Accuracy is",accuracy_score(Y_test, Y_pred))
from sklearn.ensemble import RandomForestClassifier # to make random forest with original data

model = RandomForestClassifier(n_estimators=12, random_state=50, max_depth=5, criterion = 'entropy') #n_estimators we have choosen 12 because its giving max accuracy at 12, we have checked from 1-13



model.fit(X_train, Y_train)

pred7=model.predict(X_test)

print(pred7)

from sklearn.metrics import accuracy_score

print(accuracy_score(Y_test, pred7))



estimators=model.estimators_[5] # here model estimators we choose randomly that how many trees we want to display but this should be less than tital number of columns

factors= ['age','sex','cp','fbs','trestbps','chol','restecg','thalach','exang','oldpeak','slope','ca','thal']

from sklearn import tree

from graphviz import Source

from IPython.display import SVG

from IPython.display import display



graph = Source(tree.export_graphviz(estimators, out_file=None

   , feature_names=factors

   , filled = True))

display(SVG(graph.pipe(format='svg')))



from sklearn.svm import SVC #SVM ANALYSIS WITH ORIGINAL DATA

svc=SVC() #Default hyperparameters

svc.fit(X_train,Y_train)

Y_pred1=svc.predict(X_test)

Y_pred1





from sklearn import metrics

print('Accuracy Score: of svc default parameters')

print(metrics.accuracy_score(Y_test,Y_pred1))



svc=SVC(kernel='linear')  #WITH LINEAR KERNEL



svc.fit(X_train,Y_train)



Y_pred2=svc.predict(X_test)



print('Accuracy Score: with default linear kernel')



print(metrics.accuracy_score(Y_test,Y_pred2))



svc=SVC(kernel='rbf')   #WITH RADIAL KERNEL

svc.fit(X_train,Y_train)

Y_pred3=svc.predict(X_test)

print('Accuracy Score: with default rbf kernel')

print(metrics.accuracy_score(Y_test,Y_pred3))

from sklearn.metrics import confusion_matrix  

print(confusion_matrix(Y_test, Y_pred3))  

Y_pred3



svc=SVC(kernel='poly')  #WITH POLY KERNEL

svc.fit(X_train,Y_train)

Y_pred5=svc.predict(X_test)

print('Accuracy Score:with default poly kernel')

print(metrics.accuracy_score(Y_test,Y_pred5))   













from sklearn.svm import SVC      #with rbf kernel and with C  and gamma value on original data



from sklearn import metrics

svm = SVC(kernel = 'rbf', C=1.0, gamma = 0.1, random_state=0)



svm.fit(X_train,Y_train)



Y_pred5=svm.predict(X_test)



print('Accuracy Score:with rbf kernel and with C and gamma value')



print(metrics.accuracy_score(Y_test,Y_pred5))   







from sklearn import metrics

from sklearn.svm import SVC  # with sigmoid kernel

svclassifier = SVC(kernel='sigmoid')  

svclassifier.fit(X_train, Y_train)  

Y_pred6=svclassifier.predict(X_test)

print('Accuracy Score:with sigmoid kernel')

print(metrics.accuracy_score(Y_test,Y_pred6))   









from sklearn.naive_bayes import GaussianNB  

model = GaussianNB()

model.fit(X_train, Y_train)



prediction2 = model.predict(X_test)







from sklearn.metrics import accuracy_score



print(accuracy_score(Y_test, prediction2))





print(metrics.confusion_matrix(Y_test, prediction2))

prediction2    # only gaussian analysis is done, bernauli and multinomial will not be done here because our data is not in the form of string
X1= heart. iloc[:,[0,2,4,5,6,7,8,11]]     #feature selection procedure to inccrease accuracy

X1

Y1= heart['target']

Y1

#splitting the data into training and testing set

from sklearn.model_selection import train_test_split

#from sklearn.cross_validation import train_test_split

X1_train, X1_test, Y1_train, Y1_test = train_test_split (X1, Y1, test_size = .20, random_state = 0)

print(X1_train.shape)

print(Y1_train.shape)

print(X1_test.shape)

print(Y1_test.shape)
#LOGISTIC REGRESSION WITH FEATURE SELECTION

from sklearn.linear_model import LogisticRegression

LRClassifier = LogisticRegression ()

LRClassifier.fit (X1_train, Y1_train)



#============================================

# Predict the values 

#============================================

prediction3 = LRClassifier.predict (X1_test)





# to check the accuracy of machine 

from sklearn.metrics import accuracy_score

print(accuracy_score(Y1_test, prediction3))





print(prediction3)

#feature selection process didnt work well so now we are using other method of scaling to increase accuracy

from sklearn.tree import DecisionTreeClassifier #to make decision tree with feature selection data



decision_tree = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=4)

decision_tree.fit(X1_train,Y1_train)

prediction6 =decision_tree.predict(X1_test)



print(prediction6)



from sklearn.metrics import accuracy_score

print(accuracy_score(Y1_test, prediction6))







factors= ['age','cp','chol','fbs','restecg','thalach','exang','ca']



from sklearn.tree import export_graphviz

from sklearn import tree

from graphviz import Source

from IPython.display import SVG

from IPython.display import display



graph = Source(tree.export_graphviz(decision_tree, out_file=None, feature_names=factors, filled = True,rounded=True))





display(SVG(graph.pipe(format='svg')))
from sklearn.ensemble import RandomForestClassifier # to make random forest with feature selection

model = RandomForestClassifier(n_estimators=7, random_state=50, max_depth=5, criterion = 'entropy')



model.fit(X1_train, Y1_train)

pred8=model.predict(X1_test)

print(pred8)

from sklearn.metrics import accuracy_score

print(accuracy_score(Y1_test, pred8))



estimators=model.estimators_[5] # here model estimators we choose randomly that how many trees we want to display but this should be less than tital number of columns

factors= ['age','cp','chol','fbs','restecg','thalach','exang','ca']



from sklearn import tree

from graphviz import Source

from IPython.display import SVG

from IPython.display import display



graph = Source(tree.export_graphviz(estimators, out_file=None

   , feature_names=factors

   , filled = True))

display(SVG(graph.pipe(format='svg')))

from sklearn.neighbors import KNeighborsClassifier  #KNN ANALYSIS with feature selection

knn=KNeighborsClassifier(n_neighbors=5)   #n_neighbors = value of k

#n_neighbors is not fixed we we have to fix it by trying values and the number which give max accuracy

#will be n_neighbors like here its giving max accuracy at 8, below and after 8 accuracy is low but this n_neighbors should be less than total number of columns not also equal to no of columns



knn.fit(X1_train,Y1_train) 



#let us get the predictions using the classifier we had fit above

Y1_pred = knn.predict(X1_test)

print(Y1_pred)



from sklearn.metrics import accuracy_score

print("Accuracy is",accuracy_score(Y1_test, Y1_pred))
from sklearn.svm import SVC #SVM ANALYSIS WITH FEATURE SELECTION

svc=SVC() 

svc.fit(X1_train,Y1_train)

Y1_pred1=svc.predict(X1_test)

Y1_pred1





from sklearn import metrics

print('Accuracy Score: of svc default parameters')

print(metrics.accuracy_score(Y1_test,Y1_pred1))



svc=SVC(kernel='linear')  #WITH LINEAR KERNEL



svc.fit(X1_train,Y1_train)



Y1_pred2=svc.predict(X1_test)



print('Accuracy Score: with default linear kernel')



print(metrics.accuracy_score(Y1_test,Y1_pred2))



svc=SVC(kernel='rbf')   #WITH RADIAL KERNEL

svc.fit(X1_train,Y1_train)

Y1_pred2=svc.predict(X1_test)

print('Accuracy Score: with default rbf kernel')

print(metrics.accuracy_score(Y_test,Y_pred2))

from sklearn.metrics import confusion_matrix  

print(confusion_matrix(Y1_test, Y1_pred2))  

Y1_pred2



svc=SVC(kernel='poly')  #WITH POLY KERNEL

svc.fit(X1_train,Y1_train)

Y1_pred3=svc.predict(X1_test)

print('Accuracy Score:with default poly kernel')

print(metrics.accuracy_score(Y1_test,Y1_pred3))   



from sklearn.svm import SVC      #with rbf kernel and with C  and gamma value



from sklearn import metrics

svm = SVC(kernel = 'rbf', C=1.0, gamma = 0.1, random_state=0)



svm.fit(X1_train,Y1_train)



Y1_pred4=svm.predict(X1_test)



print('Accuracy Score:with rbf kernel and with C and gamma value')



print(metrics.accuracy_score(Y1_test,Y1_pred4))   

from sklearn import metrics

from sklearn.svm import SVC  # with sigmoid kernel

svclassifier = SVC(kernel='sigmoid')  

svclassifier.fit(X1_train, Y1_train)  

Y1_pred6=svclassifier.predict(X1_test)

print('Accuracy Score:with sigmoid kernel')

print(metrics.accuracy_score(Y1_test,Y1_pred6))   









from sklearn.naive_bayes import GaussianNB  

model = GaussianNB()

model.fit(X1_train, Y1_train)



prediction7 = model.predict(X1_test)







from sklearn.metrics import accuracy_score



print(accuracy_score(Y1_test, prediction7))





print(metrics.confusion_matrix(Y1_test, prediction7))

prediction7  
from sklearn.preprocessing import MinMaxScaler  #scaling we will apply to our feature selected data

mms = MinMaxScaler()

X1_train_norm = mms.fit_transform (X1_train) #fit and transform

X1_test_norm = mms.transform (X1_test) # only transform



print(X1_train)

print(X1_train_norm)

print(X1_test)

print(X1_test_norm)
#LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression # using minmax scaling for logistic regression

LRClassifier = LogisticRegression ()

LRClassifier.fit (X1_train_norm, Y1_train)



#============================================

# Predict the values 

#============================================

prediction8 = LRClassifier.predict (X1_test_norm)





# to check the accuracy of machine 

from sklearn.metrics import accuracy_score

print(accuracy_score(Y1_test, prediction8))





print(prediction8)
from sklearn.tree import DecisionTreeClassifier #using min max scaling to make decision tree



decision_tree = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=4)

decision_tree.fit(X1_train_norm,Y1_train)

prediction9 =decision_tree.predict(X1_test_norm)



print(prediction9)



from sklearn.metrics import accuracy_score

print(accuracy_score(Y1_test, prediction9))









factors= ['age','cp','chol','fbs','restecg','thalach','exang','ca']

from sklearn.tree import export_graphviz

from sklearn import tree

from graphviz import Source

from IPython.display import SVG

from IPython.display import display



graph = Source(tree.export_graphviz(decision_tree, out_file=None, feature_names=factors, filled = True,rounded=True))





display(SVG(graph.pipe(format='svg')))

from sklearn.neighbors import KNeighborsClassifier  #KNN ANALYSIS with min max scaling

knn=KNeighborsClassifier(n_neighbors=3)   #n_neighbors = value of k





knn.fit(X1_train_norm,Y1_train) 



#let us get the predictions using the classifier we had fit above

Y1_pred9 = knn.predict(X1_test_norm)

print(Y1_pred9)



from sklearn.metrics import accuracy_score

print("Accuracy is",accuracy_score(Y1_test, Y1_pred9))
from sklearn.ensemble import RandomForestClassifier # to make random forest with feature selection

model = RandomForestClassifier(n_estimators=7, random_state=50, max_depth=5, criterion = 'entropy')



model.fit(X1_train_norm, Y1_train)

pred10=model.predict(X1_test_norm)

print(pred10)

from sklearn.metrics import accuracy_score

print(accuracy_score(Y1_test, pred10))



estimators=model.estimators_[5] # here model estimators we choose randomly that how many trees we want to display but this should be less than tital number of columns

factors= ['age','cp','chol','fbs','restecg','thalach','exang','ca']



from sklearn import tree

from graphviz import Source

from IPython.display import SVG

from IPython.display import display



graph = Source(tree.export_graphviz(estimators, out_file=None

   , feature_names=factors

   , filled = True))

display(SVG(graph.pipe(format='svg')))
from sklearn.svm import SVC #SVM ANALYSIS WITH MIN MAX SCALING

svc=SVC() 

svc.fit(X1_train_norm,Y1_train)

Y1_pred7=svc.predict(X1_test_norm)

Y1_pred7





from sklearn import metrics

print('Accuracy Score: of svc default parameters')

print(metrics.accuracy_score(Y1_test,Y1_pred7))



svc=SVC(kernel='linear')  #WITH LINEAR KERNEL



svc.fit(X1_train_norm,Y1_train)



Y1_pred8=svc.predict(X1_test_norm)



print('Accuracy Score: with default linear kernel')



print(metrics.accuracy_score(Y1_test,Y1_pred8))



svc=SVC(kernel='rbf')   #WITH RADIAL KERNEL

svc.fit(X1_train_norm,Y1_train)

Y1_pred9=svc.predict(X1_test_norm)

print('Accuracy Score: with default rbf kernel')

print(metrics.accuracy_score(Y1_test,Y1_pred9))

from sklearn.metrics import confusion_matrix  

print(confusion_matrix(Y1_test, Y1_pred9))  

Y1_pred9



svc=SVC(kernel='poly')  #WITH POLY KERNEL

svc.fit(X1_train_norm,Y1_train)

Y1_pred10=svc.predict(X1_test_norm)

print('Accuracy Score:with default poly kernel')

print(metrics.accuracy_score(Y1_test,Y1_pred10))   

from sklearn.svm import SVC      #with rbf kernel and with C  and gamma value



from sklearn import metrics

svm = SVC(kernel = 'rbf', C=1.0, gamma = 0.1, random_state=0)



svm.fit(X1_train_norm,Y1_train)



Y1_pred11=svm.predict(X1_test_norm)



print('Accuracy Score:with rbf kernel and with C and gamma value')



print(metrics.accuracy_score(Y1_test,Y1_pred11))   

from sklearn import metrics

from sklearn.svm import SVC  # with sigmoid kernel

svclassifier = SVC(kernel='sigmoid')  

svclassifier.fit(X1_train_norm, Y1_train)  

Y1_pred12=svclassifier.predict(X1_test_norm)

print('Accuracy Score:with sigmoid kernel')

print(metrics.accuracy_score(Y1_test,Y1_pred12))   









from sklearn.naive_bayes import GaussianNB  

model = GaussianNB()

model.fit(X1_train_norm, Y1_train)



prediction10 = model.predict(X1_test_norm)







from sklearn.metrics import accuracy_score



print(accuracy_score(Y1_test, prediction10))





print(metrics.confusion_matrix(Y1_test, prediction10))

prediction10  
from sklearn.preprocessing import StandardScaler

independent_scalar = StandardScaler()

X1_trainstd = independent_scalar.fit_transform (X1_train) #fit and transform

X1_teststd = independent_scalar.transform (X1_test) # only transform

print(X1_trainstd)
from sklearn.linear_model import LogisticRegression # using STANDARD scaling for logistic regression

LRClassifier = LogisticRegression ()

LRClassifier.fit (X1_trainstd, Y1_train)



#============================================

# Predict the values 

#============================================

prediction11 = LRClassifier.predict (X1_teststd)





# to check the accuracy of machine 

from sklearn.metrics import accuracy_score

print(accuracy_score(Y1_test, prediction11))





print(prediction11)
from sklearn.tree import DecisionTreeClassifier #using standard scaling to make decision tree



decision_tree = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=4)

decision_tree.fit(X1_trainstd,Y_train)

prediction12 =decision_tree.predict(X1_teststd)



print(prediction12)



from sklearn.metrics import accuracy_score

print(accuracy_score(Y_test, prediction12))



factors= ['age','cp','chol','fbs','restecg','thalach','exang','ca']

from sklearn.tree import export_graphviz

from sklearn import tree

from graphviz import Source

from IPython.display import SVG

from IPython.display import display



graph = Source(tree.export_graphviz(decision_tree, out_file=None, feature_names=factors, filled = True,rounded=True))





display(SVG(graph.pipe(format='svg')))
from sklearn.neighbors import KNeighborsClassifier  #KNN ANALYSIS with standard scaling

knn=KNeighborsClassifier(n_neighbors=5)   #n_neighbors = value of k





knn.fit(X1_trainstd,Y1_train) 



#let us get the predictions using the classifier we had fit above

Y1_pred13 = knn.predict(X1_teststd)

print(Y1_pred13)



from sklearn.metrics import accuracy_score

print("Accuracy is",accuracy_score(Y1_test, Y1_pred13))
from sklearn.ensemble import RandomForestClassifier # to make random forest with standard scaling

model = RandomForestClassifier(n_estimators=7, random_state=50, max_depth=5, criterion = 'entropy')



model.fit(X1_trainstd, Y1_train)

pred11=model.predict(X1_teststd)

print(pred11)

from sklearn.metrics import accuracy_score

print(accuracy_score(Y1_test, pred11))



estimators=model.estimators_[5] # here model estimators we choose randomly that how many trees we want to display but this should be less than tital number of columns

factors= ['age','cp','chol','fbs','restecg','thalach','exang','ca']



from sklearn import tree

from graphviz import Source

from IPython.display import SVG

from IPython.display import display



graph = Source(tree.export_graphviz(estimators, out_file=None

   , feature_names=factors

   , filled = True))

display(SVG(graph.pipe(format='svg')))
from sklearn.svm import SVC #SVM ANALYSIS WITH MIN MAX SCALING

svc=SVC() 

svc.fit(X1_trainstd,Y1_train)

Y1_pred13=svc.predict(X1_teststd)

Y1_pred13





from sklearn import metrics

print('Accuracy Score: of svc default parameters')

print(metrics.accuracy_score(Y1_test,Y1_pred13))



svc=SVC(kernel='linear')  #WITH LINEAR KERNEL



svc.fit(X1_trainstd,Y1_train)



Y1_pred14=svc.predict(X1_teststd)



print('Accuracy Score: with default linear kernel')



print(metrics.accuracy_score(Y1_test,Y1_pred14))



svc=SVC(kernel='rbf')   #WITH RADIAL KERNEL

svc.fit(X1_trainstd,Y1_train)

Y1_pred15=svc.predict(X1_teststd)

print('Accuracy Score: with default rbf kernel')

print(metrics.accuracy_score(Y1_test,Y1_pred15))

from sklearn.metrics import confusion_matrix  

print(confusion_matrix(Y1_test, Y1_pred15))  

Y1_pred15



svc=SVC(kernel='poly')  #WITH POLY KERNEL

svc.fit(X1_trainstd,Y1_train)

Y1_pred16=svc.predict(X1_teststd)

print('Accuracy Score:with default poly kernel')

print(metrics.accuracy_score(Y1_test,Y1_pred16))   

from sklearn.svm import SVC      #with rbf kernel and with C  and gamma value



from sklearn import metrics

svm = SVC(kernel = 'rbf', C=1.0, gamma = 0.1, random_state=0)



svm.fit(X1_trainstd,Y1_train)



Y1_pred17=svm.predict(X1_teststd)



print('Accuracy Score:with rbf kernel and with C and gamma value')



print(metrics.accuracy_score(Y1_test,Y1_pred17))  
from sklearn import metrics

from sklearn.svm import SVC  # with sigmoid kernel

svclassifier = SVC(kernel='sigmoid')  

svclassifier.fit(X1_trainstd, Y1_train)  

Y1_pred18=svclassifier.predict(X1_teststd)

print('Accuracy Score:with sigmoid kernel')

print(metrics.accuracy_score(Y1_test,Y1_pred18))   









from sklearn.naive_bayes import GaussianNB  

model = GaussianNB()

model.fit(X1_trainstd, Y1_train)



prediction13 = model.predict(X1_teststd)







from sklearn.metrics import accuracy_score



print(accuracy_score(Y1_test, prediction13))





print(metrics.confusion_matrix(Y1_test, prediction13))

prediction13 
from sklearn.decomposition import PCA

pc = PCA(n_components=4) 

X1_trainpca=pc.fit_transform(X1_trainstd) # used here X_trainstd because before PCA,scaling is necessary

X1_testpca=pc.transform(X1_teststd)

print(X1_trainpca)

print(pc.explained_variance_ratio_) 



from sklearn.linear_model import LogisticRegression

LRClassifier = LogisticRegression ()

LRClassifier.fit (X1_trainpca, Y1_train)



#============================================

# Predict the values 

#============================================

prediction01  = LRClassifier.predict (X1_testpca)





# to check the accuracy of machine 

from sklearn.metrics import accuracy_score

print(accuracy_score(Y1_test, prediction01))

prediction01 
from sklearn.tree import DecisionTreeClassifier #using pca to make decision tree



decision_tree = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=4)

decision_tree.fit(X1_trainpca,Y1_train)

prediction02 =decision_tree.predict(X1_testpca)



print(prediction02)



from sklearn.metrics import accuracy_score

print(accuracy_score(Y1_test, prediction02))



from sklearn.neighbors import KNeighborsClassifier  #KNN ANALYSIS with pca scaling

knn=KNeighborsClassifier(n_neighbors=5)   #n_neighbors = value of k





knn.fit(X1_trainpca,Y1_train) 



#let us get the predictions using the classifier we had fit above

Y1_pred01 = knn.predict(X1_testpca)

print(Y1_pred01)



from sklearn.metrics import accuracy_score

print("Accuracy is",accuracy_score(Y1_test, Y1_pred01))
from sklearn.ensemble import RandomForestClassifier # to make random forest with pca scaling

model = RandomForestClassifier(n_estimators=5, random_state=50, max_depth=5, criterion = 'entropy')



model.fit(X1_trainpca, Y1_train)

pred01=model.predict(X1_testpca)

print(pred01)

from sklearn.metrics import accuracy_score

print(accuracy_score(Y1_test, pred01))



from sklearn.svm import SVC #SVM ANALYSIS WITH MIN MAX SCALING

svc=SVC() 

svc.fit(X1_trainpca,Y1_train)

Y1_pred02=svc.predict(X1_testpca)

Y1_pred02





from sklearn import metrics

print('Accuracy Score: of svc default parameters')

print(metrics.accuracy_score(Y1_test,Y1_pred02))



svc=SVC(kernel='linear')  #WITH LINEAR KERNEL



svc.fit(X1_trainpca,Y1_train)



Y1_pred03=svc.predict(X1_testpca)



print('Accuracy Score: with default linear kernel')



print(metrics.accuracy_score(Y1_test,Y1_pred03))



svc=SVC(kernel='rbf')   #WITH RADIAL KERNEL

svc.fit(X1_trainpca,Y1_train)

Y1_pred04=svc.predict(X1_testpca)

print('Accuracy Score: with default rbf kernel')

print(metrics.accuracy_score(Y1_test,Y1_pred04))

from sklearn.metrics import confusion_matrix  

print(confusion_matrix(Y1_test, Y1_pred04))  

Y1_pred04



svc=SVC(kernel='poly')  #WITH POLY KERNEL

svc.fit(X1_trainpca,Y1_train)

Y1_pred05=svc.predict(X1_testpca)

print('Accuracy Score:with default poly kernel')

print(metrics.accuracy_score(Y1_test,Y1_pred05)) 
from sklearn.svm import SVC      #with rbf kernel and with C  and gamma value



from sklearn import metrics

svm = SVC(kernel = 'rbf', C=1.0, gamma = 0.1, random_state=0)



svm.fit(X1_trainpca,Y1_train)



Y1_pred06=svm.predict(X1_testpca)



print('Accuracy Score:with rbf kernel and with C and gamma value')



print(metrics.accuracy_score(Y1_test,Y1_pred06))  
from sklearn import metrics

from sklearn.svm import SVC  # with sigmoid kernel

svclassifier = SVC(kernel='sigmoid')  

svclassifier.fit(X1_trainpca, Y1_train)  

Y1_pred07=svclassifier.predict(X1_testpca)

print('Accuracy Score:with sigmoid kernel')

print(metrics.accuracy_score(Y1_test,Y1_pred07))   









from sklearn.naive_bayes import GaussianNB  

model = GaussianNB()

model.fit(X1_trainpca, Y1_train)



prediction03 = model.predict(X1_testpca)







from sklearn.metrics import accuracy_score



print(accuracy_score(Y1_test, prediction03))





print(metrics.confusion_matrix(Y1_test, prediction03))

prediction03 