import warnings

warnings.filterwarnings('ignore')



# data analysis and wrangling

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
Train = pd.read_csv('../input/train.csv')

Test = pd.read_csv('../input/test.csv')

# Check the data type of each of the data file

Train.info();



## check the number of rows and columns

Train.shape



## For test set

Test.info();

Test.shape
# Descriptive Statistics

Train.describe()

Test.describe()
###Do we have any missing value in Train and test

Train.isnull().sum()
Test.isnull().sum()
##Lets make new variable Family size

Train['Family']=Train['SibSp']+Train['Parch']+1

Test['Family']=Test['SibSp']+Test['Parch']+1



print (Train['Family'].value_counts())
## Just to avoid any overfitting, will group this family size variable

Train.loc[Train["Family"]==1,"Family_grp"]="Single"

Train.loc[(Train["Family"]>1) & (Train["Family"]<4),"Family_grp"]="Small"

Train.loc[(Train["Family"]>3) ,"Family_grp"]="Large"



Test.loc[Test["Family"]==1,"Family_grp"]="Single"

Test.loc[(Test["Family"]>1) & (Test["Family"]<4),"Family_grp"]="Small"

Test.loc[(Test["Family"]>3) ,"Family_grp"]="Large"



print (Train['Family_grp'].value_counts())



# Lets replace Embarked 2 misiing column by mode

Mode_F=Train.Embarked.dropna().mode()[0]



for i in Train:

    Train['Embarked']=Train['Embarked'].fillna(Mode_F)

    

print (Train['Embarked'].value_counts())    

##So good to go as missing value replaced



Test['Fare'].fillna(Test['Fare'].dropna().median(),inplace=True) 
##We decided to not to use Age with so many missing value

##Since its one of the important varianle, we definitelly need the proxiy for the same

##we gone to use intial in the name column for the same



Train['intial'] =0

for i in Train:

    Train['intial']=Train.Name.str.extract('([A-Za-z]+)\.')

    

print (Train['intial'].value_counts())        



##So there are so many rare value just to avoid any over fit, will

Train.loc[Train["intial"] == "Mlle", "intial"] = 'Miss'

Train.loc[Train["intial"] == "Ms", "intial"] = 'Miss'

Train.loc[Train["intial"] == "Lady", "intial"] = 'Miss'

Train.loc[Train["intial"] == "Dona", "intial"] = 'Miss'

Train.loc[Train["intial"] == "Mme", "intial"] = 'Mrs'



Train.loc[Train["intial"] == "Capt", "intial"] = 'Officer'

Train.loc[Train["intial"] == "Col", "intial"] = 'Officer'

Train.loc[Train["intial"] == "Major", "intial"] = 'Officer'

Train.loc[Train["intial"] == "Dr", "intial"] = 'Officer'

Train.loc[Train["intial"] == "Rev", "intial"] = 'Officer'

Train.loc[Train["intial"] == "Don", "intial"] = 'Officer'

Train.loc[Train["intial"] == "the Countess", "intial"] = 'Officer'

Train.loc[Train["intial"] == "Jonkheer", "intial"] = 'Officer'

Train.loc[Train["intial"] == "Sir", "intial"] = 'Officer'

Train.loc[Train["intial"] == "Countess", "intial"] = 'Officer'



print (Train['intial'].value_counts()) 


Test['intial'] =0

for i in Test:

    Test['intial']=Test.Name.str.extract('([A-Za-z]+)\.')

    

print (Test['intial'].value_counts())        







##So there are so many rare value just to avoid any over fit, will

Test.loc[Test["intial"] == "Mlle", "intial"] = 'Miss'

Test.loc[Test["intial"] == "Ms", "intial"] = 'Miss'

Test.loc[Test["intial"] == "Lady", "intial"] = 'Miss'

Test.loc[Test["intial"] == "Dona", "intial"] = 'Miss'

Test.loc[Test["intial"] == "Mme", "intial"] = 'Mrs'



Test.loc[Test["intial"] == "Capt", "intial"] = 'Officer'

Test.loc[Test["intial"] == "Col", "intial"] = 'Officer'

Test.loc[Test["intial"] == "Major", "intial"] = 'Officer'

Test.loc[Test["intial"] == "Dr", "intial"] = 'Officer'

Test.loc[Test["intial"] == "Rev", "intial"] = 'Officer'

Test.loc[Test["intial"] == "Don", "intial"] = 'Officer'

Test.loc[Test["intial"] == "the Countess", "intial"] = 'Officer'

Test.loc[Test["intial"] == "Jonkheer", "intial"] = 'Officer'

Test.loc[Test["intial"] == "Sir", "intial"] = 'Officer'

Test.loc[Test["intial"] == "Countess", "intial"] = 'Officer'



print (Test['intial'].value_counts())        

##Lets create Fare band

Train['FareBand'] = pd.qcut(Train['Fare'], 4)

print (Train['FareBand'].value_counts())
Train.loc[Train["Fare"]<= 7.91, "Fare1"] = 1

Train.loc[(Train["Fare"] >7.91) & (Train["Fare"] <=14.454), "Fare1"] = 2

Train.loc[(Train["Fare"] >14.454) & (Train["Fare"] <=31), "Fare1"] = 3

Train.loc[Train["Fare"]>= 31, "Fare1"] = 4

print (Train['Fare1'].value_counts())



print (Train['FareBand'].value_counts())
Test.loc[Test["Fare"]<= 7.91, "Fare1"] = 1

Test.loc[(Test["Fare"] >7.91) & (Test["Fare"] <=14.454), "Fare1"] = 2

Test.loc[(Test["Fare"] >14.454) & (Test["Fare"] <=31), "Fare1"] = 3

Test.loc[Test["Fare"]>= 31, "Fare1"] = 4

print (Test['Fare1'].value_counts())
##Select only the required feauters as per our EDA in R version of Kernel



Train_sub=Train[["Pclass","Fare1","Sex","Embarked","Family_grp","intial" ,"Survived"]]

Test_sub=Test[["Pclass","Fare1","Sex","Embarked","Family_grp","intial" ]]
##Data Pre-processing

from sklearn import preprocessing

lb=preprocessing.LabelBinarizer();



Train_sub['Sex']=lb.fit_transform(Train_sub['Sex']);



Test_sub['Sex']=lb.fit_transform(Test_sub['Sex']);



Train_sub.head(5)

#Test_sub["Sex"].head(5)

###Creating dummy variable

Train_fin=pd.get_dummies(Train_sub,columns=["Embarked","Family_grp","intial"],drop_first=True)

Test_fin=pd.get_dummies(Test_sub,columns=["Embarked","Family_grp","intial"],drop_first=True)
##Spilt train and test

X = Train_fin.drop("Survived",axis=1)

y = Train_fin["Survived"]
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

###################################################

##############Fit Logistic regression model 

###################################################

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)

classifier.coef_

    

# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

###Confusion Matrix

print ("Confusion Matrix\n",cm)



acc_train = round(classifier.score(X_train, y_train) * 100, 2)

print ("Accuracy of training data\n ",acc_train)

acc_test = round(classifier.score(X_test, y_test) * 100, 2)

print ("Accuracy of testing data\n ",acc_test);

# k-Fold Cross Validation



# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10);

print ("Cross validation accuracy\n",accuracies.mean());

print ("Cross validation SD\n",accuracies.std());
##Lets try to improve it by Tuning cost parameter

from sklearn.model_selection import GridSearchCV

parameters = [{'C': [1,5,8, 10,15,25,50,100], 'penalty': ['l1']},

              {'C': [1,5,8, 10,15,25,50,100], 'penalty': ['l2']}]

grid_search = GridSearchCV(estimator = classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10)



grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_







acc_test = round(grid_search.score(X_test, y_test) * 100, 2)

print ("Grid search test accuracy\n",acc_test)
############################################################

###### K-Nearest Neighbors (K-NN)

############################################################

# Fitting K-NN to the Training set

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)



##Training and Test accurcay score

acc_train = round(classifier.score(X_train, y_train) * 100, 2)

acc_test = round(classifier.score(X_test, y_test) * 100, 2)



print ("Confusion Matrix\n",cm)

print ("Accuracy of training data\n ",acc_train)

print ("Accuracy of testing data\n ",acc_test);



# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)



print ("Cross validation accuracy\n",accuracies.mean());

print ("Cross validation SD\n",accuracies.std());





##Tuning Hyper Pararmter

from sklearn.model_selection import GridSearchCV

parameters = [{'n_neighbors': [3,5,7,9,11,15,19,25,27,29,31,33,35]}]

grid_search = GridSearchCV(estimator = classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10)



grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

###Applying best model to test data

y_pred = grid_search.predict(X_test)



acc_test = round(grid_search.score(X_test, y_test) * 100, 2)

print ("Accuracy of test scrore \n",acc_test)
###################################

#################SVM - Linear

###################################





from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)





##Training and Test accurcay score

acc_train = round(classifier.score(X_train, y_train) * 100, 2)

acc_test = round(classifier.score(X_test, y_test) * 100, 2)



print ("Confusion Matrix\n",cm)

print ("Accuracy of training data\n ",acc_train)

print ("Accuracy of testing data\n ",acc_test);

# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)



print ("Cross validation accuracy\n",accuracies.mean());

print ("Cross validation SD\n",accuracies.std());





# Applying Grid Search to find the best model and the best parameters

from sklearn.model_selection import GridSearchCV

parameters = [{'C': [1,3,4, 5,6,7,8,9], 'kernel': ['linear']}]





grid_search = GridSearchCV(estimator = classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10)



grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_



###Applying best model to test data

y_pred = grid_search.predict(X_test)



acc_test = round(grid_search.score(X_test, y_test) * 100, 2)

print ("Accuracy of test scrore \n",acc_test)

#####################################

########Rbf

######################################

from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)





##Training and Test accurcay score

acc_train = round(classifier.score(X_train, y_train) * 100, 2)

acc_test = round(classifier.score(X_test, y_test) * 100, 2)



print ("Confusion Matrix\n",cm)

print ("Accuracy of training data\n ",acc_train)

print ("Accuracy of testing data\n ",acc_test);

# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print ("Cross validation accuracy\n",accuracies.mean());

print ("Cross validation SD\n",accuracies.std());



# Applying Grid Search to find the best model and the best parameters

from sklearn.model_selection import GridSearchCV

parameters = [{'C': [0.01,0.05,0.07,1,2, 3], 'kernel': ['rbf'], 'gamma': [0.01,0.05,0.1,0.2,0.3]}]







grid_search = GridSearchCV(estimator = classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10)



grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_





acc_test = round(grid_search.score(X_test, y_test) * 100, 2)

print ("Accuracy of test scrore \n",acc_test)

##############################################

##################Sigmoid#####################

##############################################



from sklearn.svm import SVC

classifier = SVC(kernel = 'sigmoid', random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)





##Training and Test accurcay score

acc_train = round(classifier.score(X_train, y_train) * 100, 2)

acc_test = round(classifier.score(X_test, y_test) * 100, 2)



print ("Confusion Matrix\n",cm)

print ("Accuracy of training data\n ",acc_train)

print ("Accuracy of testing data\n ",acc_test);



# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print ("Cross validation accuracy\n",accuracies.mean());

print ("Cross validation SD\n",accuracies.std());



# Applying Grid Search to find the best model and the best parameters

from sklearn.model_selection import GridSearchCV

parameters = [{'C': [0.01,0.05,0.07,1,2, 3], 'kernel': ['sigmoid'], 'gamma': [0.01,0.05,0.1,0.2,0.3]}]



grid_search = GridSearchCV(estimator = classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10)



grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_



acc_test = round(grid_search.score(X_test, y_test) * 100, 2)

print ("Accuracy of test scrore \n",acc_test)

################################################

#####################Naive Bayes

################################################



# Fitting Naive Bayes to the Training set

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)





##Training and Test accurcay score

acc_train = round(classifier.score(X_train, y_train) * 100, 2)

acc_test = round(classifier.score(X_test, y_test) * 100, 2)





print ("Confusion Matrix\n",cm)

print ("Accuracy of training data\n ",acc_train)

print ("Accuracy of testing data\n ",acc_test);

# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print ("Cross validation accuracy\n",accuracies.mean());

print ("Cross validation SD\n",accuracies.std());





##There is no hyper parmerer here for Tuning



###################################################

#################Decision Tree

###################################################

# Fitting Decision Tree Classification to the Training set

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

##Training and Test accurcay score

acc_train = round(classifier.score(X_train, y_train) * 100, 2)

acc_test = round(classifier.score(X_test, y_test) * 100, 2)



print ("Confusion Matrix\n",cm)

print ("Accuracy of training data\n ",acc_train)

print ("Accuracy of testing data\n ",acc_test);



# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print ("Cross validation accuracy\n",accuracies.mean());

print ("Cross validation SD\n",accuracies.std());



#####################################################

#####################RANDOM FOREST

####################################################

classifier= RandomForestClassifier(criterion = "gini", 

                                       min_samples_leaf = 1, 

                                       min_samples_split = 10,   

                                       n_estimators=100, 

                                       max_features='auto', 

                                       oob_score=True, 

                                       random_state=0)





classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)



##Training and Test accurcay score

acc_train = round(classifier.score(X_train, y_train) * 100, 2)

acc_test = round(classifier.score(X_test, y_test) * 100, 2)



print ("Confusion Matrix\n",cm)

print ("Accuracy of training data\n ",acc_train)

print ("Accuracy of testing data\n ",acc_test);



# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print ("Cross validation accuracy\n",accuracies.mean());

print ("Cross validation SD\n",accuracies.std());

importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(classifier.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances.plot.bar()

##The intial_office not playing imporat role, lets drop and run the model again

X_train=X_train.drop("intial_Officer",axis=1)

X_test=X_test.drop("intial_Officer",axis=1)



##Fit Model again



classifier= RandomForestClassifier(criterion = "gini", 

                                       min_samples_leaf = 1, 

                                       min_samples_split = 10,   

                                       n_estimators=100, 

                                       max_features='auto', 

                                       oob_score=True, 

                                       random_state=0)





classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)



##Training and Test accurcay score

acc_train = round(classifier.score(X_train, y_train) * 100, 2)

acc_test = round(classifier.score(X_test, y_test) * 100, 2)



print ("Confusion Matrix\n",cm)

print ("Accuracy of training data\n ",acc_train)

print ("Accuracy of testing data\n ",acc_test);

# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print ("Cross validation accuracy\n",accuracies.mean());

print ("Cross validation SD\n",accuracies.std());



print("oob score:", round(classifier.oob_score_, 4)*100, "%")



###Precion and Recall

from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(y_test, y_pred))

print("Recall:",recall_score(y_test, y_pred))





##F1 score



from sklearn.metrics import f1_score

print("F1 score:",f1_score(y_test, y_pred))

###ROC Curve



# getting the probabilities of our predictions

y_scores = classifier.predict_proba(X_train)

y_scores = y_scores[:,1]



from sklearn.metrics import roc_curve



false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_scores)



plt.figure(figsize=(14, 7))

plt.plot(false_positive_rate, true_positive_rate)



plt.plot([0,1],[0,1],'r',linewidth=4)

plt.xlim([0.0,1])

plt.ylim([0.0,1])

plt.title("ROC Curve for Diabetic classifier")

plt.xlabel("FP(1-Spec")

plt.ylabel("TP(Senistivity)")

plt.grid(True)





###ROC AUC Score

from sklearn.metrics import roc_auc_score

r_a_score = roc_auc_score(y_train, y_scores)

print("ROC-AUC-Score:", r_a_score)
##Decision tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



def print_score(clf, X_train, y_train, X_test, y_test, train=True):

    '''

    print the accuracy score, classification report and confusion matrix of classifier

    '''

    if train:

        '''

        training performance

        '''

        print("Train Result:\n")

        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))

        print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))

        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))



        res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')

        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))

        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))

        

    elif train==False:

        '''

        test performance

        '''

        print("Test Result:\n")        

        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))

        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))

        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test))))    

##Decision tree

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)

classifier.fit(X_train, y_train)



##Train accuracy

print_score(classifier, X_train, y_train, X_test, y_test, train=True)



##Test accuracy

print_score(classifier, X_train, y_train, X_test, y_test, train=False) # Test

##Bagging, oobd score False

bag_clf = BaggingClassifier(base_estimator=classifier,n_estimators=1000,

                            bootstrap=True,

                            random_state=42)

      

bag_clf.fit(X_train, y_train)



##Train score

print_score(bag_clf, X_train, y_train, X_test, y_test, train=True)
##Test score

print_score(bag_clf, X_train, y_train, X_test, y_test, train=False)

##Bagging (OOB=True)



bag_clf = BaggingClassifier(base_estimator=classifier,n_estimators=1000,

                            bootstrap=True, oob_score=True,

                             random_state=0)



bag_clf.fit(X_train, y_train)



bag_clf.oob_score_



##Train score

print_score(bag_clf, X_train, y_train, X_test, y_test, train=True)



##Test Result

print_score(bag_clf, X_train, y_train, X_test, y_test, train=False)
##Extra tree 



from sklearn.ensemble import ExtraTreesClassifier



xt_clf=ExtraTreesClassifier(random_state=42)

xt_clf.fit(X_train,y_train)



##Train result

print_score(xt_clf, X_train, y_train, X_test, y_test, train=True)

##Test Result

print_score(xt_clf, X_train, y_train, X_test, y_test, train=False)
### AdaBoost / Adaptive Boosting¶



from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier()

ada_clf.fit(X_train, y_train)



##Train result

print_score(ada_clf, X_train, y_train, X_test, y_test, train=True)

##Test Result

print_score(ada_clf, X_train, y_train, X_test, y_test, train=False)
##Add boost with RF+best parameters

from sklearn.ensemble import RandomForestClassifier

ada_clf = AdaBoostClassifier(RandomForestClassifier(bootstrap=False,

 criterion= 'gini',

 max_depth= None,

 min_samples_leaf= 7,

 min_samples_split= 2))





ada_clf.fit(X_train, y_train)





print_score(ada_clf, X_train, y_train, X_test, y_test, train=True)



##Test Result

print_score(ada_clf, X_train, y_train, X_test, y_test, train=False)


####Graidient Boosting/

from sklearn.ensemble import GradientBoostingClassifier



gbc_clf = GradientBoostingClassifier()

gbc_clf.fit(X_train, y_train)



print_score(gbc_clf, X_train, y_train, X_test, y_test, train=True)
##Test result

print_score(gbc_clf, X_train, y_train, X_test, y_test, train=False)


##XG booost

import xgboost as xgb



xgb_clf = xgb.XGBClassifier(max_depth=5, n_estimators=10000, learning_rate=0.3)



xgb_clf.fit(X_train,y_train)

##Train results

print_score(xgb_clf, X_train, y_train, X_test, y_test, train=True)

##Test Results

print_score(xgb_clf, X_train, y_train, X_test, y_test, train=False)