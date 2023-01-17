import pandas as pd

import seaborn as sns

import numpy as np



import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline





def warn(*args, **kwargs):

    pass

import warnings

warnings.warn = warn



#LABEL ENCODER

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()



from sklearn.neighbors import KNeighborsClassifier



from sklearn.utils import shuffle



from sklearn.metrics import mean_squared_error

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score, recall_score, f1_score



from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB



from math import sqrt

import math

from sklearn.metrics import classification_report



from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from mlxtend.plotting import plot_decision_regions



from sklearn import metrics

from sklearn.model_selection import cross_validate





from sklearn.tree import DecisionTreeClassifier 





from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO  

from IPython.display import Image  

import pydotplus



from sklearn.metrics import accuracy_score
#Function for drawing category





def draw(name, restrict=0):

    category_count = data[name].value_counts()

    if restrict > 0:

        category_count= category_count[:restrict]

    plt.figure(figsize=(10,5))

    sns.barplot(category_count.index, category_count.values, alpha=0.8)

    plt.ylabel('Number of '+name, fontsize=12)

    plt.xlabel(name, fontsize=5)

    plt.show()



#DOWNLOAD THE DATA

data = pd.read_csv('https://query.data.world/s/txomiyjwpcnn3debgfyk5sshdiasmh')



### DRAW BEFORE THE DROP ####



draw("Age")

draw("Demographic information")

draw("Status")



### DROP FEATURES ###

data = data.drop([0])

data = data.drop(data.columns[[4,5,7,8,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,62,63]], axis=1)

data = data.dropna()



###AFTER feature extraction our age scale between 55-75 age

draw("Age")

draw("Demographic information")

draw("Status")



## FEATURE SCALING ##

data['Age'].unique()

data['Age'] = data['Age'].apply(lambda x: int(x))

print(data['Demographic information'].unique())

data["Demographic information"] = encoder.fit_transform(data["Demographic information"])

print(data['Tremor'].unique())

data["Tremor"] = encoder.fit_transform(data["Tremor"])

print(data['Medication'].unique())

data["Medication"] = encoder.fit_transform(data["Medication"])

print(data['Medication.3'].unique())

data["Medication.3"] = encoder.fit_transform(data["Medication.3"])

print(data['Medication.5'].unique())

data["Medication.5"] = encoder.fit_transform(data["Medication.3"])



def toFloat(name):

    data[name] = data[name].apply(lambda x: float(x))

toFloat("Speech examination: speaking task of monologue")

toFloat("Speech examination: speaking task of monologue.1")

toFloat("Speech examination: speaking task of monologue.2")

toFloat("Speech examination: speaking task of monologue.3")

toFloat("Speech examination: speaking task of monologue.4")

toFloat("Speech examination: speaking task of monologue.5")

toFloat("Speech examination: speaking task of monologue.6")

toFloat("Speech examination: speaking task of monologue.7")

toFloat("Speech examination: speaking task of monologue.8")

toFloat("Speech examination: speaking task of monologue.9")







#print(data[name].unique())

toFloat('Speech examination: speaking task of reading passage')

toFloat('Speech examination: speaking task of reading passage.1')

toFloat('Speech examination: speaking task of reading passage.2')

toFloat('Speech examination: speaking task of reading passage.3')

toFloat('Speech examination: speaking task of reading passage.4')

toFloat('Speech examination: speaking task of reading passage.5')

toFloat('Speech examination: speaking task of reading passage.6')

toFloat('Speech examination: speaking task of reading passage.7')

toFloat('Speech examination: speaking task of reading passage.8')

toFloat('Speech examination: speaking task of reading passage.9')

toFloat('Speech examination: speaking task of reading passage.10')

toFloat('Speech examination: speaking task of reading passage.11')





print(data['Status'].unique())

data["Status"] = encoder.fit_transform(data["Status"])





print(data['History '].unique())

data["History "] = encoder.fit_transform(data["History "])





print(data['Status'].value_counts())
corr = data.corr()

corr['Status']
## SPLIT X AND Y 

data = shuffle(data)

x,y = data.loc[:,data.columns != 'Status'], data.loc[:,'Status']
## Function for test train split

def splitTestTrain():

    return train_test_split(x, y, test_size = 0.2, random_state = 42)



## Function for single precision, recall, f1 etc



def scorer(y_test,prediction):

    print('Confusion Matrix')

    print(confusion_matrix(y_test, prediction))

    print()

    print(classification_report(y_test,prediction))



## Cross validation x,y 



def cross_val_multiple(func):

    print('\033[91mK FOLD CROSS VALIDATION\033[0m')

    accuracy = cross_val_score(func, x, y, cv=5, scoring="accuracy")

    precision = cross_val_score(func, x, y, cv=5, scoring="precision")

    recall = cross_val_score(func, x, y, cv=5, scoring="recall")

    f1 = cross_val_score(func, x, y, cv=5, scoring="f1")

    print('accuracy_score: ',np.mean(precision))

    print('precision_score:',np.mean(precision))

    print('recall_score:',np.mean(recall))

    print('f1_score:',np.mean(f1))





## Cros val for svm

def cross_val_for_svm(func):

    x_train, x_test, y_train, y_test = splitTestTrain()

    

    sc = StandardScaler()

    x_scaled = sc.fit_transform(x)

    x_test_scaled = sc.transform(x_test)

    x_scaled = sc.transform(x)

    

    pca = PCA(n_components=2)

    pca.fit(x_train_scaled)

    

    x_pca_train = pca.transform(x_train_scaled)

    x_pca_test = pca.transform(x_test_scaled)

    x_pca =  pca.transform(x_scaled)

    

    print('\033[91mK FOLD CROSS VALIDATION\033[0m')

    accuracy = cross_val_score(func, x_pca, y, cv=5, scoring="accuracy")

    precision = cross_val_score(func, x_pca, y, cv=5, scoring="precision")

    recall = cross_val_score(func, x_pca, y, cv=5, scoring="recall")

    f1 = cross_val_score(func, x_pca, y, cv=5, scoring="f1")

    print('accuracy_score: ',np.mean(accuracy))

    print('precision_score:',np.mean(precision))

    print('recall_score:',np.mean(recall))

    print('f1_score:',np.mean(f1))

    



##DRAW Logistic Regression



def sigmoid(x):

    a = []

    for item in x:

        a.append(1/(1+math.exp(-item)))

    return a



def bound(x):

    a = []

    for item in x:

        a.append(0.5)

    return a

  

def drawLGR(y_proba, y_test):

    scaler = MinMaxScaler(feature_range=(-5, 5))

    a = scaler.fit_transform(np.array([x_test["Speech examination: speaking task of reading passage"].tolist()]).T)

    pp = np.arange(-5, 5, 0.01)

    sig = sigmoid(pp)

    bo = bound(pp)

    plt.plot(pp,sig)

    plt.plot(pp,bo)

    i = 0

    for z,o in y_proba > 0.5:

        real = y_test.tolist()[i]

        ft = a[i]

        if real == 1:

            c = "r"

        if real == 0:

            c = "g"

        if z == True:

            p = 0

        if z == False:

            p = 1

        plt.scatter(ft, p, s=10, c=c)

    i+=1

    plt.show()

'''

KNN TEST WITH K = 3 

'''



x_train, x_test, y_train, y_test = splitTestTrain()

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

print('With KNN (K=3) accuracy is: ',accuracy_score(y_test,prediction))

scorer(y_test, prediction)

##K FOLD CROS VALIDATION

cross_val_multiple(knn)



##KNN GRID SEARCH

x_train, x_test, y_train, y_test = splitTestTrain()

k_range = list(range(1, 31))

param_grid = dict(n_neighbors=k_range)



scoring = ['accuracy','f1','precision','recall']

grid = GridSearchCV(knn, param_grid, cv=5, scoring=scoring, refit='recall')



best_model = grid.fit(x_train, y_train)

print('\033[91mBest estimator paramaters\033[0m')

print(best_model.best_estimator_.get_params())



pd.DataFrame(grid.cv_results_).to_csv('export.csv')

y_pred = best_model.best_estimator_.predict(x_test)



scorer(y_test, prediction)

cross_val_multiple(best_model.best_estimator_)
#Naive bayes there is a no hyperparamters for navie bayes

x_train, x_test, y_train, y_test = splitTestTrain()

nb = GaussianNB()

nb.fit(x_train,y_train)

prediction = nb.predict(x_test)



print('Naive bayes Accuracy is: ',accuracy_score(y_test,prediction))

scorer(y_test, prediction)

cross_val_multiple(nb)

## LOGISTIC REGRESSION

x_train, x_test, y_train, y_test = splitTestTrain()

lr = LogisticRegression()

lr.fit(x_train,y_train)

prediction = lr.predict(x_test)

print('Default LOGISTIC REGRESSION accuracy is: ',accuracy_score(y_test,prediction))

scorer(y_test, prediction)



##K FOLD CROS VALIDATION

cross_val_multiple(lr)

##Logistic regression grid search

x_train, x_test, y_train, y_test = splitTestTrain() 

grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge

logreg = LogisticRegression()

logreg_cv = GridSearchCV(logreg,grid,cv=10)

logreg_cv.fit(x_train,y_train)



print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)

print("accuracy :",logreg_cv.best_score_)
## Train with best parameters

x_train, x_test, y_train, y_test = splitTestTrain()

lr = LogisticRegression(C=0.01,penalty="l1")

lr.fit(x_train,y_train)

prediction = lr.predict(x_test)

print('Default LOGISTIC REGRESSION accuracy is: ',accuracy_score(y_test,prediction))

scorer(y_test, prediction)



##K FOLD CROS VALIDATION

cross_val_multiple(lr)
## SVM WITHOUT FEATURE SCALING

x_train, x_test, y_train, y_test = splitTestTrain()



clf = SVC()

clf.fit(x_train, y_train)

prediction = clf.predict(x_test)



print('SVM accuracy is: ',accuracy_score(y_test,prediction))

scorer(y_test, prediction)

cross_val_multiple(clf)
## SVM WITH FEATURE SCALING WITH Linear Kernel



x_train, x_test, y_train, y_test = splitTestTrain()



##Scale the feautres

sc = StandardScaler()

x_train_scaled = sc.fit_transform(x_train)

x_test_scaled = sc.transform(x_test)



pca = PCA(n_components=2)

pca.fit(x_train_scaled)



x_pca_train = pca.transform(x_train_scaled)

x_pca_test = pca.transform(x_test_scaled)



clf = SVC(kernel='linear')

clf.fit(x_pca_train, y_train)



prediction = clf.predict(x_pca_test)



print('SVM accuracy is: ',accuracy_score(y_test,prediction))

scorer(y_test, prediction)





cross_val_for_svm(clf)

# Plotting decision regions

plot_decision_regions(x_pca_train, y_train.as_matrix(), clf=clf, legend=2)



# Adding axes annotations

plt.xlabel('Scaled X ')

plt.ylabel('Scaled Y')

plt.title('Linear svm Train')

plt.show()
# Plotting decision regions

plot_decision_regions(x_pca_test, y_test.as_matrix(), clf=clf, legend=2)

# Adding axes annotations

plt.xlabel('Scaled X ')

plt.ylabel('Scaled Y')

plt.title('Linear svm Test')

plt.show()
Cs = [0.001, 0.01, 0.1, 1, 10]

gammas = [0.001, 0.01, 0.1, 1]

param_grid = {'C': Cs, 'gamma' : gammas}

grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)



grid_search.fit(x_pca_train, y_train)

grid_search.best_params_

print("tuned hpyerparameters :(best parameters for svm kernel=rbf) ",grid_search.best_params_)
clf = SVC(kernel='rbf', C=10, gamma=1)

clf.fit(x_pca_train, y_train)



prediction = clf.predict(x_pca_test)



print('SVM accuracy is: ',accuracy_score(y_test,prediction))

scorer(y_test, prediction)



cross_val_for_svm(clf)

# Plotting decision regions

plot_decision_regions(x_pca_train, y_train.as_matrix(), clf=clf, legend=2)



# Adding axes annotations

plt.xlabel('Scaled X ')

plt.ylabel('Scaled Y')

plt.title('Rbf svm Train')

plt.show()
# Plotting decision regions

plot_decision_regions(x_pca_test, y_test.as_matrix(), clf=clf, legend=2)

# Adding axes annotations

plt.xlabel('Scaled X ')

plt.ylabel('Scaled Y')

plt.title('Rbf svm Test')

plt.show()
##DesicionTree



x_train, x_test, y_train, y_test = splitTestTrain()

clf = DecisionTreeClassifier()



# Train Decision Tree Classifer

clf = clf.fit(x_train,y_train)

#Predict the response for test dataset

prediction = clf.predict(x_test)

# Model Accuracy, how often is the classifier correct?

#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print('Default DecisionTree is: ',accuracy_score(y_test,prediction))

scorer(y_test, prediction)



##K FOLD CROS VALIDATION

cross_val_multiple(clf)

dot_data = StringIO()

export_graphviz(clf, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())
##DesicionTree



x_train, x_test, y_train, y_test = splitTestTrain()

clf = DecisionTreeClassifier(criterion='entropy')



# Train Decision Tree Classifer

clf = clf.fit(x_train,y_train)

#Predict the response for test dataset

prediction = clf.predict(x_test)

# Model Accuracy, how often is the classifier correct?

#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print('Default DecisionTree is: ',accuracy_score(y_test,prediction))

scorer(y_test, prediction)



##K FOLD CROS VALIDATION

cross_val_multiple(clf)

dot_data = StringIO()

export_graphviz(clf, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())
## GRID SEARCH DECISION TREE

x_train, x_test, y_train, y_test = splitTestTrain()



dt = DecisionTreeClassifier()

param_grid = [{'max_depth':np.arange(3, 20),

              'min_samples_leaf':[1, 5, 10, 20, 50, 100]}]

scoring = ['accuracy','f1','precision','recall']

grid = GridSearchCV(dt, param_grid, cv=5, scoring=scoring, refit='recall')



best_model = grid.fit(x_train, y_train)

print(best_model.best_estimator_.get_params())

##DesicionTree



x_train, x_test, y_train, y_test = splitTestTrain()

clf = DecisionTreeClassifier(criterion='gini',max_depth=3, min_samples_leaf=1, min_samples_split=2,presort=False,splitter='best' )



# Train Decision Tree Classifer

clf = clf.fit(x_train,y_train)

#Predict the response for test dataset

prediction = clf.predict(x_test)

# Model Accuracy, how often is the classifier correct?

#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print('Default DecisionTree is: ',accuracy_score(y_test,prediction))

scorer(y_test, prediction)



##K FOLD CROS VALIDATION

cross_val_multiple(clf)

dot_data = StringIO()

export_graphviz(clf, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())