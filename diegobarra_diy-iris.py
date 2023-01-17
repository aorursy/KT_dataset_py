# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt



plt.style.use("ggplot")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# First I import the file 

file = "../input/Iris.csv"

df = pd.read_csv(file)
 # I dropped the id column because it is not useful



df.drop('Id',axis=1,inplace=True)

print(df.info())

print(df.head(2))

print(df.describe())

# Performing some EDA for relation between Sepal length and sepal width on each of the 3 species.



fig = df[df.Species == "Iris-setosa"].plot(kind="scatter",x = "SepalLengthCm",y="SepalWidthCm",color="green",label="Setosa")

df[df.Species == "Iris-versicolor"].plot(kind="scatter",x = "SepalLengthCm" ,y="SepalWidthCm",color="blue",label="Versicolor",ax=fig)

df[df.Species == "Iris-virginica"].plot(kind="scatter",x = "SepalLengthCm",y="SepalWidthCm",color ="orange",label="Virginica",ax=fig)

fig.set_xlabel("Sepal Length (Cm)")

fig.set_ylabel("Sepal Width (Cm)")

fig.set_title("Sepal Length vs Sepal Width for three species of Iris")

fig=plt.gcf

plt.show()



# Performing some EDA for relation between Petal Length and Petal Width on each of the 3 species



fig1 = df[df.Species == "Iris-setosa"].plot(kind="scatter",x="PetalLengthCm",y="PetalWidthCm",color="green",label="Setosa")

df[df.Species == "Iris-versicolor"].plot(kind="scatter",x="PetalLengthCm",y="PetalWidthCm",color="blue",label="Versicolor",ax=fig1)

df[df.Species == "Iris-virginica"].plot(kind="scatter",x="PetalLengthCm",y="PetalWidthCm",color="orange",label="Virginica",ax=fig1)

fig1.set_xlabel("Petal Length (Cm)")

fig1.set_ylabel("Petal Width (Cm)")

fig1.set_title("Petal Length vs Petal Width for three species of Iris")

fig=plt.gcf

plt.show()



plt.figure(figsize=(10,10))

plt.subplot(2,2,1)

sns.violinplot(x="Species",y="PetalLengthCm",data=df)

plt.subplot(2,2,2)

sns.violinplot(x="Species",y="PetalWidthCm",data=df)

plt.subplot(2,2,3)

sns.violinplot(x="Species",y="SepalLengthCm",data=df)

plt.subplot(2,2,4)

sns.violinplot(x="Species",y="SepalLengthCm",data=df)

plt.show()





sns.heatmap(df.corr(),annot=True,cmap='cubehelix_r')
# STEP a: We set up the data for our model by removing the "SepalWidthCm" column from the dataset

# Then we get our X (independent variables or features) and y (target) for our classifiers.



df.drop(['SepalWidthCm'], axis = 1, inplace = True)



X = df.drop("Species",axis=1).values

y = df.Species.values



# STEP b: Train test split and some exploration with our first model



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)



from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(X_train,y_train)



print("Preliminary model score:")

print("")

print(knn.score(X_test,y_test))



neighbors = np.arange(1, 9)

train_accuracy = np.empty(len(neighbors))

test_accuracy = np.empty(len(neighbors))



for i, k in enumerate(neighbors):

    # We instantiate the classifier

    knn = KNeighborsClassifier(n_neighbors=k)



    # Fit the classifier to the training data

    knn.fit(X_train,y_train)

    

    # Compute accuracy on the training set

    train_accuracy[i] = knn.score(X_train, y_train)



    # Compute accuracy on the testing set

    test_accuracy[i] = knn.score(X_test, y_test)



# Visualization of k values vs accuracy



plt.title('k-NN: Varying Number of Neighbors')

plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')

plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.show()
# STEP d: Create the model for deployment



from sklearn.model_selection import cross_val_score



knn_defmodel = KNeighborsClassifier(n_neighbors=5)

cv_scores = cross_val_score(knn_defmodel,X,y,cv=5)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))





# This small code is intended to ignore deprecation warnings

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)



#Now we import the modules required for our SVM model



from sklearn import svm

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.preprocessing import scale



# Setting the parameters for gridsearch. By default, C value will be 1 in the GridSearch,

# but if we want to fine tune the parameters, it is advisable to use a logarithmic scale on C,

# as well as for gamma



C_range = np.logspace(-2, 10, 13)

gamma_range = np.logspace(-9, 3, 13)

param_grid = dict(gamma=gamma_range, C=C_range)



# for SVM is highly recommended to rescale the training and testing X values, so we import scale module from sklearn.preprocessing

Xs_train = scale(X_train)

Xs_test = scale(X_test)



for kernel in ("rbf","poly"):

    # svm model instantiation

    svmmodel = svm.SVC(kernel = kernel) # here we choose which kernel we want to use (Radial Base Function and Polynomial)

    

    # Another form of train_test_split 

    strat = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)



    SVM_cv = GridSearchCV(svmmodel, param_grid=param_grid, cv=strat)

    SVM_cv.fit(Xs_train,y_train)



    # We show the best parameters selected for our SVM model,

    print("Parameters, accuracy and score from " +kernel +" Kernel:")

    print("")

    print("   Tuned Parameters: {}".format(SVM_cv.best_params_))

    print("   Tuned Accuracy: {}".format(SVM_cv.best_score_))





    # Now we create the model with our best parameters

    Xs = scale(X)



    SVM_C = SVM_cv.best_params_["C"]

    SVM_gamma = SVM_cv.best_params_["gamma"]

    SVM_defmodel = svm.SVC(gamma = SVM_gamma, C = SVM_C)

    SVM_defmodel.fit(Xs,y)

    print("   Model Score: {}".format(SVM_defmodel.score(Xs,y)))


