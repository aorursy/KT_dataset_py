import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



from pandas.plotting import radviz

from sklearn.preprocessing import StandardScaler

from sklearn.cross_validation import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.decomposition import PCA

from matplotlib.colors import ListedColormap
# Define decision boundary plotting method

def plotDecisionBoundary (classifier, X, y, colorMap, labels, title = 'decision plot', featureX = 'feature X', featureY = 'feature Y') :

    X1, X2 = np.meshgrid(

        np.arange(start = X[:, 0].min() - 1, stop = X[:, 0].max() + 1, step = 0.01),

        np.arange(start = X[:, 1].min() - 1, stop = X[:, 1].max() + 1, step = 0.01)

    )



    plt.contourf(

            X1, 

            X2, 

            classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 

            alpha = 0.5, 

            cmap = colorMap

    )



    plt.xlim(X1.min(), X1.max())

    plt.ylim(X2.min(), X2.max())



    for i,j in enumerate(np.unique(y)):

        plt.scatter(

                X[y == j, 0], 

                X[y == j, 1], 

                c = colorMap(i), 

                label = labels[j]

    )



    plt.title(title)  

    plt.xlabel(featureX)

    plt.ylabel(featureY)

    plt.legend()

    plt.show()

    



# Define model score display method

def plotModelScore(classifier, X, y, modelName = 'model'):

    predictions = classifier.predict(X)

    classificationReport = classification_report(y_test, predictions)

    confusionMatrix = confusion_matrix(y_test, predictions)

    accuracyScores = cross_val_score(estimator = classifier, X = X, y = y, cv = 10)

    

    print(modelName)

    print('Classification Report')

    print(classificationReport)

    print('Confusion Matrix')

    print(confusionMatrix)

    print('CV Accuracy Scores')

    print(accuracyScores) 

    print('Overall accuracy')

    print(np.mean(accuracyScores))

# Import and plot dataset



dataset = pd.read_csv('../input/Iris.csv')

dataset.head()

dataset["Species"].value_counts()



sns.set(style="white", color_codes=True)

sns.FacetGrid(dataset, hue="Species", size=5).map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()

plt.show()

sns.FacetGrid(dataset, hue="Species", size=5).map(plt.scatter, "PetalLengthCm", "PetalWidthCm").add_legend()

plt.show()

radviz(dataset.drop("Id", axis=1), "Species")

plt.show()



classes = {

   'Iris-setosa' : 0,

   'Iris-virginica' : 1, 

   'Iris-versicolor': 2

}



X = dataset.iloc[:, 1 : 5]

y = dataset.iloc[:, 5]

y = list(map(lambda classAsString : classes[classAsString], y))
# Preprocess Dataset

standardScaler = StandardScaler()

X = standardScaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Select best KNN model with cross validation

possibleKNNParameters = [{ 

        'n_neighbors': np.arange(start = 1, stop = 50, step = 1)

}]

    

knnModel     = KNeighborsClassifier()

gridSearchCV = GridSearchCV(estimator = knnModel, param_grid = possibleKNNParameters)

gridSearchCV.fit(X_train, y_train).score(X_test, y_test)



bestNeighborsK = gridSearchCV.best_estimator_.n_neighbors;

knnModel       = KNeighborsClassifier(n_neighbors = bestNeighborsK)

knnModel.fit(X_train, y_train)

# Select best SVC model with cross validation

possibleSVCParameters = [{ 

        'C': np.logspace(1, 10, 10), 

        'gamma': [0.01, 0.001, 0.0001, 0.00001], 

        'kernel': ['rbf']

}]



svcModel = SVC()



gridSearchCV = GridSearchCV(estimator = svcModel, param_grid = possibleSVCParameters)

gridSearchCV.fit(X_train, y_train).score(X_test, y_test)



bestC       = gridSearchCV.best_estimator_.C;

bestGamma   = gridSearchCV.best_estimator_.gamma;



svcModel = SVC(C = bestC, gamma = bestGamma)

svcModel.fit(X_train, y_train)
# Select best RandomForest model with cross validation

possibleRandomForestParameters = [{ 

        'n_estimators': [10, 25, 50, 100], 

        'max_depth': [5, 10, 50, 100, 250, 500, 1000], 

        'criterion': ['gini', 'entropy']

}]



randomForestModel = RandomForestClassifier()



gridSearchCV = GridSearchCV(estimator = randomForestModel, param_grid = possibleRandomForestParameters)

gridSearchCV.fit(X_train, y_train).score(X_test, y_test)



bestEstimators = gridSearchCV.best_estimator_.n_estimators

bestMaxDepth   = gridSearchCV.best_estimator_.max_depth

bestCriterion  = gridSearchCV.best_estimator_.criterion



randomForestModel = RandomForestClassifier(n_estimators = bestEstimators, max_depth = bestMaxDepth, criterion = bestCriterion)

randomForestModel.fit(X_train, y_train)
# Select best neural network model

possibleNeuralNetworkParameters = [{  

        'hidden_layer_sizes' : [(100), (500), (100, 100), (250,250), (100,100,50), (100,250,50), (250,250,100)],

        'activation' :['relu', 'logistic'],

        'alpha' : [0.00001, 0.0001, 0.001, 0.01],

        'learning_rate' : ['constant', 'adaptive']

}]



neuralNetworkModel = MLPClassifier()



gridSearchCV = GridSearchCV(estimator = neuralNetworkModel, param_grid = possibleNeuralNetworkParameters)

gridSearchCV.fit(X_train, y_train).score(X_test, y_test)



neuralNetworkModel = MLPClassifier(

        hidden_layer_sizes  = gridSearchCV.best_estimator_.hidden_layer_sizes, 

        activation          = gridSearchCV.best_estimator_.activation, 

        alpha               = gridSearchCV.best_estimator_.alpha,

        learning_rate       = gridSearchCV.best_estimator_.learning_rate

) 

neuralNetworkModel.fit(X_train, y_train)

# Display models score reports

plotModelScore(svcModel, X_test, y_test, 'SVC model performance')

plotModelScore(knnModel, X_test, y_test, 'KNN model performance')

plotModelScore(randomForestModel, X_test, y_test, 'Random forest model performance')

plotModelScore(neuralNetworkModel, X_test, y_test, 'Neural Network model performance')



# Display models decisions boundaries

pca = PCA(n_components=2)

pca.fit(X_train)



X_display = pca.transform(X_train)

y_display = y_train

    

svcModel = SVC(C = bestC, gamma = bestGamma)

svcModel.fit(X_display, y_display)

knnModel.fit(X_display, y_display)

randomForestModel.fit(X_display, y_display)

neuralNetworkModel.fit(X_display, y_display)



plotDecisionBoundary(

        classifier = svcModel, 

        X = X_display, 

        y = y_display, 

        colorMap = ListedColormap(['red', 'green', 'blue']), 

        title = 'SVC on training set', 

        featureX = 'PCA 1',

        featureY = 'PCA 2',

        labels = {

                0 : 'Iris-setosa',

                1 : 'Iris-virginica', 

                2 : 'Iris-versicolor'

        }

)



plotDecisionBoundary(

        classifier = knnModel, 

        X = X_display, 

        y = y_display, 

        colorMap = ListedColormap(['red', 'green', 'blue']), 

        title = 'KNN on training set', 

        featureX = 'PCA 1',

        featureY = 'PCA 2',

        labels = {

                0 : 'Iris-setosa',

                1 : 'Iris-virginica', 

                2 : 'Iris-versicolor'

        }

)



plotDecisionBoundary(

        classifier = randomForestModel, 

        X = X_display, 

        y = y_display, 

        colorMap = ListedColormap(['red', 'green', 'blue']), 

        title = 'Random forest on training set', 

        featureX = 'PCA 1',

        featureY = 'PCA 2',

        labels = {

                0 : 'Iris-setosa',

                1 : 'Iris-virginica', 

                2 : 'Iris-versicolor'

        }

)



plotDecisionBoundary(

        classifier = neuralNetworkModel, 

        X = X_display, 

        y = y_display, 

        colorMap = ListedColormap(['red', 'green', 'blue']), 

        title = 'Neural network on training set', 

        featureX = 'PCA 1',

        featureY = 'PCA 2',

        labels = {

                0 : 'Iris-setosa',

                1 : 'Iris-virginica', 

                2 : 'Iris-versicolor'

        }

)