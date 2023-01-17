# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


#Als erstes lesen wir die .arff-Datei ein, die unsere Trainingsdaten enthält
def readData(filename):
    f = open(filename)
    data_line = False
    data = []
    for l in f:
        l = l.strip() 
        if data_line:
            content = [float(x) for x in l.split(',')]
            if len(content) == 3:
                data.append(content)
        else:
            if l.startswith('@DATA'):
                data_line = True
        #print(data)
    return data
    
# Hierbei werden die Werte für die X und Y die Test und die Trainingsdaten gesplittet    
def dataSplit(data_arff):
    df_data = pd.DataFrame({'x':[item[0] for item in train], 'y':[item[1] for item in train], 'Category':[item[2] for item in train]})
    X = df_data[["x","y"]].values
    Y = df_data["Category"].values
    colors = {-1:'red',1:'green'}
    
    # Aufteilung  in Trainings- und Testdaten 
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y, random_state=0, test_size = 0.5)
    plt.scatter(X[:,0],X[:,1],c=df_data["Category"].apply(lambda x: colors[x]))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    return (X_Train, Y_Train, X_Test, Y_Test)

#Traning der Datensätze mit Visualisierung  
def plotModel(XTrain, YTrain, XTest, YTest):
    test_accuracy = []
    neighbors_range = range(1,40)
    for n_neighbors in neighbors_range:
        
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(XTrain, YTrain)
        test_accuracy.append(clf.score(XTest, YTest))
        
    plt.plot(neighbors_range, test_accuracy, label='Genauigkeit bei den Testdaten')
    plt.ylabel('Genauigkeit')
    plt.xlabel('Anzahl der Nachbarn')
    plt.legend()
    #return (XTra, YTra)
    
# Berechnung der Scorewert für das Model
def modelCalculation(XTest, YTest, XTrain, YTrain):
    model = KNeighborsClassifier(n_neighbors = 14)
    model.fit(XTrain, YTrain)
    print(model.score(XTest,YTest))

# Diese Methode plotet Die Entscheidung    
def plotDecisionComparation(model, X, y):
    cmap_light = ListedColormap(['#AAAAFF', '#AAFFAA', '#FFAAAA']) 
    cmap_bold = ListedColormap(['#0000FF', '#00FF00', '#FF0000'])
    h = .02  # step size in the mesh
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
              edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(model.__class__.__name__)

    plt.show()
    
# In dieser Methode werden die Trainierte Daten in 2 Breichen geteilt
def trainingShowScopes(X_train, Y_train, X_test, Y_test):
    model = KNeighborsClassifier(13) #Anzahl betrachteter Nachbarn
    model.fit(X_train, Y_train)
    plotDecisionComparation(model, X_train, Y_train)
    print ('train accuracy: {}'.format(model.score(X_train, Y_train)))
    print ('test accuracy: {}'.format(model.score(X_test, Y_test)))
    print("\n################################################################################\n")
    return model
    
# Diese Funktion vesucht ein Vorhersagen zu bestimmen    
def predictionResult(model):
    ######### hier versuchen wir nun das Vorhersagen#######
    testdf = pd.read_csv("../input/kiwhs-comp-1-complete/test.csv")

    testX = testdf[["X","Y"]].values
    model.predict(testX)
    ######################################################

    ######## Anschließend Speichern wir unsere Vorhersage ab #######
    prediction = pd.DataFrame()
    id = []
    for i in range(len(testX)):
        id.append(i)
        i = i + 1
    prediction["Id (String)"] = id 
    prediction["Category (String)"] = model.predict(testX).astype(int)
    print(prediction[:10])
    prediction.to_csv("agwn_submission_knn.csv", index=False, header=True)

if __name__== "__main__":
    import os
    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    from sklearn.model_selection import train_test_split #  Split arrays or matrices into random train and test subsets
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KNeighborsClassifier
    import importlib.machinery
    
    #Source: Competition --> Discussion
    #Visualization
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.pylab as pylab
    import seaborn as sns
    from pandas.tools.plotting import scatter_matrix
    # Compare
    # http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html
    from matplotlib.colors import ListedColormap
    
    train = readData("../input/kiwhs-comp-1-complete/train.arff")
    (XTr, YTr,XTe, YTe) = dataSplit(train)
    plotModel(XTr, YTr, XTe, YTe)
    modelCalculation(XTe, YTe, XTr, YTr)
    model = trainingShowScopes(XTr, YTr, XTe,YTe)
    predictionResult(model)
    


