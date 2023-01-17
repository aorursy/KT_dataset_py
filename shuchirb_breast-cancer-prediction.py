### This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.tree import DecisionTreeClassifier

from sklearn.cross_validation import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score  

from matplotlib.mlab import PCA

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.neural_network import MLPClassifier

from subprocess import check_output

from pandas import read_csv



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





def readDataSet():

    return pd.read_csv('../input/data.csv')



def cleanseDataSet(bcData):

    df_x=bcData.loc[:,"radius_mean":"fractal_dimension_worst"]

    df_x=df_x.astype(np.float32)

    df_y=bcData.loc[:,"diagnosis"]

    df_y=df_y.replace("M",1)

    df_y=df_y.replace("B",0)

    return df_x, df_y



#Normalized Data

def normalizeData(x):

    for idx in ("radius_mean","fractal_dimension_worst"):

        x[idx]=x[idx]-min(x[idx])/(max(x[idx])-min(x[idx]))

    return x



def performPCA(x):

    pca=PCA(n_components=9,whiten=True)

    x=pca.fit(x).transform(x)

    return x



def createTrainTestDataSet(df_x,df_y):

   x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=.2,random_state=4)

   return x_train,x_test,y_train,y_test



def decisionTreeClssification(x_train,y_train):

    model=DecisionTreeClassifier()

    fittedModel=model.fit(x_train, y_train)

    return fittedModel



def kmeansClustering(x_train,y_train):

    model=KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300)

    fittedModel=model.fit(x_train, y_train)

    return fittedModel



def neuralNetwork(x_train,y_train):

    model = MLPClassifier(

    activation='tanh',

    solver='lbfgs',

    alpha=1e-5,

    early_stopping=False,

    hidden_layer_sizes=(40,40),

    random_state=1,

    batch_size='auto',

    max_iter=20000,

    learning_rate_init=1e-5,

    power_t=0.5,

    tol=1e-4,

    beta_1=0.9,

    beta_2=0.999,

    epsilon=1e-08

    )

    fittedModel=model.fit(x_train, y_train)

    return fittedModel



def envSetup():    

    print("Read data from CSV ::")

    bcData=readDataSet()

    print("Dataset created successfully!!!")

    print("Cleanse dataset ::")

    df_x,df_y=cleanseDataSet(bcData)

    print("Dataset cleansed successfully!!!")

    print("Normalize dataset ::")

    df_x=normalizeData(df_x)

    print("Dataset Normalized successfully!!!")

    print("Find correlation using PCA ::")

    df_x=performPCA(df_x)

    print("Dataset reduced to 9 columns")

    print("Create Train & Test Data set ::")

    x_train,x_test,y_train,y_test=createTrainTestDataSet(df_x,df_y)

    print("Dataset divided as 80% train dataset & 20 test dataset")

    return x_train,x_test,y_train,y_test



def choiceDecisionTree():

    print("++++++++++++++++++ DECISION TREE CLASSIFICATION ++++++++++++++++++")

    print("Perform DecisionTree Classification ::")

    fittedModel=decisionTreeClssification(x_train,y_train)

    print("DecisionTree train model created successfully!!!")

    print("Validate the classificaiton model ::")

    predictions=getPrediction(fittedModel,x_test)

    print("Classification model tested successfully!!!")

    print("Get the Confussion Matrix ::")

    confusion_mat=getConfusionMatrix(y_test, predictions)

    print(confusion_mat)

    print("Get the Accuracy ::")

    accuracy=getAccuracy(y_test, predictions)

    print(accuracy)



def choiceKMeans():

    print("++++++++++++++++++ K_MEANS CLUSTERING ++++++++++++++++++")

    print("Perform K_Means Classification ::")

    fittedModel=kmeansClustering(x_train,y_train)

    print("K-Means train model created successfully!!!")

    print("Validate the clustering model ::")

    predictions=getPrediction(fittedModel,x_test)

    print("Clustering model tested successfully!!!")

    print("Get the Confussion Matrix ::")

    confusion_mat=getConfusionMatrix(y_test, predictions)

    print(confusion_mat)

    print("Get the Accuracy ::")

    accuracy=getAccuracy(y_test, predictions)

    print(accuracy)



def choiceNeuralNetwork():

    print("++++++++++++++++++ Neural Network ++++++++++++++++++")

    print("Perform Neural Network Model ::")

    fittedModel=neuralNetwork(x_train,y_train)

    print("Neural Network train model created successfully!!!")

    print("Validate the Neural Network model ::")

    predictions=getPrediction(fittedModel,x_test)

    print("Neural Network model tested successfully!!!")

    print("Get the Confussion Matrix ::")

    confusion_mat=getConfusionMatrix(y_test, predictions)

    print(confusion_mat)

    print("Get the Accuracy ::")

    accuracy=getAccuracy(y_test, predictions)

    print(accuracy)





def getPrediction(fittedModel,x_test):

    predictions=fittedModel.predict(x_test)

    return predictions



def getConfusionMatrix(y_test, predictions):

    return confusion_matrix(y_test,predictions)



def getAccuracy(y_test, predictions):

    return accuracy_score(y_test,predictions)







x_train,x_test,y_train,y_test=envSetup()



ans="0"

while ans!="4":

    print(ans)

    print ("Select a model from the below list ::\n 1.K-Means \n 2.DecisionTree \n 3.Neural Network \n 4.Exit/Quit")

    ans=input("What would you like to do? ") 

    if ans=="1": 

      choiceKMeans()

    elif ans=="2":

      choiceDecisionTree() 

    elif ans=="3":

      choiceNeuralNetwork()

    elif ans=="4":

      print("\n Goodbye") 

    elif ans !="":

      print("\n Not Valid Choice Try again") 