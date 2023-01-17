# 1 LOAD LIBRARIES



import pandas as pd

import numpy as np

import pickle as pk



# for dataset spliting

from sklearn.model_selection import train_test_split

#

from sklearn.model_selection import cross_val_score





# visualization

import matplotlib.pyplot as plt



#

from pandas.plotting import scatter_matrix



# 

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC





# metrics

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



# 2 LOAD DATA



data = pd.read_csv('../input/Iris.csv')

data.head(2)
# 3 Summarization of dataset: Descriptive Stats

data.describe().T
# checking if any null values 

data.isnull().values.any()
# How manu classes of target col 'class'

data['Species'].unique()
data.groupby('Species').size()
# Data preprocessing



def pre_processing(data):

    X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

    y = data['Species']



    xtrain,xtest, ytrain, ytest = train_test_split(X,y,test_size=0.33)

    

    return  xtrain,xtest, ytrain, ytest

    
# 4 Data visualization



col_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']



data[col_names].plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)



plt.show()
data[col_names].hist()
scatter_matrix(data[col_names])
# 4 Data preprocessing

xtrain,xtest, ytrain, ytest = pre_processing(data)
# Building Models for iris flower classification



np.random.seed(1000)

# making a list of ml classification models

models = []



def classification_Models(xtrain,xtest, ytrain, ytest ):

    



    

    models.append( ('LR',  LogisticRegression()) )

    models.append( ('CART',DecisionTreeClassifier()) )

    models.append( ('KNN', KNeighborsClassifier()) )

    models.append( ('NB',  GaussianNB()) )

    models.append( ('LDA',  LinearDiscriminantAnalysis()) )

    models.append( ('SVM',  SVC()) )



    modeloutcomes = []

    modelnames = []

    for name,model in models:

        v_results = cross_val_score(model, xtrain, ytrain, cv = 3, 

                                     scoring='accuracy', n_jobs = -1, verbose = 0)

        print(name,v_results.mean())

        modeloutcomes.append(v_results)

        modelnames.append(name)

        

    print(modeloutcomes)

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.set_xticklabels(modelnames)

    plt.boxplot(modeloutcomes)

        

classification_Models(xtrain,xtest, ytrain, ytest)
# Evaluating and predicting models





for name,model in models:

    trainedmodel = model.fit(xtrain,ytrain)

    

    # prediction

    ypredict = trainedmodel.predict(xtest)

    

    acc = accuracy_score(ytest,ypredict)

    classreport = classification_report(ytest,ypredict)

    confMat = confusion_matrix(ytest,ypredict)

    

    print('\n****************************'+name)

    print('The accuracy: {}'.format(acc))

    print('The Classification Report:\n {}'.format(classreport))

    print('The Confusion Matrix:\n {}'.format(confMat))

    

    

    # save models

    import pickle as pk

    

    with open('model_'+name+'.pickle','wb') as f:

        pk.dump(trainedmodel,f)

    