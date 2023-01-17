#!/usr/bin/env python3
'''
Randomforest - plot: # of trees - accuracy for various training set sizes
             - plot: training set size - accuracy
Modifications by: Noah Carter
Date 4/25/16
Original Author: Hideki Ikeda
Date 7/11/15
Design: Uses an outer loop of training set size and an inner loop of forest size to capture the performance data
for each model; outputs the results;
then, outputs the graph of the highest forest size's performance for different training set sizes
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

def main():
    '''
    Define: Reads the training data and allows for the initialization of a few commonly changed variables and parameters. Calls performTests(...)
    Design: axs is a container which keeps track of figures which will display once the program finishes.
      n_trees is the set of forest sizes. ADJUSTING n_trees TO SMALLER OR FEWER ELEMENTS WILL EXPEDITE THE PROGRAM 
    '''
    # loading training data
    print('Loading training data')
    data = pd.read_csv('../input/train.csv')
    fig,axs=plt.subplots(1,2,figsize=(20,14))
    #n_trees = [10, 15, 20, 25, 30, 40]#, 50, 70, 100, 150]
    #I elected to use the below forest sizes (I couldn't use all sizes due to the 1200 second time cap)
    #n_trees=[10,50,100]
    #however, FOR DEMONSTRATION, THESE FOREST SIZES ARE USED
    n_trees=[10,20,25]
    performTests(data,42000,18,n_trees,False,True,axs)
def plotThisForestAccuracy(n_trees,scores,sc_array,std_array,showStd,axs):
    '''
    Define: Plots classificatory accuracy for the model which was built from by largest random forest as 
      a function of training set size.
    Design: n_trees: the number of trees in the forest; 
     scores: the scores as a list;sc_array: scores as a numpy array;
     std_array: standard deviations of scores as numpy array;showStd: whether or not graph should show lines for
     standard deviations; axs:the object containing the figures to show
    '''    
    pd.Series(scores, index=n_trees).plot(ax=axs[0])
    if (showStd):
        pd.Series(sc_array + std_array,index=n_trees).plot(style="b--",ax=axs[0])
        pd.Series(sc_array - std_array,index=n_trees).plot(style="b--",ax=axs[0])
    axs[0].set_xlabel("# of trees")
    axs[0].set_ylabel("CV score")
    axs[0].set_title("Forest Size vs. Average Accuracy for Varying Training Set Sizes")
    ###plt.savefig('cv_trees.png')
def plotByTrainingSetSize(n_obsInTrainingSet,scores,sc_array,std_array,showStd,axs):
    '''
    Define: Plots classificatory accuracy for the model which was built from the largest random forest as 
      a function of training set size.
    Design: n_obsInTrainingSet: the number of observations in the training set; 
     scores: the scores as a list;sc_array: scores as a numpy array;
     std_array: standard deviations of scores as numpy array;showStd: whether or not graph should show lines for
     standard deviations; axs:the object containing the figures to show
    Document: Creates and modifies a plot. Conditionally plots the standard deviations. 
    '''
    pd.Series(scores, index=n_obsInTrainingSet).plot(ax=axs[1])
    if (showStd):
        pd.Series(sc_array + std_array,index=n_obsInTrainingSet).plot(style="b--",ax=axs[1])
        pd.Series(sc_array - std_array,index=n_obsInTrainingSet).plot(style="b--",ax=axs[1])
    axs[1].set_xlabel("Size of Training Set")
    axs[1].set_ylabel("CV score")
    axs[1].set_title("Training Set Size vs. Average Accuracy for Largest Forest Size")
    #######plt.savefig('cv_trees2.png')
    
def trainAndTest(n_trees,X_tr,y_tr,scores,scores_std):
    '''
    Define: Builds each tree in each forest, tests them, and reports the score data by appending to the parameters
    Design: n_trees: a list of the sizes of each forest to train; 
     X_tr: a result of the "values" method on the data, contains the class for each observation in the training set;
     y_tr: a result of the "values" method on the data, contains the pixel fields for each observation in the training set;
     scores: a list which will hold the scores for each forest;
     scores_std: a list which will hold the stds for the scores of each forest
    '''
    for n_tree in n_trees:
        print(n_tree)

        recognizer = RandomForestClassifier(n_tree)
        score = cross_val_score(recognizer, X_tr, y_tr)
        scores.append(np.mean(score))
        scores_std.append(np.std(score))
def performTests(data,numberOfObservations,numberOfDivisions,n_trees,showStd1,showStd2,axs):
    '''
    Define: Manages the other functions and the general procedure
    Design: data: the numpy dataframe;
     numberOfObservations: number of observations to recognize in the data;
     numberOfDivisions: desired number of different training set sizes;
     n_trees: list of different forest sizes to build;
     showStd1: whether or not to show the lines for the standard deviations on the first graph;
     showStd2: whether or not to show the lines for the standard deviations on the second graph;
     axs: the object containing the figures to show;
    Document: Calculates the positions of the breaks for the different training set sizes.
     In a loop, gets the performance data for each training set for each of the forests.
     In this loop, also calls for the display and plotting of the performances
     Finally, prints and plots the performance of the largest forest and calls for the printing of slopes.
    '''
    sizeOfGap=int(numberOfObservations/numberOfDivisions)
    
    scoresListByTrainingSet=list()
    stdListByTrainingSet=list()
    n_obsInTrainingSet=list(range(sizeOfGap,numberOfObservations+1,sizeOfGap))
    #f1=plt.figure() #create a new figure
    #get the data and plot by number of trees
    for i in range(sizeOfGap,numberOfObservations+1,sizeOfGap):
        #note that the values in the training set are not selected randomly, but sequentially
        X_tr = data.values[:i, 1:].astype(float)
        y_tr = data.values[:i, 0]
        
        scores = list()
        scores_std = list()
        
        print('Start learning for training set of size ',i,'...')
                
        trainAndTest(n_trees,X_tr,y_tr,scores,scores_std)
        
        sc_array = np.array(scores)
        std_array = np.array(scores_std)
        print('Scores: ', sc_array)
        print('Stds  : ', std_array)
        
        plotThisForestAccuracy(n_trees,scores,sc_array,std_array,showStd1,axs)
        
        scoresListByTrainingSet.append(scores[len(n_trees)-1])  #store the mean and std performance
        stdListByTrainingSet.append(scores_std[len(n_trees)-1])      #for the largest forest size on this
                                                                     #training set size
    #use the data to plot by training set size for the largest forest
    scoresByTrainingSet=np.array(scoresListByTrainingSet)
    stdByTrainingSet=np.array(stdListByTrainingSet)
    print("Scores of Highest Forest Size: ",scoresByTrainingSet)
    print("Standard deviations of Scores of Highest Forest Size: ",stdByTrainingSet)
    plotByTrainingSetSize(n_obsInTrainingSet,scoresListByTrainingSet,scoresByTrainingSet,stdByTrainingSet,showStd2,axs)
    
    printSlopes(range(sizeOfGap,numberOfObservations+1,sizeOfGap),scoresListByTrainingSet,axs)
def printSlopes(xList,yList,axs):
    '''
    Define: Calculates and prints slopes
    Design: If the coordinates are not equal in number, breaks. Otherwise, loops through the coordinates to
      calculate slopes. Prints the slopes.
    Document: xList: list of x coordinates, yList: list of y coordinates,
      axs: the object containing the figures to show
    '''
    slopeList=list()
    if (len(xList)!=len(yList)):
        print ("Lists must be of same length.")
        return
    for i in range(0,len(xList)):
        if(i+1<len(xList)):
            slopeList.append((yList[i+1]-yList[i])/(xList[i+1]-xList[i])*5000)
    slopeNumpyArray=np.array(slopeList)
    print ("Rate of Change in Accuracy per 5000 Added Training Observations: ",slopeNumpyArray)
    
if __name__ == '__main__':
    main()


#
