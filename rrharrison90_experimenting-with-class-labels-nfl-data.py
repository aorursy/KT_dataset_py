#Import Dependencies

%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib

import numpy as np

from sklearn.model_selection import train_test_split

from numpy import genfromtxt

import math

import collections

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import classification_report,confusion_matrix



File_Loc ="../input/"



#Import NFL Play by Play data set

D = pd.read_csv(File_Loc + 'NFL2015_Working2.csv')



#Import NFL Player Roster data set (for encoding purposes)

P = pd.read_csv(File_Loc + 'Players.csv')



#Import NFL Team List data set (for encoding purposes)

T = pd.read_csv(File_Loc + 'Teams.csv')
#Visualize the distribution target value "Yards.Gained"

plt.hist(D["Yards.Gained"],bins=200)

plt.xlabel("Yards Gained")

plt.ylabel("freq")

plt.title("Yards Gained Histogram")

plt.show()



#View preview of data prefore any transformations are applied

D.head(n=50)
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler 

from sklearn.preprocessing import LabelEncoder

        

#CUSTOM DATA TRANSFORMATION CLASS: Encodes the values of specified columns of a dataframe        

class BulkEncoder(BaseEstimator, TransformerMixin):

    def __init__(self,Cols,Base,Bundle):

        self.Cols = Cols

        self.Base = Base

        self.Bundle = Bundle

    def fit(self,x,y=None):

        return self

    def transform(self,x):

        le =LabelEncoder()

        if self.Bundle == "yes":

            le.fit(self.Base)

            for c in self.Cols:

                x[c] = le.transform(x[c])

        elif self.Bundle == "no":

            for c in self.Cols:

                le.fit(x[c])

                x[c] = le.transform(x[c])

        return x

    



#fill in missing strings with and x and missing int with 0

D = D.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna("x"))





#Transformation pipeline

PreProcessing_Pipline = Pipeline(

  [



    # - encode all columns containing the name of players as an int

   ("team_encoder",BulkEncoder(["posteam","DefensiveTeam","SideofField"]

                               ,T["Teams"],"yes")),

   ("misc_encoder",BulkEncoder(["RunLocation","RunGap","PlayType",

                                "PassLength","PassLocation"],"none","no")),  

  ]

)



#Transformed dataset

D = PreProcessing_Pipline.fit_transform(D)
#Splitting data set into test and train for each trial

x  = D.iloc[:,0:19]

Predictors = list(x.columns.values)

print(Predictors)

y1 = D['Yards.Gained.Cat.20Plus']

y2 = D['Yards.Gained.Cat.10Plus']

y3 = D['Yards.Gained.Cat.4Class']

y4 = D["Yards.Gained.Cat.1Plus"]



scaler = StandardScaler()

scaler.fit(x)

x = scaler.transform(x)



x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y1

                                                        ,test_size=0.20

                                                        ,shuffle=True

                                                        ,random_state=2)

x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y2

                                                        ,test_size=0.20

                                                        ,shuffle=True

                                                        ,random_state=2)

x_train3, x_test3, y_train3, y_test3 = train_test_split(x, y3

                                                        ,test_size=0.20

                                                        ,shuffle=True

                                                        ,random_state=2)

x_train4, x_test4, y_train4, y_test4 = train_test_split(x, y4

                                                        ,test_size=0.20

                                                        ,shuffle=True

                                                        ,random_state=2)
#Dictionaries to house the scores from out trials with different models

import json

Trials = ['1Plus','10Plus','20Plus','4Class']



Training_And_Testing_Sets = {'Train':

                                     {

                                      'Data':{

                                               '20Plus':x_train1

                                               ,'10Plus':x_train2

                                               ,'4Class':x_train3

                                               ,'1Plus':x_train4

                                              }

                                      ,'Target':{

                                               '20Plus':y_train1

                                               ,'10Plus':y_train2

                                               ,'4Class':y_train3

                                               ,'1Plus':y_train4

                                               }

                                     }

                             ,'Test':

                                     {

                                      'Data':{

                                               '20Plus':x_test1

                                               ,'10Plus':x_test2

                                               ,'4Class':x_test3

                                               ,'1Plus':x_test4

                                              }

                                      ,'Target':{

                                               '20Plus':y_test1

                                               ,'10Plus':y_test2

                                               ,'4Class':y_test3

                                               ,'1Plus':y_test4

                                               }

                                     }

                            }



Model_Scores = {

            'multi-label':{'Acc':{}

                          ,'Pr':{}

                          ,'RC':{}

                           ,'F1':{}

            }

           ,'20Plus':{'Acc':{}

                     ,'Pr':{}

                     ,'RC':{}

                     ,'F1':{}}

           ,'10Plus':{'Acc':{}

                     ,'Pr':{}

                     ,'RC':{}

                     ,'F1':{}}

           ,'4Class':{'Acc':{}

                     ,'Pr':{}

                     ,'RC':{}

                     ,'F1':{}}

         ,'1Plus':{'Acc':{}

                     ,'Pr':{}

                     ,'RC':{}

                     ,'F1':{}}

}
from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.svm import LinearSVC,SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score,precision_recall_fscore_support,classification_report

from sklearn.multiclass import OneVsRestClassifier



KNN = KNeighborsClassifier(weights='distance',n_neighbors=2000)

MLP = MLPClassifier(activation='relu'

                       ,hidden_layer_sizes=(10,10,10)

                       ,max_iter=2000

                       ,verbose=False

                       ,batch_size=350

                       ,random_state=1

                       ,warm_start= False

                       )

RF = RandomForestClassifier(n_estimators=1000,max_depth=25

                            ,criterion='gini')

SVM = OneVsRestClassifier(SVC())



TrialModel = {

                'MLP': MLP

               ,'Random Forest': RF 

               ,'SVM': SVM

               

}

    

FeatureImportance = []

for Trial in Trials:

    train_x = Training_And_Testing_Sets['Train']['Data'][Trial]

    train_y = Training_And_Testing_Sets['Train']['Target'][Trial] 

    test_y  = Training_And_Testing_Sets['Test']['Target'][Trial]

    test_x  = Training_And_Testing_Sets['Test']['Data'][Trial]

    

    acc_list = []

    pr_list = []

    rc_list = []

    F1_list = []

    for TR in TrialModel:

        print(TR,": ", Trial)

        Model = TrialModel[TR]

        Model.fit(train_x,train_y)

        Prediction = Model.predict(test_x)

        Prediction1 = Model.predict(train_x)

        if hasattr(Model,'feature_importances_'):

            FeatureImportance.append(Model.feature_importances_)

        print("Accuracy test: ", accuracy_score(test_y,Prediction)*100,'%')

        print("Accuracy Train: ", accuracy_score(train_y,Prediction1)*100,'%')

        print("Precision ",precision_recall_fscore_support(test_y,Prediction,average=None)[0])

        print("Recall ",precision_recall_fscore_support(test_y,Prediction,average=None)[1])

        print("f1 ",precision_recall_fscore_support(test_y,Prediction,average=None)[2])

        acc_list.append(accuracy_score(test_y,Prediction))

        pr,rc,f1,sup = precision_recall_fscore_support(test_y,Prediction,average=None)

        print(classification_report(test_y,Prediction))

        pr_list.append(pr.tolist())

        rc_list.append(rc.tolist())

        F1_list.append(f1.tolist())

        print(' ')

        print(' ')

    Model_Scores[Trial]['Acc']=acc_list

    Model_Scores[Trial]['Pr']=pr_list

    Model_Scores[Trial]['RC']=rc_list

    Model_Scores[Trial]['F1']=F1_list

def autolabel(rects):

    """

    Attach a text label above each bar displaying its height

    """

    for rect in rects:

        height = float(rect.get_height())

        ax.text(rect.get_x() + rect.get_width()/4., 1.05*height,

                '%d' % float(height),

                ha='left', va='bottom')


N = 19





ind = np.arange(N)  # the x locations for the groups

width = 0.20       # the width of the bars



for i in range(len(FeatureImportance)):

    fig, ax = plt.subplots()

    rects1 = ax.bar(ind, FeatureImportance[i].tolist(), width, color='r')

    ax.set_xticks(ind)

    ax.set_title(Trials[i])

    ax.set_xticklabels(Predictors,rotation=45)

    fig.set_figwidth(15)  

    plt.show()
N = 3

Trials = ['1Plus','10Plus','20Plus','4Class']





P1 = [i*100 for i in Model_Scores['1Plus']['Acc']]

P1V = [.05*i for i in P1]

P10 = [i*100 for i in Model_Scores['10Plus']['Acc']]

P10V = [.05*i for i in P10]

P20 = [i*100 for i in Model_Scores['20Plus']['Acc']]

P20V = [.05*i for i in P20]

C3 = [i*100 for i in Model_Scores['4Class']['Acc']]

C3V = [.05*i for i in C3]



ind = np.arange(N)  # the x locations for the groups

width = 0.17       # the width of the bars



fig, ax = plt.subplots()

rects2 = ax.bar(ind + width, P1, width, color='b',yerr=P1V)

rects3 = ax.bar(ind + width + width, P10, width, color='y',yerr=P10V)

rects4 = ax.bar(ind + width + width + width, P20, width, color='g',yerr=P20V)

rects5 = ax.bar(ind + width + width + width + width, C3, width, color='k',yerr=C3V)

# add some text for labels, title and axes ticks

ax.set_ylabel('Scores')

ax.set_title('Accuracy Scores')

ax.set_xticks(ind + width)

ax.set_xticklabels(('MLP', 'RF', 'SVM'))





autolabel(rects2)

autolabel(rects3)

autolabel(rects4)

autolabel(rects5)

fig.set_figheight(5)

fig.set_figwidth(10)  

fig.legend(( rects2[0],rects3[0],rects4[0],rects5[0]), Trials, ncol=5,loc="upper center")



plt.show()
N = 3



ind = np.arange(N)  # the x locations for the groups

width = 0.20       # the width of the bars

w=width

metrics = ['Pr','RC','F1']

Barcolors = ['r','b','t']



for metric in metrics:

        fig, axes = plt.subplots(nrows=1, ncols=4)

        for ax,Trial in zip(axes.flat[0:],Trials):               

                #for i in range(0,2):

                #w=width*i 

                scores = [x[0]*100 for x in Model_Scores[Trial][metric]]

                scores2 =  [x[1]*100 for x in Model_Scores[Trial][metric]]              

                rects1 = ax.bar(ind + w, scores, width, color='r')

                rects2 = ax.bar(ind + w + w, scores2, width, color='b')

                if Trial == "4Class":

                    scores3 =  [x[2]*100 for x in Model_Scores[Trial][metric]]

                    rects3 = ax.bar(ind + w + w + w, scores3, width, color='y')

                    scores4=  [x[3]*100 for x in Model_Scores[Trial][metric]]

                    rects4 = ax.bar(ind + w + w + w + w, scores4, width, color='g')

                ax.set_ylim(0,100)

                ax.set_ylabel('Scores')

                ax.set_title(metric+': '+Trial)

                ax.set_xticks(ind + width)

                ax.set_xticklabels(('MLP', 'RF', 'SVM'))

                autolabel(rects1)

                autolabel(rects2)

                if Trial == '4Class':

                    autolabel(rects3)

                    autolabel(rects4)

        fig.set_figheight(3)

        fig.set_figwidth(16)

        fig.legend((rects1[0], rects2[0],rects3[0],rects4[0]), ['class 0','class 1','class 2','class3']

                   , ncol=4,loc="upper center")

        fig.tight_layout()   

        plt.show()