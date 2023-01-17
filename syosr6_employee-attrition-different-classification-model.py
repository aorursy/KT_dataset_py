# Installing pydotplus package that could provide a python interface to graphviz's dot language.

# We will use this package later on to plot the feature importance.

!pip install pydotplus
##Importing the packages

#Data processing packages

import numpy as np 

import pandas as pd 



#Visualization packages

import matplotlib.pyplot as plt

import seaborn as sns 



#Machine Learning packages

from sklearn.svm import SVC,NuSVC

#from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB,MultinomialNB

from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix



import os

print(os.listdir("../input"))
# Importing the Dataset

dataset = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")

dataset.head()
# Shuffle the Dataset.

dataset = dataset.sample(frac=1,random_state=4)
# Check if our Dataset has any nan values

dataset.isnull().values.any()
sns.countplot(x = dataset['Age'], y=None, hue =dataset['Attrition']).set_title( "Attrition rate with the Age")
sns.countplot(x = dataset['Department'], y=None, hue =dataset['Attrition']).set_title( "Attrition distribution in different department")
sns.countplot(x = dataset['DistanceFromHome'], y=None, hue =dataset['Attrition']).set_title( "Attrition distribution depending on distance from home to company")
sns.countplot(x = dataset['YearsAtCompany'], y=None, hue =dataset['Attrition']).set_title( "Age distribution depending on years at the company")
# Build a histogram that shows the attrition in the company 

sns.countplot(x = 'Attrition',data= dataset,palette = "Set2").set_title('Attrition')
ToDrop = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]

dataset = dataset.drop(ToDrop, axis=1)
Y = dataset.loc[:, "Attrition"]
X = dataset.loc[:, dataset.columns !="Attrition"]



Continuous = ["Age", "DailyRate", "DistanceFromHome", "HourlyRate", "MonthlyIncome", "MonthlyRate", 

              "PercentSalaryHike","TotalWorkingYears", "TrainingTimesLastYear", "YearsAtCompany", "YearsInCurrentRole", 

              "YearsSinceLastPromotion", "YearsWithCurrManager"]



ContData = X[Continuous].copy()

ContData.head()
Discreet = ['BusinessTravel', 'Department', 'Education', 'EducationField', 'EnvironmentSatisfaction',

            'Gender', 'JobInvolvement', 'JobLevel', 'JobRole','JobSatisfaction', 'MaritalStatus',

            'NumCompaniesWorked', 'OverTime', 'PerformanceRating', 'RelationshipSatisfaction', 

            'StockOptionLevel', 'WorkLifeBalance']



DisData = X[Discreet].copy()

DisData.head()
plt.figure(figsize =(15,8))

sns.heatmap(ContData.corr(),annot=True,cmap='viridis')

plt.show()
Test = DisData["Department"]

Test1 = pd.get_dummies(Test, drop_first= True)

Test1
X, Y = dataset.loc[:, dataset.columns !="Attrition"], dataset.loc[:, "Attrition"]

X = pd.get_dummies(X, drop_first= True) # X has 44 columns

Y = pd.get_dummies(Y, drop_first= True)
X.head()
# transform the Y collumn to an array

Y = np.ravel(Y)

Y.shape
#importing the necessary librairies

from sklearn.feature_selection import GenericUnivariateSelect

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import SelectPercentile

from sklearn.feature_selection import chi2



# printing the indexes of the selected features

#V = GenericUnivariateSelect(chi2, 'k_best', param=40).fit(X,Y).get_support(indices=True)

#X.iloc[:,V].columns
# Build the function that selects the best k features based on chi2 statistical test

def selectbest_chi2(X,Y,k):

    #Selecting the best k features

    X_new = GenericUnivariateSelect(chi2, 'k_best', param=k).fit_transform(X, Y)

    ## X_new = GenericUnivariateSelect(chi2, 'percentile', param=k).fit_transform(X, Y)

    return X_new
#importing the necessary librairie

from sklearn.feature_selection import GenericUnivariateSelect

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import SelectPercentile

from sklearn.feature_selection import f_classif
# Build the function that selects the best k features based on F statistical test.

def selectbest_fclassif(X,Y,k):

    #Selecting the best k features

    X_new = GenericUnivariateSelect(f_classif, 'k_best', param=k).fit_transform(X, Y)

    return X_new
#importing the necessary librairies

from sklearn.feature_selection import GenericUnivariateSelect

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import SelectPercentile

from sklearn.feature_selection import mutual_info_classif
# Build the function that selects the best k features based on mutual information between the features.

def selectbest_mutualinfoclassif (X,Y,k):

    #Selecting the best k features

    X_new = GenericUnivariateSelect(mutual_info_classif, 'k_best', param=k).fit_transform(X, Y)

    return X_new
def recursive_elimination(model,X,Y):

    from sklearn.model_selection import StratifiedKFold

    from sklearn.feature_selection import RFECV

    from sklearn.datasets import make_classification



    # The "accuracy" scoring is proportional to the number of correct

    # classifications

    rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(2),scoring='accuracy')

    rfecv.fit(X, Y)



    print("Optimal number of features : %d" % rfecv.n_features_)



    # Plot number of features VS. cross-validation scores

    plt.figure()

    plt.xlabel("Number of features selected")

    plt.ylabel("Cross validation score (nb of correct classifications)")

    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

    plt.show()
# Import statements required for Plotly 

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls
# Ranking function specified only to the models which are based on trees (models in DecisionTreeClassifier)

def Ranking(model,title):

    trace = go.Scatter(

        y = model.feature_importances_,

        x = X.columns.values,

        mode='markers',

        marker=dict(

            sizemode = 'diameter',

            sizeref = 1,

            size = 13,

            #size= rf.feature_importances_,

            #color = np.random.randn(500), #set color equal to a variable

            color = model.feature_importances_,

            colorscale='Portland',

            showscale=True

        ),

        text = X.columns.values

    )

    data = [trace]



    layout= go.Layout(

        autosize= True,

        hovermode= 'closest',

         xaxis= dict(

             ticklen= 5,

             showgrid=False,

            zeroline=False,

            showline=False

         ),

        yaxis=dict(

            title= 'Feature Importance',

            showgrid=False,

            zeroline=False,

            ticklen= 5,

            gridwidth= 2

        ),

        showlegend= False

    )

    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig,filename='scatter2010')
def scaling(X_new):

    #feature scaling

    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()

    X_new = sc.fit_transform(X_new)

    return X_new
def splitting(X_new,Y): 

    # Splitting the dataset into the Training set and Test set

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size = 0.25, random_state = 0)

    return X_train, X_test, y_train, y_test
#Function to Train and Test Machine Learning Model

def train_test_ml_model(X_train,y_train,X_test,model,Model):

    import time

    start = time.time()



    model.fit(X_train,y_train) #Train the Model

    y_pred = model.predict(X_test) #Use the Model for prediction

    end = time.time()



    # Test the Model

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test,y_pred)

    

    # Test the model accuracy

    from sklearn.model_selection import cross_val_score

    #accuracy = cross_val_score(estimator = model, X = X_new, y = y_train, cv = 4) #It gives you the accuracy of each training

    accuracy = round(100*np.trace(cm)/np.sum(cm),1)

    

    # Test the RocAUC of the model

    #from sklearn.metrics import roc_auc_score

   # roc_value = roc_auc_score(y_test, y_pred)

    

    #Precision,Recall,F1_score

    Precision = cm[0][0] / (cm[0][0] + cm[0][1])

    Recall = cm[0][0] / (cm[0][0] + cm[1][0])

    F1 = 2*(Precision*Recall)/(Precision+Recall)



    #Mean square error

    #from sklearn.metrics import mean_squared_error

    #MSE = mean_squared_error(y_test, y_pred)

    

    #Plot/Display the results

    cm_plot(cm,Model)

    

    print('The Wrong predicted points are:')

    Wrong_Prediction(X_test,y_test,y_pred)



    Results = pd.DataFrame(columns=['accuracy','Precision = TP/(TP+FP)','Recall = TP/(TP+FN)',

                                    'F1_score','Time of execution'], index=[Model])

    results = {'accuracy':accuracy,'Precision = TP/(TP+FP)':Precision,

         'Recall = TP/(TP+FN)':Recall,'F1_score':F1,'Time of execution':end - start} 

    Results.loc[Model] = pd.Series(results)

    return (Results)
#Function to plot Confusion Matrix

def cm_plot(cm,Model):

    plt.clf()

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.viridis)

    classNames = ['Negative','Positive']

    plt.title('Comparison of Prediction Result for '+ Model)

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    tick_marks = np.arange(len(classNames))

    plt.xticks(tick_marks, classNames, rotation=45)

    plt.yticks(tick_marks, classNames)

    s = [['TN','FP'], ['FN', 'TP']]

    for i in range(2):

        for j in range(2):

            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))

    plt.show()
def Wrong_Prediction (X_test,y_test,y_pred):

    WIndex = []

    for i in range(len(y_test)):

        if y_test[i] != y_pred[i] :

            WIndex.append(i)

    print (WIndex)
X_train, X_test, y_train, y_test = splitting(X,Y)
from sklearn.linear_model import LogisticRegression  #Import package related to Model

Model = "LogisticRegression"

model= LogisticRegression(C=0.35000000000000003, solver="newton-cg", max_iter=200) #Create the Model
X_new = selectbest_chi2(X,Y,40)

X_new = scaling(X_new)

X_train, X_test, y_train, y_test = splitting(X_new,Y)

Results = pd.DataFrame()

Results = train_test_ml_model(X_train,y_train,X_test,model,Model)
recursive_elimination(model,X_train,y_train)
X_train, X_test, y_train, y_test = splitting(X,Y)
X_new = selectbest_chi2(X,Y,37)

X_new = scaling(X_new)

X_train, X_test, y_train, y_test = splitting(X_new,Y)

results_logistic = pd.DataFrame()

results_logistic = train_test_ml_model(X_train,y_train,X_test,model,Model)



Results = Results.append(results_logistic.iloc[0,:])

Results
from sklearn.ensemble import RandomForestClassifier  #Import package related to Model

Model = "RandomForestClassifier"

model= RandomForestClassifier(n_estimators=100, bootstrap = True, max_features = 'sqrt') #Create the Model
X_train, X_test, y_train, y_test = splitting(X,Y)

X_new = selectbest_chi2(X,Y,40)

X_new = scaling(X_new)

X_train, X_test, y_train, y_test = splitting(X_new,Y)

results = pd.DataFrame()

results = train_test_ml_model(X_train,y_train,X_test,model,Model)

recursive_elimination(model,X_train,y_train)
X_train, X_test, y_train, y_test = splitting(X,Y)

X_new = selectbest_chi2(X,Y,25)

X_new = scaling(X_new)

X_train, X_test, y_train, y_test = splitting(X_new,Y)

results_forest = pd.DataFrame()

results_forest = train_test_ml_model(X_train,y_train,X_test,model,Model)



results = results.append(results_forest.iloc[0,:])

results
title= 'Random Forest Feature Importance'

Ranking(model,title)
from sklearn import svm  #Import package related to Model

Model = "LinearSVM"

model = svm.SVC(kernel='linear', C=1.0)  # Create the Model
X_train, X_test, y_train, y_test = splitting(X,Y)

X_new = selectbest_chi2(X,Y,40)

X_new = scaling(X_new)

X_train, X_test, y_train, y_test = splitting(X_new,Y)



results = pd.DataFrame()

results = train_test_ml_model(X_train,y_train,X_test,model,Model)
recursive_elimination(model,X_train,y_train)
X_train, X_test, y_train, y_test = splitting(X,Y)

X_new = selectbest_chi2(X,Y,27)

X_new = scaling(X_new)

X_train, X_test, y_train, y_test = splitting(X_new,Y)

results_linearSVM = pd.DataFrame()

results_linearSVM = train_test_ml_model(X_train,y_train,X_test,model,Model)



results = results.append(results_linearSVM.iloc[0,:])

results
from xgboost import XGBClassifier  #Import package related to Model

Model = "XGBClassifier"

model=XGBClassifier() #Create the Model
X_train, X_test, y_train, y_test = splitting(X,Y)

X_new = selectbest_chi2(X,Y,40)

X_new = scaling(X_new)

X_train, X_test, y_train, y_test = splitting(X_new,Y)



results = pd.DataFrame()

results = train_test_ml_model(X_train,y_train,X_test,model,Model)
recursive_elimination(model,X_train,y_train)
X_train, X_test, y_train, y_test = splitting(X,Y)

X_new = selectbest_chi2(X,Y,30)

X_new = scaling(X_new)

X_train, X_test, y_train, y_test = splitting(X_new,Y)

results_XGBC = pd.DataFrame()

results_XGBC = train_test_ml_model(X_train,y_train,X_test,model,Model)



results = results.append(results_XGBC.iloc[0,:])

results
title= 'XGBoost Feature Importance'

Ranking(model,title)
import matplotlib.gridspec as gridspec

import itertools

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from mlxtend.classifier import EnsembleVoteClassifier

from mlxtend.data import iris_data

from mlxtend.plotting import plot_decision_regions



# Initializing Classifiers

clf1 = LogisticRegression(C=0.35000000000000003, solver="newton-cg", max_iter=200) #Create the First Model

clf2 = RandomForestClassifier(n_estimators=100, bootstrap = True, max_features = 'sqrt') #Create the Second Model

clf3 = svm.SVC(kernel='linear', C=1.0, probability=True)  # Create the Third Model



eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=[3, 1.5, 2], voting='soft')





model = eclf

Model = "Combination1"
X_train, X_test, y_train, y_test = splitting(X,Y)

X_new = selectbest_chi2(X,Y,40)

X_new = scaling(X_new)

X_train, X_test, y_train, y_test = splitting(X_new,Y)



results_Combination1 = pd.DataFrame()

results_Combination1 = train_test_ml_model(X_train,y_train,X_test,model,Model)

results_Combination1
import matplotlib.gridspec as gridspec

import itertools

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from xgboost import XGBClassifier  #Import package related to Model

from mlxtend.classifier import EnsembleVoteClassifier

from mlxtend.data import iris_data

from mlxtend.plotting import plot_decision_regions



# Initializing Classifiers

clf1 = LogisticRegression(C=0.35000000000000003, solver="newton-cg", max_iter=200) #Create the First Model

clf2 = XGBClassifier()  #Create the Second Model

clf3 = svm.SVC(kernel='linear', C=1.0, probability=True)  # Create the Third Model



eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=[3, 1.5, 2], voting='soft')





model = eclf

Model = "LR+XGB+SVM"
X_train, X_test, y_train, y_test = splitting(X,Y)

X_new = selectbest_chi2(X,Y,40)

X_new = scaling(X_new)

X_train, X_test, y_train, y_test = splitting(X_new,Y)



results_Combination2 = pd.DataFrame()

results_Combination2 = train_test_ml_model(X_train,y_train,X_test,model,Model)

results_Combination2
Results = pd.DataFrame

Results = results_logistic

T = [results_linearSVM,results_forest,results_XGBC,results_Combination2]

for i in range(len (T)):

    Results = Results.append(T[i])



Results