!pip install seaborn --upgrade #Update Seaborn for Plotting
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import missingno as msno #Visualize null



#Plotting Functions

import matplotlib.pyplot as plt



#Aesthetics

import seaborn as sns

sns.set_style('ticks') #No grid with ticks

print(sns.__version__)
#Data Import

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
fetal=pd.read_csv('/kaggle/input/fetal-health-classification/fetal_health.csv')

fetal.info()

fetal.head()
#Dropping duplicates

fetal_dup=fetal.copy()

fetal_dup.drop_duplicates(inplace=True)

print('Total number of replicates are:', fetal.shape[0] - fetal_dup.shape[0])

fetal=fetal_dup.copy()

fetal
def Plotter(plot, x_label, y_label, x_rot=None, y_rot=None,  fontsize=12, fontweight=None, legend=None, save=False,save_name=None):

    """

    Helper function to make a quick consistent plot with few easy changes for aesthetics.

    Input:

    plot: sns or matplot plotting function

    x_label: x_label as string

    y_label: y_label as string

    x_rot: x-tick rotation, default=None, can be int 0-360

    y_rot: y-tick rotation, default=None, can be int 0-360

    fontsize: size of plot font on axis, defaul=12, can be int/float

    fontweight: Adding character to font, default=None, can be 'bold'

    legend: Choice of including legend, default=None, bool, True:False

    save: Saves image output, default=False, bool

    save_name: Name of output image file as .png. Requires Save to be True.

               default=None, string: 'Insert Name.png'

    Output: A customized plot based on given parameters and an output file

    

    """

    #Ticks

    ax.tick_params(direction='out', length=5, width=3, colors='k',

               grid_color='k', grid_alpha=1,grid_linewidth=2)

    plt.xticks(fontsize=fontsize, fontweight=fontweight, rotation=x_rot)

    plt.yticks(fontsize=fontsize, fontweight=fontweight, rotation=y_rot)



    #Legend

    if legend==None:

        pass

    elif legend==True:

        

        plt.legend()

        ax.legend()

        pass

    else:

        ax.legend().remove()

        

    #Labels

    plt.xlabel(x_label, fontsize=fontsize, fontweight=fontweight, color='k')

    plt.ylabel(y_label, fontsize=fontsize, fontweight=fontweight, color='k')



    #Removing Spines and setting up remianing, preset prior to use.

    ax.spines['top'].set_color(None)

    ax.spines['right'].set_color(None)

    ax.spines['bottom'].set_color('k')

    ax.spines['bottom'].set_linewidth(3)

    ax.spines['left'].set_color('k')

    ax.spines['left'].set_linewidth(3)

    

    if save==True:

        plt.savefig(save_name)
fig, ax=plt.subplots()#Required outside of function. This needs to be activated first when plotting in every code block

plot=sns.countplot(data=fetal, x='fetal_health', hue='fetal_health', palette=['b','r','g'])#count plot

Plotter(plot, 'fetal_health level', 'count', legend=True, save=True, save_name='fetal health count.png')#Plotter function for aesthetics

plot
fig, ax=plt.subplots(figsize=(12,12))#Required outside of function. This needs to be activated first when plotting in every code block

plot=sns.heatmap(fetal.corr(),annot=True, cmap='Blues', linewidths=1)

Plotter(plot, None, None, 90,legend=False, save=True, save_name='Corr.png')
from sklearn.feature_selection import SelectKBest #Feature Selector

from sklearn.feature_selection import f_classif #F-ratio statistic for categorical values
#Feature Selection

X=fetal.drop(['fetal_health'], axis=1)

Y=fetal['fetal_health']

bestfeatures = SelectKBest(score_func=f_classif, k='all')

fit = bestfeatures.fit(X,Y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Feature','Score']  #naming the dataframe columns



#Visualize the feature scores

fig, ax=plt.subplots(figsize=(7,7))

plot=sns.barplot(data=featureScores, x='Score', y='Feature', palette='viridis',linewidth=0.5, saturation=2, orient='h')

Plotter(plot, 'Score', 'Feature', legend=False, save=True, save_name='Feature Importance.png')#Plotter function for aesthetics

plot
#Selection method

selection=featureScores[featureScores['Score']>=200]#Selects features that scored more than 200

selection=list(selection['Feature'])#Generates the features into a list

selection.append('fetal_health')#Adding the Level string to be used to make new data frame

new_fetal=fetal[selection] #New dataframe with selected features

new_fetal.head() #Lets take a look at the first 5
new_name_fetal=new_fetal.rename(columns = {'percentage_of_time_with_abnormal_long_term_variability':'%_ab_long_var', 

                                           'abnormal_short_term_variability': 'short_var'}) #Reduce the size of names for plotting

sns.pairplot(new_name_fetal, hue='fetal_health')
#Splitting

from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(fetal.drop(['fetal_health'], axis=1), fetal['fetal_health'],test_size=0.30, random_state=0, 

                                                 stratify=fetal['fetal_health'])



#Checking the shapes

print("X_train shape :",X_train.shape)

print("Y_train shape :",y_train.shape)

print("X_test shape :",X_test.shape)

print("Y_test shape :",y_test.shape)



#Scaling

from sklearn import preprocessing

scaler=preprocessing.StandardScaler()



X_train_scaled=scaler.fit_transform(X_train) #Scaling and fitting the training set to a model

X_test_scaled=scaler.transform(X_test) #Transformation of testing set based off of trained scaler model
#Packages for metrics and search

"""These packages are required for the functions below

"""

from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV #Paramterizers

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix #Accuracy metrics

import itertools #Used for iterations
def Searcher(estimator, param_grid, search, train_x, train_y, test_x, test_y,label=None,cv=10):

    """

    This is a helper function for tuning hyperparameters using the two search methods.

    Methods must be GridSearchCV or RandomizedSearchCV.

    Inputs:

        estimator: Any Classifier

        param_grid: Range of parameters to search

        search: Grid search or Randomized search

        train_x: input variable of your X_train variables 

        train_y: input variable of your y_train variables

        test_x: input variable of your X_test variables

        test_y: input variable of your y_test variables

        label: str to print estimator, default=None

        cv: cross-validation replicates, int, default=10

    Output:

        Returns the estimator instance, clf

        

    Modified from: https://www.kaggle.com/crawford/hyperparameter-search-comparison-grid-vs-random#To-standardize-or-not-to-standardize

    

    """   

    

    try:

        if search == "grid":

            clf = GridSearchCV(

                estimator=estimator, 

                param_grid=param_grid, 

                scoring=None,

                n_jobs=-1, 

                cv=cv, #Cross-validation at 10 replicates

                verbose=0,

                return_train_score=True

            )

        elif search == "random":           

            clf = RandomizedSearchCV(

                estimator=estimator,

                param_distributions=param_grid,

                n_iter=10,

                n_jobs=-1,

                cv=cv,

                verbose=0,

                random_state=1,

                return_train_score=True

            )

    except:

        print('Search argument has to be "grid" or "random"')

        sys.exit(0) #Exits program if not grid or random

        

    # Fit the model

    print('Start model fitting for', label)

    clf.fit(X=train_x, y=train_y)

    

    #Testing the model

    

    try:

        if search=='grid':

            cfmatrix=confusion_matrix(

            y_true=test_y, y_pred=clf.predict(test_x))

        

            #Defining prints for accuracy metrics of grid

            print("**Grid search results of", label,"**")

            print("The best parameters are:",clf.best_params_)

            print("Best training accuracy:\t", clf.best_score_)

            print('Classification Report:')

            print(classification_report(y_true=test_y, y_pred=clf.predict(test_x))

             )

        elif search == 'random':

            cfmatrix=confusion_matrix(

            y_true=test_y, y_pred=clf.predict(test_x))



            #Defining prints for accuracy metrics of grid

          

            print("**Random search results of", label,"**")

            print("The best parameters are:",clf.best_params_)

            print("Best training accuracy:\t", clf.best_score_)

            print('Classification Report:')

            print(classification_report(y_true=test_y, y_pred=clf.predict(test_x))

             )

    except:

        print('Search argument has to be "grid" or "random"')

        sys.exit(0) #Exits program if not grid or random

        

    return clf, cfmatrix; #Returns a trained classifier with best parameters
def plot_confusion_matrix(cm, label,color=None,title=None):

    """

    Plot for Confusion Matrix:

    Inputs:

        cm: sklearn confusion_matrix function for y_true and y_pred as seen in https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

        title: title of confusion matrix as a 'string', default=None

        label: the unique label that represents classes for prediction can be done as sorted(dataframe['labels'].unique()).

        color: confusion matrix color, default=None, set as a plt.cm.color, based on matplot lib color gradients

    """

    

    classes=sorted(label)

    plt.imshow(cm, interpolation='nearest', cmap=color)

    plt.title(title)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)

    plt.ylabel('Actual')

    plt.xlabel('Predicted')

    thresh = cm.mean()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j]), 

                 horizontalalignment="center",

                 color="white" if cm[i, j] < thresh else "black")
from sklearn.svm import SVC #Support Vector Classifier



#Grid Search SVM Parameters

svm_param = {

    "C": [.01, .1, 1, 5, 10, 100], #Specific parameters to be tested at all combinations

    "gamma": [0, .01, .1, 1],

    "kernel": ["rbf","linear","poly"],

    "degree": [3,4],

    "random_state": [1]}



#Randomized Search SVM Parameters

svm_dist = {

    "C": np.arange(0.01,100, 0.01),   #By using np.arange it will select from randomized values

    "gamma": np.arange(0,1, 0.01),

    "kernel": ["rbf","linear","poly"],

    "degree": [3,4],

    "random_state": [1]}



"""

Following the code above, we can set the parameters for both grid search and randomized search. The grid search will evaluate all specified 

parameters while the randomized search will look at the parameters labeled in random order at the best training accuracy. The np.arange function

allows for a multitude of points to be looked at between the set start and end values of 0.01 to 1. """



#Grid Search SVM

svm_grid, cfmatrix_grid= Searcher(SVC(), svm_param, "grid", X_train_scaled, y_train, X_test_scaled, y_test,label='SVC Grid')



print('_____'*20)#Spacer



#Random Search SVM

svm_rand, cfmatrix_rand= Searcher(SVC(), svm_dist, "random", X_train_scaled, y_train, X_test_scaled, y_test,label='SVC Random')



#Plotting the confusion matrices

plt.subplots(1,2)

plt.subplots_adjust(left=-0.5, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

plot_confusion_matrix(cfmatrix_rand, title='Random Search Confusion Matrix',label=fetal['fetal_health'].unique(), color=plt.cm.cividis) #grid matrix function

plt.subplot(121)

plot_confusion_matrix(cfmatrix_grid, title='Grid Search Confusion Matrix', label=fetal['fetal_health'].unique(), color=plt.cm.cividis) #randomized matrix function
from sklearn.ensemble import RandomForestClassifier as RFC



#Grid Search RFC Parameters

rfc_param = {

    "n_estimators": [10, 50, 75, 100, 150,200], #Specific parameters to be tested at all combinations

    "criterion": ['entropy','gini'],

    "random_state": [1],

    "max_depth":np.arange(1,16,1)}



#Randomized Search RFC Parameters

rfc_dist = {

    "n_estimators": np.arange(10,200, 10),   #By using np.arange it will select from randomized values

    "criterion": ['entropy','gini'],

    "random_state": [1],

    "max_depth":np.arange(1,16,1)}



#Grid Search RFC

rfc_grid, cfmatrix_grid= Searcher(RFC(), rfc_param, "grid", X_train_scaled, y_train, X_test_scaled, y_test,label='RFC Grid')



print('_____'*20)#Spacer



#Random Search RFC

rfc_rand, cfmatrix_rand= Searcher(RFC(), rfc_dist, "random", X_train_scaled, y_train, X_test_scaled, y_test,label='RFC Random')



#Plotting the confusion matrices

plt.subplots(1,2)

plt.subplots_adjust(left=-0.5, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

plot_confusion_matrix(cfmatrix_rand, title='Random Search Confusion Matrix',label=fetal['fetal_health'].unique(), color=plt.cm.cividis) #grid matrix function

plt.subplot(121)

plot_confusion_matrix(cfmatrix_grid, title='Grid Search Confusion Matrix', label=fetal['fetal_health'].unique(), color=plt.cm.cividis) #randomized matrix function
from sklearn.neural_network import MLPClassifier as MLP



#Grid Search MLP Parameters

mlp_param = {

    "hidden_layer_sizes": [(6,),(6,4),(6,4,2)], #Specific parameters to be tested at all combinations

    "activation": ['identity', 'logistic', 'tanh', 'relu'],

    "max_iter":[200,400,600,800,1000],

    "solver":['lbfgs', 'sgd', 'adam'],

    "learning_rate_init":[0.001],

    "learning_rate":['constant','adaptive'],

    "random_state": [1]}



#Randomized Search MLP Parameters

mlp_dist = {

    "hidden_layer_sizes": [(6,),(6,4),(6,4,2)], #Specific parameters to be tested at all combinations

    "activation": ['identity', 'logistic', 'tanh', 'relu'],

    "max_iter":np.arange(100,1000, 100),

    "solver":['lbfgs', 'sgd', 'adam'],

    "learning_rate_init":[0.001],

    "learning_rate":['constant','adaptive'],

    "random_state": [1]}





#Grid Search SVM

rfc_grid, cfmatrix_grid= Searcher(MLP(), mlp_param, "grid", X_train_scaled, y_train, X_test_scaled, y_test,label='MLP Grid')



print('_____'*20)#Spacer



#Random Search SVM

rfc_rand, cfmatrix_rand= Searcher(MLP(), mlp_dist, "random", X_train_scaled, y_train, X_test_scaled, y_test,label='MLP Random')



#Plotting the confusion matrices

plt.subplots(1,2)

plt.subplots_adjust(left=-0.5, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

plot_confusion_matrix(cfmatrix_rand, title='Random Search Confusion Matrix',label=fetal['fetal_health'].unique(), color=plt.cm.cividis) #grid matrix function

plt.subplot(121)

plot_confusion_matrix(cfmatrix_grid, title='Grid Search Confusion Matrix', label=fetal['fetal_health'].unique(), color=plt.cm.cividis) #randomized matrix function