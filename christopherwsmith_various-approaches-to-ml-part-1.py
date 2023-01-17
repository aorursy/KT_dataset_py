!pip install seaborn --upgrade #Update Seaborn for plotting
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Plotting Functions

import matplotlib.pyplot as plt



#Aesthetics

import seaborn as sns

sns.set_style('ticks') #No grid with ticks

print(sns.__version__)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
heart=pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

heart.info()

heart.head()
heart_list=['sex','cp', 'fbs', 'restecg','exang', 'slope','ca','thal']



for label in heart_list:

    if heart[label].max()==1:

        print(label, 'is only categorical!')

    else:

        print(label, 'is ordinal with values of', sorted(heart[label].unique()))
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

plot=sns.countplot(data=heart, x='target', hue='sex', palette=sns.color_palette("mako"))

Plotter(plot, 'target', 'count', legend=True, save=True, save_name='Heart Disease Count.png')
fig, ax=plt.subplots()

plot=sns.histplot(data=heart, x='age', hue='target', element='step',stat='density',kde=True, palette=sns.color_palette("mako",2))

Plotter(plot, 'age', 'density', legend=None, save=True, save_name='Age_Hist.png')#For histplots set legend to None. I do not know why the function does not work properly for histplots
fig, ax=plt.subplots()

plot=sns.scatterplot(data=heart, x='age', y='trestbps',hue='target', palette=sns.color_palette("mako",2))

Plotter(plot, 'age', 'trestbps', legend=True, save=True, save_name='Age_Trest.png')
fig, ax=plt.subplots()

plot=sns.boxplot(data=heart, x='thal', y='thalach',hue='target', palette=sns.color_palette("mako",2))

Plotter(plot, 'thal', 'thalach', legend=True, save=True, save_name='target_thal.png')
sns.pairplot(data=heart, hue='target',palette=sns.color_palette("mako",2))
fig, ax=plt.subplots(figsize=(9,9))#Required outside of function. This needs to be activated first when plotting in every code block

plot=sns.heatmap(heart.corr(),annot=True, cmap='Blues', linewidths=1)

Plotter(plot, None, None, 90,legend=False, save=True, save_name='Corr.png')
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(heart.drop(['target'], axis=1), heart['target'],test_size=0.30, random_state=0)



#Scaling the Data

from sklearn import preprocessing



scaler=preprocessing.StandardScaler()

X_train_ss=scaler.fit_transform(X_train)

X_test_ss=scaler.transform(X_test)
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from sklearn.ensemble import RandomForestClassifier as RFC



feature_names=list(X_train.columns)

feature_names

rfc=RFC(n_estimators=250, criterion='entropy', random_state=1)





#Setting SFS

sfs=SFS(estimator=rfc,

       k_features='best',

       forward=False,#Backwards elimination

       floating=True,#floating true, takes whole feature set

       scoring='accuracy',

       cv=5,

       verbose=0)#Shows progress, I set this to 0 to save space



#Fitting the model

sfs.fit(X_train_ss, y_train, custom_feature_names=feature_names)



#Results

print('Best Features are', sfs.k_feature_names_)

print('Best Features by index are', sfs.k_feature_idx_)

print('Best Score', sfs.k_score_)



#Transforming data

X_train_fet=sfs.transform(X_train_ss)#Set new variables

X_test_fet=sfs.transform(X_test_ss)

print('New training dimensions are',X_train_fet.shape, 'While testing dimensions are', X_test_fet.shape)



output=pd.DataFrame.from_dict(sfs.get_metric_dict()).T

output.sort_values('avg_score', ascending=False)#Print table with metrics
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV #Paramterizers

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix #Accuracy metrics

import itertools #Used for iterations
def Searcher(estimator, param_grid, search, train_x, train_y, test_x, test_y,label=None):

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

    Output:

        Returns the estimator instance, clf

    """   

    

    try:

        if search == "grid":

            clf = GridSearchCV(

                estimator=estimator, 

                param_grid=param_grid, 

                scoring=None,

                n_jobs=-1, 

                cv=10, #Cross-validation at 10 replicates

                verbose=0,

                return_train_score=True

            )

        elif search == "random":           

            clf = RandomizedSearchCV(

                estimator=estimator,

                param_distributions=param_grid,

                n_iter=10,

                n_jobs=-1,

                cv=10,

                verbose=0,

                random_state=1,

                return_train_score=True

            )

    except:

        print('Search argument has to be "grid" or "random"')

        sys.exit(0) #Exits program if not grid or random

        

    # Fit the model

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
from sklearn.linear_model import LogisticRegression as LR



#Setting searcher parameters

#Grid Search

log_param={"C":[.01, .1, 1, 5, 10, 100],#Specific parameters to be tested at all combinations

          "max_iter":[100,250,500,750,1000],

          "random_state":[1]}



log_grid, cfmatrix_grid= Searcher(LR(), log_param, "grid", X_train_fet, y_train, X_test_fet, y_test,label='LogReg')



print('_____'*20)



#Randomized Search 

log_dist = {

    "C": np.arange(0.01,100, 0.01),   #By using np.arange it will select from randomized values

    "max_iter": np.arange(100,1000, 5),

    "random_state": [1]}



log_rand, cfmatrix_rand= Searcher(LR(), log_dist, "random", X_train_fet, y_train, X_test_fet, y_test, label='LogReg')



#Plotting the confusion matrices

plt.subplots(1,2)

plt.subplots_adjust(left=-0.5, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

plot_confusion_matrix(cfmatrix_rand, title='Random Search Confusion Matrix',label=heart['target'].unique(), color=plt.cm.cividis) #grid matrix function

plt.subplot(121)

plot_confusion_matrix(cfmatrix_grid, title='Grid Search Confusion Matrix', label=heart['target'].unique(), color=plt.cm.cividis) #randomized matrix function



plt.savefig('LogReg_confusion.png')
from sklearn.tree import DecisionTreeClassifier as DTC



depth=np.arange(1,20, 1)

#Setting searcher parameters

#Grid Search

dtc_param={"criterion":['entropy',"gini"],

           'max_depth':[None, depth],

    "min_samples_split":np.arange(2,20, 1),

          "min_samples_leaf":np.arange(2,20, 1),

          "random_state":[1]}



dtc_grid, cfmatrix_grid= Searcher(DTC(), dtc_param, "grid", X_train_fet, y_train, X_test_fet, y_test,label='Tree')



print('_____'*20)



#Randomized Search 

dtc_dist = {

    "criterion":['entropy',"gini"],

           'max_depth':[None, depth],

    "min_samples_split":np.arange(2,20, 1),

          "min_samples_leaf":np.arange(2,20, 1),

          "random_state":[1]}



dtc_rand, cfmatrix_rand= Searcher(DTC(), dtc_dist, "random", X_train_fet, y_train, X_test_fet, y_test, label='Tree')



#Plotting the confusion matrices

plt.subplots(1,2)

plt.subplots_adjust(left=-0.5, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

plot_confusion_matrix(cfmatrix_rand, title='Random Search Confusion Matrix',label=heart['target'].unique(), color=plt.cm.cividis) #grid matrix function

plt.subplot(121)

plot_confusion_matrix(cfmatrix_grid, title='Grid Search Confusion Matrix', label=heart['target'].unique(), color=plt.cm.cividis) #randomized matrix function



plt.savefig('DTC_confusion.png')
from sklearn.neighbors import KNeighborsClassifier as KNN



#Setting searcher parameters

#Grid Search

knn_param={"n_neighbors":[1,2,3,4,5, 10, 15, 20],

           'weights':['uniform','distance'],

    "algorithm":['ball_tree', 'kd_tree', 'brute'],

          "p":[1,2],

          }



knn_grid, cfmatrix_grid= Searcher(KNN(), knn_param, "grid", X_train_fet, y_train, X_test_fet, y_test,label='KNN')



print('_____'*20)



#Randomized Search 

knn_dist = {"n_neighbors":np.arange(1,40,1),

           'weights':['uniform','distance'],

    "algorithm":['ball_tree', 'kd_tree', 'brute'],

          "p":[1,2],

          }



knn_rand, cfmatrix_rand= Searcher(KNN(), knn_dist, "random", X_train_fet, y_train, X_test_fet, y_test, label='KNN')



#Plotting the confusion matrices

plt.subplots(1,2)

plt.subplots_adjust(left=-0.5, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

plot_confusion_matrix(cfmatrix_rand, title='Random Search Confusion Matrix',label=heart['target'].unique(), color=plt.cm.cividis) #grid matrix function

plt.subplot(121)

plot_confusion_matrix(cfmatrix_grid, title='Grid Search Confusion Matrix', label=heart['target'].unique(), color=plt.cm.cividis) #randomized matrix function



plt.savefig('KNN_confusion.png')
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



#Grid Search 

svm_grid, cfmatrix_grid= Searcher(SVC(), svm_param, "grid", X_train_fet, y_train, X_test_fet, y_test,label='SVC')



print('_____'*20)#Spacer



#Random Search

svm_rand, cfmatrix_rand= Searcher(SVC(), svm_dist, "random", X_train_fet, y_train, X_test_fet, y_test,label='SVC')



#Plotting the confusion matrices

plt.subplots(1,2)

plt.subplots_adjust(left=-0.5, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

plot_confusion_matrix(cfmatrix_rand, title='Random Search Confusion Matrix',label=heart['target'].unique(), color=plt.cm.cividis) #grid matrix function

plt.subplot(121)

plot_confusion_matrix(cfmatrix_grid, title='Grid Search Confusion Matrix', label=heart['target'].unique(), color=plt.cm.cividis) #randomized matrix function

plt.savefig('SVM_confusion.png')
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

rfc_grid, cfmatrix_grid= Searcher(RFC(), rfc_param, "grid", X_train_fet, y_train, X_test_fet, y_test,label='RFC')



print('_____'*20)#Spacer



#Random Search RFC

rfc_rand, cfmatrix_rand= Searcher(RFC(), rfc_dist, "random", X_train_fet, y_train, X_test_fet, y_test,label='RFC')



#Plotting the confusion matrices

plt.subplots(1,2)

plt.subplots_adjust(left=-0.5, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

plot_confusion_matrix(cfmatrix_rand, title='Random Search Confusion Matrix',label=heart['target'].unique(), color=plt.cm.cividis) #grid matrix function

plt.subplot(121)

plot_confusion_matrix(cfmatrix_grid, title='Grid Search Confusion Matrix', label=heart['target'].unique(), color=plt.cm.cividis) #randomized matrix function

plt.savefig('RFC_confusion.png')
from sklearn.neural_network import MLPClassifier as MLP



#Grid Search MLP Parameters

mlp_param = {

    "hidden_layer_sizes": [(9,),(9,6),(9,6,3),(9,6,3,1)], #Specific parameters to be tested at all combinations

    "activation": ['identity', 'logistic', 'tanh', 'relu'],

    "max_iter":[200,400,600,800,1000],

    "solver":['lbfgs', 'sgd', 'adam'],

    "learning_rate_init":[0.01],

    "learning_rate":['constant','adaptive'],

    "random_state": [1]}



#Randomized Search MLP Parameters

sini=np.arange(6,9,1)

on_si=np.arange(1,6,1)

on_th=np.arange(1,3,1)

mlp_dist = {

    "hidden_layer_sizes": [(9,),(9,6),(9,6,3),(9,6,3,1)], #Specific parameters to be tested at all combinations

    "activation": ['identity', 'logistic', 'tanh', 'relu'],

    "max_iter":np.arange(100,1000, 100),

    "solver":['lbfgs', 'sgd', 'adam'],

    "learning_rate_init":np.arange(0.001,0.01,0.001),

    "learning_rate":['constant','adaptive'],

    "random_state": [1]}



#Grid Search SVM

mlp_grid, cfmatrix_grid= Searcher(MLP(), mlp_param, "grid", X_train_fet, y_train, X_test_fet, y_test,label='MLP')



print('_____'*20)#Spacer



#Random Search SVM

mlp_rand, cfmatrix_rand= Searcher(MLP(), mlp_dist, "random", X_train_fet, y_train, X_test_fet, y_test,label='MLP')



#Plotting the confusion matrices

plt.subplots(1,2)

plt.subplots_adjust(left=-0.5, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

plot_confusion_matrix(cfmatrix_rand, title='Random Search Confusion Matrix',label=heart['target'].unique(), color=plt.cm.cividis) #grid matrix function

plt.subplot(121)

plot_confusion_matrix(cfmatrix_grid, title='Grid Search Confusion Matrix', label=heart['target'].unique(), color=plt.cm.cividis) #randomized matrix function

plt.savefig('MLP_confusion.png')