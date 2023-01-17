import os

import itertools

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, KFold

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.metrics import log_loss, confusion_matrix

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score



# ignore deprecation warnings in sklearn



import warnings

warnings.filterwarnings('ignore')



# Default configurations



plt.style.use('seaborn-pastel')

pd.set_option('display.max_columns', 500)

colPlots = ['skyblue','lightgreen']



print('Libraries imported and default configuration set!')
def plotBar(df, fig, col, fontS = 9):



    """ Function that draw a tuned bar plot 

    

    Arguments

    ---------

    df:    Pandas DataFrame

    fig:   Figure to tune

    col:   Column of the dataframe to plot

    fontS: Font size of the lables that will be printed above the bars

    

    """

    

    nGroups = tmpDF.shape[0]

    eArr = np.array(tmpDF['e'])

    pArr = np.array(tmpDF['p'])

    index = np.arange(nGroups)

    bar_width = 0.425

    opacity = 0.65

    

    plt.tick_params(top='off', bottom='on', left='off', right='off', labelleft='off', labelbottom='on')



    plt.gca().spines['top'].set_visible(False)

    plt.gca().spines['left'].set_visible(False)

    plt.gca().spines['right'].set_visible(False)



    bar1 = plt.bar(index, eArr, bar_width, alpha=opacity, color = colPlots[0], label='e')



    for bar in bar1:

        height = bar.get_height()

        if height > 0 :

            plt.gca().text(bar.get_x() + bar.get_width()/2, (bar.get_height()+42), str(int(height)),

                           ha='center', color='black', fontsize=fontS)



    bar2 = plt.bar(index + bar_width, pArr, bar_width, alpha = opacity,color = colPlots[1], label = 'p')



    for bar in bar2:

        height = bar.get_height()

        if height > 0 :

            plt.gca().text(bar.get_x() + bar.get_width()/2, (bar.get_height()+40), str(int(height)),

                           ha='center', color='black', fontsize=fontS)

    b,t = plt.ylim()

    plt.ylim(top=(t*1.10))



    plt.title('Class count per ' + col, fontstretch = 'semi-condensed', fontsize = 16)

    plt.xticks(index + bar_width/2, list(tmpDF.index))

    plt.tight_layout()

   

    return fig

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, figSize = [8,6], x_rot = 45, normalize=True):

    

    """ given a sklearn confusion matrix (cm), make a nice plot



    Arguments

    ---------

    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2], the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm

    figSize:     Size of the confusion matrix plot (*)

    x_rot:        Degree of rotation of the X-axis labels (*)

    normalize:    If False, plot the raw numbers, If True, plot the proportions

    

    

    Credit --> https://www.kaggle.com/grfiv4/plot-a-confusion-matrix

    (*)    --> Changes I've included in the function.

    """

    

    accuracy = np.trace(cm) / float(np.sum(cm))

    misclass = 1 - accuracy



    if cmap is None:

        cmap = plt.get_cmap('Blues')

    

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    plt.figure(figsize=(figSize))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()



    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names, rotation=x_rot)

        plt.yticks(tick_marks, target_names)



    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.4f}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black",

                     fontweight = 'demibold')

        else:

            plt.text(j, i, "{:,}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black",

                     fontweight = 'demibold')





    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    plt.show()
# Search files in the folder



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Import the dataset into a pandas dataframe



fullDF = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')

print('Data imported!')
# Let's see the head of the dataset

fullDF.head()
# Check shape and statistics of the dataset

print(fullDF.shape)

fullDF.describe()
# Check if there're NaN values



fullDF.info()
# Let's see the different unique values per column



for col in fullDF.columns:

    print(col, '(',len(fullDF[col].unique()),' values ) -->', fullDF[col].unique())

    if col == 'class':

        intTotal = 1

    elif len(fullDF[col].unique()) > 1:

        intTotal = intTotal*(len(fullDF[col].unique())-1)   
fullDF.groupby('class').count().iloc[:,0].plot(kind='pie', startangle = 90, textprops={'size': 12},

                                               autopct='%1.1f%%', figsize=(7,7), colors = colPlots,

                                               explode = (0, 0.025), labels = ['',''], wedgeprops={'alpha' : 0.65})

plt.axis('off')

plt.legend(['e','p'],loc=1, frameon=False, fontsize = 12)

plt.title('Target distribution', fontsize = 15)

plt.show()
# Create bar plots



i = 1

fig1 = plt.figure(figsize=(20, 12))



for col in fullDF.columns:

    

    if len(fullDF[col].unique()) <= 3 and col != 'class' and col != 'veil-type' :

        tmpDF = fullDF.groupby([col, 'class']).count()['veil-type'].reset_index()

        tmpDF = tmpDF.pivot(index=col, columns='class', values='veil-type').fillna(0)

        plotBar(tmpDF,fig1.add_subplot(3, 2, i), col, 15)

        if i == 1 or i == 3 :

            plt.legend(frameon=False, bbox_to_anchor=(-0.1,1), loc="upper left", fontsize = 14)

        elif i == 2 or i == 4 : 

            plt.legend(frameon=False, bbox_to_anchor=(1.04,1), loc="upper left", fontsize = 14)

        i += 1

            

i=1

fig2 = plt.figure(figsize=(20,68))



for col in fullDF.columns:

    

    if len(fullDF[col].unique()) > 3 :

        

        tmpDF = fullDF.groupby([col, 'class']).count()['veil-type'].reset_index()

        tmpDF = tmpDF.pivot(index=col, columns='class', values='veil-type').fillna(0)

        plotBar(tmpDF,fig2.add_subplot(17, 1, i), col, 15)

        plt.legend(frameon=False, bbox_to_anchor=(-0.1,1), loc="upper left", fontsize = 14)

        i = i + 1



print('Plot process completed.')
# Drop the column veil-type and encode the target value into a binary variable (0/1)



fullDF['class'] = fullDF['class'].map({'e' : 0, 'p' : 1})

fullDF.drop('veil-type', axis = 1, inplace=True)

fullDF.info()
# First, we create the new features with pandas dummies

# Then, split into train and test the dataset.

# Finally we check both the train and test datasets.



fullDF = pd.get_dummies(data=fullDF, drop_first=True)

colFeat = list(fullDF.select_dtypes(include=['int64', 'uint8']).columns)

testDF = fullDF[colFeat]

testDF.info()
train, test = train_test_split(testDF, test_size=0.25, random_state=16, stratify=testDF['class'])

train.head()
test.head()
# Sanity check



print('Full dataset shape              -->', fullDF.shape)

print('Train dataset shape             -->', train.shape)

print('Test dataset shape              -->', test.shape)

print('Sum of train and test datasets  --> (',(train.shape[0] + test.shape[0]), ')')
# Hyperparameters lists



criterion_list = ['gini', 'entropy']

max_feature_list = ['auto', 'log2', None]

max_depth_list = np.arange(3,56)



# Create the parameter grid



param_grid = {'max_depth' : max_depth_list, 

              'max_features' : max_feature_list, 

              'criterion' : criterion_list} 



# Create a random search object



randomRF_class = RandomizedSearchCV(estimator = RandomForestClassifier(n_estimators=100, 

                                                                       random_state = 16),

                                    param_distributions = param_grid, n_iter = 100,

                                    scoring='accuracy', cv = 4, return_train_score= True)



# Fit to the training data



randomRF_class.fit(train.drop('class', axis= 1), train['class'])



# Get top-10 results and print them



clf_DF = pd.DataFrame(randomRF_class.cv_results_)

clf_DF = clf_DF.sort_values(by=['rank_test_score']).head(10)



print('Top 10 results (Random Forest - All features)\n')

for index, row in clf_DF.iterrows():

        print('Accuracy score (test): {:.8f} - Parameters: criterion    --> {}'.format(row['mean_test_score'], 

                                                                                       row['param_criterion'])) 

        print('                                                max_features --> {}'.format(row['param_max_features']))

        print('                                                max_depth    --> {}'.format(row['param_max_depth']))
# Create the classifier



classRF = RandomForestClassifier(n_estimators=100, random_state = 16, 

                                 max_features = randomRF_class.best_params_['max_features'], 

                                 max_depth = randomRF_class.best_params_['max_depth'], 

                                 criterion = randomRF_class.best_params_['criterion'])



# Fit and predict



classRF.fit(train.drop('class', axis= 1), train['class'])

classRF_Pred = classRF.predict(test.drop('class', axis= 1))

classRF_PP = classRF.predict_proba(test.drop('class', axis= 1))[:,1]



# Create and plot the confusion matrix



confMatrix = confusion_matrix(classRF_Pred, test['class'])

plot_confusion_matrix(confMatrix, ['e','p'], cmap='GnBu', figSize = [6,4], x_rot = 0)



# Compute and print the scores and metrics



print('Accuracy score  --> {:.8f}'.format(accuracy_score(test['class'], classRF_Pred)))

print('Precision score --> {:.8f}'.format(precision_score(test['class'], classRF_Pred)))

print('Recall score    --> {:.8f}'.format(recall_score(test['class'], classRF_Pred)))

print('F1 score        --> {:.8f}'.format(f1_score(test['class'], classRF_Pred)))

print('Log Loss error  --> {:.8f}'.format(log_loss(test['class'], classRF_PP)))
# Hyperparameters lists



kernel_list = ['linear', 'rbf']

C_list = [0.001, 0.01, 0.1, 1, 5, 10, 20, 50, 100]

gamma_list = [0.0001, 0.001, 0.01, 0.1, 1, 5, 10]



# Create the parameter grid



param_grid = {'kernel' : kernel_list, 'C' : C_list, 'gamma' : gamma_list} 



# Create a random search object



randomSVM_class = RandomizedSearchCV(estimator = SVC(random_state = 16),

                                     param_distributions = param_grid, n_iter = 25,

                                     scoring='accuracy', cv = 4, 

                                     return_train_score= True)



# Fit to the training data



randomSVM_class.fit(train.drop('class', axis= 1), train['class'])



# Get top-10 results and print them



clf_DF = pd.DataFrame(randomSVM_class.cv_results_)

clf_DF = clf_DF.sort_values(by=['rank_test_score']).head(10)



print('Top 10 results (Random Forest - All features)\n')

for index, row in clf_DF.iterrows():

        print('Accuracy score (test): {:.8f} - Parameters: C        --> {}'.format(row['mean_test_score'], 

                                                                                  row['param_C'])) 

        print('                                                gamma    --> {}'.format(row['param_gamma']))

        print('                                                kernel   --> {}'.format(row['param_kernel']))

# Create the classifier



classSVC = SVC(kernel = randomSVM_class.best_params_['kernel'], 

               C = randomSVM_class.best_params_['C'], 

               gamma = randomSVM_class.best_params_['gamma'], 

               random_state = 16, 

               probability = True) # to be able to create the predict_proba



# Fit and predict



classSVC.fit(train.drop('class', axis= 1), train['class'])

classSVC_Pred = classSVC.predict(test.drop('class', axis= 1))

classSVC_PP = classSVC.predict_proba(test.drop('class', axis= 1))[:,1]



# Create and plot the confusion matrix



confMatrix = confusion_matrix(classSVC_Pred, test['class'])

plot_confusion_matrix(confMatrix, ['e','p'], cmap='GnBu', figSize = [6,4], x_rot = 0)



# Compute and print the scores and metrics



print('Accuracy score  --> {:.8f}'.format(accuracy_score(test['class'], classSVC_Pred)))

print('Precision score --> {:.8f}'.format(precision_score(test['class'], classSVC_Pred)))

print('Recall score    --> {:.8f}'.format(recall_score(test['class'], classSVC_Pred)))

print('F1 score        --> {:.8f}'.format(f1_score(test['class'], classSVC_Pred)))

print('Log Loss error  --> {:.8f}'.format(log_loss(test['class'], classSVC_PP)))