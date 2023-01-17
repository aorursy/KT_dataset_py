import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sys,csv,os

import seaborn as sns

#NLP libraries

import re

import nltk #importing the tools (list of irrelevant words) which has to be removed

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer



# Text Processing

from sklearn.feature_extraction.text import CountVectorizer



# Classifiers

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier



# Data Preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



# Metrics

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict, cross_validate, RandomizedSearchCV, learning_curve

from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, classification_report, make_scorer, fbeta_score, matthews_corrcoef



# Visualization

from matplotlib.pyplot import cm

from funcsigs import signature

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import average_precision_score



import warnings

warnings.filterwarnings('ignore')



# Classifier Details

classifiers = [

{

    'label': 'Logistic Regression Classifier',

    'model': LogisticRegression(),

    'parameters': {"penalty": ['l1', 'l2'], 'C': [0.01, 0.1, 0.5, 1, 10]},

    'g_cv' : 10

    

},

{

    'label': 'Naive- Bayes Classifier',

    'model': GaussianNB(),

    'parameters': {},

    'g_cv' : 10

    

},

{

    'label': 'Support Vector Classifier',

    'model': SVC(),

    'parameters': {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']},

    'g_cv' : 10

},

{

    'label': 'K-Nearest Neighbor',

    'model': KNeighborsClassifier(),

    'parameters':{"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},

    'g_cv' : 10

},

{

    'label': 'Decision Tree Classifier',

    'model': DecisionTreeClassifier(),

    'parameters': {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 

              "min_samples_leaf": list(range(5,7,1))},

    'g_cv' : 10

    

},

{

    'label': 'Random Forest Classifier',

    'model': RandomForestClassifier(),

    'parameters': { 'n_estimators': [10,100,300,500],

                   "criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 

              "min_samples_leaf": list(range(5,7,1))},

    'g_cv' : 5    

 },



{

    'label': 'XGBoost Classifier',

    'model': XGBClassifier(),

    'parameters': { 'learning_rate': [0.01], 'n_estimators':[100,500],

                   'gamma': [0.5, 1, 1.5], 'subsample': [0.6, 0.8, 1.0], 

                   'colsample_bytree': [0.6, 0.8, 1.0], 'max_depth': [2]},

    'g_cv' : 2

    

}

]
#Converting csv to tsv

def convert_to_tsv(csv_file, tsv_file):

    """

    Converts a comma-separated file to tab_separated file

    

    Args:    

        csv_file: path to csv_file

        tsv_file: path to new tsv_file



    Returns:    

        tsv_file: path to new tsv_file

    

    """           

    csv.writer(open(tsv_file, 

                    'w+',

                    encoding="utf-8"),

    delimiter='\t').writerows(csv.reader(open(csv_file,

                              encoding="utf8")))

    return tsv_file
def create_corpus(dataset):

    """Creates corpus from the input dataset

    

    Args:

        Dataset: Input data as pandas Dataframe

        

    Returns:

        corpus: List of preprocessed input data

        

    """

    corpus = [] 

    for i in range(0,len(dataset)):

        #remove all characters except a-z, removed charac will be replaced by space

        comment = re.sub(pattern = '[^a-zA-Z]',repl = ' ' , string = dataset['CONTENT'][i]) 

        

        #to lower case

        comment = comment.lower() 

        

        #splitting each  comment sentence into list of words

        comment = comment.split() 

        

        #Stemming 

        ps = PorterStemmer()

        comment = [ps.stem(word) for word in comment if not word in set(stopwords.words('english'))] 

        

        #Joining the words back 

        comment =' '.join(comment)

        corpus.append(comment)

    return corpus
def bag_of_words(corpus):

    """Creating Bag of Words Model

    

    Creates bag of words using CountVectorizer, which is a sparse matrix

    with all the words from corpus. Each cell will contain its own frequency

    in the corresponding comment.

    

    Args:

        corpus, a list containing processed input dataset

        

    Returns:

        The sparse matrix (bag of words) X and labels y

    

    """   

    # tokenizer

    cv = CountVectorizer()

    X = cv.fit_transform(corpus).toarray() 

    

    # re-initializing to add max_features so that we can filter out irrelevant words which has very less frequency

    cv = CountVectorizer(max_features = (X.shape[1] - 50))

    X = cv.fit_transform(corpus).toarray() 

    y = dataset.iloc[:,-1].values



    return X,y
def best_estimator(classifiers, X_train, y_train, filename):

    """Finding the best estimator 

    

    Uses GridSearchCV to search for the best parameters for all

    the classifiers used to find the best optimized classifier

    

    Args: 

        classifiers: List of classifier dictionaries with names and parameter details

                

        X_train: X values of training data

        

        y_train: Actual labels of training data

        

        filename: String

        Name of Youtube accounts whose comments are being classified

        

    Returns:

        best_estimator: Dictionary of classifiers parameters, optimized for each dataset

    

    """

    

    scoring = {'acc': 'accuracy',

               'AUC': 'roc_auc',

              'prec_macro': 'precision_macro',

               'rec_micro': 'recall_micro',

               'f1_score': 'f1_micro'}

    best_estimators = dict()

    best_scores_df = pd.DataFrame()

     

    for c in classifiers:

            classifier = c['model']

            label = c['label']

            print('\n\n Optimized ', label, 'for ', filename)

            print('---------------------------------------')

            

            #print('%s Best Values' % (c['label']))

            grid_search = GridSearchCV(estimator = classifier,

                       param_grid = c['parameters'],

                       scoring = scoring,

                       refit='acc',

                       cv = c['g_cv'],

                       return_train_score=True,

                       verbose =1)

            if label == 'Random Forest Classifier':

                grid_search.fit(X_train, y_train, sample_weight = None)

            else:

                grid_search.fit(X_train, y_train)

            results = grid_search.cv_results_

            print('\n')

            #print('Best Accuracy Score: ',round(grid_search.best_score_*100,2),'%')

            print('Best Parameters: ',grid_search.best_params_)

            

            best_estimators[label] = grid_search.best_estimator_

            

            data = [[label, round(grid_search.best_score_*100,2)]]

            df2 = pd.DataFrame(data, columns = ['Classifier','Accuracy']) 

            best_scores_df = best_scores_df.append(df2)

            

            #best_scores[label] = round(grid_search.best_score_*100,2)

            

            print('\n')

            for key,scorer in scoring.items():

                #print('{} scores:\n '.format(scorer))

                for sample in ('train','test'):

                    sample_score_mean = round(results['mean_%s_%s' % (sample, key)].mean()*100,2)

                    sample_score_std = round(results['std_%s_%s' % (sample, key)].mean()*100,2)

                    if(sample == 'train'):

                        to_print = 'Training'

                    else:

                        to_print = 'Validation'

                    print(to_print,' ',scorer,' : ',sample_score_mean, '% (+/-)', sample_score_std,'%')

                print('\n')

                   

            print('-----------------------------------------')

                

    print('Comparing the Best Cross-Validated Accuracy between Classifiers')

    best_scores_df

    #print('\t'.join(['{0}{1} % \n'.format(k, v) for k,v in best_scores.items()]))

    return best_estimators
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

                     

    """Generate a simple plot of the test and training learning curve.

    

    Reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html



    Args:

        estimator: object type that implements the "fit" and "predict" methods

            An object of that type which is cloned for each validation.



        title: string

            Title for the chart.



        X: array-like, shape (n_samples, n_features)

            Training vector, where n_samples is the number of samples and

            n_features is the number of features.



        y: array-like, shape (n_samples) or (n_samples, n_features), optional

            Target relative to X for classification or regression;

            None for unsupervised learning.



        ylim: tuple, shape (ymin, ymax), optional

            Defines minimum and maximum yvalues plotted.



        cv: int, cross-validation generator or an iterable, optional

            Determines the cross-validation splitting strategy.



        n_jobs: int or None, optional (default=None)

            Number of jobs to run in parallel.

           

        train_sizes: array-like, shape (n_ticks,), dtype float or int

            Relative or absolute numbers of training examples that will be used to

            generate the learning curve. 

            

    Returns:

        plt: an object of the generated plot

    

    """



    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")



    plt.legend(loc="best")

    return plt
def simple_fit_predict(classifiers, best_estimators, X_train, X_test, y_train, y_test, filename):

    """Fit the train data to classifier and predict on test data

    

    Args:

        classifiers: List of classifier dictionaries with names and parameter details

                

        best_estimators: Dictionary of classifier names and optimized parameters

        

        X_train: X values of training data

        

        X_test: X values of test data

        

        y_train: y values of train daa

        

        y_test: y values of test data

        

        filename: String

        Name of Youtube accounts whose comments are being classified    

        

    Returns:

        y_preds: Predicted values for test input

    

    """

    print('Validation Scores:\n')

    y_preds = dict()

    for key, model in best_estimators.items():

        #classifier = c

        label = model.__class__.__name__

        if label == 'Random Forest Classifier':

            model.fit(X_train, y_train, weight = None)

        else:

            model.fit(X_train, y_train)    

        train_predictions = model.predict(X_train)

        test_predictions = model.predict(X_test)

        y_preds[label] = test_predictions

        

        print('Predicting test data for ', filename,': Using ',label)

        print('\n')

        print('Precision:')

        print('Training score: ',round(precision_score(train_predictions,y_train)*100,2),'%', '\t Testing score: ',  round(precision_score(test_predictions,y_test)*100,2),'%')

        print('Recall:')

        print('Training score: ',round(recall_score(train_predictions,y_train)*100,2),'%', '\t Testing score: ',  round(recall_score(test_predictions,y_test)*100,2),'%')

        print('F1 Score:')

        print('Training score: ',round(f1_score(train_predictions,y_train)*100,2),'%', '\t Testing score: ',  round(f1_score(test_predictions,y_test)*100,2),'%')

        print('Accuracy:')

        print('Training score: ',round(accuracy_score(train_predictions,y_train)*100,2),'%', '\t Testing score: ',  round(accuracy_score(test_predictions,y_test)*100,2),'%')

        print('==========================================')

        print('\n')

    return y_preds
def roc_curves(classifiers, best_estimators, X_test, y_test, filename):

    """Plot ROC Curves



    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html

    

    Displays the Receiver Operating Characteristic (ROC) curve for the classifiers

    

    Args: 

        classifiers: List of classifier dictionaries with names and parameter details

                

        best_estimators: Dictionary of classifier names and optimized parameters

        

        X_test: X values of test data

        

        y_test: y values of test data

        

        filename: String

        Name of Youtube accounts whose comments are being classified  



    Returns:

        plt: An object of the plot

        

    """

    color=iter(cm.rainbow(np.linspace(0,15,100)))

    for key, model in best_estimators.items():

        y_pred = model.predict(X_test) # predict the test data

        # Compute False postive rate, and True positive rate

        if hasattr(model, "decision_function"):

            y_pred = model.decision_function(X_test)

        else:

            y_pred = model.predict_proba(X_test)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, y_pred)

        # Calculate Area under the curve to display on the plot

        auc = roc_auc_score(y_test,y_pred)

        # Now, plot the computed values

        plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (model.__class__.__name__, auc))

    

    # Custom settings for the plot 

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('1-Specificity(False Positive Rate)')

    plt.ylabel('Sensitivity(True Positive Rate)')

    plt.title('Receiver Operating Characteristic - '+ filename)

    plt.legend(loc="lower right")

    plt.show()               
def get_confusion_matrix_values(y_test, y_pred):

    """Creates confusion matrix

    

    Args:

        y_test: y values of test data

        

        y_pred: Predicted y values

        

    Returns: 

        Array of confusion matrix values

    

    """

    cm = confusion_matrix(y_test, y_pred)

    return(cm[0][0], cm[0][1], cm[1][0], cm[1][1])

def performance_comparison(best_estimators, y_preds, y_test, filename):

    """Preparing evaluation metrics for all data - classifier combinations

    

    Args: 

        classifiers: List of classifier dictionaries with names and parameter details

        

        best_estimators: Dictionary of classifier names and optimized parameters

        

        X_test : X values of test data

        

        y_test : y values of test data

        

        filename : String

        Name of Youtube accounts whose comments are being classified 

        

    Returns:

        df: Dataframe of evaluation metrics of all classifiers for each dataset    

    

    """

    df = pd.DataFrame()

    df1=[]

    for key, y_pred in y_preds.items():

        label = key

        

        TN, FP, FN, TP = get_confusion_matrix_values(y_test, y_pred)

        

        precision = round(TP/ (TP + FP)*100,2)

        recall = round(TP / (TP + FN)*100,2)

        accuracy = round((TP+TN)/(TP+TN+FP+FN)*100,2)

        spam_caught_rate = round(TP/ (TP+FP)*100,2)

        blocked_ham = round(FN / (TN + FN)*100,2)

        matthews_coefficient = round(matthews_corrcoef(y_test, y_pred)*100,2) 

        f1_score = round(2 * precision * recall/(precision + recall),2)

        

        data = [[label, accuracy, spam_caught_rate, blocked_ham, matthews_coefficient, f1_score]]

        df2 = pd.DataFrame(data, columns = ['Classifier','Accuracy','Spam_Caught_Rate', 'Blocked_Ham','Matthews_Coefficient', 'F1 Score']) 

        df1.append(df2)

        

    df = pd.concat(df1,ignore_index = True)

    return df
#Data Input

base_dir = '../input/'

data_files = [os.path.join(base_dir,f) for f in os.listdir(base_dir)] 

files = os.listdir(base_dir)

input_dir = './ directory'

#os.mkdir(input_dir)



df = pd.DataFrame()

df_all = pd.DataFrame()

df_list =[]

for csv_file,file in zip(data_files, files):

    filename, file_extension = os.path.splitext(file)

    tsv_file = input_dir+ filename + '_input.tsv'

    

    print('PROCESSING DATASET.........', filename)

    print('===========================================================\n')

        

    #convert csv files to tsv format

    input_file = convert_to_tsv(csv_file, tsv_file)

    

    if filename == 'Youtube04-Eminem':

        dataset = pd.read_csv(input_file,delimiter = '\t', skiprows = range(270,276), quoting = 3)

        #skipping content with nan values

    else:

        #Reading the tsvfile ignoring quotes

        dataset = pd.read_csv(input_file,delimiter = '\t', quoting = 3)

    #Viewing the columns

    print(dataset.head())



#APPLYING NLP

    print('APPLYING NLP ON .........', filename)

    print('===========================================================\n')

       

    #Cleaning texts 

    corpus = create_corpus(dataset)

    

    X,y = bag_of_words(corpus)



#DATA PRE-PROCESSING FOR CLASSIFICATION

    # Splitting the dataset into the Training set and Test set

    print('SPLITTING INTO TRAIN AND TEST.........', filename)

    print('===========================================================\n')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    

#OPTIMIZING PARAMETERS FOR CLASSIFIERS

    #Find the best hyper parameters using GridSearchCV method

    print('OPTIMIZATION OF CLASSIFIERS THROUGH GRIDSEARCHCV FOR ', filename)

    print('===========================================================\n')

       

    best_estimators = best_estimator(classifiers, X_train, y_train, filename)



#FIT & PREDICT DATA TO THE BEST VERSION OF CLASSIFIERS

    #Simple fit_predict to see how each classifier performs

    print('FITTING THE BEST CLASSIFIER AND PREDICTING TEST SET RESULT FOR ', filename)

    print('===========================================================\n')

       

    y_preds = simple_fit_predict(classifiers,best_estimators, X_train, X_test, y_train, y_test, filename)

        

#EVALUATING CLASSIFIERS

    

    #Classification report

    #report(best_estimators, X_test, y_test)



    #SPAM DETECTION PERFORMANCE COMPARISON OF CLASSIFIERS

    print('PERFORMANCE COMPARISON OF CLASSIFIERS FOR ', filename)

    print('===========================================================\n')

       

    df = performance_comparison(best_estimators, y_preds, y_test, filename)

    print(df.head(8))

    

    df_list.append(df)

    print('===========================================================')

    

    #Plot ROC curves

    roc_curves(classifiers, best_estimators, X_test, y_test, filename)    



df_all = pd.concat(df_list, ignore_index = True)

    
def highlight_max(s):

    '''

    highlight the maximum in a Series yellow.

    '''

    is_max = s == s.max()

    return ['background-color: yellow' if v else '' for v in is_max]
from tabulate import tabulate

print('======================================\n')

print('FINAL PERFORMANCE COMPARISON GRID\n')

print('======================================\n')

headers = ['Classifier','Accuracy','Spam Caught','Blocked Ham','Mathews Coeff','F1 Score']

[print("Dataset Name: ",f,"\n\n",tabulate(x, headers = headers, tablefmt='psql', numalign="right",floatfmt=".2f"),"\n\n") for f,x in zip(files,df_list)]