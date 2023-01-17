import pandas as panda

from sklearn.model_selection import learning_curve, train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report, confusion_matrix, f1_score, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


from matplotlib import pyplot as plot
import seaborn as sns


from numpy import bincount, linspace, mean, std, arange, squeeze

import itertools, time, datetime

import warnings
warnings.simplefilter('ignore')

%matplotlib inline
remote_location = "../input/UCI_Credit_Card.csv"

def downLoadData():
    """
    
    Downloads the excel data from remote location. Reads one particular sheet, converts all columns names 
    to lower case and then returns the data
    
    """

    data = panda.read_csv(remote_location)

    data.rename(str.lower, inplace = True, axis = 'columns')

    print(data.dtypes)

    return data

data = downLoadData()
## check for varied data types. there may be alphabetical data types or numeric data written as string eg "4"".
## in such cases reformatting may be required
print(data.dtypes.value_counts()) ## all values are numeric and no formatting of data types are required in that case
data.shape

## since all fields are numeric we can get away with normal describe. else we would have required describe(include='all')
data.describe()
print(data['default.payment.next.month'].value_counts())

data.drop(['id'], inplace=True, axis =1)

'id' not in data.columns.tolist()
## divide up our x and y axis

_y_target = data['default.payment.next.month'].values

columns = data.columns.tolist()
columns.remove('default.payment.next.month')

_x_attributes = data[columns].values


## meaning of stratify = _y_target. returns test and training data having the same proportions of class label '_y_target'
_x_train,_x_test,_y_train, _y_test = train_test_split(_x_attributes, _y_target, test_size =0.30, stratify = _y_target, random_state = 1)

## lets check the distribution. we can see 4times the lower value as was the case before as well. train/test set distributed well
print("label counts in y train %s" %bincount(_y_train))
print("label counts in y test %s" %bincount(_y_test))
class CodeTimer:
    
    """
        Utility custom contextual class for calculating the time 
        taken for a certain code block to execute
    
    """
    def __init__(self, name=None):
        self.name = " '"  + name + "'" if name else ''

    def __enter__(self):
        self.start = time.clock()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = (time.clock() - self.start) * 1000.0
        time_taken = datetime.timedelta(milliseconds = self.took)
        print('Code block' + self.name + ' took(HH:MM:SS): ' + str(time_taken))
## cv is essentially value of K in k fold cross validation
    
## n_jobs = 1 is  non parallel execution    , -1 is all parallel , any other number say 2 means execute in 2 cpu cores

def plotLearningCurve(_x_train, _y_train, learning_model_pipeline,  k_fold = 10, training_sample_sizes = linspace(0.1,1.0,10), jobsInParallel = 1):
    
    training_size, training_score, testing_score = learning_curve(estimator = learning_model_pipeline, \
                                                                X = _x_train, \
                                                                y = _y_train, \
                                                                train_sizes = training_sample_sizes, \
                                                                cv = k_fold, \
                                                                n_jobs = jobsInParallel) 


    training_mean = mean(training_score, axis = 1)
    training_std_deviation = std(training_score, axis = 1)
    testing_std_deviation = std(testing_score, axis = 1)
    testing_mean = mean(testing_score, axis = 1 )



    plot.plot(training_size, training_mean, label= "Training Data", marker= '+', color = 'blue', markersize = 8)
    plot.fill_between(training_size, training_mean+ training_std_deviation, training_mean-training_std_deviation, color='blue', alpha =0.12 )

    plot.plot(training_size, testing_mean, label= "Testing/Validation Data", marker= '*', color = 'green', markersize = 8)
    plot.fill_between(training_size, testing_mean+ training_std_deviation, testing_mean-training_std_deviation, color='green', alpha =0.14 )

    plot.title("Scoring of our training and testing data vs sample sizes")
    plot.xlabel("Number of Samples")
    plot.ylabel("Accuracy")
    plot.legend(loc= 'best')
    plot.show()
def runGridSearchAndPredict(pipeline, x_train, y_train, x_test, y_test, param_grid, n_jobs = 1, cv = 10, score = 'accuracy'):
    
    response = {}
    training_timer       = CodeTimer('training')
    testing_timer        = CodeTimer('testing')
    learning_curve_timer = CodeTimer('learning_curve')
    predict_proba_timer  = CodeTimer('predict_proba')
    
    with training_timer:
        gridsearch = GridSearchCV(estimator = pipeline, param_grid = param_grid, cv = cv, n_jobs = n_jobs, scoring = score)

        search = gridsearch.fit(x_train,y_train)

        print("Grid Search Best parameters ", search.best_params_)
        print("Grid Search Best score ", search.best_score_)
            
    with testing_timer:
        y_prediction = gridsearch.predict(x_test)
            
    print("Accuracy score %s" %accuracy_score(y_test,y_prediction))
    print("F1 score %s" %f1_score(y_test,y_prediction))
    print("Classification report  \n %s" %(classification_report(y_test, y_prediction)))
    
    with learning_curve_timer:
        plotLearningCurve(_x_train, _y_train, search.best_estimator_)
        
    with predict_proba_timer:
        if hasattr(gridsearch.best_estimator_, 'predict_proba'):
            
            y_probability = gridsearch.predict_proba(x_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_probability[:,1])
            response['roc_auc_score'] = roc_auc_score(y_test, y_probability[:,1])
            response['roc_curve'] = (false_positive_rate, true_positive_rate)
    
        else: 
            
            response['roc_auc_score'] = 0
            response['roc_curve'] = None
    
    response['learning_curve_time'] = learning_curve_timer.took
    response['testing_time'] = testing_timer.took
    response['_y_prediction'] = y_prediction
    response['accuracy_score'] = accuracy_score(y_test,y_prediction)
    response['training_time'] = training_timer.took
    response['f1_score']  = f1_score(y_test, y_prediction)
    
    
    return response
    

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plot.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plot.imshow(cm, interpolation='nearest', cmap=cmap)
    plot.title(title)
    plot.colorbar()
    tick_marks = arange(len(classes))
    plot.xticks(tick_marks, classes, rotation=45)
    plot.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plot.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plot.ylabel('True label')
    plot.xlabel('Predicted label')
#     plot.tight_layout()
    plot.show()


classifiers = [

    LogisticRegression(random_state = 1),
    DecisionTreeClassifier(random_state = 1, criterion = 'gini'),
    RandomForestClassifier(random_state = 1, criterion = 'gini'),
    KNeighborsClassifier(metric = 'minkowski'),
    SVC(random_state = 1, kernel = 'rbf'),    
]


classifier_names = [
          
            'logisticregression',
            'decisiontreeclassifier',
            'randomforestclassifier',
            'kneighborsclassifier',
            'svc',               
    
]

classifier_param_grid = [
            
           
            {'logisticregression__C':[100,200,300,50,20,600]},
            {'decisiontreeclassifier__max_depth':[6,7,8,9,10,11]},
            {'randomforestclassifier__n_estimators':[1,2,3,5,6]} ,
            {'kneighborsclassifier__n_neighbors':[4,6,7,8]},
            {'svc__C':[1], 'svc__gamma':[0.01]},
    
]


    

timer = CodeTimer(name='overalltime')
model_metrics = {}

with timer:
    for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):

        pipeline = Pipeline([
                ('scaler', StandardScaler()),
                (model_name, model)
        ])

        result = runGridSearchAndPredict(pipeline, _x_train, _y_train, _x_test, _y_test, model_param_grid , score = 'f1')

        _y_prediction = result['_y_prediction']

        _matrix = confusion_matrix(y_true = _y_test ,y_pred = _y_prediction)

        model_metrics[model_name] = {}
        model_metrics[model_name]['confusion_matrix'] = _matrix
        model_metrics[model_name]['training_time'] = result['training_time']
        model_metrics[model_name]['testing_time'] = result['testing_time']
        model_metrics[model_name]['learning_curve_time'] = result['learning_curve_time']
        model_metrics[model_name]['accuracy_score'] = result['accuracy_score']
        model_metrics[model_name]['f1_score'] = result['f1_score']
        model_metrics[model_name]['roc_auc_score'] = result['roc_auc_score']
        model_metrics[model_name]['roc_curve'] = result['roc_curve']
        
        
print(timer.took)


model_estimates = panda.DataFrame(model_metrics).transpose()

## convert model_metrics into panda data frame
## print out across model estimations and accuracy score bar chart


model_estimates['learning_curve_time'] = model_estimates['learning_curve_time'].astype('float64')
model_estimates['testing_time'] = model_estimates['testing_time'].astype('float64')
model_estimates['training_time'] = model_estimates['training_time'].astype('float64')
model_estimates['f1_score'] = model_estimates['f1_score'].astype('float64')
model_estimates['roc_auc_score'] = model_estimates['roc_auc_score'].astype('float64')

#scaling time parameters between 0 and 1
model_estimates['learning_curve_time'] = (model_estimates['learning_curve_time']- model_estimates['learning_curve_time'].min())/(model_estimates['learning_curve_time'].max()- model_estimates['learning_curve_time'].min())
model_estimates['testing_time'] = (model_estimates['testing_time']- model_estimates['testing_time'].min())/(model_estimates['testing_time'].max()- model_estimates['testing_time'].min())
model_estimates['training_time'] = (model_estimates['training_time']- model_estimates['training_time'].min())/(model_estimates['training_time'].max()- model_estimates['training_time'].min())

print(model_estimates)
model_estimates.plot(kind='barh',figsize=(12, 10))
plot.title("Scaled Estimates across different classifiers used")
plot.show()
def plotROCCurveAcrossModels(positive_rates_sequence, label_sequence):
    

    for plot_values, label_name in zip(positive_rates_sequence, label_sequence):
        
        plot.plot(list(plot_values[0]), list(plot_values[1]),  label = "ROC Curve for model: "+label_name)
        
    plot.plot([0, 1], [0, 1], 'k--', label = 'Random Guessing') #
    plot.title('ROC Curve across models')
    plot.xlabel('False Positive Rate')
    plot.ylabel('True Positive Rate')
    plot.legend(loc='best')
    plot.show()
    

roc_curve_input = {}
for i , j in enumerate(model_metrics):
    
    _matrix = model_metrics[j]['confusion_matrix']
    plot_confusion_matrix(_matrix, classes = [0,1], title = 'Confusion Matrix for %s'%j)
    if model_metrics[j]['roc_curve']:
        roc_curve_input[j]= model_metrics[j]['roc_curve']
    

plotROCCurveAcrossModels(list(roc_curve_input.values()), list(roc_curve_input.keys()))
    
    

classifier = DecisionTreeClassifier(random_state = 1, criterion = 'gini', max_depth = 7, class_weight = 'balanced')

pipeline = Pipeline([('scaler', StandardScaler()), ('decisiontreeclassifier', classifier)])

pipeline.fit(_x_train,_y_train)

y_prediction = pipeline.predict(_x_test)

y_probability = pipeline.predict_proba(_x_test)

print("Accuracy score accounting for unbalanced classes", accuracy_score(_y_test, y_prediction))
print("F1 score accounting for unbalanced classes", f1_score(_y_test, y_prediction))
print("ROC AUC score accounting for unbalanced classes", roc_auc_score(_y_test, y_probability[:, 1]))

from sklearn.ensemble import  AdaBoostClassifier

boostClassifier = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(random_state = 1, criterion = 'gini', max_depth = 7, class_weight = 'balanced'), random_state = 1)

pipeline = Pipeline([('scaler', StandardScaler()), ('boostClassifier', boostClassifier)])

param_grid = [{'boostClassifier__n_estimators':[ 500], 'boostClassifier__learning_rate' :[0.1]}]

result = runGridSearchAndPredict(pipeline, _x_train, _y_train, _x_test, _y_test, param_grid , score = 'f1')

print("Accuracy score accounting for unbalanced classes", result['accuracy_score'])
print("F1 score accounting for unbalanced classes", result['f1_score'])
print("ROC AUC score accounting for unbalanced classes", result['roc_auc_score'])