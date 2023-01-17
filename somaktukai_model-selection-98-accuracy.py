import pandas as panda

from sklearn.model_selection import learning_curve, train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report, confusion_matrix, f1_score, roc_curve, roc_auc_score
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

from scipy import stats


from matplotlib import pyplot as plot
import matplotlib.patches as mpatches

import seaborn as sns


from numpy import bincount, linspace, mean, std, arange, squeeze

import itertools, time, datetime
from collections import Counter

import warnings
warnings.simplefilter('ignore')

%matplotlib inline
remote_location_test_data = "../input/test.csv"
remote_location_training_data = "../input/train.csv"
test_data = panda.read_csv(remote_location_test_data)
train_data = panda.read_csv(remote_location_training_data)

print('Testing Data Shape: ', test_data.shape, ' , Training Data Shape: ' , train_data.shape)
target_attribute = 'species'
target_spread = train_data[target_attribute].value_counts()

print('Spread of unique leaf attributes in training data: \n', target_spread)
train_data[target_attribute].describe(include = 'all')


target_spread.plot(kind = 'barh', figsize = (10,20))
plot.title("Distribution of Target Values")
plot.xlabel(" COunt")
plot.ylabel("Leaf")
plot.show()

encoder = LabelEncoder()
encoder.fit(train_data[target_attribute])
_y_train = encoder.transform(train_data[target_attribute])
Y = _y_train
_y_train
all_columns = train_data.columns.tolist()
columns_unwanted = ['id', 'species']
columns_wanted = [i for i in all_columns if i not in columns_unwanted]
print('Number of unique data types in the data \n', train_data[columns_wanted ].dtypes.value_counts())


empty_values = train_data[columns_wanted ].isnull().sum().to_frame()
empty_values= empty_values.assign(column_type = train_data.dtypes)
print('Number of empty columns: ', set(empty_values[0].values))


train_data_description = train_data[columns_wanted ].describe(include = "all")
train_data_description = train_data_description.transpose()[['25%', '50%', '75%', 'max', 'mean']]
train_data_description

# train_data_description.iloc[0 : 64, :]
# train_data_description.loc[['margin1', 'margin2'], ['max']]

exclude_numbers = list(map(str, range(0,10)))
column_identifiers = map(lambda x:''.join([i for i in x if i not in exclude_numbers]), columns_wanted)
column_identifiers = set(column_identifiers)

## lets start plotting 25%, 50%, 75% and mean of the indivual column identifiers. we know from our data description that there are 64 of each

for col in column_identifiers:
    train_data_description.loc[[col + str(i) for i in range(1,65)]].plot(kind='bar', figsize = (30,40))
    plot.title("Distribution of data for columns starting with identifier " + col)
    plot.xlabel("Count")
    plot.ylabel(col)
    plot.show()



print('value counts for column margin8 \n', train_data['margin8'].value_counts()\
          , '\n for column margin16 \n', train_data['margin16'].value_counts()\
          , '\n for column margin23 \n', train_data['margin23'].value_counts())

print('value counts for column texture23 \n', train_data['texture23'].value_counts()\
          , '\n for column texture36 \n', train_data['texture36'].value_counts()\
          , '\n for column texture56 \n', train_data['texture56'].value_counts() \
          , '\n for column texture61 \n', train_data['texture61'].value_counts())

def calculateCorrelationCoefficientsAndpValues(x_data, y_data, xlabel):
    
    pearson_coef, p_value = stats.pearsonr(x_data, y_data)
    print("The Pearson Correlation Coefficient for %s is %s with a P-value of P = %s" %(xlabel,pearson_coef, p_value))
    
    return (pearson_coef,p_value)


def plotRegressionBetweenTwoVariables(x_label,y_label, x_y_data, pearson_coef, p_value):
    
    plot.figure(figsize=(15,15))
    
    sns.regplot(x = x_label , y = y_label , data = x_y_data)


    # plot.text(x = 1, y = 40000 , s ="Pearson Correlation Coefficient = %s"%pearson_coef, fontsize = 12 )
    # plot.text(x = 1, y = 38000 , s ="P value = %s"%p_value, fontsize = 12 )

    blue_patch = mpatches.Patch(color='blue', label='Pearson Correlation Coefficient = %s, p value is %s '%(pearson_coef, p_value))
    plot.legend(handles=[blue_patch], loc ='best')
    plot.title("Regression Plot %s vs %s"%(x_label, y_label))
X = train_data[columns_wanted]
# Y = _y_train

_x_train, _x_test, _y_train, _y_test = train_test_split(X, Y, test_size =0.30, stratify = Y, random_state = 1)

##using COunter object we check to see if test and training has been properly distributed and we find it is.
print(Counter(Y))
print(Counter(_y_train))
print(Counter(_y_test))
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

    ## we have got the estimator in this case the perceptron running in 10 fold validation with 
    ## equal division of sizes betwwen .1 and 1. After execution, we get the number of training sizes used, 
    ## the training scores for those sizes and the test scores for those sizes. we will plot a scatter plot 
    ## to see the accuracy results and check for bias vs variance

    # training_size : essentially 10 sets of say a1, a2, a3,,...a10 sizes (this comes from train_size parameter, here we have given linespace for equal distribution betwwen 0.1 and 1 for 10 such values)
    # training_score : training score for the a1 samples, a2 samples...a10 samples, each samples run 10 times since cv value is 10
    # testing_score : testing score for the a1 samples, a2 samples...a10 samples, each samples run 10 times since cv value is 10
    ## the mean and std deviation for each are calculated simply to show ranges in the graph

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
    print("F1 score %s" %f1_score(y_test,y_prediction, average = 'macro'))
    print("Classification report  \n %s" %(classification_report(y_test, y_prediction)))
    
    with learning_curve_timer:
        plotLearningCurve(_x_train, _y_train, search.best_estimator_, k_fold = 7)
                   
    response['learning_curve_time'] = learning_curve_timer.took
    response['testing_time'] = testing_timer.took
    response['_y_prediction'] = y_prediction
    response['accuracy_score'] = accuracy_score(y_test,y_prediction)
    response['training_time'] = training_timer.took
    response['f1_score']  = f1_score(y_test, y_prediction, average= 'macro')
    
    
    return response
    
classifiers = [
    
    LogisticRegression(random_state = 1),
    LogisticRegression(random_state = 1, solver = 'lbfgs', multi_class = 'multinomial'),
    DecisionTreeClassifier(random_state = 1, criterion = 'gini'),
    RandomForestClassifier(random_state = 1, criterion = 'gini'),
    KNeighborsClassifier(metric = 'minkowski'),
    SVC(random_state = 1, kernel = 'rbf'), 
    LinearDiscriminantAnalysis()
     
]


classifier_names = [
            'logisticregression',
            'multinomiallogisticregression',
            'decisiontreeclassifier',
            'randomforestclassifier',
            'kneighborsclassifier',
            'svc', 
            'lda',
    
]

classifier_param_grid = [
            
            {'logisticregression__C':[100,200,300,50,20,600]},
            {'multinomiallogisticregression__C':[100,200,300,50,20,600], 'multinomiallogisticregression__penalty':['l2'], 'multinomiallogisticregression__max_iter':[100,200,300,400]},
            {'decisiontreeclassifier__max_depth':[1,2,4,6,7,8,9,10,11]},
            {'randomforestclassifier__n_estimators':[1,2,3,5,6]} ,
            {'kneighborsclassifier__n_neighbors':[4,6,7,8]},
            {'svc__C':[1, 10, 100, 200], 'svc__gamma':[0.01 , 0.1, 0.05]},
            {'lda__n_components':[4,5,6]},
    
]


    

timer = CodeTimer(name='overalltime')
model_metrics = {}

with timer:
    for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):

        pipeline = Pipeline([
                ('scaler', StandardScaler()),
                (model_name, model)
        ])

        result = runGridSearchAndPredict(pipeline, _x_train, _y_train, _x_test, _y_test, model_param_grid , cv = 7,score = 'accuracy')

        _y_prediction = result['_y_prediction']

        _matrix =  confusion_matrix(y_true = _y_test ,y_pred = _y_prediction)

        model_metrics[model_name] = {}
        model_metrics[model_name]['confusion_matrix'] = _matrix
        model_metrics[model_name]['training_time'] = result['training_time']
        model_metrics[model_name]['testing_time'] = result['testing_time']
        model_metrics[model_name]['learning_curve_time'] = result['learning_curve_time']
        model_metrics[model_name]['accuracy_score'] = result['accuracy_score']
        model_metrics[model_name]['f1_score'] = result['f1_score']
        model_metrics[model_name]['classes'] = encoder.inverse_transform(_y_prediction)
        
        
        
        
print(timer.took)




model_estimates = panda.DataFrame(model_metrics).transpose()


## convert model_metrics into panda data frame
## print out across model estimations and accuracy score bar chart


model_estimates['learning_curve_time'] = model_estimates['learning_curve_time'].astype('float64')
model_estimates['testing_time'] = model_estimates['testing_time'].astype('float64')
model_estimates['training_time'] = model_estimates['training_time'].astype('float64')
model_estimates['f1_score'] = model_estimates['f1_score'].astype('float64')

#scaling time parameters between 0 and 1
model_estimates['learning_curve_time'] = (model_estimates['learning_curve_time']- model_estimates['learning_curve_time'].min())/(model_estimates['learning_curve_time'].max()- model_estimates['learning_curve_time'].min())
model_estimates['testing_time'] = (model_estimates['testing_time']- model_estimates['testing_time'].min())/(model_estimates['testing_time'].max()- model_estimates['testing_time'].min())
model_estimates['training_time'] = (model_estimates['training_time']- model_estimates['training_time'].min())/(model_estimates['training_time'].max()- model_estimates['training_time'].min())

print(model_estimates)
model_estimates.plot(kind='barh',figsize=(12, 10))
plot.title("Scaled Estimates across different classifiers used")
plot.show()

X_test = test_data[columns_wanted]
ids = test_data['id'].values


pipeline = Pipeline([('scaler', StandardScaler()),('lineardiscriminantanalysis', LogisticRegression(random_state = 1, solver = 'lbfgs', multi_class = 'multinomial', C = 100, penalty = 'l2', max_iter = 100))])

param_grid = [{}]

search = GridSearchCV(estimator = pipeline, param_grid = param_grid, scoring = 'accuracy', cv = 10)

search.fit(train_data[columns_wanted], Y) ## fit on the entire training data

probabilities = search.predict_proba(X_test)


classes = train_data['species'].unique()

# print(classes)
classes = classes.tolist()
## it is important to sort the classes, since predict_proba returns array-like, shape = [n_samples, n_classes]
# Returns the probability of the sample for each class in the model, where classes are ordered as they are in self.classes_.
classes.sort() 

submission = panda.DataFrame(probabilities, columns=classes)
submission.insert(0, 'id', ids)
submission.reset_index()

# Export Submission
# submission.to_csv('C:/Users/somak/Documents/somak_python/real-world-use-cases/supervised/classification/kaggle/leaf_classification/submission.csv', index = False)
submission
