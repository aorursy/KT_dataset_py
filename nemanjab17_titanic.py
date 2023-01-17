# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, confusion_matrix,  roc_curve, precision_recall_curve, accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve, train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
import itertools

warnings.filterwarnings('ignore') #ignore warning messages 

import os
data_train_file = "../input/train.csv"
data_test_file = "../input/test.csv"

df_train = pd.read_csv(data_train_file)
df_test = pd.read_csv(data_test_file)

#SibSP values (1,0,3,4,2,5,8)
#891 observations in train set and 418 in test set
#test set has no outcome variable
#there is 248 unique fares, how do they impact their probability of death? Higher fares, better cabins? Or did they get boats because they were "VIP"?
#male and female change to 1 and 0
#ticket columns random numbers and letter
#

df_train.Sex.replace(to_replace = dict(male = 1, female = 0), inplace = True)
df_test.Sex.replace(to_replace = dict(male = 1, female = 0), inplace = True)
df_train.dropna()
df_test.dropna()
dummies = pd.get_dummies(df_train['Embarked'])
df_train = df_train.join(dummies, how='outer')
df_train = df_train.drop('Embarked',axis=1)
df_test = df_test.join(dummies, how='outer')
df_test = df_test.drop('Embarked',axis=1)
df_train.head()
# Any results you write to the current directory are saved as output.

def plot_distribution(data_select) : 
    sns.set_style("ticks")
    s = sns.FacetGrid(df_train, hue = 'Survived',aspect = 2.5, palette ={0 : 'lightskyblue', 1 :'gold'})
    s.map(sns.kdeplot, data_select, shade = True, alpha = 0.5)
    s.set(xlim=(0, df_train[data_select].max()))
    s.add_legend()
    s.set_axis_labels(data_select, 'proportion')
    s.fig.suptitle(data_select)
    plt.show()
    
def plot_distribution2(data_select) : 
    sns.set_style("ticks")
    s = sns.FacetGrid(df_train, hue = 'Survived',aspect = 2.5, palette ={0 : 'lightskyblue', 1 :'gold'})
    s.map(sns.kdeplot, data_select, shade = True, alpha = 0.5)
    s.set(xlim=(0, 200))
    s.add_legend()
    s.set_axis_labels(data_select, 'proportion')
    s.fig.suptitle(data_select)
    plt.show()
plot_distribution('Age')
plot_distribution('SibSp')
plot_distribution('Parch')
plot_distribution2('Fare')
# Correlation matrices by outcome
f, (ax1, ax2) = plt.subplots(1,2,figsize =( 18, 8))
sns.heatmap((df_train.loc[df_train['Survived'] ==1]).corr(), vmax = .8, square=True, ax = ax1, cmap = 'magma', linewidths = .1, linecolor = 'grey');
ax1.invert_yaxis();
ax1.set_title('Yes')
sns.heatmap((df_train.loc[df_train['Survived'] ==0]).corr(), vmax = .8, square=True, ax = ax2, cmap = 'YlGnBu', linewidths = .1, linecolor = 'grey');
ax2.invert_yaxis();
ax2.set_title('No')
plt.show()
palette ={0 : 'lightblue', 1 : 'gold'}
edgecolor = 'grey'
fig = plt.figure(figsize=(12,12))

plt.subplot(221)
ax1 = sns.scatterplot(x = df_train['Age'], y = df_train['Fare'], hue = "Survived",
                    data = df_train, palette =palette, edgecolor=edgecolor)
plt.title('Age vs Fare')

# PCA
df_train.dropna()
df_train = df_train[np.isfinite(df_train['Age'])]
#df_train = df_train[~np.isnan(df_train)]

target_pca = df_train['Survived']
data_pca = df_train.drop(['Survived', 'PassengerId','Pclass','Name','Ticket','Cabin'], axis=1)

print(data_pca.columns.values)

#To make a PCA, normalize data is essential
X_pca = data_pca.values

X_std = StandardScaler().fit_transform(X_pca)

# Select 6 components
pca = PCA(n_components = 8)
pca_std = pca.fit(X_std, target_pca).transform(X_std)

colors = ['gold', 'lightgreen', 'lightcoral', 'lightskyblue', 'lightgrey','gold','gold','gold']
explode = (0.1, 0.1, 0.1, 0.1, 0.1,0.1,0.1,0.1)
labels = ['comp 1','comp 2','comp 3','comp 4','comp 5','comp 6','comp 7','comp 8']

plt.figure(figsize=(25,12))
plt.subplot(121)
ax1 = plt.pie(pca.explained_variance_ratio_, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', textprops = { 'fontsize': 12}, shadow=True, startangle=140)
plt.title("PCA : components and explained variance (5 comp)", fontsize = 20)
plt.show()
#creating confusion matrix, it shows number of rows survived and not survived,
#compared to our predicted values, look it up online or just wait for a clear picture, it is understandable
def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Confusion matrix"',
                          cmap = plt.cm.Blues) :
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 0)
    plt.yticks(tick_marks, classes)
#choosing between different options, selecting data shown etc.. 
#really technical part
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :
        plt.text(j, i, cm[i, j],
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# Show metrics 
def show_metrics():
    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    tn = cm[0,0]
    print('Accuracy  =     {:.3f}'.format((tp+tn)/(tp+tn+fp+fn)))
    print('Precision =     {:.3f}'.format(tp/(tp+fp)))
    print('Recall    =     {:.3f}'.format(tp/(tp+fn)))
    print('F1_score  =     {:.3f}'.format(2*(((tp/(tp+fp))*(tp/(tp+fn)))/
                                                 ((tp/(tp+fp))+(tp/(tp+fn))))))
# Def X and Y
y = np.array(df_train.Survived.tolist())
X = np.array(data_pca.as_matrix())
scaler = StandardScaler()
X = scaler.fit_transform(X)
#import model from the library
from sklearn.linear_model import LogisticRegression
#random generator class, this part is not of concern to general public
random_state = 600
#selecting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.12, random_state = random_state)
#call upon algorithm
log_clf = LogisticRegression(random_state = random_state)
#give it parameters, this is engineers only concern
param_grid = {
            'penalty' : ['l2','l1'],  
            'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            }
#engineers only concern
CV_log_clf = GridSearchCV(estimator = log_clf, param_grid = param_grid , scoring = 'accuracy', verbose = 1, n_jobs = -1)
CV_log_clf.fit(X_train, y_train)
#give us the best a and b in football players analogy
best_parameters = CV_log_clf.best_params_
print('The best parameters for using this model is', best_parameters)
#this is where we connect our parameters givven with matrix we created, so now we can see the results.
#Log with best hyperparameters
CV_log_clf = LogisticRegression(C = best_parameters['C'], 
                                penalty = best_parameters['penalty'], 
                                random_state = random_state)

CV_log_clf.fit(X_train, y_train)
y_pred = CV_log_clf.predict(X_test)
y_score = CV_log_clf.decision_function(X_test)

# Confusion maxtrix & metrics
cm = confusion_matrix(y_test, y_pred)
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cm, 
                      classes=class_names, 
                      title='Logistic Confusion matrix')
plt.savefig('6')
plt.show()

show_metrics()


log2_clf = LogisticRegression(random_state = random_state)
param_grid = {
            'penalty' : ['l2','l1'],  
            'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            }

CV_log2_clf = GridSearchCV(estimator = log2_clf, param_grid = param_grid , scoring = 'recall', verbose = 1, n_jobs = -1)
CV_log2_clf.fit(X_train, y_train)

best_parameters = CV_log2_clf.best_params_
print('The best parameters for using this model is', best_parameters)
CV_log2_clf = LogisticRegression(C = best_parameters['C'], 
                                 penalty = best_parameters['penalty'], 
                                 random_state = random_state)


CV_log2_clf.fit(X_train, y_train)

y_pred = CV_log2_clf.predict(X_test)
y_score = CV_log2_clf.decision_function(X_test)
# Confusion maxtrix & metrics
cm = confusion_matrix(y_test, y_pred)
class_names = [0,1]
#Voting Classifier
voting_clf = VotingClassifier (
        estimators = [('log1', CV_log_clf), ('log_2', CV_log2_clf)],
                     voting='soft', weights = [1, 1])
    
voting_clf.fit(X_train,y_train)

y_pred = voting_clf.predict(X_test)
y_score = voting_clf.predict_proba(X_test)[:,1]

# Confusion maxtrix
cm = confusion_matrix(y_test, y_pred)
class_names = [0,1]
show_metrics()
def GridSearchModel(X, Y, model, parameters, cv):
    CV_model = GridSearchCV(estimator = model, param_grid = parameters, cv = cv)
    CV_model.fit(X, Y)
    CV_model.cv_results_
    print("Best Score:", CV_model.best_score_," / Best parameters:", CV_model.best_params_)
    
# Learning curve
def LearningCurve(X, y, model, cv, train_sizes):

    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv = cv, n_jobs = 4, 
                                                            train_sizes = train_sizes)

    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std  = np.std(train_scores, axis = 1)
    test_scores_mean  = np.mean(test_scores, axis = 1)
    test_scores_std   = np.std(test_scores, axis = 1)
    
    train_Error_mean = np.mean(1- train_scores, axis = 1)
    train_Error_std  = np.std(1 - train_scores, axis = 1)
    test_Error_mean  = np.mean(1 - test_scores, axis = 1)
    test_Error_std   = np.std(1 - test_scores, axis = 1)

    Scores_mean = np.mean(train_scores_mean)
    Scores_std = np.mean(train_scores_std)
    
    _, y_pred, Accuracy, Error, precision, recall, f1score = ApplyModel(X, y, model)
    
    plt.figure(figsize = (16,4))
    plt.subplot(1,2,1)
    ax1 = Confuse(y, y_pred, classes)
    plt.subplot(1,2,2)
    plt.fill_between(train_sizes, train_Error_mean - train_Error_std,train_Error_mean + train_Error_std, alpha = 0.1,
                     color = "r")
    plt.fill_between(train_sizes, test_Error_mean - test_Error_std, test_Error_mean + test_Error_std, alpha = 0.1, color = "g")
    plt.plot(train_sizes, train_Error_mean, 'o-', color = "r",label = "Training Error")
    plt.plot(train_sizes, test_Error_mean, 'o-', color = "g",label = "Cross-validation Error")
    plt.legend(loc = "best")
    plt.grid(True)
     
    return (model, Scores_mean, Scores_std )

def ApplyModel(X, y, model):
    
    model.fit(X, y)
    y_pred  = model.predict(X)

    Accuracy = round(np.median(cross_val_score(model, X, y, cv = cv)),2)*100
 
    Error   = 1 - Accuracy
    
    precision = precision_score(y_train, y_pred) * 100
    recall = recall_score(y_train, y_pred) * 100
    f1score = f1_score(y_train, y_pred) * 100
    
    return (model, y_pred, Accuracy, Error, precision, recall, f1score)  
    
def Confuse(y, y_pred, classes):
    cnf_matrix = confusion_matrix(y, y_pred)
    
    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis = 1)[:, np.newaxis]
    c_train = pd.DataFrame(cnf_matrix, index = classes, columns = classes)  

    ax = sns.heatmap(c_train, annot = True, cmap = cmap, square = True, cbar = False, 
                          fmt = '.2f', annot_kws = {"size": 20})
    return(ax, c_train)

def PrintResults(model, X, y, title, limitleafs=2):
    
    model, y_pred, Accuracy, Error, precision, recall, f1score = ApplyModel(X, y, model)
    
    _, Score_mean, Score_std = LearningCurve(X, y, model, cv, train_size)
    Score_mean, Score_std = Score_mean*100, Score_std*100
    
    
    print('Scoring Accuracy: %.2f %%'%(Accuracy))
    print('Scoring Mean: %.2f %%'%(Score_mean))
    print('Scoring Standard Deviation: %.4f %%'%(Score_std))
    print("Precision: %.2f %%"%(precision))
    print("Recall: %.2f %%"%(recall))
    print('f1-score: %.2f %%'%(f1score))
    print('Limited leafs:' + str(limitleafs))
    print(' ')
    
    
    Summary = pd.DataFrame({'Model': title,
                       'Accuracy': Accuracy, 
                       'Score Mean': Score_mean, 
                       'Score St Dv': Score_std, 
                       'Precision': precision, 
                       'Recall': recall, 
                       'F1-Score': f1score,
                       'Limited leafs': limitleafs}, index = [0])
    return (model, Summary)
train_size = np.linspace(.1, 1.0, 15)
cv = ShuffleSplit(n_splits = 100, test_size = 0.25, random_state = 0)

classes = ['Dead','Survived']
cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
model = DecisionTreeClassifier()
model,Summary_DT = PrintResults(model, X_train, y_train, 'DT')

#model = DecisionTreeClassifier(criterion='entropy',min_samples_split=5)
#model,Summary_DT = PrintResults(model, X_train, y_train, 'DT')
accu = []
prec= []
rec = []
tre = []
for i in range(2,25):
    model = DecisionTreeClassifier(criterion='entropy',min_samples_leaf=i)
    model,Summary_DT = PrintResults(model, X_train, y_train, 'DT',i)
    prec.append(float(Summary_DT['Precision']))
    rec.append(float(Summary_DT['Recall']))
    tre.append(float(Summary_DT['Limited leafs']))
    accu.append(float(Summary_DT['Accuracy']))
print(len(accu),len(prec),len(rec),len(tre))

data = pd.DataFrame(
    {'Limited leafs': tre,
     'Precision': prec,
     'Recall': rec,
     'Accuracy': accu
    })
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='Limited leafs',y='Precision',data=data,color='lime',alpha=0.8)
sns.pointplot(x='Limited leafs',y='Recall',data=data,color='red',alpha=0.8)
sns.pointplot(x='Limited leafs',y='Accuracy',data=data,color='blue',alpha=0.8)
plt.title('PRA',fontsize = 20,color='blue')
plt.grid()

    
    

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
param_dist = {"max_depth": [3, 5],
              "max_features": sp_randint(1, 6),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11)
             }
# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(model, param_distributions=param_dist,
                                   n_iter=n_iter_search)
random_search.fit(X_train,y_train)
report(random_search.cv_results_)
model = DecisionTreeClassifier(max_depth=5,max_features=4,min_samples_leaf=4,min_samples_split=3)
model,Summary_DT = PrintResults(model, X_train, y_train, 'DT')
