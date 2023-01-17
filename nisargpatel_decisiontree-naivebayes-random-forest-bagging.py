#import libraries
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import os
#Load the data
data = pd.read_csv("../input/Spotify_Combined.csv",index_col='index')

# Any results you write to the current directory are saved as output.
#rename the target column name
data = data.rename(columns={'mood(s)':'moods'})
print("Number of records/cases in the dataset: ",data.shape[0])
print("Number of attributes/features in the dataset: ",data.shape[1])
data.describe(include = "all").T  # T is for transpose and include is to see all features.
#Check distributions of numerical attributes
def plot_histogram(series,nbins=50):
    plt.hist(series,bins=nbins,histtype='bar')
    plt.show
plot_histogram(data['duration_ms'])
#convert duration_ms in minutes to get clearer picture
ms = data['duration_ms']
sec = ms/1000
minute = sec/60
plot_histogram(minute,nbins=30)
idx = data.index[data['moods']=='dinner, workout']
idx2= data.index[data['moods']=='dinner, party']
data_spotify = data.drop(idx)
data_spotify = data_spotify.drop(idx2)
#remove non-significant columns
data_spotify.drop(['id','name','uri','artist'],axis=1,inplace=True)
def plot_corr_matrix(df,size=10):
    f, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax, vmin = 0)
plot_corr_matrix(data_spotify)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#from imblearn.over_sampling import SMOTE
#before train test split, let us split our target variable from attributes
colnames = data_spotify.columns.values
y = data_spotify['moods']
x = data_spotify[colnames[:-1]]
#let us split the data now using train_test_split data
xtrain,xtest,ytrain, ytest = train_test_split(x,y,test_size = 0.33,stratify = y,random_state=10)
xtrain.shape,xtest.shape,ytrain.shape,ytest.shape
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix   
from sklearn.cross_validation import cross_val_score

import matplotlib.pyplot as plt
import numpy as np
import itertools
def plot_confusion_matrix(cm,target_names,title='Confusion matrix'):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),\
                     horizontalalignment="center",\
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
def calc_sensitivity_specificity(cm,class_names=None):
    '''
    Calculate sensitivity and specificity for each class and return the mean sensitivity and specificity.
    '''
    
    #variables initialization
    sensitivity = []
    specificity = []
    
    for Id,_ in enumerate(class_names) if class_names!=None else enumerate(range(cm.shape[0])):
        TP = cm[Id,Id]
        TN = np.trace(cm) - TP
        totalPos = np.sum(cm[Id:Id+1]) #TP+FN
        totalNeg = TN + np.sum(cm[:,Id:Id+1]) #TN+FP
        sensitivity.append(TP/totalPos)
        specificity.append(TN/totalNeg)
    #print('sensitivity:{}\nspecificity:{}'.format(sensitivity,specificity))
    return np.mean(sensitivity),np.mean(specificity)
    
#fucntion to train the model, predict the outcome and evaluate the model on test
def fit_and_evaluate(x_train,y_train,x_test,y_test,model,class_names=None,show_cm =False):
    
    # Train the model
    model.fit(x,y)
    # cross_val_score(model, x, y, cv=cv)
    
    # Make predictions and evalute
    model_pred = model.predict(x_test)
    
    #calculate confusion matrix
    cm = confusion_matrix(y_true=y_test,y_pred=model_pred,labels=class_names)
    model_accuracy = np.trace(cm)/float(np.sum(cm))
    misclass = 1 - model_accuracy
    
    #calculate sensitivity and specificity for each class
    mean_sensitivity,mean_specificity = calc_sensitivity_specificity(cm,class_names)
    
    #plot confusion matrix
    if show_cm:
        plot_confusion_matrix(cm,class_names)
    
    #return accuracy,sensitivity and specificity
    return model_accuracy,mean_sensitivity,mean_specificity, misclass
dt = DecisionTreeClassifier(random_state = 10)
class_names = ['dinner','party','party, workout','sleep','workout']
#check performance of model on train dataset
accuracy,sensitivity,specificity,misclass = fit_and_evaluate(xtrain,ytrain,xtrain,ytrain,dt,class_names= class_names)
print('Model accuracy: {:.4f}, Mean-Sensitivity: {:.4f}, Mean-Specificity: {:.4f}, Error: {:.4f}'.format(accuracy,sensitivity,specificity,misclass))
#check performance on test dataset
accuracy,sensitivity,specificity, misclass = fit_and_evaluate(xtrain,ytrain,xtest,ytest,dt,class_names=class_names)
print('Model accuracy: {:.4f}, Mean-Sensitivity: {:.4f}, Mean-Specificity: {:.4f}, Error: {:.4f}'.format(accuracy,sensitivity,specificity,misclass))
#create parameters to test:
param_grid = {'criterion':['gini','entropy'],
              'max_depth':[5,10,15,30,40],
              'min_sample_leaf': [1,5,10,20],
              'max_features': ['sqrt','log2',None],
              'random_state': [10]
}
def run_model_evaluations(param_grid,class_names):
    '''
    Run model with different parameter combinations as provided in the arguments. 
    Arguments:
        param_grid: [a dict object] a dictionary with parameters to change. Currenty only excepting four parameters criterion,max_depth,min_sampels_leaf,max_features
        class_names: [list of classes] list of target class names
    Returns:
        A dictionary object with parameter combinations and their results for train and test sets. 
        Results include accuracy, sensitivity and specificity 
    Limitations:
        Currently, this function is limited to
        1. DecisionTreeClassifier() and may need to update for other models.
        2. For DT only four parameters criterion,max_depth,min_sampels_leaf,max_features are expected in param_grid
    
    '''
    criteria = param_grid['criterion']
    max_depth = param_grid['max_depth']
    min_leaf_nodes=param_grid['min_sample_leaf']
    max_features = param_grid['max_features']
    if class_names!=None:
        class_names = ['dinner','party','party, workout','sleep','workout']
    #else:
    #    raise ValueError('Please provide a list of classes in target variable.')

    result_dict = {'criterion':[],'max_depth':[],'min_sample_leaf':[],'max_features':[],\
                   'accuracy':[],'sensitivity':[],'specificity':[],'test_results':[]}
    for param_set in itertools.product(criteria,max_depth,min_leaf_nodes,max_features):
        crit = param_set[0]
        depth = param_set[1]
        msl = param_set[2]
        mf = param_set[3]
        #create model with set of parameters
        model = DecisionTreeClassifier(criterion=crit,max_depth=depth,\
                                       min_samples_leaf=msl,max_features=mf)


        #calculate model performance for both train and test data
        for testset in [False,True]:
            if testset:
                accuracy,sensitivity,specificity,misclass=fit_and_evaluate(xtrain,ytrain,xtest,ytest,model,class_names,show_cm=False)
            else:
                accuracy,sensitivity,specificity,misclass=fit_and_evaluate(xtrain,ytrain,xtrain,ytrain,model,class_names,show_cm=False)

            #store parameters in results_dict
            result_dict['criterion'].append(crit)
            result_dict['max_depth'].append(depth)
            result_dict['min_sample_leaf'].append(msl)
            result_dict['max_features'].append(mf)
            #store model evaluation results in results dictionary
            result_dict['accuracy'].append(accuracy)
            result_dict['sensitivity'].append(sensitivity)
            result_dict['specificity'].append(specificity)
            result_dict['test_results'].append(testset)
    
    #return result dictionary
    return result_dict
class_names = ['dinner','party','party, workout','sleep','workout']
result = run_model_evaluations(param_grid=param_grid,class_names=class_names)
result_df = pd.DataFrame(result)
result_df[result_df.test_results==True].\
    sort_values(['accuracy','sensitivity','specificity','max_depth',\
                 'min_sample_leaf'],ascending=False).head(13)
dtclf = DecisionTreeClassifier(criterion='entropy',max_depth=15,max_features=None,min_samples_leaf=1,random_state=10)
#use best parameters to report misclassification matrix, accuracy, sensitivity and specificity on test data
accuracy,sensitivity,specificity,misclass = fit_and_evaluate(xtrain,ytrain,xtest,ytest,dtclf,class_names)
print('Model accuracy: {:.4f}, Mean-Sensitivity: {:.4f}, Mean-Specificity: {:.4f}, Error: {:.4f}'.format(accuracy,sensitivity,specificity,misclass))
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
#Scale the attributes
scaler = MinMaxScaler()
xtrain_scaled = scaler.fit(xtrain)
xtest_scaled = scaler.fit(xtest)
# Misclassification matrix, accuracy, sensitivity, and specificity on training data
accuracy,sensitivity,specificity,misclass = fit_and_evaluate(xtrain,ytrain,xtrain,ytrain,nb,class_names)
print('Model accuracy: {:.4f}, Mean-Sensitivity: {:.4f}, Mean-Specificity: {:.4f}, Error: {:.4f}'.format(accuracy,sensitivity,specificity,misclass))
# Misclassification matrix, accuracy, sensitivity, and specificity on testing data
accuracy,sensitivity,specificity,misclass=fit_and_evaluate(xtrain,ytrain,xtest,ytest,nb,class_names)
print('Model accuracy: {:.4f}, Mean-Sensitivity: {:.4f}, Mean-Specificity: {:.4f}, Error: {:.4f}'.format(accuracy,sensitivity,specificity,misclass))
from sklearn.model_selection import StratifiedShuffleSplit
from scipy import stats
dtf = DecisionTreeClassifier(criterion='entropy',max_depth=15,min_samples_leaf=1,random_state=10)
nb1 = GaussianNB()
tot_accuracy_dt= 0;tot_accuracy_nb= 0
tot_accuracy_dtt= 0;tot_accuracy_nbt= 0
tot_error_dt= 0;tot_error_nb = 0
tot_error_dtt= 0;tot_error_nbt = 0
acc_arr_dt = np.array([])
acc_arr_nb = np.array([])
acc_arr_dtt = np.array([])
acc_arr_nbt = np.array([])

np.float64(tot_accuracy_dt),np.float64(tot_accuracy_nb),np.float64(tot_error_dt),np.float64(tot_error_nb)

for i in range(30):
    xtrain,xtest,ytrain, ytest = train_test_split(x,y,test_size = 0.33,stratify = y,random_state=i)
    
    # Decision Tree
    accuracy,sensitivity,specificity, misclass =fit_and_evaluate(xtrain,ytrain,xtrain,ytrain,dtf,class_names)
    accuracy1,sensitivity1,specificity1, misclass1 =fit_and_evaluate(xtrain,ytrain,xtest,ytest,dtf,class_names)
        # cal accuracy for training and testing
    acc_arr_dt = np.append(acc_arr_dt,accuracy)
    acc_arr_dtt = np.append(acc_arr_dtt,accuracy)
    tot_accuracy_dt = tot_accuracy_dt+accuracy 
    tot_accuracy_dtt = tot_accuracy_dtt+accuracy1 
    tot_error_dt = tot_error_dt+misclass
    tot_error_dtt = tot_error_dtt+misclass1
    
    # Naive Bayes
    accuracy3,sensitivity3,specificity3, misclass3=fit_and_evaluate(xtrain,ytrain,xtrain,ytrain,nb1,class_names)
    accuracy4,sensitivity4,specificity4, misclass4=fit_and_evaluate(xtrain,ytrain,xtest,ytest,nb1,class_names)
    acc_arr_nb = np.append(acc_arr_nb,accuracy3)
    acc_arr_nbt = np.append(acc_arr_nbt,accuracy4)
    tot_accuracy_nb = tot_accuracy_nb+accuracy3
    tot_accuracy_nbt = tot_accuracy_nbt+accuracy4
    tot_error_nb = tot_error_nb+misclass3 
    tot_error_nbt = tot_error_nbt+misclass4 
    
print("DT accuracy array:",acc_arr_dt)
print("NB accuracy array",acc_arr_nb,"\n")

print("Average Training Accuracy/Means for Decision Tree: {:.4f}" .format(tot_accuracy_dt/30))
print("Average Test Accuracy/Means for Decision Tree: {:.4f}" .format(tot_accuracy_dtt/30))
print("Average Training Accuracy/Means for Naive Bayes: {:.4f}" .format(tot_accuracy_nb/30))
print("Average Test Accuracy/Means for Naive Bayes: {:.4f}" .format(tot_accuracy_nbt/30), "\n")

print("Average Training Error for Decision Tree: {:.4f}" .format(tot_error_dt/30))
print("Average Test Error for Decision Tree: {:.4f}" .format(tot_error_dtt/30))
print("Average Training Error for Naive Bayes: {:.4f}" .format(tot_error_nb/30))
print("Average Test Error for Naive Bayes: {:.4f}" .format(tot_error_nbt/30),"\n")
var_dt = np.var(acc_arr_dt)
var_dtt = np.var(acc_arr_dtt)
var_nb = np.var(acc_arr_nb)
var_nbt = np.var(acc_arr_nbt)
print("Decision Tree Training Accuracy Variance is:",var_dt)
print("Decision Tree Test Accuracy Variance is:",var_dtt)
print("Naive Bayes Training Accuracy Variance is:",var_nb)
print("Naive Bayes Test Accuracy Variance is:",var_nbt,"\n")
#t_bounds = t.interval(0.95, len(data) - 1)
#ci = [mean + critval * stddev / sqrt(len(data)) for critval in t_bounds]
std_dt = np.std(acc_arr_dt)
std_dtt = np.std(acc_arr_dtt)
std_nb = np.std(acc_arr_nb)
std_nbt = np.std(acc_arr_nbt)

ci_dt = 1.95*(std_dt/5.47)
ci_dtt = 1.95*(std_dtt/5.47)
ci_nb = 1.95*(std_nb/5.47)
ci_nbt = 1.95*(std_nbt/5.47)

print("95% CI for Trainging data:  {:.4f}(-+){:.4f}" .format((tot_accuracy_dt/30),ci_dt))
print("95% CI for Testing data:  {:.4f}(-+){:.4f}" .format((tot_accuracy_dt/30),ci_dtt))
print("95% CI for Trainging data:  {:.4f}(-+){:.4f}" .format((tot_accuracy_nb/30),ci_nb))
print("95% CI for Testing data:  {:.4f}(-+){:.4f}" .format((tot_accuracy_nb/30),ci_nbt),"\n")
tt = stats.ttest_rel(acc_arr_dt,acc_arr_nb)
ttt = stats.ttest_rel(acc_arr_dtt,acc_arr_nbt)
print ("paired t_test between decision tree and naive bayes training data:\n",tt,"\n")
print ("paired t_test between decision tree and naive bayes testing data:\n",ttt)
print("\n Note: statistic is the t -test value")
test = [0.75,0.65,0.55,0.45,0.35,0.25,0.15]
acc_err_dt = [] # substitution_dt
acc_err_dtt = [] # generalization_dt
acc_err_nb = [] # substitution_nb
acc_err_nbt = [] # generalization_nb

for j in range(7):
        xtrain,xtest,ytrain, ytest = train_test_split(x,y,test_size = test[j],stratify = y,random_state=30)
        # Decision Tree
        accuracy,sensitivity,specificity, misclass =fit_and_evaluate(xtrain,ytrain,xtrain,ytrain,dtf,class_names)
        acc_err_dt = np.append(misclass,acc_err_dt)
        accuracy,sensitivity,specificity, misclass =fit_and_evaluate(xtrain,ytrain,xtest,ytest,dtf,class_names)
        acc_err_dtt = np.append(misclass,acc_err_dtt)
        # Naive Bayes
        accuracy,sensitivity,specificity, misclass=fit_and_evaluate(xtrain,ytrain,xtrain,ytrain,nb1,class_names)
        acc_err_nb = np.append(misclass,acc_err_nb)
        accuracy,sensitivity,specificity, misclass=fit_and_evaluate(xtrain,ytrain,xtest,ytest,nb1,class_names)
        acc_err_nbt = np.append(misclass,acc_err_nbt)

train = [25,35,45,55,65,75,85]
plt.plot(train, acc_err_dt, color='red')
plt.plot(train, acc_err_dtt, color='blue')
plt.plot(train, acc_err_nb, color='black')
plt.plot(train, acc_err_nbt, color='green')
plt.xlabel('Training Ratio')
plt.ylabel('Error in training and test')
plt.title('Substitution and Generalization Error')
plt.legend()

from sklearn.metrics import roc_curve, auc
data_spotify['moods'].value_counts()
df_roc_trg= data_spotify[data_spotify['moods'] == 'dinner']
df_roc_rst = data_spotify[(data_spotify['moods'] != 'dinner')]
df_roc_trg['moods'] = 0
df_roc_rst['moods'] = 1
# combining the  two target together into one dataframe
df = [df_roc_trg,df_roc_rst]
df_roc = pd.concat(df)
df_roc['moods'].value_counts()
#df_roc.drop('index',axis=1)
colnames = df_roc.columns.values
y_roc = df_roc['moods']
X_roc = df_roc[colnames[:-1]]
X_train,X_test,y_train,y_test = train_test_split(X_roc,y_roc,test_size=0.33,stratify = y,random_state=10)
nb1.fit(X_train,y_train)
y_score_test = nb1.predict(X_test)
y_score_train = nb1.predict(X_train)
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = metrics.roc_curve(y_train,y_score_train)
fprt, tprt, thresholdst = metrics.roc_curve(y_test,y_score_test)
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='green', label='Train ROC curve (area = %0.2f)' % auc(fpr, tpr))
plt.plot(fprt, tprt, color='orange', label='Test ROC curve (area = %0.2f)' % auc(fprt, tprt))
plt.plot([0, 1], [0, 1], linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=14)
plt.ylabel('True Positive Rate (TPR)', fontsize=14)
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()
#Lift Charts
import scikitplot as skplt
predicted_lc_train = nb.predict_proba(X_train)
predicted_lc_test = nb.predict_proba(X_test)

skplt.metrics.plot_lift_curve(y_train,predicted_lc_train)
skplt.metrics.plot_lift_curve(y_test,predicted_lc_test)
plt.show()
from time import time
from sklearn.metrics import f1_score

wdbc = pd.read_csv("../input/wdbc.csv")

pd.set_option('display.max_columns', 500)
wdbc = wdbc.drop(['id'],axis=1)
wdbc.head()
# malignant:1 and benign:0
wdbc['diagnosis'] = wdbc['diagnosis'].map({'M': 1,'B': 0})
wdbc.head()

# no. of cases
cases = wdbc.shape[0]
# no. of features
features = wdbc.shape[1]-1
# no. of malignant cases
n_M = wdbc[wdbc['diagnosis']==1].shape[0]
# no. of benign cases
n_B = wdbc[wdbc['diagnosis']==0].shape[0]

print ("Total number of cases: {}".format(cases))
print ("Number of features: {}".format(features))
print ("malignant cases: {}".format(n_M))
print ("benign cases: {}".format(n_B))

#before train test split, let us split our target variable from attributes
col_names = wdbc.columns.values
X = wdbc[col_names[1:31]]
Y = wdbc['diagnosis']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33,stratify = Y)
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
cart=DecisionTreeClassifier()
num_trees = 50
ensemble_clfs = [("BaggingClassifier, max_features=None",
BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=42, oob_score=True))]

#("BaggingClassifier, max_features='log2'",
# BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=42, max_features="log2", oob_score=True)),
#("BaggingClassifier, max_features='None'",
#BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=42, max_features=None, oob_score=True)) ]
# Map a classifier name to a list of (<n_of trees>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
min_estimators = 15
max_estimators = 200

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(X,Y)

        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_tree")
plt.ylabel("error rate")
plt.legend(loc="upper right")
plt.show()
