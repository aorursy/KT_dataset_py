from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, cohen_kappa_score,  roc_curve
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
def calculate_metrics(TP, FP, TN,FN):
    '''
    Function to calculate performance metrics of classifier
    Parameters
    ----------
    TP : int
        True positives value
    FP : int
        False positives value
    TN : int
        True negatives value
    FN : int
        False negatives value
    
    Returns
    -------
    Sensitivity, Specificity, Precision, Accuracy, F1_Score
    '''     
    try:
        Sensitivity = round(TP*1.0/(TP+FN),2)
    except:
        Sensitivity = np.Inf
    try:
        Specificity = round(TN*1.0/(TN+FP),2)
    except:
        Specificity = np.Inf
    try:
        Precision = round(TP*1.0/(TP+FP),2)
    except:
        Precision = np.Inf
    try:
        Accuracy =  round((TP+TN)*1.0/(TP+TN+FP+FN),2)
    except:
        Accuracy = np.Inf
    try:
        F1_Score = round(2/((1/Sensitivity)+(1/Specificity)),2)
    except:
        F1_Score = np.Inf
    return Sensitivity, Specificity, Precision, Accuracy, F1_Score
def threshold_calculations(df):
    ''' Function to generate calcualte True/False Positives, True/False Negatives and compute performance metrics at various thresholds
        Parameters
        ---------
        df: dataframe with original class, preedicted class, 0_class_probability, 1_class_probability
        Returns
        -------
        dictionary  with True/False postive, True/False Negatives, sensitivity, specificity, preicision, accuracy at different probability values between 0 to 1
     '''   
    Results = dict()
    for x in np.arange(0,1.05,0.1):
        temp1 = df[df['Probability_0']>x]
        temp2 = df[df['Probability_0']<=x]
        temp1['Prediction'] =0
        temp2['Prediction']= 1
        TN=  temp1[temp1['Original']==temp1['Prediction']].shape[0]
        FP=  temp1[temp1['Original']!=temp1['Prediction']].shape[0]
        TP=  temp2[temp2['Original']==temp2['Prediction']].shape[0]
        FN= temp2[temp2['Original']!=temp2['Prediction']].shape[0]
        Sensitivity, Specificity, Precision, Accuracy, F1_Score = calculate_metrics(TP,FP,TN,FN)
        Results[x] = [TP, FP, TN, FN ,Sensitivity, Specificity, Precision,Accuracy]
    return Results
        

def print_confusion_matrix_at_threshold(dataframe_predictions, threshold):
    ''' Function to print confustion matrix at given probability threshold
    Parameters:
    --------
    dataframe_predictions: pandas dataframe with predictions from model
    threshold: probability threshold
    Returns:
    --------
    Confusion matrix: cross tab from pandas dataframe
    '''
    Positives = dataframe_predictions[dataframe_predictions['Probability_1']>=threshold]
    Positives.Prediction=1
    Negatives = dataframe_predictions[dataframe_predictions['Probability_1']<threshold]
    Negatives.Prediction=0
    tmp = pd.concat([Positives,Negatives],axis=0)
    print(pd.crosstab(tmp.Original, tmp.Prediction))

def plot_threshold_graph(Recall_list, Specificity_list,Precision_list ,Accuracy_list):
    ''' Function to plot performance metrics at various thresholds
    Parameters:
    ---------
    Recall_list : List
        List with mean of recall values at different probability values across cross validation folds
    Specificity_list: List
        List with specifity values at different probability values across cross validation folds
    Precision: List
        List with Precision values at different probability values across cross validation folds
    Accuracy_list: List
        List with Accuracy values at different probability values across cross validation folds

    Returns:
    --------
    Plot of mean of Recall, Specificity, Precision and Accuracy 
    '''
    
    plt.figure(figsize=(15,5))
    plt.xticks(np.arange(0,1.05,.1))
    plt.plot(np.arange(0,1.05,.1),Recall_list[::-1],label = 'Recall')
    plt.plot(np.arange(0,1.05,.1),Specificity_list[::-1],label='Specificity')
    plt.plot(np.arange(0,1.05,.1),Precision_list[::-1], label = "Precision")
    plt.plot(np.arange(0,1.05,.1),Accuracy_list[::-1],label='Accuracy')
    plt.legend()
    plt.show()
def fit_model(Model,features,target):
    '''Fits a model given features and target
    Parameters
    ----------
    Model - scikit learn machine learning model
    features -  array of features values
    target - array of target class
    Returns
    ------
    Pandas Dataframe with Original calss, Predicted class, zero class probability, one class probability  
    '''
    Model.fit(features,target)
    temp_predictions = NewModel.predict(features)
    temp_probability = NewModel.predict_proba(features)
    temp_dict = {'Original': target, 'Prediction': temp_predictions , 'Probability_0':temp_probability[:,0],'Probability_1':temp_probability[:,1]}
    temp_df = pd.DataFrame(data=temp_dict)
    return(temp_df)

    
def compute_metrics(Model, features, target, cv_folds=None):
    ''' Computes mean of recall, specificity, precision and accuracy across cross validation folds, if provided. other wise
        returns recall, specificity, precision and accuracy
    Parameters
    ---------
    Model - Scikit learn model
    features - array of features values / or pandas data frame with features as columns
    target - array of class values/ or pandas data series with class values
    cv_folds - cross validation folds
    Returns
    ------
    
    '''
    Results_cache= []
    Recall_list =[]
    Specificity_list = []
    Precision_list=[]
    Accuracy_list =[]

    if(cv_folds is not None):
        for train_index, test_index in cv_folds:
            if(isinstance(features,np.ndarray)):
                tempX_train, tempX_test = features[train_index], features[test_index]
                tempy_train, tempy_test = target[train_index], target[test_index]
            elif(isinstance(features,pd.DataFrame)):
                tempX_train, tempX_test = features.values[train_index], features.values[test_index]
                tempy_train, tempy_test = target.values[train_index], target.values[test_index]
            temp_df = fit_model(Model,tempX_train,tempy_train)
            temp_results = threshold_calculations(temp_df)
            Results_cache.append(temp_results)
    else: 
            temp_df = fit_model(Model,features,target)
            temp_results = threshold_calculations(temp_df)
            Results_cache.append(temp_results)


    for x in np.arange(0,1.05,0.1):
        temp_list = []
        for l in range(len(Results_cache)):
            temp_list.append(Results_cache[l][x])
        tmp= map(np.mean, zip(*temp_list))
        tmp2= [ round(elem, 2) for elem in tmp ]
        Recall_list.append(tmp2[4])
        Specificity_list.append(tmp2[5])
        Precision_list.append(tmp2[6])
        Accuracy_list.append(tmp2[7])

    return Recall_list, Specificity_list,Precision_list ,Accuracy_list
def print_performance_metrics(actual,preds,pred_probs=[]):
    '''Prints performance metrics given actual , predicted values. Predicted probability values are optional
    Parameters:
    ----------
    actual: array of y-values
    preds: array of predicted values
    pred_probs: array of predicted probabilityes
    Returns:
    ------
    None. Prints confusion matrixs and ROC curve
    
    '''
    print(confusion_matrix(actual,preds))
    print(classification_report(actual,preds))
    print("Cohen Kappa ", cohen_kappa_score(actual,preds))
    if(len(pred_probs)==0):
        fpr, tpr, thresholds = roc_curve(actual, preds)
        plt.plot(fpr, tpr,label='ROC curve (area = %0.2f)' %auc(fpr, tpr))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.show()
        print("AUC - ",auc(fpr, tpr))
    
    else:
        fpr, tpr,thresholds = roc_curve(actual, pred_probs)
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' %auc(fpr, tpr))
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")    
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        print("AUC - ",auc(fpr, tpr))
        # create the axis of thresholds (scores)
        ax2 = plt.gca().twinx()
        thresholds[thresholds>1]= thresholds[thresholds>1]-1
        ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r')
        ax2.set_ylabel('Threshold',color='r')
        ax2.set_ylim([thresholds[-1],thresholds[0]])
        ax2.set_xlim([fpr[0],fpr[-1]])
        plt.show()
def check_model_performance(MLModel,xt,yt,xv,yv):
    ''' prints performance metrics for training and validation datasets'''
    print("--Train performance -- ")
    preds = MLModel.predict(xt)
    pred_probs=MLModel.predict_proba(xt)
    print_performance_metrics(yt,preds,pred_probs[:,1])
    print("--Validation performance -- ")
    preds=MLModel.predict(xv)
    pred_probs=MLModel.predict_proba(xv)
    print_performance_metrics(yv,preds,pred_probs[:,1])
    
def create_dataset(n_samples=1000, weights=(0.01, 0.01, 0.98), n_classes=3,class_sep=0.8, n_clusters=1, n_features=2):
    ''' returns classification dataset '''
    return make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=2, n_redundant=0, n_repeated=0,
                               n_classes=n_classes,
                               n_clusters_per_class=n_clusters,
                               weights=list(weights),
                               class_sep=class_sep, random_state=0)
#Creating data set with class imbalanced classes in 90-10 ratio with 100 features 
X, y = create_dataset(n_samples=10000, weights=(0.9, 0.1),n_classes=2, n_features=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0,stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0,stratify=y_train)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, X_valid.shape, y_valid.shape)
#Fitting a random forest model with class weight option invoked

rfmodel = RandomForestClassifier(n_jobs=-1,n_estimators=10, class_weight ='balanced')
rfmodel.fit(X_train, y_train)
check_model_performance(rfmodel,X_train,y_train,X_valid,y_valid)
kf =StratifiedKFold(y_train, n_folds=3,random_state=42)
# NewModel = RandomForestClassifier(n_jobs=-1, oob_score=True,class_weight='balanced', max_features=0.5,verbose=1)
NewModel= rfmodel
Recall_list, Specificity_list,Precision_list ,Accuracy_list = compute_metrics(rfmodel,features=X_train, target=y,cv_folds=kf)
#Pritining performance metrics at various probability thresholds
plot_threshold_graph(Recall_list, Specificity_list,Precision_list ,Accuracy_list)
prediction_df = fit_model(rfmodel,X_train,y_train)
print_confusion_matrix_at_threshold(prediction_df,0.38)
print_confusion_matrix_at_threshold(prediction_df,0.5)
# Checking performance at 0.9
print_confusion_matrix_at_threshold(prediction_df,0.9)
prediction_df = fit_model(rfmodel,X_valid,y_valid)
print_confusion_matrix_at_threshold(prediction_df,0.5)
print_confusion_matrix_at_threshold(prediction_df,0.38)
