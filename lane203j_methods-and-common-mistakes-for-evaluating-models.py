import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# functions needed for pr_auc_score()
from sklearn.metrics import auc, precision_recall_curve

# functions needed for imbalanced_cross_validation_score()
from sklearn.model_selection import StratifiedKFold

# sampler objects
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# Classification models to compare
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
# load data
df = pd.read_csv("../input/creditcard.csv")
A  = df.values
x  = A[:,:-1]     
y  = A[:,-1]
def pr_auc_score(clf, x, y):
    '''
        This function computes area under the precision-recall curve. 
    '''
      
    precisions, recalls,_ = precision_recall_curve(y, clf.predict_proba(x)[:,1], pos_label=1)
    
    return auc(recalls, precisions)
def imbalanced_cross_validation_score(clf, x, y, cv, scoring, sampler):
    '''
        This function computes the cross-validation score of a given 
        classifier using a choice of sampling function to mitigate 
        the class imbalance, and stratified k-fold sampling.
        
        The first five arguments are the same as 
        sklearn.model_selection.cross_val_score.
        
        - clf.predict_proba(x) returns class label probabilities
        - clf.fit(x,y) trains the model
        
        - x = data
        
        - y = labels
        
        - cv = the number of folds in the cross validation
        
        - scoring(classifier, x, y) returns a float
        
        The last argument is a choice of random sampler: an object 
        similar to the sampler objects available from the python 
        package imbalanced-learn. In particular, this 
        object needs to have the method:
        
        sampler.fit_sample(x,y)
        
        See http://contrib.scikit-learn.org/imbalanced-learn/
        for more details and examples of other sampling objects 
        available.  
    
    '''
    
    cv_score = 0.
    train_score = 0.
    test_score = 0.
    
    # stratified k-fold creates folds with the same ratio of positive 
    # and negative samples as the entire dataset.
    
    skf = StratifiedKFold(n_splits=cv, random_state=0, shuffle=False)
    
    for train_idx, test_idx in skf.split(x,y):
        
        xfold_train_sampled, yfold_train_sampled = sampler.fit_sample(x[train_idx],y[train_idx])
        clf.fit(xfold_train_sampled, yfold_train_sampled)
        
        train_score = scoring(clf, xfold_train_sampled, yfold_train_sampled)
        test_score  = scoring(clf, x[test_idx], y[test_idx])
        
        print("Train AUPRC: %.2f Test AUPRC: %.2f"%(train_score,test_score))

        cv_score += test_score
        
    return cv_score/cv
cv = 5  

RegressionModel    = LogisticRegression()

# here are some other models you could try. You can also try grid searching their hyperparameters
RandomForrestModel = RandomForestClassifier()
ExtraTreesModel    = ExtraTreesClassifier()
AdaBoostModel      = AdaBoostClassifier()
# Logistic regression score with Random Over-sampling
print("Random over-sampling")
score = imbalanced_cross_validation_score(RegressionModel, x, y, cv, pr_auc_score, RandomOverSampler())
print("Cross-validated AUPRC score: %.2f"%score)

# Logistic regression score with SMOTE
print("SMOTE over-sampling")
score = imbalanced_cross_validation_score(RegressionModel, x, y, cv, pr_auc_score, SMOTE())
print("Cross-validated AUPRC score: %.2f"%score)

# Logistic regression score with ADASYN
print("ADASYN over-sampling")
score = imbalanced_cross_validation_score(RegressionModel, x, y, cv, pr_auc_score, ADASYN())
print("Cross-validated AUPRC score: %.2f"%score)

# Logistic regression score with Random Under Sampling
print("Random under-sampling")
score = imbalanced_cross_validation_score(RegressionModel, x, y, cv, pr_auc_score, RandomUnderSampler())
print("Cross-validated AUPRC score: %.2f"%score)
# for fun, let's plot one of the precision-recall curves that is computed above
#
sampler = SMOTE()
skf = StratifiedKFold(n_splits=cv, random_state=0, shuffle=False)
clf = RegressionModel

train_idx, test_idx = skf.split(x,y).__next__()
xfold_train_sampled, yfold_train_sampled = sampler.fit_sample(x[train_idx],y[train_idx])

clf.fit(xfold_train_sampled, yfold_train_sampled)

precisions, recalls,_ = precision_recall_curve(y[test_idx], clf.predict_proba(x[test_idx])[:,1], pos_label=1)

plt.step(recalls, precisions, color='b', alpha=0.2,
         where='post')
plt.fill_between(recalls, precisions, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve')