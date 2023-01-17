import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
sns.set_style("darkgrid")
counties=pd.read_csv('../input/counties.csv')
# Split the data
X = counties.drop(['hi_diabetes', 'hi_obesity', 'FIPS', 'State', 'County',], axis=1)
y = counties['hi_diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=42)
X_train.shape # Notice the number of rows in the training set. This should stay constant.
# List the features
X.columns
# Random Forest
rf = RandomForestClassifier()
tree_model=rf.fit(X_train, y_train)
# Specify the grid parameters
param_grid = {
        'n_estimators': [100, 150], 
        'max_depth': [3, 4, None],
        'min_samples_split': [5, 10, 15, 100],
        'min_samples_leaf': [5, 10],
#         'max_features': [10, 20, 25, 30], # This is not meaningful because I only have a few predictors
        'class_weight': [None]    
        }
# Grid Search for Best Parameters
grid = GridSearchCV(tree_model, param_grid=param_grid, n_jobs = 1, cv=3)
grid.fit(X_train, y_train);
# We should re-run the RF model with these optimal parameters:
print(grid.best_params_)
# Instantiate and Fit the Model with Optimal Settings
rf = RandomForestClassifier(class_weight= None, 
                            max_depth= 3, 
                            min_samples_leaf= 5, 
                            min_samples_split= 5, 
                            n_estimators= 100)
tree_model=rf.fit(X_train, y_train)
# The crossvalidiation score scores our performance on the training data. 
scores = cross_val_score(tree_model, X_train, y_train, cv=5)
np.mean(scores), np.std(scores) # This is the mean of the 5 cv scores, plus its standard dev.
# Predict the y values on the testing data.
y_hat = tree_model.predict(X_test)
y_hat_probs = tree_model.predict_proba(X_test)[:,1]
# ACCURACY
accuracy=100*metrics.accuracy_score(y_test, y_hat)
print(accuracy)
# A confusion matrix tells us our false positives and false negatives:
mat = confusion_matrix(y_test, y_hat)
print (mat)
# Let's interpret that.
tn, fp, fn, tp = mat.ravel()
sensitivity = 100*tp/(tp+fn)
specificity=100*tn/(tn+fp)
print('true positive rate:', sensitivity)
print('true negative rate:', specificity)
# Our ROC-AUC score measures the trade-off between specificity and sensitivity
roc_score=100*roc_auc_score(y_test, y_hat_probs)
print(roc_score)
# compute the feature importances
importances = tree_model.feature_importances_
std = np.std([tree.feature_importances_ for tree in tree_model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
# List the features by importance:
feat_imp=pd.DataFrame(importances, index=X_test.columns, columns=['importance'])
feat_imp['importance'].sort_values(ascending=False).head(10)
feat_imp.head()
top10=pd.DataFrame(feat_imp['importance'].sort_values(ascending=False).head(10))
top10
top10 = top10.rename(index={'MEDHHINC15':'Median household income','PC_FFRSALES12':'Expenditures per capita fast food','PC_FSRSALES12':'Expenditures per capita, restaurants',
                                   'FSRPTH14':'Full-service restaurants','SNAPSPTH16':'SNAP-authorized stores','PCT_WIC15':'WIC participants','PCT_65OLDER10':'% Population 65 years or older','SNAP_PART_RATE13':'SNAP participants',
                                   'PCT_NSLP15':'National School Lunch Program participants','CONVSPTH14':'Convenience stores','RECFACPTH14':'Recreation & fitness facilities','PCT_CACFP15':'Child & Adult Care'})
sns.set(style="darkgrid", color_codes=None)
# sns.palplot(sns.color_palette("RdBu", n_colors=7))
ax = top10.plot(kind='bar', legend=False, fontsize=18,  figsize=(20, 10))
plt.xticks(rotation = 90,  fontsize=18)
plt.title('Most Important Predictors',  fontsize=19)
plt.yticks(rotation = 0,  fontsize=18)
plt.ylabel('Feature Importance', rotation=90,  fontsize=18)
plt.savefig('feature_import.png', dpi=300, bbox_inches='tight')
from sklearn.metrics import roc_curve, auc
# Empty dictionaries.
FPR = dict()
TPR = dict()
ROC_AUC = dict()
# For class 1 (has WNV), find the area under the curve:
FPR[1], TPR[1], _ = roc_curve(y_test, y_hat_probs)
ROC_AUC[1] = auc(FPR[1], TPR[1])
# What is that ROC-AUC score?
print(ROC_AUC[1])
# Same but using the scikit default:
roc_auc_score(y_test, y_hat_probs)
# Let's draw that:
plt.style.use('seaborn-white')
plt.figure(figsize=[11,9])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('ROC Curve', fontsize=18)
plt.legend(loc="lower right", fontsize=18);
plt.savefig('rocauc.png', dpi=300, bbox_inches='tight')
# Make sure the all zip codes are 5 digits long (some of them start with 00)
counties['County FIPS Code'] = counties['FIPS'].apply(lambda x: str(x).zfill(5))
# Confirm that worked okay.
counties[['County FIPS Code', 'FIPS']].head(3)
# Predict the y values on the entire dataset.
y_hat = tree_model.predict(X)
y_hat_probs = tree_model.predict_proba(X)[:,1]
# create a new dataset with only the columns we need
submission=pd.DataFrame(list(zip(counties['County FIPS Code'], y_hat_probs)), columns=['FIPS','Probability'])
# convert probabilities to percentages
submission['Probability'] = submission['Probability'].apply(lambda x: round(x*100, 1))
submission.head()
# Confirm that we have the right length (one value for every county)
print(len(counties))
print(len(y_hat_probs))
len(submission)
# export
submission.to_csv('predict_proba.csv', index=False)
