pwd
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
from pandas import DataFrame
import pylab as pl
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
%matplotlib inline
train_data=pd.read_csv("../input/hranalyticsav/train_data.csv")
#test_data=pd.read_csv("C:\\Users\\ARPIT\\Desktop\\New folder\\HR Analytics\\test_data.csv")
train_data.shape
display('Train Head :',train_data.head())
#display('Test Head :',test_data.head())
numeric_data = train_data.select_dtypes(include=[np.number])
categorical_data = train_data.select_dtypes(exclude=[np.number])
print("Numeric_Column_Count =", numeric_data.shape)
print("Categorical_Column_Count =", categorical_data.shape)
train_data['education'].fillna(train_data['education'].mode()[0], inplace = True)
train_data['previous_year_rating'].fillna(train_data['previous_year_rating'].mode()[0], inplace = True)
train_data['previous_year_rating'].astype(int)
#train_data.groupby(['region']).sum().plot(kind='pie', y='avg_training_score',startangle=90,figsize=(15,10), autopct='%1.1f%%')
#plt.figure(figsize=(12,7))
#train_data['department'].value_counts().plot(kind='pie')
#train_data['previous_year_rating'].fillna(train_data['previous_year_rating'].mode()[0], inplace = True)
pd.get_dummies(train_data['gender'], prefix='G')
train_data = pd.concat([train_data, pd.get_dummies(train_data['gender'], prefix='G')], axis=1)
train_data.drop(['gender','G_f'],axis=1,inplace=True)
pd.get_dummies(train_data['recruitment_channel'], prefix='R')
train_data = pd.concat([train_data, pd.get_dummies(train_data['recruitment_channel'], prefix='R')], axis=1)
train_data.drop(['recruitment_channel','R_other'],axis=1,inplace=True)
pd.get_dummies(train_data['region'], prefix='Re')
train_data = pd.concat([train_data, pd.get_dummies(train_data['region'], prefix='Re')], axis=1)
train_data.drop(['region','Re_region_8'],axis=1,inplace=True)
pd.get_dummies(train_data['department'], prefix='Dep')
train_data = pd.concat([train_data, pd.get_dummies(train_data['department'], prefix='Dep')], axis=1)
train_data.drop(['department','Dep_Technology'],axis=1,inplace=True)
train_data.head()
replace={"Master's & above":3,"Bachelor's":2,"Below Secondary":1}
train_data['education']=train_data['education'].replace(replace)
train_data=train_data.drop(['employee_id'],axis=1,inplace=False)
train_data.head()
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
X = train_data.drop(['is_promoted'],axis = 1)
Y = train_data['is_promoted'] 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=123)
logit = LogisticRegression()  #Fit Logistic Regression model.
logit.fit(X_train, Y_train)
logit.classes_
logit.coef_
predictions=logit.predict(X_test)   #Make class predictions.
logit.score(X_test, Y_test)      #Calculate accuracy score.
1-logit.score(X_test, Y_test)     #Calculate Error rate.
import sklearn.metrics as metrics    
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, predictions)
accuracy_score(Y_test, predictions, normalize=False) #Calculate number of correctly classified observations.
len(Y_test) - accuracy_score(Y_test, predictions, normalize=False) #Calculate number of incorrectly classified observations.
from sklearn.metrics import log_loss
import numpy as np
print ("log_loss", metrics.log_loss(Y_test, predictions))   #Encode predicted classes and test labels.
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
confusion_mat = confusion_matrix(Y_test, predictions)
confusion_df = pd.DataFrame(confusion_mat, index=['Actual neg','Actual pos'], columns=['Predicted neg','Predicted pos'])
confusion_df
_=sns.heatmap(confusion_df, cmap='coolwarm', annot=True)
from sklearn.metrics import precision_score, recall_score
precision_score(Y_test, predictions)
recall_score(Y_test, predictions)
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve 
y_score = logit.decision_function(X_test)
precision, recall, _ = precision_recall_curve(Y_test, y_score)
PR_AUC = auc(recall, precision)
plt.figure(figsize=[15,8])
plt.plot(recall, precision, label='PR curve (area = %0.2f)' % PR_AUC, linewidth=4)
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.title('Precision Recall Curve for People getting Promoted', fontsize=18)
plt.legend(loc="lower right")
plt.show()

# reference: https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/#:~:text=The%20Precision%2DRecall%20AUC%20is,a%20model%20with%20perfect%20skill.
from sklearn.metrics import f1_score
f1_score(Y_test, predictions)
from sklearn.metrics import roc_curve, roc_auc_score
probs = logit.predict_proba(X_test)[::,1] #Let's take probablities from our classifier, instead of classes.
auc = roc_auc_score(Y_test, probs)
print(auc)
fpr, tpr, threshold = roc_curve(Y_test, probs)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = threshold[optimal_idx]
optimal_threshold
new_predictions = np.where(probs>optimal_threshold, 1, 0)
new_confusion_mat = confusion_matrix(Y_test, new_predictions)
new_confusion_df = pd.DataFrame(new_confusion_mat, index=['Actual neg','Actual pos'], columns=['Predicted neg','Predicted pos'])
new_confusion_df
TN=9278
FP=5798
FN=148
TP=1219
sensitivity = TP/(TP+FN)
specificity = TN/(TN+FP)
print("The probability of predicting people who will get promoted",specificity)
print("The probability of predicting people who will get promoted correctly is ",sensitivity)
accuracy_score(Y_test, new_predictions)
log_loss(Y_test, new_predictions)
tree = DecisionTreeClassifier(max_depth=3,max_features=4)
tree.fit(X_train.values, Y_train)
tree.feature_importances_
pd.Series(tree.feature_importances_,index=X.columns).sort_values(ascending=False)
predictions = tree.predict(X_test)
df=pd.DataFrame({'Actual':Y_test, 'Predicted':predictions})
df.head(5)
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
accuracy_score(Y_test, predictions) #Calculate number of correctly classified observations.
accuracy_score(Y_test, predictions, normalize=False)
confusion_mat = confusion_matrix(Y_test, predictions)
confusion_df = pd.DataFrame(confusion_mat, index=['Is_Promoted 0','Is_Promoted 1'],columns=['Is_Promoted 0','Is_Promoted 1' ])
print(confusion_df)
_=sns.heatmap(confusion_df, cmap='coolwarm', annot=True)
tn, fp, fn, tp = confusion_matrix(Y_test, predictions).ravel()
print("True Negatives: ", tn)
print("False Positives: ", fp)
print("False Negatives: ", fn)
print("True Positives: ", tp)
print(classification_report(Y_test, predictions))
Specificity = tn/(tn+fp)
print("The probability of predicting whether a person will be promoted is ",Specificity)
Sensitivity = tp/(tp+fn)
print("The probability of predicting whether a whether a person will be promoted is ",Sensitivity)
from sklearn.tree import export_graphviz
import graphviz
dot_data = export_graphviz(tree, filled=True, rounded=True, feature_names=X.columns, out_file=None)
graphviz.Source(dot_data)
from sklearn.model_selection import GridSearchCV
param_grid = [{"max_depth":[3, 4, 5, None], "max_features":[4,5,6,7,8,9,10,None]}]
gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),param_grid = param_grid,cv=10)
gs.fit(X_train, Y_train)
gs.cv_results_['params']
gs.best_params_
gs.best_estimator_
tree = DecisionTreeClassifier(max_depth=5,max_features=None)
tree.fit(X_train,Y_train)
predictions = tree.predict(X_test)
df=pd.DataFrame({'Actual':Y_test, 'Predicted':predictions})
df.head(5)
accuracy_score(Y_test, predictions)#Calculate number of correctly classified observations.
accuracy_score(Y_test, predictions, normalize=False) 
confusion_mat = confusion_matrix(Y_test, predictions)
confusion_df = pd.DataFrame(confusion_mat, index=['Is_Promoted 0','Is_Promoted 1'],columns=['Is_Promoted 0','Is_Promoted 1' ])
print(confusion_df)
_=sns.heatmap(confusion_df, cmap='coolwarm', annot=True)
from sklearn.tree import export_graphviz
import graphviz
dot_data = export_graphviz(tree, filled=True, rounded=True, feature_names=X.columns, out_file=None)
graphviz.Source(dot_data)
from sklearn.ensemble import VotingClassifier
log_model = LogisticRegression()
dtree_model = DecisionTreeClassifier()
ensemble = VotingClassifier(estimators=[('lr', log_model), ('dtree', dtree_model)], voting='hard')
ensemble.fit(X_train, Y_train)
from sklearn.metrics import accuracy_score
for model in (log_model, dtree_model, ensemble):
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    print(model.__class__.__name__, accuracy_score(Y_test, predictions))
from sklearn.ensemble import BaggingClassifier
bagged_model = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100, bootstrap=True, n_jobs=-1)
bagged_model.fit(X_train, Y_train)
predictions = bagged_model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, predictions)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rf_model.fit(X_train, Y_train)
predictions = rf_model.predict(X_test)
accuracy_score(Y_test, predictions)
rf_model.feature_importances_
pd.Series(rf_model.feature_importances_,index=X.columns).sort_values(ascending=False)
from sklearn.ensemble import ExtraTreesClassifier
extra_model = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, bootstrap=True, random_state=123)
extra_model.fit(X_train, Y_train)
predictions = extra_model.predict(X_test)
accuracy_score(Y_test, predictions)
extra_model.feature_importances_
pd.Series(extra_model.feature_importances_,index=X.columns).sort_values(ascending=False)
from sklearn.ensemble import AdaBoostClassifier
ada_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8), n_estimators=200, learning_rate=0.01, random_state=123)
ada_model.fit(X_train, Y_train)
predictions = ada_model.predict(X_test)
accuracy_score(Y_test, predictions)#Calculate number of correctly classified observations.
ada_model.feature_importances_
pd.Series(ada_model.feature_importances_,index=X.columns).sort_values(ascending=False)
from sklearn.ensemble import GradientBoostingClassifier
gb_model = GradientBoostingClassifier(max_depth=4, n_estimators=250, learning_rate=0.01, random_state=123)
gb_model.fit(X_train, Y_train)
predictions = gb_model.predict(X_test)
accuracy_score(Y_test, predictions)#Calculate number of correctly classified observations.
gb_model.feature_importances_
pd.Series(gb_model.feature_importances_,index=X.columns).sort_values(ascending=False)