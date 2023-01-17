import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv("../input/machinehack-financial-risk-prediction/Train.csv")
train.head()
train.shape
train.columns
train.isnull().sum()
train.dtypes
train.describe()
train.columns
plt.figure(figsize=(16,6))
train.boxplot(column=['Location_Score', 'Internal_Audit_Score',
       'External_Audit_Score', 'Fin_Score', 'Loss_score'])
plt.figure(figsize=(14,8))
clr=['red','blue','lime','orange','teal']
columns = ['Location_Score', 'Internal_Audit_Score', 'External_Audit_Score', 'Fin_Score', 'Loss_score']
for i,j in zip(range(1,6),columns):
    plt.subplot(2,3,i)
    train[j].hist(color = clr[i-1], label=j)
    plt.legend()
    
plt.figure(figsize=(14,8))
train[columns].plot(kind='density', subplots=True, 
                                                    layout=(2,3), sharex=False,
                                                    sharey=False, figsize=(14,6))
plt.show()
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
train.City.value_counts().plot(kind='bar', label = 'City')
plt.legend()

plt.subplot(1,2,2)
train.Past_Results.value_counts().plot(kind='bar', label = 'Past_Results')
plt.legend()

train.IsUnderRisk.value_counts().plot(kind='bar', color=['green', 'orange'])
plt.figure(figsize=(20,8))
corr = train.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
import seaborn as sns
sns.set(style="ticks")

sns.pairplot(train)



'''plt.figure(figsize=(14,10))
clr=['red','blue','green','pink','lime','orange','indigo','teal',
    'red','blue','green','pink','lime','orange','indigo','teal']
cols = ['elevation_complaints_ratio', 'elevation_violation_ratio', 'avg_adv_weather_metric',
        'adv_weather_metric_violation_ratio', 'avg_safety_score', 'safety_control_ratio', 
        'safety_turbulence_ratio', 'avg_complaints', 'avg_control_metric', 'avg_turbulence', 
        'avg_cabin_temp', 'avg_elevation', 'avg_violation', 'Total_Safety_Complaints_control_ratio',
        'Turbulence_In_gforces_Total_Safety_Complaints_ratio', 'Violations_Total_Safety_Complaints_ratio']

for i,j in zip(range(1,17),cols):
    plt.subplot(4,4,i)
    train_deduplicated[j].hist(color = clr[i-1], label=j)
    plt.legend()'''
train.columns
train.reset_index(drop=True, inplace=True)
train.head()

x = train.drop(['IsUnderRisk'], axis=1)
y = train['IsUnderRisk']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 100)

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# feature extraction
model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, 3)
fit = rfe.fit(x, y)

print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)


df_feat = pd.DataFrame(fit.ranking_, x.columns)
df_feat.rename(columns = {0:"Feature_Ranking"}, inplace=True)

df_feat.sort_values(by="Feature_Ranking").plot(kind='bar', figsize=(18,7))

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier

#making the instance
model= DecisionTreeClassifier(random_state=1234)

#Hyper Parameters Set
param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
          'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
          'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11],
          'random_state':[123]}

# Create grid search object
clf = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, cv=10)

# Fit on data
best_clf_dt = clf.fit(X_train, y_train)

#Predict
predictions = best_clf_dt.predict(X_test)

print("*******************ACCURACY***************************************************************")
#Check Prediction Score
print("Accuracy of Decision Trees: ",accuracy_score(y_test, predictions))

print("*******************CLASSIFICATION - REPORT***************************************************************")
print("Confusion matrix \n",confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))




from sklearn.ensemble import RandomForestClassifier

#making the instance
model= RandomForestClassifier(random_state=1234)

#Hyper Parameters Set
param_grid = {'criterion':['gini','entropy'],
          'n_estimators':[10,15,20,25,30],
          'min_samples_leaf':[1,2,3],
          'min_samples_split':[3,4,5,6,7], 
          'random_state':[123],
          'n_jobs':[-1]}

# Create grid search object
clf = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, cv=10)

# Fit on data
best_clf_rf = clf.fit(X_train, y_train)

#Predict
predictions = best_clf_rf.predict(X_test)

#Check Prediction Score
print("Accuracy of Random Forest: ",accuracy_score(y_test, predictions))

#Print Classification Report
print("Confusion matrix \n",confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


#RF On Full data

#making the instance
model= RandomForestClassifier(random_state=1234)

#Hyper Parameters Set
param_grid = {'criterion':['gini','entropy'],
          'n_estimators':[10,15,20,25,30],
          'min_samples_leaf':[1,2,3],
          'min_samples_split':[3,4,5,6,7], 
          'random_state':[123],
          'n_jobs':[-1]}

# Create grid search object
clf = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, cv=10)

# Fit on data
best_clf_rf1 = clf.fit(x, y)
from sklearn import svm

#making the instance
model= svm.SVC()

#Hyper Parameters Set
param_grid = {'C': [6,7,8,9,10,11,12], 
          'kernel': ['linear','rbf']}

# Create grid search object
clf = GridSearchCV(model, param_grid=param_grid, n_jobs=-1)

# Fit on data
best_clf_svm = clf.fit(X_train, y_train)

#Predict
predictions = best_clf_svm.predict(X_test)

print("*******************ACCURACY***************************************************************")
#Check Prediction Score
print("Accuracy of SVM: ",accuracy_score(y_test, predictions))

#Print Classification Report
print("Confusion matrix \n",confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

from sklearn.ensemble import AdaBoostClassifier

#making the instance
model= AdaBoostClassifier()

#Hyper Parameters Set
param_grid = {'n_estimators':[500,1000,2000],'learning_rate':[.001,0.01,.1]}

# Create grid search object
clf = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, cv=10)

# Fit on data
best_clf_adab = clf.fit(X_train, y_train)

#Predict
predictions = best_clf_adab.predict(X_test)

#Check Prediction Score
print("Accuracy of Adaboost Classifier: ",accuracy_score(y_test, predictions))

#Print Classification Report
print("Confusion matrix \n",confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

#making the instance
model= AdaBoostClassifier()

#Hyper Parameters Set
param_grid = {'n_estimators':[500,1000,2000],'learning_rate':[.001,0.01,.1]}

# Create grid search object
clf = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, cv=10)

# Fit on data
best_clf_adab = clf.fit(x, y)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()


#making the instance
clf= VotingClassifier(estimators=[
                                    ('lr', clf1), 
                                    ('rf', clf2), 
                                    ('gnb', clf3)], voting='hard')

# Fit on data
best_clf = clf.fit(X_train, y_train)

#Predict
predictions = best_clf.predict(X_test)

#Check Prediction Score
print("Accuracy of Voting Classifier: ",accuracy_score(y_test, predictions))

#Print Classification Report
print("Confusion matrix \n",confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


from sklearn.neural_network import MLPClassifier
from scipy.stats import randint as sp_randint
from random import uniform

#making the instance
model= MLPClassifier()

#Hyper Parameters Set
param_grid = {'hidden_layer_sizes': [(sp_randint.rvs(100,600,1),sp_randint.rvs(100,600,1),), 
                                          (sp_randint.rvs(100,600,1),)],
    'activation': ['tanh', 'relu', 'logistic'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [uniform(0.0001, 0.9)],
    'learning_rate': ['constant','adaptive']}

# Create grid search object
clf = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, cv=10)

# Fit on data
best_clf_mlp = clf.fit(X_train, y_train)

#Predict
predictions = best_clf_mlp.predict(X_test)

#Check Prediction Score
print("Accuracy of MLP Classifier: ",accuracy_score(y_test, predictions))

#Print Classification Report
print("Confusion matrix \n",confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

from xgboost import XGBClassifier
from sklearn import metrics
model = XGBClassifier()

# Fit on data
best_clf_xgb = model.fit(X_train, y_train)

predictions = best_clf_xgb.predict(X_test)

#Check Prediction Score
print("Accuracy of MLP Classifier: ",accuracy_score(y_test, predictions))

#Print Classification Report
print("Confusion matrix \n",confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

# Fit on data
best_clf_xgb1 = model.fit(x, y)
test = pd.read_csv('../input/machinehack-financial-risk-prediction/Test.csv')

test.shape
test.head(5)
test.columns
test_for_prediction = test[['City', 'Location_Score', 'Internal_Audit_Score',
       'External_Audit_Score', 'Fin_Score', 'Loss_score', 'Past_Results']]
'''#Predict

prediction_from_dt  = best_clf_dt.predict_proba(test_for_prediction)
df_prediction_from_dt = pd.DataFrame(prediction_from_dt)
df_prediction_from_dt.to_excel("Final_output_prediction_from_dt.xlsx")

prediction_from_rf  = best_clf_rf.predict_proba(test_for_prediction)
df_prediction_from_rf = pd.DataFrame(prediction_from_rf)
df_prediction_from_rf.to_excel("Final_output_prediction_from_rf.xlsx")
'''
'''prediction_from_rf1  = best_clf_rf1.predict_proba(test_for_prediction)
df_prediction_from_rf1 = pd.DataFrame(prediction_from_rf1)
df_prediction_from_rf1.to_excel("Final_output_prediction_from_rf1.xlsx")
'''
'''
prediction_from_adab  = best_clf_adab.predict_proba(test_for_prediction)
df_prediction_from_adab = pd.DataFrame(prediction_from_adab)
df_prediction_from_adab.to_excel("Final_output_prediction_from_adab.xlsx")
'''
def predict_file(model, model_instance, test_data):
    prediction_var = "prediction_from" + model
    file_name = "Final_output_prediction_from_" + model + ".xlsx"
    prediction_var  = model_instance.predict_proba(test_data)
    df_prediction_var = pd.DataFrame(prediction_var)
    df_prediction_var.to_excel(file_name)
    print("{} created.".format(file_name))
predict_file("xgbclassifier", best_clf_xgb1, test_for_prediction)

'''predict_file("mlpclassifier", best_clf_mlp, test_for_prediction)'''
