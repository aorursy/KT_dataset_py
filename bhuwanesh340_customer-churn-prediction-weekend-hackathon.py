import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
train = pd.read_csv("../input/insurance-churn-prediction-weekend-hackathon/Insurance_Churn_ParticipantsData/Train.csv")
train.head()
train.shape
train.columns
train.isnull().sum()
train.dtypes
train.describe()
train.columns
plt.figure(figsize=(16,6))
train.boxplot(column=['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4',
                       'feature_5', 'feature_6', 'feature_7'])
plt.figure(figsize=(12,6))
train.boxplot(column=['feature_8', 'feature_9', 'feature_10', 'feature_11', 'feature_12', 
                                   'feature_13', 'feature_14', 'feature_15'])
plt.figure(figsize=(14,10))
clr=['red','blue','green','pink','lime','orange','yellow','violet','indigo','teal','red','blue','green','pink','lime','orange']
for i,j in zip(range(1,17),train.columns[:-1]):
    plt.subplot(4,4,i)
    train[j].hist(color = clr[i-1], label=j)
    plt.legend()
    

train[['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4',
       'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9',
       'feature_10', 'feature_11', 'feature_12', 'feature_13', 'feature_14',
       'feature_15']].plot(kind='density', subplots=True, 
                                                    layout=(4,4), sharex=False,
                                                    sharey=False, figsize=(14,6))
plt.show()
train.labels.value_counts().plot(kind='bar', colors=['green', 'orange'])
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

train.reset_index(drop=True, inplace=True)
train.head()

x = train.drop(['labels'], axis=1)
y = train['labels']

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



import xgboost as xgb

xgbmodel=xgb.XGBClassifier(learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

# Fit on data
best_clf_xgb = xgbmodel.fit(X_train, y_train)

#Predict
predictions = best_clf_xgb.predict(X_test)

#Check Prediction Score
print("Accuracy of XGBoost: ",accuracy_score(y_test, predictions))

#Print Classification Report
print("Confusion matrix \n",confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))



from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(20, 3), max_iter=150, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

mlp.fit(X_train,y_train)
print("Training set score: %f" % mlp.score(X_train,y_train))

#Predict
predictions = mlp.predict(X_test)

#Check Prediction Score
print("Accuracy of MLP: ",accuracy_score(y_test, predictions))

#Print Classification Report
print("Confusion matrix \n",confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
from lightgbm import LGBMClassifier

lgbm_c = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                        learning_rate=0.5, max_depth=7, min_child_samples=20,
                        min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
                        n_jobs=-1, num_leaves=500, objective='binary', random_state=None,
                        reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
                        subsample_for_bin=200000, subsample_freq=0)

# Fit on data
best_clf_lgbm = lgbm_c.fit(X_train, y_train)

#Predict
predictions = best_clf_lgbm.predict(X_test)

print("*******************ACCURACY***************************************************************")
#Check Prediction Score
print("Accuracy of LGBM: ",accuracy_score(y_test, predictions))

#Print Classification Report
print("Confusion matrix \n",confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))



test = pd.read_csv('../input/insurance-churn-prediction-weekend-hackathon/Insurance_Churn_ParticipantsData/Test.csv')

test.shape
test.head(5)
test_for_prediction = test[['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4',
       'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9',
       'feature_10', 'feature_11', 'feature_12', 'feature_13', 'feature_14',
       'feature_15']]

prediction_from_dt  = best_clf_dt.predict(test_for_prediction)
df_prediction_from_dt = pd.DataFrame({'labels': prediction_from_dt})
df_prediction_from_dt.to_excel("Final_output_prediction_from_dt.xlsx")

prediction_from_rf  = best_clf_rf.predict(test_for_prediction)
df_prediction_from_rf = pd.DataFrame({'labels': prediction_from_rf})
df_prediction_from_rf.to_excel("Final_output_prediction_from_rf.xlsx")

prediction_from_xgb  = best_clf_rf.predict(test_for_prediction)
prediction_from_xgb = pd.DataFrame({'labels': prediction_from_xgb})
prediction_from_xgb.to_excel("Final_output_prediction_from_xgb.xlsx")


def generate_prediction(model_name, model, test_file):    
    prediction_file_name = "Final_output_prediction_from_" + model_name +".xlsx"
    prediction_from_model  = model.predict(test_file)
    prediction_from_model = pd.DataFrame({'labels': prediction_from_model})
    prediction_from_model.to_excel(prediction_file_name)

generate_prediction("lgbm", best_clf_lgbm, test_for_prediction)
generate_prediction("mlp", mlp, test_for_prediction)
