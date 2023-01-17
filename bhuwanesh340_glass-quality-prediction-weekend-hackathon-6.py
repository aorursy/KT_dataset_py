import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv("../input/glass-quality-prediction/Glass_Quality_Participants_Data/Train.csv")
train.head()
train.shape
train.columns
train.isnull().sum()
train.dtypes
train.describe()
train.columns
plt.figure(figsize=(16,10))

plt.subplot(2,2,1)
train.boxplot(column=['ymin', 'ymax'])

plt.subplot(2,2,2)
train.boxplot(column=['pixel_area', 'log_area'])

plt.subplot(2,2,3)
train.boxplot(column=['max_luminosity', 'thickness'])

plt.subplot(2,2,4)
train.boxplot(column=['xmin', 'xmax'])

plt.figure(figsize=(14,8))
clr=['red','blue','lime','orange','teal','red','blue','lime']
columns = ['max_luminosity', 'thickness', 'xmin', 'xmax', 'ymin', 'ymax', 'pixel_area', 'log_area']
for i,j in zip(range(1,9),columns):
    plt.subplot(4,2,i)
    train[j].hist(color = clr[i-1], label=j)
    plt.legend()
    
plt.figure(figsize=(14,8))
train[columns].plot(kind='density', subplots=True, 
                                                    layout=(4,2), sharex=False,
                                                    sharey=False, figsize=(14,6))
plt.show()
train.columns
plt.figure(figsize=(14,12))

plt.subplot(4,2,1)
train.grade_A_Component_1.value_counts().plot(kind='bar', label = 'grade_A_Component_1')
plt.legend()

plt.subplot(4,2,2)
train.grade_A_Component_2.value_counts().plot(kind='bar', label = 'grade_A_Component_2')
plt.legend()

plt.subplot(4,2,3)
train.x_component_1.value_counts().plot(kind='bar', label = 'x_component_1')
plt.legend()

plt.subplot(4,2,4)
train.x_component_2.value_counts().plot(kind='bar', label = 'x_component_2')
plt.legend()

plt.subplot(4,2,5)
train.x_component_3.value_counts().plot(kind='bar', label = 'x_component_3')
plt.legend()

plt.subplot(4,2,6)
train.x_component_4.value_counts().plot(kind='bar', label = 'x_component_4')
plt.legend()

plt.subplot(4,2,7)
train.x_component_4.value_counts().plot(kind='bar', label = 'x_component_4')
plt.legend()

train['class'].value_counts().plot(kind='bar')
train['class'].value_counts()
import seaborn as sns
sns.set(style="ticks")

sns.pairplot(train)

train.columns
train.reset_index(drop=True, inplace=True)
x = train.drop(['class'], axis=1)
y = train['class']
x_copy = x.copy()

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

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import classification_report

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
test = pd.read_csv('../input/glass-quality-prediction/Glass_Quality_Participants_Data/Test.csv')

test.shape
test.head(5)
test.columns
test_for_prediction = test[['grade_A_Component_1', 'grade_A_Component_2', 'max_luminosity',
       'thickness', 'xmin', 'xmax', 'ymin', 'ymax', 'pixel_area', 'log_area',
       'x_component_1', 'x_component_2', 'x_component_3', 'x_component_4',
       'x_component_5']]
#Define predict function

def predict_file(model, model_instance, test_data):
    prediction_var = "prediction_from" + model
    file_name = "Final_output_prediction_from_" + model + ".xlsx"
    prediction_var  = model_instance.predict_proba(test_data)
    df_prediction_var = pd.DataFrame(prediction_var, columns=[1,2])
    df_prediction_var.to_excel(file_name)
    print("{} created.".format(file_name))
predict_file("rf_classifier", best_clf_rf, test_for_prediction)
predict_file("rf1_classifier", best_clf_rf1, test_for_prediction)
