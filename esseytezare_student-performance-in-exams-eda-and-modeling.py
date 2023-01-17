# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import missingno as missing
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

import random
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score ,auc, plot_roc_curve
from sklearn import svm
import sklearn.metrics
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("../input/studentperformancebig/StudentsPerformanceBig.csv")

# preview the data
df.head()
df.info()
df.describe()
# lets check the no. of unique items present in the categorical column

df.select_dtypes('object').nunique()
plt.figure(figsize=(25,6))
plt.subplot(1, 3, 1)
sns.distplot(df['math score'])

plt.subplot(1, 3, 2)
sns.distplot(df['reading score'])

plt.subplot(1, 3, 3)
sns.distplot(df['writing score'])

plt.suptitle('Checking for Skewness', fontsize = 15)
plt.show()



plt.figure(figsize=(25,6))
plt.subplot(1, 3, 1)
sns.boxplot(x="race/ethnicity", y="math score", hue="gender", data=df)
plt.title('MATH SCORES')
plt.subplot(1, 3, 2)
sns.boxplot(x="race/ethnicity", y="reading score", hue="gender", data=df)
plt.title('READING SCORES')
plt.subplot(1, 3, 3)
sns.boxplot(x="race/ethnicity", y="writing score", hue="gender", data=df)
plt.title('WRITING SCORES')
plt.show()
plt.figure(figsize=(25,6))
plt.subplot(1, 3, 1)
sns.boxplot(x="lunch", y="math score", hue="gender", data=df)
plt.title('MATH SCORES')
plt.subplot(1, 3, 2)
sns.boxplot(x="lunch", y="reading score", hue="gender", data=df)
plt.title('READING SCORES')
plt.subplot(1, 3, 3)
sns.boxplot(x="lunch", y="writing score", hue="gender", data=df)
plt.title('WRITING SCORES')
plt.show()
plt.figure(figsize=(25,6))
plt.subplot(1, 3, 1)
sns.boxplot(x="parental level of education", y="math score", hue="gender", data=df)
plt.title('MATH SCORES')
plt.xticks(rotation = 90)
plt.subplot(1, 3, 2)
sns.boxplot(x="parental level of education", y="reading score", hue="gender", data=df)
plt.title('READING SCORES')
plt.xticks(rotation = 90)
plt.subplot(1, 3, 3)
sns.boxplot(x="parental level of education", y="writing score", hue="gender", data=df)
plt.title('WRITING SCORES')
plt.xticks(rotation = 90)
plt.show()

plt.figure(figsize=(25,6))
plt.subplot(1, 3, 1)
sns.boxplot(x="test preparation course", y="math score", hue="gender", data=df)
plt.title('MATH SCORES')
plt.subplot(1, 3, 2)
sns.boxplot(x="test preparation course", y="reading score", hue="gender", data=df)
plt.title('READING SCORES')
plt.subplot(1, 3, 3)
sns.boxplot(x="test preparation course", y="writing score", hue="gender", data=df)
plt.title('WRITING SCORES')
plt.show()
plt.figure(figsize=(25,6))
sns.pairplot(data=df,hue='gender',plot_kws={'alpha':0.2})
plt.show()
df['math_pass']=np.where(df['math score'] >= 65,'P','F')
df['reading_pass']=np.where(df['reading score'] >= 65,'P','F')
df['writing_pass']=np.where(df['writing score'] >= 65,'P','F')
df['Pass'] = df.apply(lambda x :1 if x['math score'] >= 65 and 
                      x['reading score'] >= 65 and 
                      x['writing score'] >= 65 
                      else 0, axis =1)
df.head()
df.Pass.value_counts()
plt.figure(figsize=(20,15))

plt.subplot(4,3,1)
sns.countplot(x='parental level of education', hue='writing_pass', data=df)
plt.xticks(rotation=45)
plt.subplot(4,3,2)
sns.countplot(x='parental level of education', hue='math_pass', data=df)
plt.xticks(rotation=45)
plt.subplot(4,3,3)
sns.countplot(x='parental level of education', hue='reading_pass', data=df)
plt.xticks(rotation=45)

plt.subplot(4,3,4)
sns.countplot(x='gender', hue='writing_pass', data=df)
plt.xticks(rotation=45)
plt.title("Gender - Writing Pass")
plt.subplot(4,3,5)
sns.countplot(x='gender', hue='math_pass', data=df)
plt.xticks(rotation=45)
plt.title("Gender - Math Pass")
plt.subplot(4,3,6)
sns.countplot(x='gender', hue='reading_pass', data=df)
plt.xticks(rotation=45)
plt.title("Gender - Reading Pass")

plt.subplot(4,3,7)
sns.countplot(x='test preparation course', hue='writing_pass', data=df)
plt.xticks(rotation=45)
plt.title("Preparation - Writing Pass")
plt.subplot(4,3,8)
sns.countplot(x='test preparation course', hue='math_pass', data=df)
plt.xticks(rotation=45)
plt.title("Preparation - Math Pass")
plt.subplot(4,3,9)
sns.countplot(x='test preparation course', hue='reading_pass', data=df)
plt.xticks(rotation=45)
plt.title("Preparation - Reading Pass")

plt.subplot(4,3,10)
sns.countplot(x='race/ethnicity', hue='writing_pass', data=df)
plt.xticks(rotation=45)
plt.title("Race - Writing Pass")
plt.subplot(4,3,11)
sns.countplot(x='race/ethnicity', hue='math_pass', data=df)
plt.xticks(rotation=45)
plt.title("Race - Math Pass")
plt.subplot(4,3,12)
sns.countplot(x='race/ethnicity', hue='reading_pass', data=df)
plt.xticks(rotation=45)
plt.title("Race - Reading Pass")

plt.tight_layout()
plt.show()

map1 = {"high school": 1, "some high school": 1,
        "associate's degree": 2,
        "some college": 3,
        "bachelor's degree": 4,
        "master's degree": 5}
df['parental level of education']  = df['parental level of education'].map(map1)

map2 = {"free/reduced": 0,
        "standard": 1}
df['lunch']  = df['lunch'].map(map2)

map3 = {"none": 0,
        "completed": 1}
df['test preparation course']  = df['test preparation course'].map(map3)

map4 = {"female": 0,
        "male": 1}
df['gender']  = df['gender'].map(map4)

map5 = {"group A": 1,
        "group B": 2,
        "group C": 3,
        "group D": 4,
        "group E": 5}
df['race/ethnicity']  = df['race/ethnicity'].map(map5)

plt.figure(figsize=(13,10))

plt.subplot(4,3,1)
sns.barplot(x = "parental level of education" , y="writing score" , data=df)
plt.title("Parental level - Writing Scores")
plt.subplot(4,3,2)
sns.barplot(x = "parental level of education" , y="math score" , data=df)
plt.title("Parental level - Math Scores")
plt.subplot(4,3,3)
sns.barplot(x = "parental level of education" , y="reading score" , data=df)
plt.title("Parental level - Reading Scores")

plt.subplot(4,3,4)
sns.barplot(x = "gender" , y="writing score" , data=df)
plt.title("Gender - Writing Scores")
plt.subplot(4,3,5)
sns.barplot(x = "gender" , y="math score" , data=df)
plt.title("Gender - Math Scores")
plt.subplot(4,3,6)
sns.barplot(x = "gender" , y="reading score" , data=df)
plt.title("Gender - Reading Scores")

plt.subplot(4,3,7)
sns.barplot(x = "test preparation course" , y="writing score" , data=df)
plt.title("Preparation - Writing Scores")
plt.subplot(4,3,8)
sns.barplot(x = "test preparation course" , y="math score" , data=df)
plt.title("Preparation - Math Scores")
plt.subplot(4,3,9)
sns.barplot(x = "test preparation course" , y="reading score" , data=df)
plt.title("Preparation - Reading Scores")

plt.subplot(4,3,10)
sns.barplot(x = "race/ethnicity" , y="writing score" , data=df)
plt.title("Race - Writing Scores")
plt.subplot(4,3,11)
sns.barplot(x = "race/ethnicity" , y="math score" , data=df)
plt.title("Race - Math Scores")
plt.subplot(4,3,12)
sns.barplot(x = "race/ethnicity" , y="reading score" , data=df)
plt.title("Race - Reading Scores")

plt.tight_layout()
plt.show()
plt.subplots(figsize=(15,10)) 
sns.heatmap(df.corr(), annot = True, fmt = ".2f")
plt.show()
dfDrop = df.drop(['math score','reading score','writing score', 'math_pass', 'reading_pass','writing_pass'], axis=1)
dfDrop.head()

dfDrop.info()
plt.subplots(figsize=(15,10)) 
sns.heatmap(dfDrop.corr(), annot = True, fmt = ".2f")
plt.show()
def plotLearningCurves(X_train, y_train, classifier, title):
    train_sizes, train_scores, test_scores = learning_curve(
            classifier, X_train, y_train, cv=5, scoring="accuracy")
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="b" ,label="Training Error")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="r" ,label="Cross Validation Error")
    
    plt.legend()
    plt.grid()
    plt.title(title, fontsize = 18, y = 1.03)
    plt.xlabel('Data Size', fontsize = 14)
    plt.ylabel('Error', fontsize = 14)
    plt.tight_layout()
def plotValidationCurves(X_train, y_train, classifier, param_name, param_range, title):
    train_scores, test_scores = validation_curve(
        classifier, X_train, y_train, param_name = param_name, param_range = param_range,
        cv=5, scoring="accuracy")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.plot(param_range, train_scores_mean, 'o-', color="b" ,label="Training Error")
    plt.plot(param_range, test_scores_mean, 'o-', color="r" ,label="Cross Validation Error")

    plt.legend()
    plt.grid()
    plt.title(title, fontsize = 18, y = 1.03)
    plt.xlabel('Complexity', fontsize = 14)
    plt.ylabel('Error', fontsize = 14)
    plt.tight_layout()
def printConfusionMatrix(y_train, pred):
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))
    print("Classification Report:",)
    print (classification_report(y_test, pred))
    print("Accuracy:", accuracy_score(y_test, pred))
X = dfDrop.iloc[:, :-1].values
y = dfDrop.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
rf = RandomForestClassifier(n_estimators = 9,
                                    max_depth=3,
                                    min_samples_split=9,
                                    min_samples_leaf=5
                                   )
rf.fit(X_train, y_train)
rf_pred1 = rf.predict(X_test)
plt.figure(figsize = (16,5))
title = 'Random Forest Learning Curve 1'
plotLearningCurves(X_train, y_train, rf, title)
title = 'Random Forest Validation Curve 1'
param_name = 'n_estimators'
param_range = [4, 6, 9]
plt.figure(figsize = (16,5))
plotValidationCurves(X_train, y_train, rf, param_name, param_range, title)
printConfusionMatrix(y_test, rf_pred1)
plot_roc_curve(rf, X_test, y_test)
plt.show()

rf = RandomForestClassifier(n_estimators = 9,
                                    max_depth=3,
                                    criterion='entropy',
                                    min_samples_split=9,
                                    min_samples_leaf=5
                                   )
rf.fit(X_train, y_train)
rf_pred2 = rf.predict(X_test)
plt.figure(figsize = (16,5))
title = 'Random Forest Learning Curve 2'
plotLearningCurves(X_train, y_train, rf, title)
plt.figure(figsize = (16,5))
title = 'Random Forest Validation Curve 2'
param_name = 'n_estimators'
param_range = [4, 6, 9]
plotValidationCurves(X_train, y_train, rf, param_name, param_range, title)
printConfusionMatrix(y_test, rf_pred2)
plot_roc_curve(rf, X_test, y_test)
plt.show()
rf = RandomForestClassifier(n_estimators = 9,
                                    max_depth=3,
                                    criterion='entropy',
                                    min_samples_split=10,
                                    min_samples_leaf=5
                                   )
rf.fit(X_train, y_train)
rf_pred3 = rf.predict(X_test)
plt.figure(figsize = (16,5))
title = 'Random Forest Learning Curve 3'
plotLearningCurves(X_train, y_train, rf, title)
title = 'Random Forest Validation Curve 3'
param_name = 'n_estimators'
param_range = [4, 6, 9]
plt.figure(figsize = (16,5))
plotValidationCurves(X_train, y_train, rf, param_name, param_range, title)
printConfusionMatrix(y_test, rf_pred3)
plot_roc_curve(rf, X_test, y_test)
plt.show()
rf = RandomForestClassifier(n_estimators = 9,
                                    max_depth=5,
                                    criterion='entropy',
                                    min_samples_split=9,
                                    min_samples_leaf=10
                                   )
rf.fit(X_train, y_train)
rf_pred4 = rf.predict(X_test)
plt.figure(figsize = (16,5))
title = 'Random Forest Learning Curve 4'
plotLearningCurves(X_train, y_train, rf, title)
title = 'Random Forest Validation Curve 4'
param_name = 'n_estimators'
param_range = [4, 6, 9]
plt.figure(figsize = (16,5))
plotValidationCurves(X_train, y_train, rf, param_name, param_range, title)
printConfusionMatrix(y_test, rf_pred4)
plot_roc_curve(rf, X_test, y_test)
plt.show()
rf = RandomForestClassifier(n_estimators = 9,
                                    max_depth=5,
                                    criterion='entropy',
                                    max_features='sqrt',
                                    min_samples_split=9,
                                    min_samples_leaf=5
                                   )
rf.fit(X_train, y_train)
rf_pred5 = rf.predict(X_test)
plt.figure(figsize = (16,5))
title = 'Random Forest Learning Curve 5'
plotLearningCurves(X_train, y_train, rf, title)

title = 'Random Forest Validation Curve 5'
param_name = 'n_estimators'
param_range = [4, 6, 9]
plt.figure(figsize = (16,5))
plotValidationCurves(X_train, y_train, rf, param_name, param_range, title)


printConfusionMatrix(y_test, rf_pred5)

plot_roc_curve(rf, X_test, y_test)
plt.show()
Classifier = RandomForestClassifier()
grid_obj = GridSearchCV(Classifier,
                        {'n_estimators': [4, 6, 9],
                         'max_features': ['log2', 'sqrt','auto'],
                         'criterion': ['entropy', 'gini'],
                         'max_depth': [2, 3, 5, 8],
                         'min_samples_split': [2, 5, 8, 10],
                         'min_samples_leaf': [1, 3, 5]
                        },
                        scoring=make_scorer(accuracy_score))
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
Classifier = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
Classifier.fit(X_train, y_train)

predictions = Classifier.predict(X_test)

print("Best Params: " , grid_obj.best_estimator_)
print("Best Score: " , grid_obj.best_score_)
X = dfDrop.iloc[:, :-1].values
y = dfDrop.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

svmC=svm.SVC(kernel = 'linear' , gamma=0.01, C=0.05)
svmC.fit(X_train,y_train)

svm_pred1=svmC.predict(X_test)
plt.figure(figsize=(16,5))
title='Support Vector Machine Learning Curve 1'
plotLearningCurves(X_train,y_train,svmC,title)
title = 'Support Vector Machine Validation Curve 1'
param_name = 'C'
param_range = [0.1,1, 10]
plt.figure(figsize = (16,5))
plotValidationCurves(X_train, y_train, svmC, param_name, param_range, title)
printConfusionMatrix(y_test, svm_pred1)
plot_roc_curve(svmC, X_test, y_test)
plt.show()
svmC=svm.SVC(kernel = 'rbf' , gamma=0.05, C=1)
svmC.fit(X_train,y_train)

svm_pred2=svmC.predict(X_test)
plt.figure(figsize=(16,5))
title='Support Vector Machine Learning Curve 2'
plotLearningCurves(X_train,y_train,svmC,title)
title = 'Support Vector Machine Validation Curve 2'
param_name = 'C'
param_range = [0.1,1, 10]
plt.figure(figsize = (16,5))
plotValidationCurves(X_train, y_train, svmC, param_name, param_range, title)
printConfusionMatrix(y_test, svm_pred2)

plot_roc_curve(svmC, X_test, y_test)
plt.show()
svmC=svm.SVC(kernel = 'sigmoid' , gamma=1, C=100)
svmC.fit(X_train,y_train)

svm_pred3=svmC.predict(X_test)

plt.figure(figsize=(16,5))
title='Support Vector Machine Learning Curve 3'
plotLearningCurves(X_train,y_train,svmC,title)
title = 'Support Vector Machine Validation Curve 3' 
param_name = 'C'
param_range = [0.1,1, 10]
plt.figure(figsize = (16,5))
plotValidationCurves(X_train, y_train, svmC, param_name, param_range, title)
printConfusionMatrix(y_test, svm_pred3)

plot_roc_curve(svmC, X_test, y_test)
plt.show()
param_grid = {'C': [0.05, 1,10, 20], 'gamma': [0.01,0.1,0.2,1],'kernel': ['sigmoid', 'rbf','linear']}
grid = GridSearchCV(svm.SVC(),param_grid,refit=True,verbose=2)
svclassifier = grid.fit(X_train,y_train)
SvcPredictions = svclassifier.predict(X_test)

print("Best Params: " , grid.best_estimator_)
print("Best Score: " , grid.best_score_)
X = dfDrop.iloc[:, :-1].values
y = dfDrop.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# Create KNN classifier
knn=KNeighborsClassifier(n_neighbors=3)
# Fit the classifier to the data
knn.fit(X_train,y_train)
#show first 5 model predictions on the test data
knn_pred1=knn.predict(X_test)
plt.figure(figsize=(16,5))
title='KNN Learning Curve 1'
plotLearningCurves(X_train,y_train,knn,title)
title = 'KNN Validation Curve 1' 
param_name = 'n_neighbors'
param_range = np.arange(1,9,2)
plt.figure(figsize = (16,5))
plotValidationCurves(X_train, y_train, knn, param_name, param_range, title)
printConfusionMatrix(y_test, knn_pred1)
plot_roc_curve(knn, X_test, y_test)
plt.show()
# Create KNN classifier
knn=KNeighborsClassifier(n_neighbors=7)
# Fit the classifier to the data
knn.fit(X_train,y_train)
#show first 5 model predictions on the test data
knn_pred2=knn.predict(X_test)
plt.figure(figsize=(16,5))
title='KNN Learning Curve 2'
plotLearningCurves(X_train,y_train,knn,title)
title = 'KNN Validation Curve 2' 
param_name = 'n_neighbors'
param_range = np.arange(1,9,2)
plt.figure(figsize = (16,5))
plotValidationCurves(X_train, y_train, knn, param_name, param_range, title)
printConfusionMatrix(y_test, knn_pred2)
plot_roc_curve(knn, X_test, y_test)
plt.show()
# Create KNN classifier
knn=KNeighborsClassifier(n_neighbors=10)
# Fit the classifier to the data
knn.fit(X_train,y_train)
#show first 5 model predictions on the test data
knn_pred3=knn.predict(X_test)
plt.figure(figsize=(16,5))
title='KNN Learning Curve 3'
plotLearningCurves(X_train,y_train,knn,title)
title = 'KNN Validation Curve 3' 
param_name = 'n_neighbors'
param_range = np.arange(1,9,2)
plt.figure(figsize = (16,5))
plotValidationCurves(X_train, y_train, knn, param_name, param_range, title)
printConfusionMatrix(y_test, knn_pred3)
plot_roc_curve(knn, X_test, y_test)
plt.show()
# Create KNN classifier
knn=KNeighborsClassifier(n_neighbors=20)
# Fit the classifier to the data
knn.fit(X_train,y_train)
#show first 5 model predictions on the test data
knn_pred4=knn.predict(X_test)
plt.figure(figsize=(16,5))
title='KNN Learning Curve 4'
plotLearningCurves(X_train,y_train,knn,title)
title = 'KNN Validation Curve 4' 
param_name = 'n_neighbors'
param_range = np.arange(1,9,2)
plt.figure(figsize = (16,5))
plotValidationCurves(X_train, y_train, knn, param_name, param_range, title)
printConfusionMatrix(y_test, knn_pred4)
plot_roc_curve(knn, X_test, y_test)
plt.show()
# Create KNN classifier
knn=KNeighborsClassifier(n_neighbors=17)
# Fit the classifier to the data
knn.fit(X_train,y_train)
#show first 5 model predictions on the test data
knn_pred5=knn.predict(X_test)
plt.figure(figsize=(16,5))
title='KNN Learning Curve 5'
plotLearningCurves(X_train,y_train,knn,title)
title = 'KNN Validation Curve 5' 
param_name = 'n_neighbors'
param_range = np.arange(1,9,2)
plt.figure(figsize = (16,5))
plotValidationCurves(X_train, y_train, knn, param_name, param_range, title)
printConfusionMatrix(y_test, knn_pred5)
plot_roc_curve(knn, X_test, y_test)
plt.show()
#create new a knn model
knn2=KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid= {'n_neighbors': np.arange(1, 20)}
#use gridsearch to test all values for n_neighbors
knn_gscv=GridSearchCV(knn2, param_grid, cv=5)
#fit model to data
knn_gscv.fit(X, y)

print("Best Params: " , knn_gscv.best_estimator_)
print("Best Score: " , knn_gscv.best_score_)
X = dfDrop.iloc[:, :-1].values
y = dfDrop.iloc[:, -1].values
# Encoding categorical inputs
encoder = OneHotEncoder(handle_unknown="ignore")
encoder.fit(X)
X = encoder.transform(X)

# 80/20 train split ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)
mlp = MLPClassifier(
    max_iter=3000,
    hidden_layer_sizes=[17, 13, 7], 
    solver="sgd", 
    random_state=1,
    verbose=False
).fit(X_train, y_train)

mlp_pred1 = mlp.predict(X_test)
def format_scores_as_dataframe(labels, train_scores, test_scores):
    learning_data = {"labels": [], "type": [], "score": []}

    for i in range(len(train_sizes)):
        for j in range(len(train_scores)):
            learning_data["labels"].append(labels[i])
            learning_data["type"].append("train")
            learning_data["score"].append(train_scores[i][j])
            learning_data["labels"].append(labels[i])
            learning_data["type"].append("test")
            learning_data["score"].append(test_scores[i][j])
            
    return pd.DataFrame.from_dict(learning_data)
train_sizes, train_scores, test_scores = learning_curve(mlp, X, y)

learning_curve_df = format_scores_as_dataframe(train_sizes, train_scores, test_scores)

# train and test learning scores results
ax = sns.lineplot(x="labels", y="score", hue="type", data=learning_curve_df, marker="o", ci=None)
ax.set_title("Learning Curve for MLP Algorithm")
dev_null = ax.set(xlabel="Samples", ylabel="Error")
scores = cross_val_score(mlp, X, y)

scores, scores.mean(), scores.std()

dev_null = sns.lineplot(x=[1,2,3,4,5], y=scores)
dev_null.set_title("Cross Score Distribution")
dev_null = dev_null.set(xlabel="# of runs", ylabel="Accuracy")

cross_val_result = cross_validate(mlp, X, y, return_train_score=True)


#validation_curve(mlp, X, y, param_name="alpha", param_range=[0.0001, 0.001, 0.05])
train_scores, test_scores = validation_curve(mlp, X, y, param_name="hidden_layer_sizes", param_range=([5], [10], [10,5], [15, 10], [25,10,5]))

val_curve_data = {"labels": [], "type": [], "scores": []}
param_ranges = ["[5]", "[10]", "[10,5]", "[15,10]", "[25,10,5]"]

for i in range(len(train_scores)):
    for j in range(len(train_scores[i])):
        val_curve_data["labels"].append(param_ranges[i])
        val_curve_data["type"].append("train")
        val_curve_data["scores"].append(train_scores[i][j])
        val_curve_data["labels"].append(param_ranges[i])
        val_curve_data["type"].append("test")
        val_curve_data["scores"].append(test_scores[i][j])
        
val_curve_df = pd.DataFrame.from_dict(val_curve_data)

ax = sns.lineplot(x="labels", y="scores", hue="type", data = val_curve_df, marker="o", ci=None)
ax.set_title("Validation Curve for our MLP model")
dev_null = ax.set(xlabel="Layers/Neurons", ylabel="Accuracy Score")

printConfusionMatrix(y_test, mlp_pred1)
plot_roc_curve(mlp, X_test, y_test)
plt.show()
mlp = MLPClassifier(
    max_iter=3000,
    hidden_layer_sizes=[17, 13, 7], 
    solver="sgd",
    activation="logistic",
    random_state=1,
    verbose=False
).fit(X_train, y_train)

mlp_pred2 = mlp.predict(X_test)
train_sizes, train_scores, test_scores = learning_curve(mlp, X, y)

learning_curve_df = format_scores_as_dataframe(train_sizes, train_scores, test_scores)

# train and test learning scores results
ax = sns.lineplot(x="labels", y="score", hue="type", data=learning_curve_df, marker="o", ci=None)
ax.set_title("Learning Curve for MLP Algorithm")
dev_null = ax.set(xlabel="Samples", ylabel="Error")
scores = cross_val_score(mlp, X, y)

scores, scores.mean(), scores.std()

dev_null = sns.lineplot(x=[1,2,3,4,5], y=scores)
dev_null.set_title("Cross Score Distribution")
dev_null = dev_null.set(xlabel="# of runs", ylabel="Accuracy")
cross_val_result = cross_validate(mlp, X, y, return_train_score=True)

#validation_curve(mlp, X, y, param_name="alpha", param_range=[0.0001, 0.001, 0.05])
train_scores, test_scores = validation_curve(mlp, X, y, param_name="hidden_layer_sizes", param_range=([5], [10], [10,5], [15, 10], [25,10,5]))

val_curve_data = {"labels": [], "type": [], "scores": []}
param_ranges = ["[5]", "[10]", "[10,5]", "[15,10]", "[25,10,5]"]

for i in range(len(train_scores)):
    for j in range(len(train_scores[i])):
        val_curve_data["labels"].append(param_ranges[i])
        val_curve_data["type"].append("train")
        val_curve_data["scores"].append(train_scores[i][j])
        val_curve_data["labels"].append(param_ranges[i])
        val_curve_data["type"].append("test")
        val_curve_data["scores"].append(test_scores[i][j])
        
val_curve_df = pd.DataFrame.from_dict(val_curve_data)

ax = sns.lineplot(x="labels", y="scores", hue="type", data = val_curve_df, marker="o", ci=None)
ax.set_title("Validation Curve for our MLP model")
dev_null = ax.set(xlabel="Layers/Neurons", ylabel="Accuracy Score")
printConfusionMatrix(y_test, mlp_pred2)

plot_roc_curve(mlp, X_test, y_test)
plt.show()
parameters = {
    "hidden_layer_sizes": [[8], [5]], #, [2], [8,8], [8,5], [5,8], [5,2], [2,2], [8,5,2], [8,5,5], [13,8,4], [17,13,7]
    "activation": ["identity", "logistic", "tanh", "relu"], 
    "solver": ["lbfgs", "sgd", "adam"], 
    "max_iter": [200, 500, ] #1000, 2000, 3000, 5000
}

# Brace yourself, this will take a while
mlp = MLPClassifier()
gs = GridSearchCV(mlp, parameters)
gs.fit(X_train, y_train)
gs.predict(X_test)

print("Best Params: " , gs.best_estimator_)
print("Best Score: " , gs.best_score_)

# Instantiate the classfiers and make a list
classifiers = [RandomForestClassifier(),
                MLPClassifier(), 
               svm.SVC(),
               KNeighborsClassifier()]

result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])


# print('auc =', auc)
lr_fpr1, lr_tpr1, _ = roc_curve(y_test, rf_pred3)
lr_fpr2, lr_tpr2, _ = roc_curve(y_test,  mlp_pred1)
lr_fpr3, lr_tpr3, _ = roc_curve(y_test, svm_pred1)
lr_fpr4, lr_tpr4, _ = roc_curve(y_test, knn_pred5)

# fpr , tpr, _= roc_curve(X_test, predict6_test)
auc1 = roc_auc_score(y_test, rf_pred3)
auc2 = roc_auc_score(y_test,  mlp_pred1)
auc3 = roc_auc_score(y_test, svm_pred1)
auc4 = roc_auc_score(y_test, knn_pred5)
 
    
result_table = result_table.append({'classifiers':RandomForestClassifier.__class__.__name__,
                                     'fpr':lr_fpr1, 
                                     'tpr':lr_tpr1, 
                                     'auc':auc1}, ignore_index=True)

result_table = result_table.append({'classifiers':MLPClassifier.__class__.__name__,
                                     'fpr':lr_fpr2, 
                                     'tpr':lr_tpr2, 
                                     'auc':auc2}, ignore_index=True)

result_table = result_table.append({'classifiers':svm.SVC.__class__.__name__,
                                     'fpr':lr_fpr3, 
                                     'tpr':lr_tpr3, 
                                     'auc':auc3}, ignore_index=True)

result_table = result_table.append({'classifiers':KNeighborsClassifier.__class__.__name__,
                                     'fpr':lr_fpr4, 
                                     'tpr':lr_tpr4, 
                                     'auc':auc4}, ignore_index=True)

fig = plt.figure(figsize=(8,6))

plt.plot(result_table.loc[0]['fpr'], 
         result_table.loc[0]['tpr'], 
         label="RandomForestClassifier, AUC={:.3f}".format( result_table.loc[0]['auc']))

plt.plot(result_table.loc[1]['fpr'], 
         result_table.loc[1]['tpr'], 
         label="MLPClassifier, AUC={:.3f}".format( result_table.loc[1]['auc']))

plt.plot(result_table.loc[2]['fpr'], 
         result_table.loc[2]['tpr'], 
         label="SVM, AUC={:.3f}".format( result_table.loc[2]['auc']))

plt.plot(result_table.loc[3]['fpr'], 
         result_table.loc[3]['tpr'], 
         label="KNeighborsClassifier, AUC={:.3f}".format( result_table.loc[3]['auc']))


plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()
    
!apt-get remove swig 
!apt-get install swig3.0 build-essential -y
!ln -s /usr/bin/swig3.0 /usr/bin/swig
!apt-get install build-essential
!pip install --upgrade setuptools
!pip install auto-sklearn
X = dfDrop.iloc[:, :-1].values
y = dfDrop.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import os  
import autosklearn.regression


automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    tmp_folder='/tmp/autosklearn_cv_example_tmp5',
    output_folder='/tmp/autosklearn_cv_example_out5',
    delete_tmp_folder_after_terminate=False,
    resampling_strategy='cv',
    resampling_strategy_arguments={'folds': 5},
)

# fit() changes the data in place, but refit needs the original data. We
# therefore copy the data. In practice, one should reload the data
automl.fit(X_train.copy(), y_train.copy(), dataset_name='Students')
# During fit(), models are fit on individual cross-validation folds. To use
# all available data, we call refit() which trains all models in the
# final ensemble on the whole dataset.
automl.refit(X_train.copy(), y_train.copy())

print(automl.show_models())

predictions = automl.predict(X_test)
print("Accuracy as per AutoML: ", sklearn.metrics.accuracy_score(y_test, predictions))
