
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
plt.style.use('ggplot')
#ggplot is R based visualisation package that provides better graphics with higher level of abstraction
#Loading the dataset
diabetes_data = pd.read_csv('../input/diabetes.csv')


## gives information about the data types,columns, null value counts, memory usage etc
## function reference : https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.info.html
diabetes_data.info(verbose=True)
## basic statistic details about the data (note only numerical columns would be displayed here unless parameter include="all")
## for reference: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html#pandas.DataFrame.describe
diabetes_data.describe()

## Also see :
##to return columns of a specific dtype: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.select_dtypes.html#pandas.DataFrame.select_dtypes
## observing the shape of the data
diabetes_data.shape
## data type analysis
#plt.figure(figsize=(5,5))
#sns.set(font_scale=2)
sns.countplot(y=diabetes_data.dtypes ,data=diabetes_data)
plt.xlabel("count of each data type")
plt.ylabel("data types")
plt.show()
## null count analysis
import missingno as msno
p=msno.bar(diabetes_data)

## checking the balance of the data by plotting the count of outcomes by their value
color_wheel = {1: "#0392cf", 
               2: "#7bc043"}
colors = diabetes_data["Outcome"].map(lambda x: color_wheel.get(x + 1))
print(diabetes_data.Outcome.value_counts())
p=diabetes_data.Outcome.value_counts().plot(kind="bar")

from pandas.tools.plotting import scatter_matrix
p=scatter_matrix(diabetes_data,figsize=(25, 25))

gi = []
stbmi = []
for i in range( 768 ):
    if( diabetes_data.Glucose[i] == 0 ) :
        gi.append(diabetes_data.Insulin[i] / 5 )
        
    else:
        gi.append(   diabetes_data.Insulin[i] /  diabetes_data.Glucose[i]  ) 
    if( diabetes_data.BMI[i] == 0 ):
        stbmi.append( diabetes_data.SkinThickness[i] / 10 )
    else:
        stbmi.append( diabetes_data.SkinThickness[i] / diabetes_data.BMI[i] )
        

seria = pd.Series( gi )
seria2 = pd.Series( stbmi )
seria2.name = "stbmi"
seria.name = "gi"
diabetes_data = diabetes_data.join(seria ).drop( 'Insulin' , 1 ).join(seria2).drop('SkinThickness',1)
from pandas.tools.plotting import scatter_matrix
p=scatter_matrix(diabetes_data,figsize=(25, 25))
p=sns.pairplot(diabetes_data, hue = 'Outcome')
plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(diabetes_data.corr(), annot=True)  # seaborn has very simple solution for heatmap
X = diabetes_data.drop("Outcome",axis = 1)
y = diabetes_data.Outcome
#importing train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42, stratify=y)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

test_scores = []
train_scores = []
validation_scores = []
X_train_values = X_train.values
y_train_values = y_train.values

## cross validation with KFold algorithm
kfold = KFold(5, shuffle=True, random_state=42)

it = 40

for i in range(1,it):

    knn = KNeighborsClassifier(i)

    tr_scores = []
    ts_scores = []
    for train_ix, test_ix in kfold.split(X_train_values):
        # define train/test X/y
        X_train_fold, y_train_fold = X_train_values[train_ix],y_train_values[train_ix]
        X_test_fold, y_test_fold = X_train_values[test_ix], y_train_values[test_ix]
        knn.fit(X_train_fold,y_train_fold)
        ts_scores.append(knn.score(X_test_fold,y_test_fold))
        tr_scores.append(knn.score(X_train_fold,y_train_fold))
    validation_scores.append(np.mean(ts_scores))
    train_scores.append(np.mean(tr_scores))
    test_scores.append(knn.score(X_test,y_test))
## score that comes from testing on the same datapoints that were used for training
train_scores
## score that comes from testing on the datapoints that were left out in KFold to be used for validation
validation_scores
## score that comes from testing on the datapoints that were split in the beginning to be used for testing solely
test_scores
plt.title('k-NN Varying number of neighbors')
plt.plot(range(1,it),test_scores,label="Test")
plt.plot(range(1,it),validation_scores,label="Validation")
plt.plot(range(1,it),train_scores,label="Train")
plt.legend()
plt.xticks(range(1,it))
plt.show()
#Setup a knn classifier with k neighbors
kfold = KFold(5, shuffle=True, random_state=42)
knn = KNeighborsClassifier(23)

for train_ix, test_ix in kfold.split(X_train_values):
        # define train/test X/y
        X_train_fold, y_train_fold = X_train_values[train_ix],y_train_values[train_ix]
        X_test_fold, y_test_fold = X_train_values[test_ix], y_train_values[test_ix]
        knn.fit(X_train_fold,y_train_fold)
#knn.fit(X_train,y_train)
knn.score(X_test,y_test)
#import confusion_matrix
from sklearn.metrics import confusion_matrix
#let us get the predictions using the classifier we had fit above
y_pred = knn.predict(X_test)
confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

#import classification_report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
from sklearn.metrics import roc_curve
y_pred_proba = knn.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Knn(n_neighbors=12) ROC curve')
plt.show()
#Area under ROC curve
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_proba)
#import GridSearchCV
from sklearn.model_selection import GridSearchCV
#In case of classifier like knn the parameter to be tuned is n_neighbors
param_grid = {'n_neighbors':np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(X,y)

print("Best Score:" + str(knn_cv.best_score_))
print("Best Parameters: " + str(knn_cv.best_params_))

