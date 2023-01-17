# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Importing necessary libraries
from mlxtend.plotting import plot_decision_regions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Apply the default default seaborn theme, scaling, and color palette.
sns.set() 
import warnings
warnings.filterwarnings('ignore')
#With this backend, the output of plotting commands is displayed inline within frontends like the Jupyter notebook, directly below the code cell that produced it. 
#The resulting plots will then also be stored in the notebook document.
%matplotlib inline

#Loading the dataset
diabetes_data = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")

#Print the first 5 rows of the dataframe
diabetes_data.head()
diabetes_data.isna().any(axis=0)

diabetes_data.info(verbose=True)
diabetes_data.describe()
diabetes_copy = diabetes_data.copy(deep = True)
diabetes_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)


print(diabetes_copy.isnull().sum())
p = diabetes_data.hist(figsize = (20,20))
diabetes_copy['Glucose'].fillna(diabetes_copy['Glucose'].mean(), inplace=True)
diabetes_copy['BloodPressure'].fillna(diabetes_copy['BloodPressure'].mean(), inplace=True)
diabetes_copy['SkinThickness'].fillna(diabetes_copy['SkinThickness'].mean(), inplace=True)
diabetes_copy['Insulin'].fillna(diabetes_copy['Insulin'].mean(), inplace=True)
diabetes_copy['BMI'].fillna(diabetes_copy['BMI'].mean(), inplace=True)

p = diabetes_copy.hist(figsize = (20,20))

print(diabetes_data["Outcome"].value_counts())
p=diabetes_data["Outcome"].value_counts().plot(kind="bar")
diabetes_data.corr()
import seaborn as sns
from matplotlib import pyplot as plt
sns.pairplot(diabetes_data,diag_kind='kde',hue='Outcome')
plt.figure(figsize=(12,10))  
p=sns.heatmap(diabetes_data.corr(), annot=True,cmap ='RdYlGn')  
plt.figure(figsize=(12,10))  
p=sns.heatmap(diabetes_copy.corr(), annot=True,cmap ='RdYlGn')  
from sklearn.preprocessing import StandardScaler
X_sc = StandardScaler()
X = pd.DataFrame(X_sc.fit_transform(diabetes_copy.drop(["Outcome"],axis = 1),),
                columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPredigreeFunction','Age'])
X.head()
y = diabetes_copy.Outcome
y.head()
#importing train_test_split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3, random_state = 42, stratify = y)
from sklearn.neighbors import KNeighborsClassifier

test_scores = []
train_scores = []

for i in range(1,15):
    
    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))
    
## score that comes from testing on the same datapoints that were used for training
max_train_score = max(train_scores)
train_scores_ind = [i for i,v in enumerate(train_scores) if v == max_train_score]
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x : x+1, train_scores_ind))))
## score that comes from testing on the datapoints that were split in the beginning to be used for testing solely
max_test_score = max(test_scores)
test_scores_ind = [i for i,v  in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100, list(map(lambda x: x+1, test_scores_ind))))
plt.figure(figsize=(12,5))
p = sns.lineplot(range(1,15),train_scores,marker='*', label='Train Score')
p = sns.lineplot(range(1,15),test_scores,marker='o',  label='Test Score')
# Setting knn clasifier with k neighbors
knn = KNeighborsClassifier(11)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)
## trying to plot decision boundary
value = 20000
width = 20000

plot_decision_regions(X.values, y.values, clf= knn, legend=2,
                     filler_feature_values={2: value, 3: value, 4:value, 5:value, 6:value, 7:value},
                     filler_feature_ranges={2: width, 3: width, 4:width, 5:width, 6:width, 7:width},
                     X_highlight=X_test.values)

plt.title('KNN with Diabetes Data')
plt.show()

# import confustion_matrix
from sklearn.metrics import confusion_matrix
#let us get the predictions using the classifier we had fit above
y_pred = knn.predict(X_test)
confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
y_pred = knn.predict(X_test)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
# import classification_report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
from sklearn.metrics import roc_curve
y_pred_proba = knn.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Knn(n_neighbors=11) ROC curve')
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