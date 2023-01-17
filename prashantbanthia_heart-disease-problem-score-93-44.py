# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")

df.head(2)
df.isna().sum()
df.target.value_counts()
df.target.value_counts().plot(kind='bar',color=['orange','green'])
df.info()
df.describe()
#Heart Disease with respect to Sex

pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(10,6))

plt.title("Heart Disease vs Sex")

plt.xlabel("0=No Disease & 1=Disease")

plt.ylabel("Count")

plt.legend(["female","Male"]);
#Age vs Max Heart rate(thalach)

plt.figure(figsize=(10,6))



plt.scatter(df.age[df.target==1],

           df.thalach[df.target==1])



plt.scatter(df.age[df.target==0],

           df.thalach[df.target==0])



plt.legend(["Disease","No Disease"])

plt.xlabel("Age")

plt.ylabel("Heart Rate")

plt.title("Age vs Heart Rate");
pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(10,6))

plt.xlabel("Chest Pain Type")

plt.title("Chest Pain vs Target")

plt.ylabel("Count")

plt.legend([" No Disease","Disease"])

plt.xticks(rotation = 0);
df.corr()
#Plot Correlation matrix using seaborn 

corr_mat=df.corr()

plt.figure(figsize=(15,10))

sns.heatmap(corr_mat,

           annot=True,

           linewidth=0.5,

           fmt= ".2f");
cp=pd.get_dummies(df['cp'],prefix='cp',drop_first=True)

thal=pd.get_dummies(df['thal'],prefix='thal',drop_first=True)

slope=pd.get_dummies(df['slope'],prefix='slope',drop_first=True)
new_df=pd.concat([df,cp,thal,slope],axis=1)

new_df=new_df.drop(['cp','thal','slope'],axis=1)

new_df.head()
X = new_df.drop("target", axis=1)



# Target variable

y = new_df['target']



#Normalize X

X = (X - X.min())/(X.max()-X.min())



#Split data into training and test data

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score,RandomizedSearchCV,GridSearchCV

#Store all models in a Dictionary

np.random.seed(42)

models={'Logistic Regression':LogisticRegression(max_iter=1000),

       'RandomForestClassifier':RandomForestClassifier(),

        "GradientBoostingClassifier":GradientBoostingClassifier(),

       "Knn":KNeighborsClassifier(),

        "GaussianNB":GaussianNB(),

       "SVC":SVC(kernel='linear')}

#Store result in an empty dictionary

scores={}



#Write a function to fit,train and test a model and store in result dictionary

def fit_and_score(models,X_train,X_test,y_train,y_test):

    for model_name,model in models.items():

        model.fit(X_train,y_train)

        scores[model_name]=model.score(X_test,y_test)

    return scores
fit_and_score(models,X_train,X_test,y_train,y_test)
scores['Knn']
pd.DataFrame(scores,index=['accuracy']).T.plot(kind='bar',figsize=(10,6))

plt.yticks(np.arange(0,1.1,0.05))

plt.grid();
# 1. Let us tune our KNN model wuth different values of n_neighbors

train_scores = []



# Create a list of test scores

test_scores = []



# Create a list of different values for n_neighbors

neighbors = range(1, 21) # 1 to 20



# Setup algorithm

knn = KNeighborsClassifier()



# Loop through different neighbors values

for i in neighbors:

    knn.set_params(n_neighbors = i) # set neighbors value

    

    # Fit the algorithm

    knn.fit(X_train, y_train)

    

    # Update the training scores

    train_scores.append(knn.score(X_train, y_train))

    

    # Update the test scores

    test_scores.append(knn.score(X_test, y_test))

    

plt.plot(neighbors, train_scores, label="Train score")

plt.plot(neighbors, test_scores, label="Test score")

plt.xticks(np.arange(1, 21, 1))

plt.xlabel("Number of neighbors")

plt.ylabel("Model score")

plt.legend()

 

knn_score=max(test_scores)

print(f"Maximum KNN score on the test data: {knn_score*100:.2f}%")
# 2. Let us tune our Logistic Regression and Random Forest model using RandomizedSearchCV

#Logistic Regression hyperparameter grid

log_reg_grid={"C":np.logspace(-4,4,20),

             "solver":['newton-cg','liblinear','sag','saga']}

#RandomForest hyerparameter grid

rf_grid={"n_estimators":np.arange(10,1000,50),

        "max_depth":[None,1,2,3,5,10,15,22,25],

        "min_samples_split":np.arange(2,40,2),

        "min_samples_leaf":np.arange(1,40,2)}
#Set up random hyperparameter search for Logistic Regression

np.random.seed(42)

rs_log_reg=RandomizedSearchCV(LogisticRegression(),param_distributions=log_reg_grid,n_iter=100,cv=5,verbose=True)



rs_log_reg.fit(X_train,y_train);
#Best parameters for LogisticRegression Model

rs_log_reg.best_params_
#Accuracy score for LogisticRegression Model

rs_log_reg_score=rs_log_reg.score(X_test,y_test)

rs_log_reg_score
np.random.seed(42)

rs_rf=RandomizedSearchCV(RandomForestClassifier(),param_distributions=rf_grid,n_iter=20,cv=5,verbose=True)



rs_rf.fit(X_train,y_train)
rs_rf.best_params_
rs_rf_score=rs_rf.score(X_test,y_test)

rs_rf_score
clf_model=['Logistic Regression','RandomForestClassifier','GradientBoostingClassifier','Knn','GaussianNB','SVC']

accuracy=[rs_log_reg_score,rs_rf_score,scores['GradientBoostingClassifier'],knn_score,scores['GaussianNB'],scores['SVC']]

plt.bar(x=clf_model,height=accuracy)

plt.title('Model Accuracy')

plt.grid()

plt.xticks(rotation=90)

plt.yticks(np.arange(0,1.1,0.1));
#Predicted values for y_test

gnb=GaussianNB()

gnb.fit(X_train,y_train)

y_preds = gnb.predict(X_test)

y_preds
y_test
#Plot ROC Curve

from sklearn.metrics import plot_roc_curve,confusion_matrix,plot_confusion_matrix,classification_report

plot_roc_curve(gnb,X_test,y_test)
#Plot confusion matrix

confusion_matrix(y_test,y_preds)
plot_confusion_matrix(gnb,X_test,y_test)
#Print Calssification Report

print(classification_report(y_test,y_preds))
from sklearn.model_selection import cross_val_score

cv_acc= cross_val_score(gnb,

                        X,

                        y,

                        cv=5,  #5-fold cross validation

                       scoring='accuracy') # scoring parameter as accuracy



cv_acc
#Take mean of these values

cv_acc=np.mean(cv_acc)

cv_acc
#Calculate rest of paramters

cv_precision=np.mean(cross_val_score(gnb,

                        X,

                        y,

                        cv=5,  

                       scoring='precision'))



cv_precision 
cv_recall=np.mean(cross_val_score(gnb,

                        X,

                        y,

                        cv=5,  

                       scoring='recall'))



cv_recall 
cv_f1=np.mean(cross_val_score(gnb,

                        X,

                        y,

                        cv=5,  

                       scoring='f1'))



cv_f1 
# Let us visualize them

cv_metrics=pd.DataFrame({'Accuracy':cv_acc,

                        'Precision':cv_precision,

                        'Recall':cv_recall,

                        'F1-score':cv_f1},index=[0])

cv_metrics.T.plot(kind='bar',legend=False)

plt.title("Classification Metrics for GaussianNB")

plt.yticks(np.arange(0,1.1,0.1))

plt.grid();