# -*- coding: utf-8 -*-

"""

Created on Wed Jun 24 17:06:46 2020



@author: DEEXITH REDDY

"""





import pandas as pd

import numpy as np



df=pd.read_csv("bank-additional-full.csv",header=0,sep=";")



##Creating dummy variables:



categorial=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']

for var in categorial:

    cat_list='var'+'_'+var

    cat_list = pd.get_dummies(df[var], prefix=var)

    df1=df.join(cat_list)

    df=df1

categorical=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']

df2=df.columns.values.tolist()

to_keep=[i for i in df2 if i not in categorical]



df3=df[to_keep]

df3.columns.values



df_final=df[to_keep]

df_final.columns.values



df=df_final





##Dropping columns since categorical columns for these have already been made



df=df.drop(['job','marital','education','default','housing','loan','contact','month','day_of_week','duration'],axis=1)





##Creating target variable for prediction

y=df(['y'])



df=df.drop(['y'],axis=1)



##Counting the actual no and yes



y.value_counts()



##Visulaizing all features

df.hist(edgecolor='black', linewidth=1.2, figsize=(20, 20))



##Visualizaing theheatmap



plt.figure(figsize=(30, 30))

sns.heatmap(df.corr(), annot=True, cmap="RdYlGn", annot_kws={"size":15})



##Splitting the dataset into training and testing



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)





##Decison Tree Classifier and predicting the model

from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier()



##Fitting the model on the dataset



clf.fit(X_train,y_train)



##Checking the decision tree classifer model without fine tuning the parameters

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score



def print_score(clf, X_train, y_train, X_test, y_test, train=True):

    if train:

        pred = clf.predict(X_train)

        print("Train Result:\n===========================================")

        print(f"accuracy score: {accuracy_score(y_train, pred):.4f}\n")

       

    elif train==False:

        pred = clf.predict(X_test)

        print("Test Result:\n===========================================")        

        print(f"accuracy score: {accuracy_score(y_test, pred)}\n")

        

print_score(clf, X_train, y_train, X_test, y_test, train=True)

print_score(clf, X_train, y_train, X_test, y_test, train=False)  



##Using grid search to finetune the model:

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV



params = {

    "criterion":("gini", "entropy"), 

    "splitter":("best", "random"), 

    "max_depth":(list(range(1, 20))), 

    "min_samples_split":[2, 3, 4], 

    "min_samples_leaf":list(range(1, 20)), 

}





model = DecisionTreeClassifier(random_state=42)

grid_search_cv = GridSearchCV(model, params, scoring="accuracy", n_jobs=-1, verbose=1, cv=3)



##Checking accuracy

print_score(grid_search_cv.best_estimator_, X_train, y_train, X_test, y_test, train=True)

print_score(grid_search_cv.best_estimator_, X_train, y_train, X_test, y_test, train=False)



##Random Forest importin and fitting

from sklearn.ensemble import RandomForestClassifier



rand_forest = RandomForestClassifier(n_estimators=100)

rand_forest.fit(X_train, y_train)



print_score(rand_forest, X_train, y_train, X_test, y_test, train=True)

print_score(rand_forest, X_train, y_train, X_test, y_test, train=False)





##Adaboost



# Classification-tree 'dt'



from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score



dt = DecisionTreeClassifier(max_depth=1, random_state=42)



# Instantiate an AdaBoost classifier 'adb_clf'

adb_clf = AdaBoostClassifier(base_estimator=dt, n_estimators=100)



# Fit 'adb_clf' to the training set

adb_clf.fit(X_train, y_train)



y_pred_proba = adb_clf.predict_proba(X_test)[:,1]



print_score(adb_clf, X_train, y_train, X_test, y_test, train=True)

print_score(adb_clf, X_train, y_train, X_test, y_test, train=False)