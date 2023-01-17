# load libraries for performing matrix operations



import pandas as pd

import numpy as np
#load data file



df = pd.read_csv("../input/data.csv")
#load libraries for data visualization



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
print("This dataset has over {0} rows and {1} columns." .format(df.shape[0], df.shape[1]))
df.columns
#lists all the columns with NaN values in it

for parameter in df.columns:

    if (df[parameter].isnull().sum() != 0):

        print(parameter)
percentage = pd.value_counts(df["diagnosis"]).apply(lambda x: x/len(df['diagnosis'])*100)



plt.pie(percentage, labels = ['Benign', 'Malignant'], autopct='%1.2f%%')
#setting up the dataset for modelling



target = df["diagnosis"].map({"M": 1, "B": 0}) #dependent variable



train = df.drop(["diagnosis", "id", "Unnamed: 32"], axis =1) #Independent variables, influence the dependent variable
from sklearn.preprocessing import StandardScaler



from sklearn.dummy import DummyClassifier



from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.metrics import confusion_matrix

import time



from sklearn.pipeline import make_pipeline
#make a function for model evaluation

cv = StratifiedKFold(n_splits = 10, random_state = 40)



SS = StandardScaler()



def specificity(model):

    total_cm = np.array([[0,0], [0,0]])

    

    start_time = time.time()

    

    for train_index, test_index in cv.split(train, target):

        #print("TRAIN:", train_index, "TEST:", test_index)

        X_train, X_test = train.iloc[train_index], train.iloc[test_index]

        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        

        cm = confusion_matrix(y_test, model.fit(SS.fit_transform(X_train), y_train).predict(SS.fit(X_train).transform(X_test)))

        total_cm = total_cm + cm 

    

    end_time = time.time()

    

    #specificity (true negative rate)  = true negative/ (true negative + false posisitve)

    specificity_score = total_cm[1,1] / (total_cm[1,1] + total_cm[0,1]) * 100

    

    print("Specificity of the  model is: {:.2f}" .format(specificity_score))

    print("Model execution time: {:.4f} seconds" .format(end_time - start_time))
baseline = DummyClassifier()
specificity(baseline)
#load linear model libraries

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



#load non linear models

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
#place models into  a single list

models = []



#linear models

models.append(('Logistic Regression', LogisticRegression()))

models.append(("Linear Discrimnant Analysis", LinearDiscriminantAnalysis()))



#non-linear models

models.append(('K Nearest Neighbors', KNeighborsClassifier()))

models.append(('Naive Bayes', GaussianNB()))

models.append(('Support Vector Machine', SVC()))



for name, classification_model in models:

    print("{0}" .format(name))

    specificity(classification_model)

    print()

    
penalty_reg = ["l2", "l1"]

C_values = [0.001, 0.01, 0.1, 1, 10, 100]



#initialize

best_score = 0

best_params = {"C": None, "penalty": None}



for penalty in penalty_reg:

    for C in C_values:

        logit = LogisticRegression(penalty = penalty, C = C) 

        

        total_cm = np.array([[0,0], [0,0]])

    

        for train_index, test_index in cv.split(train, target):

            #print("TRAIN:", train_index, "TEST:", test_index)

            X_train, X_test = train.iloc[train_index], train.iloc[test_index]

            y_train, y_test = target.iloc[train_index], target.iloc[test_index]

    

            cm = confusion_matrix(y_test, logit.fit(SS.fit_transform(X_train), y_train).predict(SS.fit(X_train).transform(X_test)))

            total_cm = total_cm + cm 



        #specificity (true negative rate)  = true negative/ (true negative + false posisitve)

        specificity_score = total_cm[1,1] / (total_cm[1,1] + total_cm[0,1]) * 100



        score = specificity_score

        

        if score > best_score:

            best_score = score

            best_params['C'] = C

            best_params['penalty'] = penalty



#print best score

best_score, best_params
final_model = LogisticRegression(penalty = 'l2', C = 0.1)
specificity(final_model)