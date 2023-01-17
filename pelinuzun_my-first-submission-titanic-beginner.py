# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Load Data

train_data = pd.read_csv("../input/train.csv")
#get info about data

train_data.info()
x = train_data.columns.values

print(x)
train_data.head(10)
#how many missing values exist in the collection 

train_data.isnull().sum()
train_data.fillna(train_data['Age'].dropna().median(), inplace = True)
train_data.shape
train_data.plot(kind = "scatter", x = "Fare", y = "Survived", color = "r", linewidth = 1)

plt.show()
sns.barplot(x = "Pclass", y = "Survived", data = train_data)
sns.barplot(x = "Sex", y = "Survived", data = train_data)
#correlation 

f, ax = plt.subplots(figsize = (18, 18))

sns.heatmap(train_data.corr(), linewidth = 1, annot = True, fmt = '.1f', ax=ax)

plt.show()
#extract title from names - train_data

train_data['Title'] = train_data['Name'].str.extract("([A-za-a]+)\.", expand = False) 
train_data

#now we have Title column as well
#delete unnecessary features from dataset

train_data.drop('Name', axis = 1, inplace = True)

train_data.head(10)
train_data
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3,

                "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3, "Countless": 3, 

                "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona": 3, "Mme": 3, "Capt": 3, "Sir": 3}



train_data['Title'] = train_data['Title'].map(title_mapping)
#Sex mapping male: 0 female: 1

sex_mapping = {"male": 0, "female": 1}

train_data['Sex'] = train_data['Sex'].map(sex_mapping)
train_data.head(10)
train_data.loc[train_data['Age'] <= 16, 'Age'] = 0

train_data.loc[(train_data['Age'] > 16) & (train_data['Age'] <= 26), 'Age'] = 1

train_data.loc[(train_data['Age'] > 26) & (train_data['Age'] <= 36), 'Age'] = 2

train_data.loc[(train_data['Age'] > 36) & (train_data['Age'] <= 62), 'Age'] = 3

train_data.loc[train_data['Age'] > 62, 'Age'] = 4
train_data.head(10)
#embarked mapping

embarked_mapping = {"S": 0, "C": 1, "Q": 2}

train_data['Embarked'] = train_data["Embarked"].map(embarked_mapping)
#fare categorising

train_data.loc[train_data['Fare'] <= 17, 'Fare'] = 0

train_data.loc[(train_data['Fare']> 17) & (train_data['Fare'] <= 29), 'Fare'] = 1

train_data.loc[(train_data['Fare']> 29) & (train_data['Fare'] <= 100), 'Fare'] = 2

train_data.loc[train_data['Fare']>100, 'Fare'] = 3
train_data.head(10)
train_data = train_data.drop(['Cabin', 'Ticket'], axis = 1)

train_data.head(10)
#create new attributes called FamilySize

train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1

train_data.head(10)
#family mapping

family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2.0, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4.0}

train_data['FamilySize'] = train_data['FamilySize'].map(family_mapping)
train_data.head(10)
#we can remove unnecessary features from the dataset

features_drop = ['SibSp', 'Parch']

train_data = train_data.drop(features_drop, axis = 1)

train_data = train_data.drop(['PassengerId'], axis = 1)

target = train_data['Survived']

train = train_data.drop(['Survived'], axis = 1)

train_data.head(10)
x_train = train_data.drop("Survived", axis = 1)

y_train = train_data['Survived']
x_train.fillna(x_train.dropna().median(), inplace = True)
y_train.fillna(y_train.dropna().median(), inplace = True)
#remove feature by using backward elimination

import statsmodels.formula.api as sm

regressor_OLS = sm.OLS(y_train, x_train).fit()

regressor_OLS.summary()
x_train = x_train.loc[:, ['Pclass', 'Sex', 'Age', 'Fare', 'Title', 'FamilySize']]

regressor_OLS = sm.OLS(y_train, x_train).fit()

regressor_OLS.summary()
x_train = x_train.loc[:, ['Pclass', 'Sex', 'Fare', 'Title', 'FamilySize']]

regressor_OLS = sm.OLS(y_train, x_train).fit()

regressor_OLS.summary()
#I am gonna create my own test-set from a training set rather than using existing test-set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2)
#Scaling the data

from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)
#Logistic Regression

from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression()

logistic_regression.fit(x_train, y_train)

y_predict = logistic_regression.predict(x_test)

#score returns the mean accuracy on the given test data and labels

accuracy_lr = round(logistic_regression.score(x_train, y_train) * 100, 2)

print("Logistic Regression Accuracy: " +str(accuracy_lr))
from sklearn.metrics import roc_curve, confusion_matrix, auc

def evalBinaryClassifier(model,x, y, labels=['Positives','Negatives']):

    '''

    Visualize the performance of  a Logistic Regression Binary Classifier.

    

    Displays a labelled Confusion Matrix, distributions of the predicted

    probabilities for both classes, the ROC curve, and F1 score of a fitted

    Binary Logistic Classifier. Author: gregcondit.com/articles/logr-charts

    

    Parameters

    ----------

    model : fitted scikit-learn model with predict_proba & predict methods

        and classes_ attribute. Typically LogisticRegression or 

        LogisticRegressionCV

    

    x : {array-like, sparse matrix}, shape (n_samples, n_features)

        Training vector, where n_samples is the number of samples

        in the data to be tested, and n_features is the number of features

    

    y : array-like, shape (n_samples,)

        Target vector relative to x.

    

    labels: list, optional

        list of text labels for the two classes, with the positive label first

        

    Displays

    ----------

    3 Subplots

    

    Returns

    ----------

    F1: float

    '''

    #model predicts probabilities of positive class

    p = model.predict_proba(x)

    if len(model.classes_)!=2:

        raise ValueError('A binary class problem is required')

    if model.classes_[1] == 1:

        pos_p = p[:,1]

    elif model.classes_[0] == 1:

        pos_p = p[:,0]

    

    #FIGURE

    plt.figure(figsize=[15,4])

    

    #1 -- Confusion matrix

    cm = confusion_matrix(y,model.predict(x))

    plt.subplot(131)

    ax = sns.heatmap(cm, annot=True, cmap='Blues', cbar=False, 

                annot_kws={"size": 14}, fmt='g')

    cmlabels = ['True Negatives', 'False Positives',

              'False Negatives', 'True Positives']

    for i,t in enumerate(ax.texts):

        t.set_text(t.get_text() + "\n" + cmlabels[i])

    plt.title('Confusion Matrix', size=15)

    plt.xlabel('Predicted Values', size=13)

    plt.ylabel('True Values', size=13)

      

    #2 -- Distributions of Predicted Probabilities of both classes

    df = pd.DataFrame({'probPos':pos_p, 'target': y})

    plt.subplot(132)

    plt.hist(df[df.target==1].probPos, density=True, 

             alpha=.5, color='green',  label=labels[0])

    plt.hist(df[df.target==0].probPos, density=True, 

             alpha=.5, color='red', label=labels[1])

    plt.axvline(.5, color='blue', linestyle='--', label='Boundary')

    plt.xlim([0,1])

    plt.title('Distributions of Predictions', size=15)

    plt.xlabel('Positive Probability (predicted)', size=13)

    plt.ylabel('Samples (normalized scale)', size=13)

    plt.legend(loc="upper right")

    

    #3 -- ROC curve with annotated decision point

    fp_rates, tp_rates, _ = roc_curve(y,p[:,1])

    roc_auc = auc(fp_rates, tp_rates)

    plt.subplot(133)

    plt.plot(fp_rates, tp_rates, color='green',

             lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='grey')

    #plot current decision point:

    tn, fp, fn, tp = [i for i in cm.ravel()]

    plt.plot(fp/(fp+tn), tp/(tp+fn), 'bo', markersize=8, label='Decision Point')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate', size=13)

    plt.ylabel('True Positive Rate', size=13)

    plt.title('ROC Curve', size=15)

    plt.legend(loc="lower right")

    plt.subplots_adjust(wspace=.3)

    plt.show()

    #Print and Return the F1 score

    tn, fp, fn, tp = [i for i in cm.ravel()]

    precision = tp / (tp + fp)

    recall = tp / (tp + fn)

    F1 = 2*(precision * recall) / (precision + recall)

    printout = (

        f'Precision: {round(precision,2)} | '

        f'Recall: {round(recall,2)} | '

        f'F1 Score: {round(F1,2)} | '

    )

    print(printout)

    return F1
F1 = evalBinaryClassifier(logistic_regression, x_test, y_predict)

data_to_submit = pd.DataFrame({

    'Survived': y_predict, 

    'Test Target': y_test

})
data_to_submit
data_to_submit.to_csv('csv_to_submit.csv', index = False)