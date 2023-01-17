# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/data.csv")

df.head()


#we dont need id coloum and last coloum unnamed

df.drop(["Unnamed: 32","id"],axis=1,inplace=True)

df.describe(include="all")
import seaborn as sns

sns.set(rc={'figure.figsize':(5,5)})

sns.heatmap(df.isnull())
import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize':(20,20)})

sns.heatmap( df.corr(),annot=True)

# Create correlation matrix

corr_matrix = df.corr().abs()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.95

positive_to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

positive_to_drop=np.asarray(positive_to_drop)



negative_to_drop = [column for column in upper.columns if any(upper[column] < -0.90)]

negative_to_drop=np.asarray(negative_to_drop)



for i in positive_to_drop:

    df.drop(i,axis=1,inplace=True)

    

for i in negative_to_drop:

    df.drop(i,axis=1,inplace=True)

    
sns.set(rc={'figure.figsize':(20,20)})

sns.heatmap( df.corr(),annot=True)
df["diagnosis"].replace("M",1,inplace=True)

df["diagnosis"].replace("B",0,inplace=True)

df=df.apply(pd.to_numeric)
fig = plt.figure(figsize = (20, 25))

j = 0

#Droping_Characters and string coloums because graph donot support them



for i in df.columns:

    plt.subplot(7, 7, j+1)

    j += 1

    sns.distplot(df[i][df['diagnosis']==1], color='r', label = 'malignant')

    sns.distplot(df[i][df['diagnosis']==0], color='g', label = 'benign')

    plt.legend(loc='best')

fig.suptitle('Breast Cancer ')

fig.tight_layout()

fig.subplots_adjust(top=0.95)

plt.show()
#outliers

sns.set(rc={'figure.figsize':(5,5)})

for column in df:

    plt.figure()

    sns.boxplot(x=df[column])


df_X=df.drop("diagnosis",axis=1)

Q1 = df_X.quantile(0.25)

Q3 = df_X.quantile(0.75)

IQR = Q3 - Q1

df_X = df_X[~((df_X < (Q1 - 1.5 * IQR)) |(df_X > (Q3 + 1.5 * IQR))).any(axis=1)]

print("With Outliers Valuse",len(df))

print("Removed Outliers Valuse",len(df_X))

print("Values Removed ",str(len(df)-len(df_X)))


df_y=df["diagnosis"]

new_df=df_X.merge(df_y.to_frame(), left_index=True, right_index=True)



new_df.head(10)
from sklearn.model_selection import train_test_split

X=new_df.drop("diagnosis",axis=1)

y=new_df["diagnosis"]





print(X.shape)

print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(

   X,y, test_size=0.1, random_state=0)



X_train=np.asarray(X_train)

X_test=np.asarray(X_test)

y_train=np.asarray(y_train)

y_test=np.asarray(y_test)



print(type(X_train))

print(type(y_train))

print(X_train.shape)

print(y_train.shape)
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=20)



results=cross_val_score(clf,X_train ,y_train , cv=10)

print(results)

Average=sum(results) / len(results) 

print("Average Accuracy :",Average)
len(df_X.columns)
clf.fit(X_train,y_train)

importance_frame = pd.DataFrame()

importance_frame['Features'] = X.columns

importance_frame['Importance'] = clf.feature_importances_

importance_frame = importance_frame.sort_values(by=['Importance'], ascending=True)

plt.barh([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], importance_frame['Importance'], align='center', alpha=0.5)

plt.yticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], importance_frame['Features'])

plt.xlabel('Importance')

plt.title('Feature Importances')

plt.show()
from sklearn import ensemble

from sklearn import gaussian_process

from sklearn import linear_model

from sklearn import naive_bayes

from sklearn import neighbors

from sklearn import svm

from sklearn import tree

from sklearn import discriminant_analysis

from sklearn import model_selection

from xgboost.sklearn import XGBClassifier 

from sklearn import metrics
#Machine Learning Algorithm (MLA) Selection and Initialization

MLA = [

    #Ensemble Methods

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),



    #Gaussian Processes

    gaussian_process.GaussianProcessClassifier(),

    

    #GLM

    linear_model.LogisticRegressionCV(),

    linear_model.PassiveAggressiveClassifier(),

    linear_model.RidgeClassifierCV(),

    linear_model.SGDClassifier(),

    linear_model.Perceptron(),

    

    #Navies Bayes

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    

    #Nearest Neighbor

    neighbors.KNeighborsClassifier(),

    

    #SVM

    svm.SVC(probability=True),

    svm.NuSVC(probability=True),

    svm.LinearSVC(),

    

    #Trees    

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    

    #Discriminant Analysis

    discriminant_analysis.LinearDiscriminantAnalysis(),

    discriminant_analysis.QuadraticDiscriminantAnalysis(),



    

    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html

    XGBClassifier()    

    ]









#create table to compare MLA metrics

MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Test Accuracy' ]

MLA_compare = pd.DataFrame(columns = MLA_columns)







#index through MLA and save performance to table

row_index = 0

for alg in MLA:



    #set name and parameters

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    

    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate

   # cv_results = model_selection.cross_validate(alg, X_train, y_train)

    alg.fit(X_train, y_train)

    y_pred=alg.predict(X_test)

    score=metrics.accuracy_score(y_test, y_pred)

    

    MLA_compare.loc[row_index, 'MLA Test Accuracy'] =score



    

    

    row_index+=1



    



MLA_compare

#MLA_predict
from sklearn.ensemble import VotingClassifier

vote_clf = VotingClassifier(estimators=[

                                        

                                        ("LR",linear_model.LogisticRegressionCV()),

                                        ("RC",linear_model.RidgeClassifierCV()),

                                        ("SVC",svm.SVC(probability=True)),

                                        ("XGB",XGBClassifier())

                                        ])

vote_clf = vote_clf.fit(X_train, y_train)


from sklearn.metrics import confusion_matrix

prediction=vote_clf.predict(X_test)

y_pred=[]



true_negative,false_positive,false_negative,true_positive=confusion_matrix(y_test, prediction).ravel()



print("true_negative: ",true_negative)

print("false_positive: ",false_positive)

print("false_negative: ",false_negative)

print("true_positive: ",true_positive)

print("\n\n Accuracy Measures\n\n")



Accuracy=(true_positive+true_negative)/(true_positive+false_positive+true_negative+false_negative)

print("Accuracy: ",Accuracy)



Sensitivity=true_positive/(true_positive+false_negative)

print("Sensitivity: ",Sensitivity)



False_Positive_Rate=false_positive/(false_positive+true_negative)

print("False_Positive_Rate: ",False_Positive_Rate)



Specificity=true_negative/(false_positive + true_negative)

print("Specificity: ",Specificity)



#FDR à 0 means that very few of our predictions are wrong

False_Discovery_Rate=false_positive/(false_positive+true_positive)

print("False_Discovery_Rate: ",False_Discovery_Rate)



Positive_Predictive_Value =true_positive/(true_positive+false_positive)

print("Positive_Predictive_Value: ",Positive_Predictive_Value)