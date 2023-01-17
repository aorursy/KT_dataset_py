#import the needed packages



from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.metrics import classification_report,roc_auc_score, confusion_matrix, accuracy_score,recall_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold, cross_val_score, train_test_split,GridSearchCV

#from sklearn.cross_validation import cross_val_predict

from sklearn.metrics import classification_report

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import RobustScaler

from sklearn.metrics import confusion_matrix

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.metrics import make_scorer

from sklearn import metrics

import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE

import scipy.stats as stats

import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib

%matplotlib inline

import warnings



import sklearn

import scipy



import sys

import os









# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings("ignore")
# Import the data to bank_df

bank_df=pd.read_csv("../input/Bank_Personal_Loan_Modelling.csv")
bank_df.head(10)
# Shape of training and test data set

def dataframe_shape(df):

    print("The dataframe has %d rows" %df.shape[0])

    print("The dataframe has %d columns" %df.shape[1])



dataframe_shape(bank_df)
# Columns/Feature in dataset

pd.DataFrame(bank_df.columns,index=None,copy=False).T
# First 3 observation

bank_df.head(3) # you can choose any number of rows by changing the number inside head function. Default it shows 5
# Last 3 observation

bank_df.tail(3) # you can choose any number of rows by changing the number inside tail function. Default it shows 5
# Random 3 observation

bank_df.sample(3) # you can choose any number of rows by changing the number inside sample function. Default it shows 1
# datatypes present into training dataset

def datatypes_insight(data):

    display(data.dtypes.to_frame().T)

    data.dtypes.value_counts().plot(kind="barh")



datatypes_insight(bank_df)
# Missing value identification



def Nan_value(data):

    display(data.apply(lambda x: sum(x.isnull())).to_frame().T)

    ##data.apply(lambda x: sum(x.isnull())).plot(kind="barh")



Nan_value(bank_df)
# Ploting the NAN values if any.

sns.heatmap(bank_df.isna(),yticklabels=False,cbar=False,cmap='viridis')
# Unique values in features

def unique_data(data):

    display(data.apply(lambda x: len(x.unique())).to_frame().T)

    data.apply(lambda x: len(x.unique())).plot(kind="barh")



unique_data(bank_df)
# check for imbalance dataset

fig, ax = plt.subplots(nrows=1, ncols=2,squeeze=True)

fig.set_size_inches(14,6)

frequency_colums= pd.crosstab(index=bank_df["Personal Loan"],columns="count")

frequency_colums.plot(kind='bar',ax=ax[0],color="c",legend=False,rot=True,fontsize=10)

frequency_colums.plot(kind='pie',ax=ax[1],subplots=True,legend=False,fontsize=10,autopct='%.2f')

ax[0].set_title('Frequency Distribution of Dependent variable: Survived',fontsize=10)

ax[1].set_title('Pie chart representation of Dependent variable: Survived',fontsize=10)



#adding the text labels

rects = ax[0].patches

labels = frequency_colums["count"].values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax[0].text(rect.get_x() + rect.get_width()/2, height +1,label, ha='center', va='bottom',fontsize=10)

plt.show()
#statistical analysis of data set

bank_df.describe().T
def distploting(df):

    col_value=df.columns.values.tolist()

    sns.set(context='notebook',style='whitegrid', palette='dark',font='sans-serif',font_scale=1.2,color_codes=True)

    

    fig, axes = plt.subplots(nrows=7, ncols=2,constrained_layout=True)

    count=0

    for i in range (7):

        for j in range (2):

            s=col_value[count+j]

            #axes[i][j].hist(df[s].values,color='c')

            sns.distplot(df[s].values,ax=axes[i][j],bins=30,color="c")

            axes[i][j].set_title(s,fontsize=17)

            fig=plt.gcf()

            fig.set_size_inches(8,20)

            plt.tight_layout()

        count=count+j+1

        

             

distploting(bank_df)
bank_df[['CreditCard', 'Personal Loan']].groupby(['CreditCard'], as_index=False).mean().sort_values(by='Personal Loan', ascending=False)
bank_df[['Online', 'Personal Loan']].groupby(['Online'], as_index=False).mean().sort_values(by='Personal Loan', ascending=False)
bank_df[['Family', 'Personal Loan']].groupby(['Family'], as_index=False).mean().sort_values(by='Personal Loan', ascending=False)
bank_df[['Education', 'Personal Loan']].groupby(['Education'], as_index=False).mean().sort_values(by='Personal Loan', ascending=False)
bank_df[['CD Account', 'Personal Loan']].groupby(['CD Account'], as_index=False).mean().sort_values(by='Personal Loan', ascending=False)
bank_df[['Securities Account', 'Personal Loan']].groupby(['Securities Account'], as_index=False).mean().sort_values(by='Personal Loan', ascending=False)
g = sns.FacetGrid(bank_df, col='Personal Loan')

g.map(plt.hist,'Income', bins=20)
g = sns.FacetGrid(bank_df, col='Personal Loan')

g.map(plt.hist,'Mortgage', bins=20)
g = sns.FacetGrid(bank_df, col='Personal Loan')

g.map(plt.hist,'CCAvg', bins=20)
g = sns.FacetGrid(bank_df, col='Personal Loan')

g.map(plt.hist,'Age', bins=20)
g = sns.FacetGrid(bank_df, col='Personal Loan')

g.map(plt.hist,'Experience', bins=20)
grid = sns.FacetGrid(bank_df, col='Personal Loan', row='Education', size=2.5, aspect=1.6)

grid.map(plt.hist, 'Income', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(bank_df, col='Personal Loan', row='Family', size=2.5, aspect=1.6)

grid.map(plt.hist, 'Income', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(bank_df, col='Personal Loan', row='Online', size=2.5, aspect=1.6)

grid.map(plt.hist, 'Income', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(bank_df, col='Personal Loan', row='CreditCard', size=2.5, aspect=1.6)

grid.map(plt.hist, 'Income', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(bank_df, col='Personal Loan', row='Family', size=2.5, aspect=1.6)

grid.map(plt.hist, 'Mortgage', alpha=.5, bins=20)

grid.add_legend();
# Compare the Age, Exp and Education for the person

pd.DataFrame(bank_df[bank_df["Experience"]>0][["Age","Education","Experience"]].sort_values("Age")).head()
#Lets see if we have any relationship bewteen Exp and Age

df = pd.DataFrame(bank_df.groupby("Age").mean()["Experience"]).reset_index()

fig.set_size_inches(20,6)

sns.lmplot(x='Age',y='Experience',data=df)

plt.ylabel("Experience(Mean)")

plt.title("Mean Experience by Age")

plt.show()
# From the plot, we can see Age and Experience has linear relationship.

#In data set the value was correct but it was captured with wrong sign.let replace the values with absolute value.

bank_df["Experience"] = bank_df["Experience"].apply(abs)
bank_df["PP_income_M"] = (((bank_df["Income"]*1000)/12)-((bank_df["CCAvg"]*1000)/12))
g = sns.FacetGrid(bank_df, col='Personal Loan')

g.map(plt.hist,'PP_income_M', bins=20)
bank_df = bank_df.drop(['ID','ZIP Code'], axis=1)
corr = bank_df.corr()



mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True



plt.figure(figsize=(20,15))

sns.heatmap(corr, mask=mask,annot=True,square=True,cmap="coolwarm")
plt.figure(figsize=(20, 20))

sns.pairplot(bank_df,hue="Personal Loan")
from sklearn.preprocessing import MinMaxScaler,StandardScaler,robust_scale

scaler = StandardScaler();



colscal=['Age', 'Experience', 'Income', 'CCAvg','PP_income_M']



scaler.fit(bank_df[colscal])

scaled_bank_df = pd.DataFrame(scaler.transform(bank_df[colscal]),columns=colscal)



bank_df =bank_df.drop(colscal,axis=1)

bank_df = scaled_bank_df.join(bank_df)
X=bank_df[['Age','Experience', 'Income', 'Family', 'CCAvg', 'Education', 'Securities Account', 'CD Account', 'Online',

       'CreditCard','PP_income_M']]

y=bank_df["Personal Loan"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=101)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predict = logmodel.predict(X_test.values)

predictProb = logmodel.predict_proba(X_test.values)

acc_log=round(metrics.accuracy_score(predict,y_test)*100,2)
import pickle

filename = 'finalized_model.sav'

pickle.dump(logmodel, open(filename, 'wb'))
print("**"*40)

print('The accuracy of the Logistic is',metrics.accuracy_score(predict,y_test))

print("__"*40)

print("confusion_matrix :\n",confusion_matrix(y_test, predict))

print("__"*40)

print("\nclassification_report :\n",classification_report(y_test, predict))

print("__"*40)

print('Recall Score',recall_score(y_test, predict))

print('ROC AUC :', roc_auc_score(y_test, predictProb[:,1]))

print('Accuracy :',accuracy_score(y_test, predict))

print("**"*40)
score1 =cross_val_score(X=X,y=y,estimator=logmodel,scoring="recall",cv=10)

score2 =cross_val_score(X=X,y=y,estimator=logmodel,scoring="roc_auc",cv=10)

score3 =cross_val_score(X=X,y=y,estimator=logmodel,scoring="accuracy",cv=10)

score4 =cross_val_score(X=X,y=y,estimator=logmodel,scoring="f1",cv=10)

score5 =cross_val_score(X=X,y=y,estimator=logmodel,scoring="average_precision",cv=10)
print("**"*40)

print("Logistic Regression Cross Validation:")

print("\nCross Validation Recall :",score1.mean())

print("Cross Validation Roc Auc :",score2.mean())

print("Cross Validation accuracy :",score3.mean())

print("Cross Validation f1 :",score4.mean())

print("Cross Validation average_precision :",score5.mean())

print("**"*40)
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)
predict = knn.predict(X_test.values)

predictProb = knn.predict_proba(X_test.values)

acc_knn=round(metrics.accuracy_score(predict,y_test)*100,2)
print("**"*40)

print('The accuracy of the KNN is',metrics.accuracy_score(predict,y_test))

print("__"*40)

print("confusion_matrix :\n",confusion_matrix(y_test, predict))

print("__"*40)

print("\nclassification_report :\n",classification_report(y_test, predict))

print("__"*40)

print('Recall Score',recall_score(y_test, predict))

print('ROC AUC :', roc_auc_score(y_test, predictProb[:,1]))

print("**"*40)
score1 =cross_val_score(X=X,y=y,estimator=knn,scoring="recall",cv=10)

score2 =cross_val_score(X=X,y=y,estimator=knn,scoring="roc_auc",cv=10)

score3 =cross_val_score(X=X,y=y,estimator=knn,scoring="accuracy",cv=10)

score4 =cross_val_score(X=X,y=y,estimator=knn,scoring="f1",cv=10)

score5 =cross_val_score(X=X,y=y,estimator=knn,scoring="average_precision",cv=10)
print("KNN Cross Validation:")

print("**"*40)

print("\nCross Validation Recall :",score1.mean())

print("Cross Validation Roc Auc :",score2.mean())

print("Cross Validation accuracy :",score3.mean())

print("Cross Validation f1 :",score4.mean())

print("Cross Validation average_precision :",score5.mean())

print("**"*40)
from sklearn.model_selection import GridSearchCV

k = np.arange(1,10,1)
parameters = {'n_neighbors': k, 

              'weights': ["uniform","distance"], 

              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']

             }



acc_scorer = make_scorer(accuracy_score)
grid_obj = GridSearchCV(knn, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X_train, y_train)
print("**"*40)

print('The accuracy of the KNN is',metrics.accuracy_score(predict,y_test))
predict = grid_obj.predict(X_test.values)

predictProb = grid_obj.predict_proba(X_test.values)
print("**"*40)

print('The accuracy of the KNN with GridSearchCV is',metrics.accuracy_score(y_test,predict))

print("__"*40)

print("confusion_matrix :\n",confusion_matrix(y_test, predict))

print("__"*40)

print("\nclassification_report :\n",classification_report(y_test, predict))

print("__"*40)

print('Recall Score',recall_score(y_test, predict))

print('ROC AUC :', roc_auc_score(y_test, predictProb[:,1]))

print('Accuracy :',accuracy_score(y_test, predict))

print("**"*40)
from sklearn import model_selection

# subsetting just the odd ones

neighbors = list(np.arange(1,20,2))



# empty list that will hold cv scores

cv_scores = []



# perform 10-fold cross validation

for k in neighbors:

    knn = KNeighborsClassifier(n_neighbors=k)

    scores =model_selection.cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')

    cv_scores.append(scores.mean())
# changing to misclassification error

MSE = [1 - x for x in cv_scores]



optimal_k = neighbors[MSE.index(min(MSE))]

print ("The optimal number of neighbors is %d" % optimal_k)



# plot misclassification error vs k

plt.plot(neighbors,MSE)

locator = matplotlib.ticker.MultipleLocator(2)

plt.gca().xaxis.set_major_locator(locator)

formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")

plt.gca().xaxis.set_major_formatter(formatter)

plt.xlabel('Number of Neighbors K')

plt.ylabel('Misclassification Error')

plt.show()
gb = GaussianNB()

gb.fit(X_train, y_train)
predict = gb.predict(X_test)

predictProb = gb.predict_proba(X_test)

acc_nb=round(metrics.accuracy_score(predict,y_test)*100,2)
print("**"*40)

print('The accuracy of the Naïve Bayes is',metrics.accuracy_score(predict,y_test))

print("__"*40)

print("confusion_matrix :\n",confusion_matrix(y_test, predict))

print("__"*40)

print("\nclassification_report :\n",classification_report(y_test, predict))

print("__"*40)

print('Recall Score',recall_score(y_test, predict))

print('ROC AUC :', roc_auc_score(y_test, predictProb[:,1]))

print('Accuracy :',accuracy_score(y_test, predict))

print("**"*40)
score1 =cross_val_score(X=X,y=y,estimator=gb,scoring="recall",cv=10)

score2 =cross_val_score(X=X,y=y,estimator=gb,scoring="roc_auc",cv=10)

score3 =cross_val_score(X=X,y=y,estimator=gb,scoring="accuracy",cv=10)

score4 =cross_val_score(X=X,y=y,estimator=gb,scoring="f1",cv=10)

score5 =cross_val_score(X=X,y=y,estimator=gb,scoring="average_precision",cv=10)
print("Naïve Bayes Cross Validation:")

print("**"*40)

print("\nCross Validation Recall :",score1.mean())

print("Cross Validation Roc Auc :",score2.mean())

print("Cross Validation accuracy :",score3.mean())

print("Cross Validation f1 :",score4.mean())

print("Cross Validation average_precision :",score5.mean())

print("**"*40)
models = pd.DataFrame({

    'Model': ['KNN', 'Logistic Regression','Naive Bayes'],

    'Score': [acc_knn, acc_log, acc_nb, 

              ]})

models.sort_values(by='Score', ascending=False)