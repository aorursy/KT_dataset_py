import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn 

%matplotlib inline

import os

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import Imputer

#importing all the required ML packages

from sklearn.linear_model import LogisticRegression #logistic regression

from sklearn import svm #support vector Machine

from sklearn.ensemble import RandomForestClassifier #Random Forest

from sklearn.neighbors import KNeighborsClassifier #KNN

from sklearn.naive_bayes import GaussianNB #Naive bayes

from sklearn.tree import DecisionTreeClassifier #Decision Tree

from sklearn import metrics #accuracy measure

from sklearn.metrics import confusion_matrix #for confusion matrix

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

print(os.listdir("../input"))

#===========================================================================================================

#Read input file

#============================================================================================================

Diabetic1=pd.read_csv("../input/diabetes.csv")

Diabetic1.head()
#===========================================================================================================

# Rows and Columns

#===========================================================================================================

Diabetic1.shape
#===========================================================================================================

# Find the basic statistical details

#===========================================================================================================

Diabetic1.describe()
#===========================================================================================================

#Check for null values

#===========================================================================================================

Diabetic2=Diabetic1.copy()

Diabetic2.isnull().sum()
Diabetic2.info()
Diabetic3=Diabetic2.copy()

Diabetic3=Diabetic3.drop('Outcome', axis=1)

Diabetic3.head()


#correlation heatmap of dataset

def correlation_heatmap(df):

    _ , ax = plt.subplots(figsize =(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        df.corr(), 

        cmap = "YlGn",

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)



correlation_heatmap(Diabetic3)


from statsmodels.stats.outliers_influence import variance_inflation_factor



class ReduceVIF(BaseEstimator, TransformerMixin):

    def __init__(self, thresh=10.0, impute=True, impute_strategy='median'):

        # From looking at documentation, values between 5 and 10 are "okay".

        # Above 10 is too high and so should be removed.

        self.thresh = thresh

        

        # The statsmodel function will fail with NaN values, as such we have to impute them.

        # By default we impute using the median value.

        # This imputation could be taken out and added as part of an sklearn Pipeline.

        if impute:

            self.imputer = Imputer(strategy=impute_strategy)



    def fit(self, X, y=None):

        print('ReduceVIF fit')

        if hasattr(self, 'imputer'):

            self.imputer.fit(X)

        return self



    def transform(self, X, y=None):

        print('ReduceVIF transform')

        columns = X.columns.tolist()

        if hasattr(self, 'imputer'):

            X = pd.DataFrame(self.imputer.transform(X), columns=columns)

        return ReduceVIF.calculate_vif(X, self.thresh)



    @staticmethod

    def calculate_vif(X, thresh=5.0):

        # Taken from https://stats.stackexchange.com/a/253620/53565 and modified

        dropped=True

        while dropped:

            variables = X.columns

            dropped = False

            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]

            

            max_vif = max(vif)

            if max_vif > thresh:

                maxloc = vif.index(max_vif)

                print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')

                X = X.drop([X.columns.tolist()[maxloc]], axis=1)

                dropped=True

        return X
#=============================================================================================================

# Remove columns having higher VIF factor or having high multicollinearity

#=============================================================================================================

transformer = ReduceVIF()

Diabetic4 = transformer.fit_transform(Diabetic3)

Diabetic4.head()
pd.crosstab(Diabetic2.Pregnancies,Diabetic2.Outcome).plot(kind='bar')

plt.title('Relationship')

plt.xlabel('Pregnancies')

plt.ylabel('Outcome')

plt.savefig('Relationship')

#==========================================================================================

# Pregnancies is a good factor for prediction

#==========================================================================================
pd.crosstab(Diabetic2.BloodPressure,Diabetic2.Outcome).plot(kind='bar')

plt.title('Relationship')

plt.xlabel('BloodPressure')

plt.ylabel('Outcome')

plt.savefig('Relationship')

#==========================================================================================

# BloodPressure is a good factor for prediction

#==========================================================================================
pd.crosstab(Diabetic2.SkinThickness,Diabetic2.Outcome).plot(kind='bar')

plt.title('Relationship')

plt.xlabel('SkinThickness')

plt.ylabel('Outcome')

plt.savefig('Relationship')

#==========================================================================================

# SkinThickness is not a good factor for prediction

#==========================================================================================

pd.crosstab(Diabetic2.Insulin,Diabetic2.Outcome).plot(kind='bar')

plt.title('Relationship')

plt.xlabel('Insulin')

plt.ylabel('Outcome')

plt.savefig('Relationship')

#==========================================================================================

# Insulin is not a good factor for prediction

#==========================================================================================
pd.crosstab(Diabetic2.DiabetesPedigreeFunction,Diabetic2.Outcome).plot(kind='bar')

plt.title('Relationship')

plt.xlabel('DiabetesPedigreeFunction')

plt.ylabel('Outcome')

plt.savefig('Relationship')

#========================================================

#DiabetesPedigreeFunction is a good factor for prediction

#========================================================
X = Diabetic2.as_matrix(['Pregnancies','BloodPressure','DiabetesPedigreeFunction'])

y = Diabetic2['Outcome']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=10)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_train= sc.transform(X_train)

X_test = sc.transform(X_test)

       #lets scale the data

X = Diabetic2.as_matrix(['Pregnancies','BloodPressure','DiabetesPedigreeFunction'])

y = Diabetic2['Outcome']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=10)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_train= sc.transform(X_train)

X_test = sc.transform(X_test)
# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

#y_pred = logreg.predict(X_test)

score_logreg = logreg.score(X_test,y_test)

print('The accuracy of the Logistic Regression is', score_logreg)
# Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

# y_pred = gaussian.predict(X_test)

score_gaussian = gaussian.score(X_test,y_test)

print('The accuracy of Gaussian Naive Bayes is', score_gaussian)
# Support Vector Classifier (SVM/SVC)

from sklearn.svm import SVC

svc = SVC(gamma=0.22)

svc.fit(X_train, y_train)

#y_pred = logreg.predict(X_test)

score_svc = svc.score(X_test,y_test)

print('The accuracy of SVC is', score_svc)
svc_radical =svm.SVC(kernel='rbf',C=1,gamma=0.22)

svc_radical.fit(X_train,y_train.values.ravel())

score_svc_radical = svc_radical.score(X_test,y_test)

print('The accuracy of Radical SVC Model is', score_svc_radical)
#===============================================================================================================

# Vary the n_neighbors paramter through 1, 3,5 and 7 as part of improving accuracy

#===============================================================================================================

# K-Nearest Neighbors

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

score_knn = knn.score(X_test,y_test)

print('The accuracy of the KNN Model is',score_knn)

accuracy_score(y_pred,y_test)

### cross validation

from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction

kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts

xyz=[]

accuracy=[]

std=[]

classifiers=['Naive Bayes','Linear Svm','Radial Svm','Logistic Regression','Decision Tree','KNN','Random Forest']

models=[GaussianNB(), svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),DecisionTreeClassifier(),

        KNeighborsClassifier(n_neighbors=9),RandomForestClassifier(n_estimators=100)]

for i in models:

    model = i

    cv_result = cross_val_score(model,X,y, cv = kfold,scoring = "accuracy")

    cv_result=cv_result

    xyz.append(cv_result.mean())

    std.append(cv_result.std())

    accuracy.append(cv_result)

models_dataframe=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       

models_dataframe