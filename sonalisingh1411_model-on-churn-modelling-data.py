# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing libraries.................

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression 

from sklearn.ensemble import RandomForestClassifier ,AdaBoostClassifier,BaggingClassifier,ExtraTreesClassifier,GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve ,KFold

from sklearn.metrics import accuracy_score,f1_score,auc,confusion_matrix

from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')





#Checking the dataset.....................

df=pd.read_csv('../input/Churn_Modelling.csv')

df.head()




#Checking missing values..............

df.isnull().sum()



#Checking dataset shape

df.shape
#Separating the dtypes based on objects and integer or float.......

g = df.columns.to_series().groupby(df.dtypes).groups

print(g)





df.info()
df.describe()
#Checking the correlation...........................

plt.figure(figsize=(11,7)) #7 is the size of the width and 4 is parts.... 

sns.heatmap(df.corr(),annot=True,cmap='cubehelix_r') #draws  heatmap with input as the correlation matrix calculted by(iris.corr())

plt.show()
#While analysing its look like "RowNumber","CustomerId","Surname" are not needing for this prediction...

df.drop(columns= ["RowNumber","CustomerId","Surname"],inplace=True)
df.head()



#Visualizing the Geography

df['Geography'].value_counts().plot(kind='bar')

#As per data i can see that france have high customer_rate
#Visualizing the gender

df['Gender'].value_counts().plot(kind='bar')

#As per data most of the customers are male


#Visualizing the Exited

df['Exited'].value_counts().plot(kind='bar')

#As per analysis most of the customer having less chances of leaving the bank. 


#Encoding Categorical Variables

from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()

for col in df.columns:

    df[col] = labelencoder.fit_transform(df[col])

#Gender-female(0),male(1)

#Geography-France(0),Spain(2),Germany(1)    

    

#Now one hot encoding

df=pd.get_dummies(df, columns=["Gender","Geography"],drop_first=False)


df.rename(columns={'Gender_0':'Female','Gender_1':'Male','Geography_0':'France','Geography_2':'Spain','Geography_1':'Germany'}, inplace=True)

#Rearranged the order of the dataframe....

df = df[['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Female','Male','France','Germany','Spain','Exited']]

df.head()

   
#Separating features and label

X = df.iloc[:,0:13].values

y = df.iloc[:,-1].values


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)



# =============================================================================

# Cross validation on differnet set of algorithm!!!

# =============================================================================

################################################################

kfold = StratifiedKFold(n_splits=8,shuffle=True, random_state=42)





rs = 15

clrs = []



clrs.append(AdaBoostClassifier(random_state=rs))

clrs.append(GradientBoostingClassifier(random_state=rs))

clrs.append(RandomForestClassifier(random_state=rs))

clrs.append(LogisticRegression(random_state = rs))

clrs.append(ExtraTreesClassifier(random_state = rs))

#clrs.append(BaggingClassifier(random_state = rs))



cv_results = []

for clr in clrs :

    cv_results.append(cross_val_score(clr, X_train, y_train , scoring = 'accuracy', cv = kfold, n_jobs=-1))



cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())

    

cv_df = pd.DataFrame({"CrossVal_Score_Means":cv_means,"CrossValerrors": cv_std,"Algo":["RandomForestClassifier","Logistic Regression","AdaBoostClassifier","Gradient Boosting",'ExtraTreesClassifier']})





import seaborn as sns

g = sns.barplot("CrossVal_Score_Means","Algo",data = cv_df,orient = "h",**{'xerr':cv_std})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")

print(cv_df)   


# Applying PCA

from sklearn.decomposition import PCA

pca = PCA(n_components = None)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_

    

print(explained_variance)



    

len(explained_variance)





with plt.style.context('dark_background'):

    plt.figure(figsize=(16, 8))

    

    plt.bar(range(13), explained_variance, alpha=0.5, align='center',label='individual explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()



# =============================================================================

# Selecting top 5 components

# =============================================================================

# Applying PCA

from sklearn.decomposition import PCA

pca = PCA(n_components = 7 )

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_

    

print(explained_variance)





rs = 15

clrs = []



clrs.append(AdaBoostClassifier(random_state=rs))

clrs.append(RandomForestClassifier(random_state=rs))

clrs.append(GradientBoostingClassifier(random_state=rs))

clrs.append(LogisticRegression(random_state = rs,))

clrs.append(ExtraTreesClassifier(random_state = rs))





cv_results = []

for clr in clrs :

    cv_results.append(cross_val_score(clr, X, y, scoring = 'accuracy', cv = kfold, n_jobs=-1))



    

cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())







cv_df = pd.DataFrame({"CrossVal_Score_Means":cv_means,"CrossValerrors": cv_std,"Algo":["AdaBoostClassifier","RandomForestClassifier","Gradient Boosting","Logistic Regression",'ExtraTreesClassifier']})



g = sns.barplot("CrossVal_Score_Means","Algo",data = cv_df,orient = "h",**{'xerr':cv_std})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")

print(cv_df)
# Gradient boosting tunning



GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1] 

              }



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 0)



gsGBC.fit(X_train,y_train)

GBC_best = gsGBC.best_estimator_



# Best score

gsGBC.best_score_,gsGBC.best_params_

# =============================================================================

# Building a model...

# =============================================================================





GBC = GradientBoostingClassifier(learning_rate= 0.05,loss='deviance',max_depth=8,max_features=0.3,min_samples_leaf=150,n_estimators=300)

GBC.fit(X_train, y_train)





#predicting the test set

y_pred = GBC.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)







print(accuracy_score(y_test,y_pred))

  



from sklearn.metrics import classification_report 

print(classification_report(y_test,y_pred))





print(f1_score(y_test,y_pred))
#Adaboosting hypertunning

from sklearn.ensemble import AdaBoostClassifier

ABC = AdaBoostClassifier()

ABC_parameter = {'n_estimators' :[100,200,300],'random_state' : [10,20,30,40,50]}

ABC = GridSearchCV(ABC,param_grid = ABC_parameter, cv=kfold, scoring="accuracy")



ABC.fit(X_train,y_train)

ABC_best = ABC.best_estimator_



# Best score

ABC.best_score_,ABC.best_params_

#Adaboosting algorithm

from sklearn.ensemble import AdaBoostClassifier



ABC = AdaBoostClassifier(n_estimators = 200,random_state = 10)

ABC.fit(X_train, y_train)



#predicting the test set

y_pred = ABC.predict(X_test)





# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)







print(accuracy_score(y_test,y_pred))

  



from sklearn.metrics import classification_report 

print(classification_report(y_test,y_pred))





print(f1_score(y_test,y_pred))


