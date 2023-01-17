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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#reading the file

df=pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

pd.options.display.max_columns = 30 #to view all the columns
#first 5 rows are obtained

df.head()

#gives the general info about the dataset like the non-null counts and the datatype of each column

df.info()
#to find whether there is any null values for each column

df.isnull().sum()
#shape gives the no:of rows and columns

df.shape
#shows the list of columns

df.columns
"""total charges column is object..so checking if there is any missing value.

there are 11 missing values in this column.

We can infer that the tenure is 0 in aall these cases.



""" 

df[df["TotalCharges"]==" "]
# Replacing the mssing value into 0 and converting the object into float value.

df['TotalCharges'] = df['TotalCharges'].replace(" ", 0).astype('float32')
#checking whether the column is converted into float type

df.info()
""" To find the sd,median,mean and count of the foating and numerical value columns.

The 50% represents the median=1394.55 and mean=2279.73 with a standard deviation of 2266.794 for Total charges.

"""

df.describe()
#Plotting the numerical columns

sns.pairplot(data=df)
#churn No = (5100/7043)*100= 72.4%

#churn Yes  = 100-72.4= 27.6%

sns.countplot(x="Churn",data=df)
"""

checking the churn status of other numerical fields using kde plot

we can see that recent joiners have a churning tendency more and high monthly charges leads to churning

"""

def kde(feature):

    plt.figure(figsize=(9,4))

    plt.title("kde plot for {}".format(feature))

    ax0=sns.kdeplot(df[df["Churn"]=="Yes"][feature],color="red",label= "Churn - Yes")

    ax1=sns.kdeplot(df[df["Churn"]=="No"][feature],color="green",label="Churn - No")

kde("tenure")

kde("MonthlyCharges")

kde("TotalCharges")


g=sns.PairGrid(df,x_vars=["MonthlyCharges","TotalCharges"],y_vars="tenure",hue="Churn",palette="coolwarm",height=5)

g.map(plt.scatter,alpha=0.5)

plt.legend(loc=(-0.3,0.6))
#Finding the total no:of males and females .We can see that churning rate and no:of males and females are more like similar.

sns.countplot(x="gender",data=df,hue="Churn",palette="coolwarm")
#To find the senior citizens.We can infer that there are less senior citizen people joined when compared to youngsters.

sns.countplot(x="SeniorCitizen",hue="gender",data=df,palette="coolwarm")
#The churning rate of senior citizens is less.

sns.countplot(x="SeniorCitizen",hue="Churn",data=df)
#The dependents has a less churning rate when compared to others.

sns.countplot(x="Dependents",hue="Churn",data=df)
#we can see that as monthly charges increases, churning increases

sns.boxplot(x="Churn",y="MonthlyCharges",data=df)
#people who are single shows more churning rate

sns.countplot(x="Partner",hue="Churn",data=df)
#people who joined for a long time are less likely to churn

sns.boxplot(y="tenure",x="Churn",data=df)


#finding the 23 columns

df.columns



"""

The columns having "No internet service" are replaced to "No"

The column "MultipleLines" having "No phone service" are replaced to "No"

"""

replace_cols=["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]

for i in replace_cols:

    df[i]  = df[i].replace('No internet service' , 'No')

df["MultipleLines"]=df["MultipleLines"].replace("No phone service","No")
#Getting histogram plot of the numerical columns

df.hist()
"""

people with or without multiple lines are showing equal churning.

so multiple lines might not be directly related to churn rate



"""

sns.countplot(x="MultipleLines",hue="Churn",data=df)
#finding the phone service count.people having phone service are showing high churning tendency

sns.countplot(x="PhoneService",hue="Churn",data=df)
"""

Visualizing using a box plot

multiple lines with high monthly charges is showing high churning rate.

Whether or not the person has multiple lines, if he has high monthly charges, he has a tendency to churn.



"""

print(sns.boxplot(x="MultipleLines",y="MonthlyCharges",hue="Churn",data=df,palette="coolwarm"))

#Similar as Box Plot, using a violin plot.

sns.violinplot(x="MultipleLines",y="MonthlyCharges",hue="Churn",data=df,palette="coolwarm",split=True)
#tenure is more or less same for the DSL and fibre optic internet services

sns.barplot(x="InternetService",y="tenure",data=df)
#churning is high for customers having fibre optic connections

sns.countplot(x="InternetService",hue="Churn",data=df)
#Fibre optic services have a high monthly charge when compared to others and so is the churn rate

sns.boxplot(x="InternetService",y="MonthlyCharges",hue="Churn",data=df,palette="coolwarm")
#To display the columns together with subplot

cols=['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',

       'StreamingTV', 'StreamingMovies',]

plt.tight_layout()

fig, axes =plt.subplots(2,3,figsize=(15,8))

sns.countplot(data=df,x=cols[0], ax=axes[0,0])

sns.countplot(data=df,x=cols[1], ax=axes[0,1])

sns.countplot(data=df,x=cols[2], ax=axes[0,2])

sns.countplot(data=df,x=cols[3], ax=axes[1,0])

sns.countplot(data=df,x=cols[4], ax=axes[1,1])

sns.countplot(data=df,x=cols[5], ax=axes[1,2])

fig.show()

#To display these columns together with subplots using for loop

cols=['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',

'StreamingTV', 'StreamingMovies',]

x=0

y=0

num=0

plt.tight_layout()

fig, axes =plt.subplots(2,3,figsize=(15,8))

for x in range(2):

    for y in range(3):

        sns.countplot(x=cols[num],data=df,ax=axes[x,y])

        num +=1

         
#To display these columns together with subplots using zip and flatten functions.



df1=df[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',

       'StreamingTV', 'StreamingMovies']].copy()

plt.tight_layout()

fig, axes =plt.subplots(2,3,figsize=(15,8))



"""

 zip() creates an iterator that will aggregate elements from two or more iterables.

Here in the case is a for loop.

The flatten() function is used to get a copy of a given array collapsed into one dimension. 

"""

for cols,ax in zip(df1, axes.flatten()[:6]):

    sns.countplot(data = df1, x = cols, ax = ax)
#people with monthly contract showing high churning rate.

sns.countplot(x="Contract",data=df,hue="Churn")
#monthly charges is high for for all kind of contracts

sns.barplot(x="Contract",y="MonthlyCharges",hue="Churn",data=df)
"""

checking the count of paperless billing.

Churn rate more for paperless billing.This can be because of more companies offering paper less billing

"""

sns.countplot(x="PaperlessBilling",hue="Churn",data=df)
#Checking the count of different payment methods

df["PaymentMethod"].value_counts()

"""

Electronic check payment method is giving more churning.

This might be because of loading issues due to traffic or there might be other complaints.

"""

sns.countplot(x="PaymentMethod",hue="Churn",data=df)

plt.tight_layout()
#Using a heatmap to find the correlation between the numerical columns

tc=df.corr()

sns.heatmap(tc,xticklabels=True,cmap="coolwarm")
#Creating dummy values for the categorical columns for visualization and modelling purpose

dummy=["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","Churn"]

df=pd.get_dummies(df, prefix=dummy, columns=dummy,drop_first=True)#to get either one of the columns.ie, yes or no
#To check whether the columns are updated or not.

df.head() 
#Splitting dataset into X and y arrays

X=df.drop(["customerID","Churn_Yes"],axis=1)#multi dimensional array 



#target column(dependent coumn) is taken i.e churn/not churn

y = df["Churn_Yes"]



"""

FeatureSelection-Method_1

to check which are the best features of this dataset that can predict the target.

"""

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



#apply SelectKBest class to extract top 12 best features

bestfeatures = SelectKBest(score_func=chi2, k=12)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)



#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['features','Score']  #naming the dataframe columns

print(featureScores.nlargest(12,'Score')) 
#FeatureSelection-Method_2

from sklearn.ensemble import RandomForestClassifier 

#you can also use "ExtraTreesClassifier".Gives almost the same results.



import matplotlib.pyplot as plt

model = RandomForestClassifier()

model.fit(X,y)

print(model.feature_importances_) #inbuilt function which enables gives feature importances



#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(10).plot(kind='barh')

plt.show()
"""

#Feature_Selection-Method_3



Heat map to find correlation between all the column variables and find the best features

green repreprents correlation = 1

red represents correlation= 0



"""

plt.figure(figsize=(12,12))

correlating=df.corr()

sns.heatmap(correlating,cmap="RdYlGn")
#Using the different methods, selecting the top 6 features

X=df[["TotalCharges","tenure","MonthlyCharges","Contract_Two year","PaymentMethod_Electronic check","InternetService_Fiber optic"]]
#Splitting the whole dataset into 2:Train and Test

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
"""

Using a standard scaler to scale all columns into a small range which makes prediction more easier and 

decreses the chances of the model getting biased.



"""

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(X_train) #Fitting and transforming the X train dataset

X_test=sc.transform(X_test)#Transforming the test dataset
#Checking the transformed values

X_train
#Classification Model-1

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

#fitting the train values( both X and y )

lr.fit(X_train,y_train)

#Predicting the y valus for the X_test

y_pred=lr.predict(X_test)
#Performance check using Confusion matrix,accuracy score and Classification Report

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix



print(confusion_matrix(y_pred,y_test))

print("\n") #correct prediction= 1425+250, wrong prediction=135+303



print(classification_report(y_pred,y_test))

print("\n") # to get the values of precision,recall,f1-score and support



print(accuracy_score(y_pred,y_test))#accuracy score is 79.2%

#Accuracy=(Total no:of correct predictions)/(Total no:of prdictions)
#Classification Model-2

from xgboost import XGBClassifier

xgb=XGBClassifier()

xgb.fit(X_train,y_train)

y_pred=xgb.predict(X_test)





print(confusion_matrix(y_pred,y_test))

print("\n")

print(classification_report(y_pred,y_test))

print("\n")

print(accuracy_score(y_pred,y_test))

#Accuracy=78.6%

#Classification Model-3

from sklearn.neighbors import KNeighborsClassifier

classifier=KNeighborsClassifier()

classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)



print(confusion_matrix(y_pred,y_test))

print("\n")

print(classification_report(y_pred,y_test))

print("\n")

print(accuracy_score(y_pred,y_test))

#Accuracy=77.4%
#Classification Model-4

from sklearn.naive_bayes import GaussianNB

gaussian=GaussianNB()

gaussian.fit(X_train,y_train)

y_pred=gaussian.predict(X_test)



print(confusion_matrix(y_pred,y_test))

print("\n")

print(classification_report(y_pred,y_test))

print("\n")

print(accuracy_score(y_pred,y_test))

#Accuracy=72.4%
#Classification Model-5

from sklearn.tree import DecisionTreeClassifier

tree=DecisionTreeClassifier(criterion="entropy",random_state=0)

tree.fit(X_train,y_train)

y_pred=tree.predict(X_test)



print(confusion_matrix(y_pred,y_test))

print("\n")

print(classification_report(y_pred,y_test))

print("\n")

print(accuracy_score(y_pred,y_test))

#Accuracy=72.4%
#Classification Model-6

from sklearn.ensemble import RandomForestClassifier

forest=RandomForestClassifier(n_estimators=10,criterion="entropy",random_state=0)

forest.fit(X_train,y_train)

y_pred=forest.predict(X_test)



print(confusion_matrix(y_pred,y_test))

print("\n")

print(classification_report(y_pred,y_test))

print("\n")

print(accuracy_score(y_pred,y_test))

#Accuracy=77.8%
#Classification Model-7

from sklearn.svm import SVC

classifier1 = SVC(kernel="rbf",C=1,gamma=0.001,random_state=0)

classifier1.fit(X_train,y_train)

y_pred=classifier1.predict(X_test)



print(confusion_matrix(y_pred,y_test))

print("\n")

print(classification_report(y_pred,y_test))

print("\n")

print(accuracy_score(y_pred,y_test))

#Accuracy=79.08%
#Classification Model-8

from sklearn.svm import SVC

classifier2 = SVC(kernel="linear",random_state=0)

classifier2.fit(X_train,y_train)

y_pred=classifier1.predict(X_test)



print(confusion_matrix(y_pred,y_test))

print("\n")

print(classification_report(y_pred,y_test))

print("\n")

print(accuracy_score(y_pred,y_test))
"""

Model Selection: using k-fold Cross validation

"""

#Checking Logistic Regression Model's accuracy

from sklearn.model_selection import cross_val_score

accuracies=cross_val_score(estimator=lr,X=X_train,y=y_train,cv=10)

print("Mean_Accuracy:",accuracies.mean())

print("\n")

print("Standard Deviation:",accuracies.std())

#Checking XGBoost Model's accuracy

from sklearn.model_selection import cross_val_score

accuracies=cross_val_score(estimator=xgb,X=X_train,y=y_train,cv=10)

print("Accuracy:",accuracies.mean())

print("\n")

print("Standard Deviation:",accuracies.std())

"""

Checking SVM model's accuracy using Grid search and finding whether the model is linearly seperable or not.

Here we get the best parameters tobe selected



"""

from sklearn.model_selection import GridSearchCV

parameters=[{"C":[0.25,0.5,0.75,1],"kernel":["linear"]},{"C":[1,10,100,1000],"kernel":["rbf"],"gamma":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]



grid_search=GridSearchCV(estimator=classifier1,param_grid=parameters,scoring="accuracy",cv=10,n_jobs=-1)



grid_search.fit(X_train,y_train)

best_score=grid_search.best_score_

best_parameters=grid_search.best_params_





print("Best_Score:",best_score)

print("\n")

print("Best_Parameters:",best_parameters)

#Getting an overall acuracy of 80%


