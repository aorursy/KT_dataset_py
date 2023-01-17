import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats



%matplotlib inline
df=pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")

df

df["Outcome"].value_counts().plot(kind='bar')
df1=df.copy()
for i in df.columns:

    plt.figure()

    plt.title(i)

    plt.hist(df[i],bins=30)

    plt.show()
df["ins"]=np.where(df["Insulin"]==0,1,0)

print(df.groupby("ins")["Outcome"].mean())

df["sk"]=np.where(df["SkinThickness"]==0,1,0)

print(df.groupby("sk")["Outcome"].mean())
df["Glucose"]=np.where(df["Glucose"]==0,df["Glucose"].mean(),df["Glucose"])

df["BloodPressure"]=np.where(df["BloodPressure"]==0,df["BloodPressure"].mean(),df["BloodPressure"])

df["BMI"]=np.where(df["BMI"]==0,df["BMI"].mean(),df["BMI"])

for i in df.columns:

    plt.figure()

    plt.title(i)

    plt.hist(df[i],bins=30)

    plt.show()
print(df1.groupby("Outcome")["SkinThickness"].mean())

df1=df[df["SkinThickness"]!=0]

a=df1.groupby("Outcome")["SkinThickness"].mean()[0]

print(a)

b=df1.groupby("Outcome")["SkinThickness"].mean()[1]

df1=df[df["Insulin"]!=0]

c=df1.groupby("Outcome")["Insulin"].mean()[0]



d=df1.groupby("Outcome")["Insulin"].mean()[1]



#df.loc[(df['First_name'] == 'Bill') | (df['First_name'] == 'Emma'), 'name_match'] = 'Match'  

df.loc[(df["SkinThickness"]==0 )&( df["Outcome"]==1),"SkinThickness"]=b

df.loc[(df["SkinThickness"]==0 )&( df["Outcome"]==0),"SkinThickness"]=a

df.loc[(df["SkinThickness"]==0 )&( df["Outcome"]==1),"SkinThickness"]=d

df.loc[(df["SkinThickness"]==0 )&( df["Outcome"]==0),"SkinThickness"]=c

df
for i in df.columns:

    plt.figure()

    plt.title(i)

    plt.hist(df[i],bins=30)

    plt.show()
sns.pairplot(df,hue="Outcome")
plt.figure(figsize=(20,20))

sns.heatmap(df.corr(),cmap="RdYlGn",annot=True)
for fet in df.columns:

    plt.figure()

    sns.boxplot(df[fet])

    plt.show()
Q1 = df.quantile(0.25)

Q3 = df.quantile(0.75)

IQR = Q3 - Q1

print(IQR)

df = df[~((df < (Q1 - 3 * IQR)) |(df > (Q3 + 3 * IQR))).any(axis=1)]
for fet in df.columns:

    plt.figure()

    sns.boxplot(df[fet])

    plt.show()
a=['DiabetesPedigreeFunction', 'Age', 'Insulin','Pregnancies']

def diagnostic_plots(df, variable):

    plt.figure(figsize=(15,6))

    plt.subplot(1, 2, 1)

    df[variable].hist(bins=20)

    plt.subplot(1, 2, 2)

    stats.probplot(df[variable], dist="norm", plot=plt)

    plt.title(variable)

    plt.show()

for i in df.columns:

    diagnostic_plots(df, i)
for i in a:

    df[i]=np.log(df[i]+1)

    diagnostic_plots(df,i)
x=df.drop("Outcome",axis=1)

y=df["Outcome"]

from sklearn.preprocessing import MinMaxScaler 

scl=MinMaxScaler()

x=scl.fit_transform(df.drop("Outcome",axis=1))
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score 

clf1=LogisticRegression()

cross_val_score(clf1,x,y,cv=20).mean()

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score

from sklearn.metrics import f1_score

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)

from sklearn.model_selection import GridSearchCV



for i in [0.1,0.5,1,2,3,5,10,50,100]:

    clf2 = LogisticRegression(C=i)

#grid_clf_acc = GridSearchCV(clf1, param_grid = grid_values,scoring = {'Accuracy': 'accuracy'})

    clf2.fit(x_train, y_train)

    print(str(i) +"- c value")

    



#Predict values based on new parameters



#



#print(list(zip(y_test,y_score)))

    y_pred=clf2.predict(x_test)

    print("confusion_matrix - "+str(confusion_matrix(y_test,y_pred)))

    print("accuracy_score - "+str(accuracy_score(y_test, y_pred)))

    print("f1_score - "+str(f1_score(y_test, y_pred,average=None)))

    print("______________________________________________________")
df.hist()
df
for fet in df.columns:

    plt.figure()

    sns.boxplot(df[fet])

    plt.show()
x=df.drop("Outcome",axis=1)

y=df["Outcome"]

from sklearn.neighbors import KNeighborsClassifier as knn

for i in [3,5,7,9,11,15,21]: 

    clf2=knn(n_neighbors=i)

    print(cross_val_score(clf2,x,y,cv=10).mean())
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

x=df.drop("Outcome",axis=1)

y=df["Outcome"]

clf3=RandomForestClassifier()

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)

#clf.fit(x,y)

cross_val_score(clf3,x,y,cv=10).mean()


from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

print(random_grid)
rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(x, y)

rf_random.best_params_
rf = RandomForestClassifier(n_estimators= 1600,

 min_samples_split= 2,

 min_samples_leaf= 4,

 max_features= 'sqrt',

 max_depth= 10,

 bootstrap= True)

cross_val_score(rf,x,y,cv=10).mean()