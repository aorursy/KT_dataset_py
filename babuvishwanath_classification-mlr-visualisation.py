#importing the necessary libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn
data_place = pd.read_csv("../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")

data_place.head(10)
#Details about the data

data_place.info()

#Totally there are 14 variables and we can remove the serial number variable as it is not important
data_place.shape
# now checking the missing value imputation

data_place.isna().sum()
#We can fill the salary missing values as zero because the students who are not placed has givven as missing values

data_place["salary"]=data_place["salary"].fillna(0.0)

data_place.isna().sum()
data_place.drop("sl_no",axis=1,inplace=True)

data_place.describe()
data_place.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
data_place.skew()# there is no problem of skewness
data_place.dtypes
sns.barplot("gender","salary",data=data_place)#we can say that the salary for male is more compared to the female


print(data_place.groupby('status')['gender'].value_counts(normalize=True))

print("\n")

print(data_place.groupby('status')['ssc_b'].value_counts(normalize=True))

print("\n")

print(data_place.groupby('status')['hsc_b'].value_counts(normalize=True))

print("\n")

print(data_place.groupby('status')['hsc_s'].value_counts(normalize=True))

print("\n")

print(data_place.groupby('status')['degree_t'].value_counts(normalize=True))

print("\n")

print(data_place.groupby('status')['workex'].value_counts(normalize=True))

print("\n")

print(data_place.groupby('status')['specialisation'].value_counts(normalize=True))
sns.countplot("gender", hue="status", data=data_place)

plt.show()

sns.barplot("gender","salary",hue="status",data=data_place)

#male candidates got high paid jobs than females
plt.figure(figsize =(18,6))

sns.boxplot("salary", "gender", data=data_place)

plt.show()
sns.barplot("ssc_p","status",data=data_place)

#We can say that on an average if a person takes above 50 percent in ssc he may get placed
sns.countplot("ssc_b",hue="status",data=data_place)

# we can say that the students in central board are placed more but the ssc education doesnt much effect the placement
plt.figure(figsize =(18,6))

sns.boxplot("salary", "ssc_b", data=data_place)

plt.show()

# the students studied in central board are got high paid salaries
sns.lineplot("ssc_p", "salary", hue="ssc_b", data=data_place)

plt.show()

#From this plot we can say that the candidates from central board with

# ssc percentage with an average of 60 are getting highest paid job
sns.barplot("hsc_p","status",data=data_place)

# we can say that on an average if student gets 50 percentage, there is possibility of getting placed
sns.countplot("hsc_b", hue="status", data=data_place)

plt.show()

#In HSC other board students are placed more
sns.boxplot("salary", "hsc_b", data=data_place)

# The salary for central board candidates are high
sns.countplot("hsc_s", hue="status", data=data_place)

plt.show()

#Arts students placed ratio is low
plt.figure(figsize =(18,6))

sns.boxplot("salary", "hsc_b", data=data_place)

plt.show()

# Salary for Central board candidates are high
sns.lineplot("hsc_p", "salary", hue="hsc_b", data=data_place)

plt.show()

# A candidate from HSC central board with 60 percentage are getting highest paid job 
plt.figure(figsize =(18,6))

sns.boxplot("salary", "hsc_s", data=data_place)

plt.show()

#The salary for the science students in HSC is more
#Kernel-Density Plot

sns.kdeplot(data_place.degree_p[ data_place.status=="Placed"])

sns.kdeplot(data_place.degree_p[ data_place.status=="Not Placed"])

plt.legend(["Placed", "Not Placed"])

plt.xlabel("Under Graduate Percentage")

plt.show()

# the placed rate will be high if the degree percentage is around 55 percentage
sns.countplot("degree_t", hue="status", data=data_place)

plt.show()

#commerce&Mmt students are more placed
plt.figure(figsize =(18,6))

sns.boxplot("salary", "degree_t", data=data_place)

plt.show()

# we can say that the students in Sci-tech get high paid jobs but Comm&mgmt students are getting good jobs
sns.countplot("workex",hue="status",data=data_place)

#So the students with experience are getting placed and their chance of not getting place is less
sns.barplot("salary","workex",data=data_place)

#Workexperinced candidates are getting high paid jobs
sns.barplot("etest_p","status",data=data_place)

#So this feature doesnt effect placement
sns.lineplot("etest_p","salary",data=data_place)

# so this doesnt effect the salary also
sns.barplot("specialisation","salary",hue="status",data=data_place)

#mkt and finance students are getting highly paid  jobs
sns.countplot("specialisation",hue="status",data=data_place)

#Market and finance candidates are getting placed more
sns.barplot("mba_p","status",data=data_place)

#this doesnt effect status
sns.lineplot('mba_p',"salary",data=data_place)

#doesnt effect salary 
data_new = data_place.drop(["hsc_b","ssc_b","salary"],axis=1)

data_new.info()
data_new["gender"].value_counts()
data_new.loc[data_new['gender']=='M','gender']= 0



data_new.loc[data_new['gender']=='F','gender']= 1



data_new["gender"] = data_new["gender"].astype(int)





data_new["hsc_s"] = data_new.hsc_s.map({"Commerce":0,"Science":1,"Arts":2})

data_new["degree_t"] = data_new.degree_t.map({"Comm&Mgmt":0,"Sci&Tech":1, "Others":2})

data_new["workex"] = data_new.workex.map({"No":0, "Yes":1})

data_new["status"] = data_new.status.map({"Not Placed":0, "Placed":1})

data_new["specialisation"] = data_new.specialisation.map({"Mkt&HR":0, "Mkt&Fin":1})
data_new.dtypes
#importing the necessary libraries

from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, fbeta_score, confusion_matrix, accuracy_score

from sklearn.neighbors import KNeighborsClassifier



#LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



#NAIVE_BAYES MODEL

from sklearn.naive_bayes import GaussianNB



#SVC 

from sklearn.svm import SVC



#XGBOOST

from xgboost import XGBClassifier

import pandas as pd
data_new.dtypes
cor=data_new.corr()

plt.figure(figsize=(14,6))

sns.heatmap(cor,annot=True)
#from correlation plot we can remove the hsc_s and degree_t

data_new.drop(["hsc_s","degree_t"],axis=1,inplace=True)
x = data_new.drop("status", axis =1).values#independent variable



y = data_new["status"].values #dependant variable

#train and test data split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 42)



print(x_train.shape, x_test.shape)

#NOTE:

#.values will store the values in the form of array

#if you not give x will store the values in series
#to feed the random state

seed = 42



#prepare models

models = []

models.append(("LR", LogisticRegression()))

models.append(("LDA", LinearDiscriminantAnalysis()))

models.append(("KNN", KNeighborsClassifier()))

models.append(("CART", DecisionTreeClassifier()))

models.append(("NB", GaussianNB()))

models.append(("RF", RandomForestClassifier()))

models.append(("SVM", SVC(gamma = 'auto')))

models.append(("XGB", XGBClassifier()))

#appending all the models with their names
import warnings 

warnings.filterwarnings("ignore")# to avoid the warnings in our data-set

result = []

names = []

scoring = 'recall'

seed = 42



for name, model in models:

    kfold = KFold(n_splits = 5, random_state =seed)# 5 split of data (value of k)

    cv_results = cross_val_score(model, x_train, y_train, cv = kfold, scoring = scoring)

    result.append(cv_results)

    names.append(name)

    msg = (name, cv_results.mean(), cv_results.std())

    print(msg)
#boxplot results for choosing our algorithm

import matplotlib.pyplot as plt

fig = plt.figure(figsize = (8,4))

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(1,1,1)

plt.boxplot(result)

ax.set_xticklabels(names)

plt.show()
#precion

import warnings 

warnings.filterwarnings("ignore")# to avoid the warnings in our data-set

result1 = []

names = []

scoring = 'precision'

seed = 42



for name, model in models:

    kfold = KFold(n_splits = 5, random_state =seed)# 5 split of data (value of k)

    cv_results = cross_val_score(model, x_train, y_train, cv = kfold, scoring = scoring)

    result1.append(cv_results)

    names.append(name)

    msg1 = (name, cv_results.mean(), cv_results.std())

    print(msg1)

#first one is mean value of a model, next one is the std deviation
#boxplot results for choosing our algorithm

fig = plt.figure(figsize = (8,4))

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(1,1,1)

plt.boxplot(result1)

ax.set_xticklabels(names)

plt.show()
# default scoring is a accuracy

import warnings 

warnings.filterwarnings("ignore")# to avoid the warnings in our data-set

result2 = []

names = []

seed = 42



for name, model in models:

    kfold = KFold(n_splits = 5, random_state =seed)# 5 split of data (value of k)

    cv_results = cross_val_score(model, x_train, y_train, cv = kfold)

    result2.append(cv_results)

    names.append(name)

    msg1 = (name, cv_results.mean(), cv_results.std())

    print(msg1)

#first one is mean value of a model, next one is the std deviation
fig = plt.figure(figsize = (8,4))

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(1,1,1)

plt.boxplot(result2)

ax.set_xticklabels(names)

plt.show()
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(criterion='entropy')

dtree.fit(x_train, y_train)

y_pred = dtree.predict(x_test)
print(classification_report(y_test,y_pred))
#Using Random Forest Algorithm

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(x_train, y_train)

y_pred = random_forest.predict(x_test)
print(classification_report(y_test,y_pred))
# creating confusion matrix heatmap



conf_mat = pd.DataFrame(confusion_matrix(y_test, y_pred))

fig = plt.figure(figsize=(10,7))

sns.heatmap(conf_mat, annot=True, annot_kws={"size": 16}, fmt='g')

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()


from sklearn.ensemble import ExtraTreesClassifier

x2= data_new.drop("status",axis=1)

y= data_new["status"]



model = ExtraTreesClassifier(n_estimators =5, criterion = 'entropy')



model.fit(x2,y)



fi = model.feature_importances_



print(fi)
fi_df = pd.DataFrame({'fi':fi, "feature":x2.columns})

fi_df.head()
fi_df.sort_values(["fi"],ascending = False)
x2_col = fi_df[fi_df["fi"]>0.05]

x2_col
#now we are going to extract only these features

x2 = x2[x2_col["feature"]]

x2.head()
#now with these values of x and y we are going to build the model

from sklearn.preprocessing import MinMaxScaler



std_data = MinMaxScaler()

std_data = std_data.fit_transform(x2)

std_data = pd.DataFrame(std_data, columns =x2.columns)

std_data.head()
x_train, x_test, y_train, y_test = train_test_split(std_data,y, test_size = 0.25, random_state = 100)

model1 = RandomForestClassifier().fit(x_train,y_train)



y_pred = model1.predict(x_test)



print(classification_report(y_test,y_pred))
# creating confusion matrix heatmap



conf_mat = pd.DataFrame(confusion_matrix(y_test, y_pred))

fig = plt.figure(figsize=(10,7))

sns.heatmap(conf_mat, annot=True, annot_kws={"size": 16}, fmt='g')

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()
from sklearn.model_selection import RandomizedSearchCV

#no of trees in randomforest

#n_estimators = [100,200,500]



#no of features to consider at every split

max_features = ['auto','sqrt']



#max number of levels in tree

max_depth = [int(x) for x in np.linspace(10,110,11)]



#minimum no of samples required at each node

min_samples_leaf = [1,2,4]





random_grid = {'max_features':max_features,'max_depth':max_depth,'min_samples_leaf':min_samples_leaf}



rf = RandomForestClassifier()

rf_random = RandomizedSearchCV(estimator=rf,param_distributions = random_grid, n_iter= 100, cv=3,verbose=1,n_jobs=2,random_state=11)



rf_random.fit(x_train,y_train)
rf_random.best_estimator_
newmod = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,

                       criterion='gini', max_depth=4, max_features='auto',

                       max_leaf_nodes=None, max_samples=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, n_estimators=200,

                       n_jobs=None, oob_score=False, random_state=42,

                       verbose=0, warm_start=False).fit(x_train,y_train)
y_pred = newmod.predict(x_test)

print(classification_report(y_test,y_pred))
data_n= data_place

data_n.dtypes
data_n["gender"] = data_n.gender.map({"M":0,"F":1})

data_n["ssc_b"] = data_n.ssc_b.map({"Others":0,"Central":1})

data_n["hsc_b"] = data_n.hsc_b.map({"Others":0,"Central":1})

data_n["hsc_s"] = data_n.hsc_s.map({"Commerce":0,"Science":1,"Arts":2})

data_n["degree_t"] = data_n.degree_t.map({"Comm&Mgmt":0,"Sci&Tech":1, "Others":2})

data_n["workex"] = data_n.workex.map({"No":0, "Yes":1})

data_n["status"] = data_n.status.map({"Not Placed":0, "Placed":1})

data_n["specialisation"] = data_n.specialisation.map({"Mkt&HR":0, "Mkt&Fin":1})

cor=data_n.corr()

plt.figure(figsize=(14,6))

sns.heatmap(cor,annot=True)
# Seperating Features and Target

X = data_n[[ 'ssc_p', 'hsc_p', 'hsc_s', 'degree_p',  'workex','etest_p', 'specialisation', 'mba_p',]]

y = data_n['status']
# Let us now split the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify =y)
from sklearn.metrics import make_scorer, accuracy_score,precision_score

from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



randomForestFinalModel = RandomForestClassifier(n_estimators=200,criterion='gini',

 max_depth= 4 ,

 max_features= 'auto',random_state=42)

randomForestFinalModel.fit(X_train, y_train)

predictions_rf = randomForestFinalModel.predict(X_test)



print(classification_report(y_test,predictions_rf))
conf_mat = pd.DataFrame(confusion_matrix(y_test, predictions_rf))

fig = plt.figure(figsize=(10,7))

sns.heatmap(conf_mat, annot=True, annot_kws={"size": 16}, fmt='g')

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score

from sklearn.model_selection import cross_val_score



df = pd.read_csv("../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")

df.head(10)
df["salary"]=df["salary"].fillna(0.0)

df.isna().sum()
X1 = df.drop(['status',"salary"], axis = 1)

y1 = df.status



from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()



#placed as 1, not placed as 0

y1 = encoder.fit_transform(y1)

X1 = pd.get_dummies(X1)
from sklearn.model_selection import train_test_split

X_train2, X_test2, y_train2, y_test2 = train_test_split(X1, y1, test_size= 0.3, random_state=41)
knn = KNeighborsClassifier(n_neighbors= 5 )

knn.fit(X_train2, y_train2)

y_pred2 = knn.predict(X_test2)

print(accuracy_score(y_test2,y_pred2))



print(classification_report(y_test2, y_pred2))
conf_mat = pd.DataFrame(confusion_matrix(y_test2, y_pred2))

fig = plt.figure(figsize=(10,7))

sns.heatmap(conf_mat, annot=True, annot_kws={"size": 16}, fmt='g')

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()
dataset = pd.read_csv("../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")

dataset
X=dataset.iloc[:,[2,4]].values # X contain columns hsc_p and ssc_p

Y=dataset.iloc[:,12].values.reshape(-1,1)
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
#fitting multiple linear regression to the training set

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(X_train,Y_train)



#predicting the test result

y_pred=regressor.predict(X_test)
#Let’s check out the coefficients for the predictors:

regressor.coef_
regressor.intercept_
from sklearn.metrics import r2_score

r2_score(Y_test, y_pred)
X1=dataset.iloc[:,[2,7]].values

Y1=dataset.iloc[:,12].values.reshape(-1,1)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X1,Y1,test_size=0.2,random_state=0)
#fitting multiple linear regression to the training set

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(X_train,Y_train)



#predicting the test result

y_pred=regressor.predict(X_test)

#Let’s check out the coefficients for the predictors:

regressor.coef_
regressor.intercept_
from sklearn.metrics import r2_score

r2_score(Y_test, y_pred)
X2=dataset.iloc[:,[4,7]].values

Y2=dataset.iloc[:,12].values.reshape(-1,1)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X2,Y2,test_size=0.2,random_state=0)



#fitting multiple linear regression to the training set

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(X_train,Y_train)



#predicting the test result

y_pred=regressor.predict(X_test)

#Let’s check out the coefficients for the predictors:

regressor.coef_

regressor.intercept_
from sklearn.metrics import r2_score

r2_score(Y_test, y_pred)
X3=dataset.iloc[:,[2,4,7]].values # X contain columns hsc_p and ssc_p

Y3=dataset.iloc[:,12].values.reshape(-1,1)
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X3,Y3,test_size=0.2,random_state=0)



#fitting multiple linear regression to the training set

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(X_train,Y_train)



#predicting the test result

y_pred=regressor.predict(X_test)
regressor.intercept_
r2_score(Y_test, y_pred)
regressor.coef_