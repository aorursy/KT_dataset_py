import numpy as np #to read the file

import pandas as pd #for numerical computations
# Importing the dataset using pandas library

dataset = pd.read_csv('../input/HR_comma_sep.csv')

#prints first 5 rows

dataset.head()
#Renaming of dataset

dataset=dataset.rename(columns={'sales':'dept'})

dataset=dataset.rename(columns={'average_montly_hours':'average_monthly_hours'})
#Gives feature names, type, entry counts, feature count, memory usage etc

dataset.info() 
#lets see if there are any more columns with missing values 

dataset.isnull().sum()
#Encoding categorical data

from sklearn.preprocessing import LabelEncoder,OneHotEncoder



labelEnc=LabelEncoder()



cat_vars=["dept","salary"]

for col in cat_vars:

    dataset[col]=labelEnc.fit_transform(dataset[col])

#showing results for less confusion

#for salary, low=1,mid=2,high=0

        
#for all the plots to be in line

%matplotlib inline

#matplot.lib for plotting 

import matplotlib.pyplot as plt

plt.style.use(style = 'default')

dataset.hist(bins=11,figsize=(10,10),grid=True)
#Assuming RandomForestClassifier is best.

from sklearn.ensemble import RandomForestClassifier



predictors = ["satisfaction_level", "last_evaluation", "number_project",

              "average_monthly_hours","time_spend_company","Work_accident", "promotion_last_5years", "dept","salary"]

rf = RandomForestClassifier(random_state=1, n_estimators=50, max_depth=9,min_samples_split=6, min_samples_leaf=4)

    

rf.fit(dataset[predictors],dataset["left"])

importances=rf.feature_importances_

std = np.std([rf.feature_importances_ for tree in rf.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]

sorted_important_features=[]

for i in indices:

    sorted_important_features.append(predictors[i])

plt.figure()

plt.title("Feature Importances By Random Forest Model")

plt.bar(range(np.size(predictors)), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(np.size(predictors)), sorted_important_features, rotation='vertical')



plt.xlim([-1, np.size(predictors)])

plt.show()
#Heat Map is drawn

import seaborn as sns

sns.set(font_scale=1)

corr=dataset.corr()

plt.figure(figsize=(10, 10))



sns.heatmap(corr, vmax=1, square=True,annot=True,cmap='cubehelix')

plt.title('Correlation between features')
import seaborn as sns

sns.set(font_scale=1)

g = sns.FacetGrid(dataset, col="number_project", row="left", margin_titles=True)

g.map(plt.hist, "satisfaction_level",color="green")
g = sns.FacetGrid(dataset, hue="left", col="time_spend_company", margin_titles=True,

                  palette={1:"black", 0:"red"})

g=g.map(plt.scatter, "satisfaction_level", "average_monthly_hours",edgecolor="w").add_legend()
g = sns.FacetGrid(dataset, hue="left", col="number_project", margin_titles=True,

                palette="Set1",hue_kws=dict(marker=["^", "v"]))

g.map(plt.scatter, "average_monthly_hours", "time_spend_company",edgecolor="w").add_legend()

plt.subplots_adjust(top=0.8)

g.fig.suptitle('Resignation by time spent in compnay, no of projects, average monthly hours spent')
g = sns.FacetGrid(dataset, hue="left", col="number_project", margin_titles=True,

                  palette={1:"brown", 0:"green"})

g=g.map(plt.scatter, "satisfaction_level", "last_evaluation",edgecolor="w").add_legend()
sns.set(font_scale=1)

g = sns.factorplot(x="number_project", y="left", col="time_spend_company",

                    data=dataset, saturation=.5,

                    kind="bar", ci=None, aspect=.6)

(g.set_axis_labels("no of projects", "leaving Rate")

    .set_xticklabels([1,2,3,4,5,6,7])

    .set_titles("{col_name} {col_var}")

    .set(ylim=(0, 1))

    .despine(left=True))  

plt.subplots_adjust(top=0.8)

g.fig.suptitle('How many employees left completing projects and time spending in company')
g = sns.FacetGrid(dataset, hue="left", col="number_project", margin_titles=True,

                  palette={1:"yellow", 0:"orange"})

g=g.map(plt.scatter, "last_evaluation", "average_monthly_hours",edgecolor="w").add_legend()
dataset['efficiency'] = ( dataset['time_spend_company'] * (12) * dataset['average_monthly_hours'] )/ dataset['number_project']

#12 months in a year

_ = sns.distplot(dataset['efficiency'])

plt.show()

x1 = np.corrcoef(x=dataset['efficiency'], y=dataset['satisfaction_level'])

y1 = np.corrcoef(x=dataset['efficiency'], y=dataset['left']) 

z1 = np.corrcoef(x=dataset['left'], y=dataset['satisfaction_level']) 

print(x1,y1,z1)

X = dataset.iloc[:,[0,1,2,3,4,5,7,8,9]].values

y = dataset.iloc[:,6].values



#splitting the dataset into training set and test set

#Splitting 75%:25%

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state = 0)



#Applying feature scaling to the dataset

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.fit_transform(X_test)

#Logistic Regression

#fitting the logistic regression model to the training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train,y_train)



#Predicting the test result

y_pred = classifier.predict(X_test)



#making confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)



#Mathewws correlation metrics 

from sklearn.metrics import matthews_corrcoef

print(matthews_corrcoef(y_test, y_pred) )

print(cm)
#after categorical encoding

X = dataset.iloc[:,[0,1,2,3,4,5,7,8,9]].values

onehotencoder = OneHotEncoder(categorical_features = [[7],[8]])

X = onehotencoder.fit_transform(X).toarray()
#Splitting 75%:25%

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state = 0)



#Applying feature scaling to the dataset

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.fit_transform(X_test)



#Logistic Regression

#fitting the logistic regression model to the training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train,y_train)



#Predicting the test result

y_pred = classifier.predict(X_test)



#making confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)



#Mathewws correlation metrics 

from sklearn.metrics import matthews_corrcoef

print(matthews_corrcoef(y_test, y_pred) )

print(cm)
#with extra featured variable I have created above "efficiency"

X = dataset.iloc[:,[0,1,2,3,4,5,7,8,9,10]].values

onehotencoder = OneHotEncoder(categorical_features = [[7],[8]])

X = onehotencoder.fit_transform(X).toarray()
#Splitting 75%:25%

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state = 0)



#Applying feature scaling to the dataset

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.fit_transform(X_test)



#Logistic Regression

#fitting the logistic regression model to the training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train,y_train)



#Predicting the test result

y_pred = classifier.predict(X_test)



#making confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)



#Mathewws correlation metrics 

from sklearn.metrics import matthews_corrcoef

print(matthews_corrcoef(y_test, y_pred) )

print(cm)
X = dataset.iloc[:,[0,1,2,3,4,5,7,8,9]].values

onehotencoder = OneHotEncoder(categorical_features = [[7],[8]])

X = onehotencoder.fit_transform(X).toarray()

#Splitting 75%:25%

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state = 0)



#Applying feature scaling to the dataset

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.fit_transform(X_test)

#KNN algorithm

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski',p=2)

classifier.fit(X_train, y_train)



#Predicting the results

y_pred = classifier.predict(X_test)

print(matthews_corrcoef(y_test,y_pred))
#SVM 

from sklearn.svm import SVC

classifier = SVC(kernel='linear',random_state=0)

classifier.fit(X_train, y_train)



#Predicting the results

y_pred = classifier.predict(X_test)

print(matthews_corrcoef(y_test,y_pred))
#kernel SVM 

from sklearn.svm import SVC

classifier = SVC(kernel='rbf',random_state=0)

classifier.fit(X_train, y_train)



#Predicting the results

y_pred = classifier.predict(X_test)

print(matthews_corrcoef(y_test,y_pred))
#Naive bayes

from sklearn.naive_bayes import GaussianNB

classifier=GaussianNB()

classifier.fit(X_train,y_train)



#Predicting the test results

y_pred = classifier.predict(X_test)

print(matthews_corrcoef(y_test,y_pred))
#Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier as dt

classifier = dt(criterion='entropy',random_state=0)

classifier.fit(X_train,y_train)



#Predicting the results

y_pred=classifier.predict(X_test)



print(matthews_corrcoef(y_test,y_pred))
#random forest classification

from sklearn.ensemble import RandomForestClassifier as rfc

#for 5 Trees in forest

classifier=rfc(n_estimators=5,criterion='entropy',random_state=0)

classifier.fit(X_train,y_train)

#Predicting the results

y_pred=classifier.predict(X_test)

print("5 tress =",matthews_corrcoef(y_test,y_pred))



#for 10 Trees in forest

classifier=rfc(n_estimators=10,criterion='entropy',random_state=0)

classifier.fit(X_train,y_train)

#Predicting the results

y_pred=classifier.predict(X_test)

print("10 tress =",matthews_corrcoef(y_test,y_pred))



#for 20 Trees in forest

classifier=rfc(n_estimators=20,criterion='entropy',random_state=0)

classifier.fit(X_train,y_train)

#Predicting the results

y_pred=classifier.predict(X_test)

print("20 tress =",matthews_corrcoef(y_test,y_pred))



#for 30 Trees in forest

classifier=rfc(n_estimators=30,criterion='entropy',random_state=0)

classifier.fit(X_train,y_train)

#Predicting the results

y_pred=classifier.predict(X_test)

print("30 tress =",matthews_corrcoef(y_test,y_pred))



#for 40 Trees in forest

classifier=rfc(n_estimators=40,criterion='entropy',random_state=0)

classifier.fit(X_train,y_train)

#Predicting the results

y_pred=classifier.predict(X_test)

print("40 tress =",matthews_corrcoef(y_test,y_pred))

"""

#Artificial neural networks

import keras

from keras.models import Sequential

from keras.layers import Dense



#Initialising the ANN

classifier = Sequential()



#Adding the input layer and hidden layer

classifier.add(Dense(output_dim=1,init='uniform',activation='relu',input_dim=20))



#Adding the output layer

classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))



#Compiling ANN - Applying the stochastic gradient 

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



#Fitting the ANN to the training set

classifier.fit(X_train,y_train,batch_size=10,epochs=100)



#Predicting the test results

y_pred=classifier.predict(X_test)

y_pred=(y_pred>0.5)



from sklearn.metrics import matthews_corrcoef

print(matthews_corrcoef(y_test, y_pred) )



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

"""

print("After running this neural network, neural network accuracy = 80.23%, loss = 34.88% Mattews coefficient accuracy is 59.22% ") 