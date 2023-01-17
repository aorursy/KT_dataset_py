import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



import os

print(os.listdir())



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv("../input/heart.csv")
type(data)
data.shape
data.head()
data.describe()
data.info()
data.sample(5)
data.isnull().sum()
data.isnull().sum().sum()
print(data.corr()["target"].abs().sort_values(ascending=False))
y = data["target"]
ax = sns.countplot(data["target"])

target_temp = data.target.value_counts()

print(target_temp)
print("Percentage of patience without heart problems: "+str(round(target_temp[0]*100/303,2)))

print("Percentage of patience with heart problems: "+str(round(target_temp[1]*100/303,2)))
data["sex"].unique()
sns.barplot(data["sex"],data["target"])
def plotAge():

    facet_grid = sns.FacetGrid(data, hue='target')

    facet_grid.map(sns.kdeplot, "age", shade=True, ax=axes[0])

    legend_labels = ['disease false', 'disease true']

    for t, l in zip(axes[0].get_legend().texts, legend_labels):

        t.set_text(l)

        axes[0].set(xlabel='age', ylabel='density')



    avg = data[["age", "target"]].groupby(['age'], as_index=False).mean()

    sns.barplot(x='age', y='target', data=avg, ax=axes[1])

    axes[1].set(xlabel='age', ylabel='disease probability')



    plt.clf()
fig_age, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 8))



plotAge()
countFemale = len(data[data.sex == 0])

countMale = len(data[data.sex == 1])

print("Percentage of Female Patients:{:.2f}%".format((countFemale)/(len(data.sex))*100))

print("Percentage of Male Patients:{:.2f}%".format((countMale)/(len(data.sex))*100))
categorial = [('sex', ['female', 'male']), 

              ('cp', ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic']), 

              ('fbs', ['fbs > 120mg', 'fbs < 120mg']), 

              ('restecg', ['normal', 'ST-T wave', 'left ventricular']), 

              ('exang', ['yes', 'no']), 

              ('slope', ['upsloping', 'flat', 'downsloping']), 

              ('thal', ['normal', 'fixed defect', 'reversible defect'])]
def plotGrid(isCategorial):

    if isCategorial:

        [plotCategorial(x[0], x[1], i) for i, x in enumerate(categorial)] 

    else:

        [plotContinuous(x[0], x[1], i) for i, x in enumerate(continuous)] 
def plotCategorial(attribute, labels, ax_index):

    sns.countplot(x=attribute, data=data, ax=axes[ax_index][0])

    sns.countplot(x='target', hue=attribute, data=data, ax=axes[ax_index][1])

    avg = data[[attribute, 'target']].groupby([attribute], as_index=False).mean()

    sns.barplot(x=attribute, y='target', hue=attribute, data=avg, ax=axes[ax_index][2])

    

    for t, l in zip(axes[ax_index][1].get_legend().texts, labels):

        t.set_text(l)

    for t, l in zip(axes[ax_index][2].get_legend().texts, labels):

        t.set_text(l)

fig_categorial, axes = plt.subplots(nrows=len(categorial), ncols=3, figsize=(15, 30))



plotGrid(isCategorial=True)
continuous = [('trestbps', 'blood pressure in mm Hg'), 

              ('chol', 'serum cholestoral in mg/d'), 

              ('thalach', 'maximum heart rate achieved'), 

              ('oldpeak', 'ST depression by exercise relative to rest'), 

              ('ca', '# major vessels: (0-3) colored by flourosopy')]
def plotContinuous(attribute, xlabel, ax_index):

    sns.distplot(data[[attribute]], ax=axes[ax_index][0])

    axes[ax_index][0].set(xlabel=xlabel, ylabel='density')

    sns.violinplot(x='target', y=attribute, data=data, ax=axes[ax_index][1])
fig_continuous, axes = plt.subplots(nrows=len(continuous), ncols=2, figsize=(15, 22))



plotGrid(isCategorial=False)
pd.crosstab(data.age,data.target).plot(kind="bar",figsize=(20,6))

plt.title('Heart Disease Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.savefig('heartDiseaseAndAges.png')

plt.show()
pd.crosstab(data.sex,data.target).plot(kind="bar",figsize=(20,10),color=['blue','#AA1111' ])

plt.title('Heart Disease Frequency for Sex')

plt.xlabel('Sex (0 = Female, 1 = Male)')

plt.xticks(rotation=0)

plt.legend(["Don't have Disease", "Have Disease"])

plt.ylabel('Frequency')

plt.show()
data.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',

       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
data.head()
pd.crosstab(data.fasting_blood_sugar,data.target).plot(kind="bar",figsize=(20,10),color=['#4286f4','#f49242'])

plt.title("Heart disease according to FBS")

plt.xlabel('FBS- (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')

plt.xticks(rotation=90)

plt.legend(["Don't Have Disease", "Have Disease"])

plt.ylabel('Disease or not')

plt.show()
data["chest_pain_type"].unique()
plt.figure(figsize=(26, 10))

sns.barplot(data["chest_pain_type"],y)
data["resting_blood_pressure"].unique()
plt.figure(figsize=(26, 10))

sns.barplot(data["resting_blood_pressure"],y)
data["rest_ecg"].unique()
plt.figure(figsize=(26, 15))

sns.barplot(data["rest_ecg"],y)
data["exercise_induced_angina"].unique()
plt.figure(figsize=(10, 10))

sns.barplot(data["exercise_induced_angina"],y)
data["st_slope"].unique()
plt.figure(figsize=(25, 10))

sns.barplot(data["st_slope"],y)
data["num_major_vessels"].unique()
sns.countplot(data["num_major_vessels"])
sns.barplot(data["num_major_vessels"],y)
data["thalassemia"].unique()
sns.distplot(data["thalassemia"])
sns.barplot(data["thalassemia"],y)
plt.figure(figsize=(20,10))

sns.scatterplot(x='cholesterol',y='thalassemia',data=data,hue='target')

plt.show()
plt.figure(figsize=(20,10))

sns.scatterplot(x='thalassemia',y='resting_blood_pressure',data=data,hue='target')

plt.show()
plt.figure(figsize=(20, 10))

plt.scatter(x=data.age[data.target==1], y=data.thalassemia[(data.target==1)], c="green")

plt.scatter(x=data.age[data.target==0], y=data.thalassemia[(data.target==0)])

plt.legend(["Disease", "Not Disease"])

plt.xlabel("Age")

plt.ylabel("Maximum Heart Rate")

plt.show()
sns.pairplot(data=data)
data.hist()
# store numeric variables in cnames

cnames=['age','resting_blood_pressure','cholesterol','max_heart_rate_achieved','st_depression','num_major_vessels']
#Set the width and height of the plot

f, ax = plt.subplots(figsize=(7, 5))



#Correlation plot

df_corr = data.loc[:,cnames]

#Generate correlation matrix

corr = df_corr.corr()



#Plot using seaborn library

sns.heatmap(corr, annot = True, cmap='coolwarm',linewidths=.1)

plt.show()
df_corr = data.loc[:,cnames]

df_corr
from sklearn.model_selection import train_test_split



predictors = data.drop("target",axis=1)

target = data["target"]



X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)

print("Training features have {0} records and Testing features have {1} records.".\

      format(X_train.shape[0], X_test.shape[0]))
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape
from sklearn.metrics import accuracy_score
def train_model(X_train, y_train, X_test, y_test, classifier, **kwargs):

    

    """

    Fit the chosen model and print out the score.

    

    """

    

    # instantiate model

    model = classifier(**kwargs)

    

    # train model

    model.fit(X_train,y_train)

    

    # check accuracy and print out the results

    fit_accuracy = model.score(X_train, y_train)

    test_accuracy = model.score(X_test, y_test)

    

    print(f"Train accuracy: {fit_accuracy:0.2%}")

    print(f"Test accuracy: {test_accuracy:0.2%}")

    

    return model
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()



logreg.fit(X_train, Y_train)



y_pred_lr = logreg.predict(X_test)

print(y_pred_lr)
score_lr = round(accuracy_score(y_pred_lr,Y_test)*100,2)



print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

print(confusion_matrix(Y_test,y_pred_lr))

print(classification_report(Y_test,y_pred_lr))

print("Accuracy:",accuracy_score(Y_test, y_pred_lr))
# Logistic Regression

from sklearn.linear_model import LogisticRegression

model = train_model(X_train, Y_train, X_test, Y_test, LogisticRegression)
#Logistic Regression supports only solvers in ['liblinear', 'newton-cg'<-93.44, 'lbfgs'<-91.8, 'sag'<-72.13, 'saga'<-72.13]

clf = LogisticRegression(random_state=0, solver='newton-cg').fit(X_test, Y_test)

#The solver for weight optimization.

#'lbfgs' is an optimizer in the family of quasi-Newton methods.

clf.score(X_test, Y_test)
from sklearn.metrics import confusion_matrix
matrix= confusion_matrix(Y_test, y_pred_lr)
sns.heatmap(matrix,annot = True, fmt = "d")
from sklearn.metrics import precision_score
precision = precision_score(Y_test, y_pred_lr)
print("Precision: ",precision)
from sklearn.metrics import recall_score
recall = recall_score(Y_test, y_pred_lr)
print("Recall is: ",recall)
print((2*precision*recall)/(precision+recall))
from sklearn.ensemble import RandomForestClassifier

randfor = RandomForestClassifier(n_estimators=100, random_state=0)



randfor.fit(X_train, Y_train)



y_pred_rf = randfor.predict(X_test)

print(y_pred_rf)
from sklearn.model_selection import learning_curve

# Create CV training and test scores for various training set sizes

train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(), 

                                                        X_train, 

                                                        Y_train,

                                                        # Number of folds in cross-validation

                                                        cv=10,

                                                        # Evaluation metric

                                                        scoring='accuracy',

                                                        # Use all computer cores

                                                        n_jobs=-1, 

                                                        # 50 different sizes of the training set

                                                        train_sizes=np.linspace(0.01, 1.0, 50))



# Create means and standard deviations of training set scores

train_mean = np.mean(train_scores, axis=1)

train_std = np.std(train_scores, axis=1)



# Create means and standard deviations of test set scores

test_mean = np.mean(test_scores, axis=1)

test_std = np.std(test_scores, axis=1)



# Draw lines

plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")

plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")



# Draw bands

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")

plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")



# Create plot

plt.title("Learning Curve")

plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")

plt.tight_layout()

plt.show()

score_rf = round(accuracy_score(y_pred_rf,Y_test)*100,2)



print("The accuracy score achieved using Random Forest is: "+str(score_rf)+" %")
#Random forest with 100 trees

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=0)

rf.fit(X_train, Y_train)

print("Accuracy on training set: {:.3f}".format(rf.score(X_train, Y_train)))

print("Accuracy on test set: {:.3f}".format(rf.score(X_test, Y_test)))
rf1 = RandomForestClassifier(max_depth=3, n_estimators=100, random_state=0)

rf1.fit(X_train, Y_train)

print("Accuracy on training set: {:.3f}".format(rf1.score(X_train, Y_train)))

print("Accuracy on test set: {:.3f}".format(rf1.score(X_test, Y_test)))
from sklearn.metrics import confusion_matrix
matrix= confusion_matrix(Y_test, y_pred_rf)
sns.heatmap(matrix,annot = True, fmt = "d")
from sklearn.metrics import precision_score
precision = precision_score(Y_test, y_pred_rf)
print("Precision: ",precision)
from sklearn.metrics import recall_score
recall = recall_score(Y_test, y_pred_rf)
print("Recall is: ",recall)
print((2*precision*recall)/(precision+recall))
from sklearn.naive_bayes import GaussianNB

nb = train_model(X_train, Y_train, X_test, Y_test, GaussianNB)



nb.fit(X_train, Y_train)



y_pred_nb = nb.predict(X_test)

print(y_pred_nb)
score_nb = round(accuracy_score(y_pred_nb,Y_test)*100,2)



print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")
#Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

model = train_model(X_train, Y_train, X_test, Y_test, GaussianNB)
from sklearn.metrics import confusion_matrix
matrix= confusion_matrix(Y_test, y_pred_nb)
sns.heatmap(matrix,annot = True, fmt = "d")
from sklearn.metrics import precision_score
precision = precision_score(Y_test, y_pred_nb)
print("Precision: ",precision)
from sklearn.metrics import recall_score
recall = recall_score(Y_test, y_pred_nb)
print("Recall is: ",recall)
print((2*precision*recall)/(precision+recall))
from sklearn.neighbors import KNeighborsClassifier

knn = train_model(X_train, Y_train, X_test, Y_test, KNeighborsClassifier, n_neighbors=8)



knn.fit(X_train, Y_train)



y_pred_knn = knn.predict(X_test)

print(y_pred_knn)
score_knn = round(accuracy_score(y_pred_knn,Y_test)*100,2)



print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")
# KNN

from sklearn.neighbors import KNeighborsClassifier

model = train_model(X_train, Y_train, X_test, Y_test, KNeighborsClassifier)
# Seek optimal 'n_neighbours' parameter

for i in range(1,10):

    print("n_neigbors = "+str(i))

    train_model(X_train, Y_train, X_test, Y_test, KNeighborsClassifier, n_neighbors=i)
from sklearn.metrics import confusion_matrix
matrix= confusion_matrix(Y_test, y_pred_knn)
sns.heatmap(matrix,annot = True, fmt = "d")
from sklearn.metrics import precision_score
precision = precision_score(Y_test, y_pred_knn)
print("Precision: ",precision)
from sklearn.metrics import recall_score
recall = recall_score(Y_test, y_pred_knn)
print("Recall is: ",recall)
print((2*precision*recall)/(precision+recall))
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=3, random_state=0)



dt.fit(X_train, Y_train)



y_pred_dt = dt.predict(X_test)

print(y_pred_dt)
score_dt = round(accuracy_score(y_pred_dt,Y_test)*100,2)



print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")
from sklearn.tree import DecisionTreeClassifier

tree1 = DecisionTreeClassifier(random_state=0)

tree1.fit(X_train, Y_train)

print("Accuracy on training set: {:.3f}".format(tree1.score(X_train, Y_train)))

print("Accuracy on test set: {:.3f}".format(tree1.score(X_test, Y_test)))
tree1 = DecisionTreeClassifier(max_depth=3, random_state=0)

tree1.fit(X_train, Y_train)

print("Accuracy on training set: {:.3f}".format(tree1.score(X_train, Y_train)))

print("Accuracy on test set: {:.3f}".format(tree1.score(X_test, Y_test)))
from sklearn.metrics import confusion_matrix
matrix= confusion_matrix(Y_test, y_pred_dt)
sns.heatmap(matrix,annot = True, fmt = "d")
from sklearn.metrics import precision_score
precision = precision_score(Y_test, y_pred_dt)
print("Precision: ",precision)
from sklearn.metrics import recall_score
recall = recall_score(Y_test, y_pred_dt)
print("Recall is: ",recall)
print((2*precision*recall)/(precision+recall))
# initialize an empty list

accuracy = []



# list of algorithms names

classifiers = ['KNN', 'Decision Trees', 'Logistic Regression', 'Naive Bayes', 'Random Forests']



# list of algorithms with parameters

models = [KNeighborsClassifier(n_neighbors=8), DecisionTreeClassifier(max_depth=3, random_state=0), LogisticRegression(), 

        GaussianNB(), RandomForestClassifier(n_estimators=100, random_state=0)]



# loop through algorithms and append the score into the list

for i in models:

    model = i

    model.fit(X_train, Y_train)

    score = model.score(X_test, Y_test)

    accuracy.append(score)
# create a dataframe from accuracy results

summary = pd.DataFrame({'accuracy':accuracy}, index=classifiers)       

summary
scores = [score_lr,score_nb,score_knn,score_dt,score_rf]

algorithms = ["Logistic Regression","Naive Bayes","K-Nearest Neighbors","Decision Tree","Random Forest"] 

sns.set(rc={'figure.figsize':(15,8)})

plt.xlabel("Algorithms")

plt.ylabel("Accuracy score")



sns.barplot(algorithms,scores)