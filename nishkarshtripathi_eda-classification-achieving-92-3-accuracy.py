# importing libraries



import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve,auc
data = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")



data.drop("sl_no", axis=1, inplace=True) # Removing Serial Number
print("Number of rows in data :",data.shape[0])

print("Number of columns in data :", data.shape[1])
data.head()
data.info()
# Percentage of null values present in salary column



p = data['salary'].isnull().sum()/(len(data))*100



print(f"Salary column has {p.round(2)}% null values.")
data.describe()
# Let's peek at the object data types seperately



data.select_dtypes(include=['object']).head()
# getting the object columns

object_columns = data.select_dtypes(include=['object']).columns



# iterating over each object type column

for col in object_columns:

    print('-' * 40 + col + '-' * 40 , end='-')

    display(data[col].value_counts())
sns.countplot("gender", data = data)

plt.show()
# Let's look at more important plot i.e gender vs status (target)



sns.countplot("gender", hue="status", data=data)

plt.show()
sns.countplot("ssc_b", data = data)

plt.show()
# Let's see the impact of taking a spcific board in 10th grade on placements



sns.set(rc={'figure.figsize':(8.7,5.27)})



sns.countplot("ssc_b", hue="status", data=data)

plt.show()
# Let's plot percentage vs status to see how much effect they make



sns.barplot(x="status", y="ssc_p", data=data)
# Let's see the how much percentage was scored by students in different boards



sns.barplot(x="ssc_b", y="ssc_p", data=data)
# Let's look at how many students opted for central this time?



sns.countplot("hsc_b", data = data)

plt.show()
# Let's see the impact of a spcific board on placements



sns.set(rc={'figure.figsize':(8.7,5.27)})



sns.countplot("hsc_b", hue="status", data=data)

plt.show()
# Let's plot percentage vs status to see how much effect they make



sns.barplot(x="status", y="hsc_p", data=data)
# Let's see the how much percentage was scored by students in 12th grade in different boards



sns.barplot(x="hsc_b", y="hsc_p", data=data)
# Let's see what count of students opted for in 12th grade



sns.countplot("hsc_s", data=data)
# Let's look at how well each specialisation students performed



ax = sns.barplot(x="hsc_s", y="hsc_p", data=data)
# Let's see the impact of taking a spcific branch on placements



sns.countplot("hsc_s", hue="status", data=data)
# Let's see what count of students opted for what after 12th grade



sns.countplot("degree_t", data=data)
# Let's look at how well each field students performed



sns.barplot(x="degree_t", y="degree_p", data=data)
# Let's see the impact of taking a field on placements



sns.countplot("degree_t", hue="status", data=data)
# Let's see if the work experience impacts on placements or not



data['status'] = data['status'].map( {'Placed':1, 'Not Placed':0})



sns.barplot(x="workex", y="status", data=data)
sns.barplot(x="status", y="etest_p", data=data)
# Let's see how specialisation effects the placement of candidates



sns.countplot("specialisation", hue="status", data=data)
sns.barplot(x="status", y="mba_p", data=data)

plt.title("Salary vs MBA Percentage")
# Let's look at the distribution of salary



plt.figure(figsize=(10,5))

sns.distplot(data['salary'], bins=50, hist=False)

plt.title("Salary Distribution")

plt.show()
sns.barplot(x="gender", y="salary", data=data)

plt.title("Salary vs gender")
sns.violinplot(x=data["gender"], y=data["salary"], hue=data["specialisation"])

plt.title("Salary vs Gender based on specialisation")
sns.violinplot(x=data["gender"], y=data["salary"], hue=data["workex"])

plt.title("Gender vs Salary based on work experience")
sns.violinplot(x=data["gender"], y=data["salary"], hue=data["ssc_b"])

plt.title("Salary vs Gender based on Board in 10th grade")
sns.violinplot(x=data["gender"], y=data["salary"], hue=data["hsc_b"])

plt.title("Salary vs Gender based on Board in 12th grade")
sns.violinplot(x=data["gender"], y=data["salary"], hue=data["degree_t"])

plt.title("Salary vs Gender based on Degree Type")
# Dropping useless columns



data.drop(['ssc_b','hsc_b', 'salary'], axis=1, inplace=True)
# Using simple binary mapping on two class categorical variables (gender, workerx, specialisation)



data["gender"] = data.gender.map({"M":0,"F":1})

data["workex"] = data.workex.map({"No":0, "Yes":1})

data["specialisation"] = data.specialisation.map({"Mkt&HR":0, "Mkt&Fin":1})
# Using get dummies for 3 class categorical variables (hsc_s and degree_t)



for column in ['hsc_s', 'degree_t']:

    dummies = pd.get_dummies(data[column])

    data[dummies.columns] = dummies
# Now let's clean up the left overs (already encoded so no use now)



data.drop(['degree_t','hsc_s'], axis=1, inplace=True)
# Now let us look at the data



data.head()
# Let's do a sanity check by peeking at the data



data.head()
# Let's plot correlation matrix to find out less correlated variable to drop them



cor=data.corr()

plt.figure(figsize=(14,6))

sns.heatmap(cor,annot=True)
# From the correlation matrix we can see that some of the features are not much useful like "Others" and "Arts" which are negatively 

# correlated as well as have low value.



# Another reason to remove these variables is the so called Dummy variable trap which occurs when we do encoding of multiclass features



data.drop(['Others', 'Arts'], axis=1, inplace=True)
# target vector

y = data['status']



# dropping as it is not a predictor

data.drop('status', axis = 1, inplace = True)



# scaling the data so as to get rid of any dramatic results during modelling

sc = StandardScaler()



# predictors

X = sc.fit_transform(data)



# Let us now split the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)





print("X-Train:",X_train.shape)

print("X-Test:",X_test.shape)

print("Y-Train:",y_train.shape)

print("Y-Test:",y_test.shape)
# creating our model instance

log_reg = LogisticRegression()



# fitting the model

log_reg.fit(X_train, y_train)
# predicting the target vectors



y_pred=log_reg.predict(X_test)
# creating confusion matrix heatmap



conf_mat = pd.DataFrame(confusion_matrix(y_test, y_pred))

fig = plt.figure(figsize=(10,7))

sns.heatmap(conf_mat, annot=True, annot_kws={"size": 16}, fmt='g')

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()
# getting precision, recall and f1-score via classification report



print(classification_report(y_test, y_pred))
# let's look at our accuracy



accuracy = accuracy_score(y_pred, y_test)



print(f"The accuracy on test set using Logistic Regression is: {np.round(accuracy, 3)*100.0}%")
# plotting the ROC curve



auc_roc = roc_auc_score(y_test, log_reg.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, log_reg.predict_proba(X_test)[:,1])



plt.plot(fpr, tpr, color='darkorange', lw=2, 

         label='Average ROC curve (area = {0:0.3f})'.format(auc_roc))

plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', 

         label= 'Average ROC curve (area = 0.500)')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
# calculate auc 

auc_score = auc(fpr, tpr)

print(f"Our auc_score came out to be {round(auc_score, 3)}.")
# creating a list of depths for performing Decision Tree

depth = list(range(1,10))



# list to hold the cv scores

cv_scores = []



# perform 10-fold cross validation with default weights

for d in depth:

  dt = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)

  scores = cross_val_score(dt, X_train, y_train, cv=10, scoring='accuracy', n_jobs = -1)

  cv_scores.append(scores.mean())



# finding the optimal depth

optimal_depth = depth[cv_scores.index(max(cv_scores))]

print("The optimal depth value is: ", optimal_depth)
# plotting accuracy vs depth

plt.plot(depth, cv_scores)

plt.xlabel("Depth of Tree")

plt.ylabel("Accuracy")

plt.title("Accuracy vs depth Plot")

plt.grid()

plt.show()



print("Accuracy scores for each depth value is : ", np.round(cv_scores, 3))
# create object of classifier

dt_optimal = DecisionTreeClassifier(criterion="gini", max_depth=optimal_depth, random_state=42)



# fit the model

dt_optimal.fit(X_train,y_train)



# predict on test vector

y_pred = dt_optimal.predict(X_test)



# evaluate accuracy score

accuracy = accuracy_score(y_test, y_pred)*100

print(f"The accuracy on test set using optimal depth = {optimal_depth} is {np.round(accuracy, 3)}%")
# creating a list of our models

ensembles = [log_reg, dt_optimal]



# Train each of the model

for estimator in ensembles:

    print("Training the", estimator)

    estimator.fit(X_train,y_train)
# Find the scores of each estimator



scores = [estimator.score(X_test, y_test) for estimator in ensembles]



scores
# Training a voting classifier with hard voting and using logistic regression and decision trees as estimators



from sklearn.ensemble import VotingClassifier



named_estimators = [

    ("log_reg",log_reg),

    ("dt_tree", dt_optimal),



]
# getting an instance for our Voting classifier



voting_clf = VotingClassifier(named_estimators)
# fit the classifier



voting_clf.fit(X_train,y_train)
# Let's look at our accuracy

acc = voting_clf.score(X_test,y_test)



print(f"The accuracy on test set using voting classifier is {np.round(acc, 4)*100}%")