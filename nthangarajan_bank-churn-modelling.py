# Importing Librarys

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV # to split the data

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score, roc_curve #To evaluate our model

from sklearn.metrics import plot_confusion_matrix

from sklearn.externals import joblib

from scipy import stats

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from pylab import rcParams



import warnings

warnings.filterwarnings("ignore")
# Using pandas 

# Data downloaded from above link and stored in local folder

dataframe = pd.read_csv("../input/bank-customer-churn-modeling/Churn_Modelling.csv")
dataframe.head(5)
# Find out the total number of rows and columns

dataframe.shape
#columns and their corresponding data types,along with finding whether they contain null values or not

dataframe.info()
# The describe() function in pandas is very handy in getting various summary statistics.

# This function returns the count, mean, standard deviation, minimum and maximum values and the quantiles of the data

dataframe.describe()
# Checking data sampling

dataframe.Exited.unique()

dataframe.Exited.value_counts()
# Data Visualization

import seaborn as sns

sns.countplot(dataframe['Exited'],label="Count")

plt.show()
# Data to plot

male_dataframe = dataframe[dataframe.Gender=="Male"]

sizes = male_dataframe['Exited'].value_counts(sort = True)

colors = ["Green","Red"] 

rcParams['figure.figsize'] = 5,5

# Plot

plt.pie(sizes,  colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=270,)

plt.title('Percentage of Churn in Dataset for Male')

plt.show()
# Data to plot

female_dataframe = dataframe[dataframe.Gender=="Female"]

sizes = female_dataframe['Exited'].value_counts(sort = True)

colors = ["Green","Red"] 

rcParams['figure.figsize'] = 5,5

# Plot

plt.pie(sizes,  colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=270,)

plt.title('Percentage of Churn in Dataset for Female')

plt.show()
correlations = dataframe.corr()

# plot correlation matrix

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(correlations, vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,9,1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(dataframe.columns)

ax.set_yticklabels(dataframe.columns)

rcParams['figure.figsize'] = 40,15

plt.show()
rcParams['figure.figsize'] = 10,10

boxplot = dataframe.boxplot(column=['EstimatedSalary', 'Balance'])
rcParams['figure.figsize'] = 10,8

sns.boxplot(x="Geography",y="Exited",data=dataframe,palette='rainbow')
# removing null values to avoid errors  

dataframe.dropna(inplace = True)
#remove the fields from the data set that we don't

# want to include in our model

del dataframe['RowNumber']

del dataframe['CustomerId']

del dataframe['Surname']

from sklearn.preprocessing import LabelEncoder
label1 = LabelEncoder()

dataframe['Geography'] = label1.fit_transform(dataframe['Geography'])
label2 = LabelEncoder()

dataframe['Gender'] = label2.fit_transform(dataframe['Gender'])
features_dataframe = pd.get_dummies(dataframe, columns=['Geography'])
#Remove the Exited from the feature data

del features_dataframe['Exited']
X = features_dataframe.values

y = dataframe['Exited']

features_dataframe.columns

# apply near miss

import imblearn

from imblearn.under_sampling import NearMiss 

nr = NearMiss() 

X, y = nr.fit_sample(X, y)

X[100]
y[100]
#Split the data set in a traning set (80%) and a test set(20%)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
# Algorithmns models to be compared

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier
%%time

# to feed the random state

seed = 7

results = []

names = []

scoring = 'recall'



# Models = [RandomForestClassifier(), LogisticRegression(), DecisionTreeClassifier(), KNeighborsClassifier(), 

# LinearDiscriminantAnalysis(),GaussianNB(), SVC(), GradientBoostingClassifier(), XGBClassifier()]

# prepare models

models = []

models.append(('LogisticRegression         :', LogisticRegression()))

models.append(('LinearDiscriminantAnalysis :', LinearDiscriminantAnalysis()))

models.append(('KNeighborsClassifier       :', KNeighborsClassifier()))

models.append(('DecisionTreeClassifier     :', DecisionTreeClassifier()))

models.append(('GaussianNB                 :', GaussianNB()))

models.append(('RandomForestClassifier     :', RandomForestClassifier()))

models.append(('SVC                        :', SVC(gamma='auto')))

models.append(('GradientBoostingClassifier :', GradientBoostingClassifier()))

models.append(('XGBClassifier              :', XGBClassifier()))

print('Accuracy_Score :')

print('----------------')

for name, model in models:

    model.fit(X_train,y_train)

    kfold = KFold(n_splits=10, random_state=seed)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    print(name,accuracy_score(y_test, model.predict(X_test))*100)

# From above models we can see, 

# We are getting highest accuracy and better values of precision and recall for XGBClassifier

model = GradientBoostingClassifier()

model.fit(X_train, y_train)

print('Accuracy = ',accuracy_score(y_test, model.predict(X_test)))

print('classification_report = ',classification_report(y_test, model.predict(X_test)))
%%time

parameters = {

    "learning_rate": [0.01, 0.025, 0.05],

    "max_depth":[1,3,5],

    "max_features":["log2","sqrt"],

    "n_estimators":[10,50,100,150]

    }



grid_search = GridSearchCV(GradientBoostingClassifier(), parameters, cv=10, n_jobs=-1)

grid_search.fit(X_train, y_train)

print(grid_search.score(X_train, y_train))

print(grid_search.best_params_)
%%time

#Seting the Hyper Parameters

model = GradientBoostingClassifier(learning_rate= 0.05, 

                                   max_depth = 5, 

                                   max_features = 'sqrt', 

                                   n_estimators= 50)

model.fit(X_train, y_train)

print('Accuracy = ',accuracy_score(y_test, model.predict(X_test)))

print('classification_report = ',classification_report(y_test, model.predict(X_test)))
print('Accuracy = ',accuracy_score(y_test, model.predict(X_test)))

print('classification_report = ',classification_report(y_test, model.predict(X_test)))
# Plot non-normalized confusion matrix

titles_options = [("Confusion matrix, without normalization", None),

                  ("Normalized confusion matrix", 'true')]

for title, normalize in titles_options:

    disp = plot_confusion_matrix(model, X_test, y_test,

                                 display_labels=['Exited','Not Exited'],

                                 cmap=plt.cm.Blues,

                                 normalize=normalize)

    disp.ax_.set_title(title)



    print(title)

    print(disp.confusion_matrix)



plt.show()
y_pred_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic')

plt.show()
# Save the trained model to a file so we can use it in other programs

joblib.dump(model,"customer_churn_mlmodel.pkl")
# These are the features labels from out data set

feature_labels = np.array(['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',

       'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_0','Geography_1', 'Geography_2'])
# Create a numpy array based on the model's feature importances

importance = model.feature_importances_
# Sort the feature labels based on the feature importance rankings from the model

feature_indexes_by_importance = importance.argsort()
# Print each feature label, from most important to least important (reverse order)

for index in feature_indexes_by_importance:

    print("{} - {:.2f}%".format(feature_labels[index], (importance[index] * 100.0)))