# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from numpy import *
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/adult-census-income/adult.csv")
df.head()
# The amount of rows and columns od data-set
print(df.shape)
df.count()[1]
# Information about data-set
df.info()
#Before dropping the duplicate Rows
print(df.shape)
#After Dropping the duplicate Rows
df = df.drop_duplicates(keep = 'first')
df.shape
# Checking the null values in the columns
df.isnull().sum()
# This Code will Count the occuring of the '?' in all the columns
def check(x):
    return sum(x=='?')
df.apply(check)
# Dropping the rows whose workclass is '?' 
df = df[df.workclass != '?']

df['workclass'].value_counts()
# Dropping the rows whose occupation is '?' 
df = df[df.occupation != '?']

df['occupation'].value_counts()
# Dropping the rows whose country is '?' 
df = df[df['native.country'] != '?']

df['native.country'].value_counts()
#Checking for cleanliness
df.apply(check)
#Counting the values of column "sex"
df['sex'].value_counts()
#Showing the plot
sns.countplot(df['sex'])
#Showing the age histogram
df['age'].hist()

#Showing the sex histogram where sex = 'female'

df[df['sex']=='Female'].age.hist()

#Showing the age mean where sex = 'female'
df[df['sex']=='Female'].age.mean()
#Showing the sex histogram where sex = 'male'
df[df['sex']=='Male'].age.hist()
#Showing the age mean where sex = 'male'
df[df['sex']=='Male'].age.mean()
# This distribution plot shows the distribution of Age of people across the Data Set
plt.rcParams['figure.figsize'] = [12, 8]
sns.set(style = 'whitegrid')

sns.distplot(df['age'], bins = 90, color = 'mediumslateblue')
plt.ylabel("Distribution", fontsize = 15)
plt.xlabel("Age", fontsize = 15)
plt.margins(x = 0)

print ("The maximum age is", df['age'].max())
print ("The minimum age is", df['age'].min())
#Distribution of age according to their workclass
fig=sns.FacetGrid(df,hue='workclass',aspect=3)
fig.map(sns.kdeplot,'age',shade=True)
a=df['age'].max()
fig.set(xlim=(0,a))
fig.add_legend()
df.loc[df['native.country']!='United-States','native.country'] = 'non_usa'

#Creating different plots
fig, ((a,b),(c,d),(e,f)) = plt.subplots(3,2,figsize=(15,20))
plt.xticks(rotation=45)
sns.countplot(df['workclass'],hue=df['income'],ax=f)
sns.countplot(df['relationship'],hue=df['income'],ax=b)
sns.countplot(df['marital.status'],hue=df['income'],ax=c)
sns.countplot(df['race'],hue=df['income'],ax=d)
sns.countplot(df['sex'],hue=df['income'],ax=e)
sns.countplot(df['native.country'],hue=df['income'],ax=a)

#Showing amount of hours per week according to their workclass
df.groupby(by='workclass')['hours.per.week'].mean()
# This heatmap shows the Correlation between the different variables
plt.rcParams['figure.figsize'] = [10,7]
sns.heatmap(df.corr(), annot = True);
# This shows the hours per week according to the education of the person
sns.set(rc={'figure.figsize':(12,8)})
sns_grad = sns.barplot(x = df['education'], y = df['hours.per.week'], data = df)
plt.setp(sns_grad.get_xticklabels(), rotation=90);
# This bar graph shows the difference of hours per week between male and female 
sns.set(style = 'whitegrid', rc={'figure.figsize':(8,6)})
sns.barplot(x = df['sex'], y = df['hours.per.week'], data = df,
            estimator = mean, hue = 'sex', palette = 'winter');
# Changing the income column into Numerical Value
df['income'] = df['income'].map({'<=50K':0, '>50K':1})
# Changing the Categorical Values to Numerical values using the Label Encoder
from sklearn.preprocessing import LabelEncoder

categorical_features = list(df.select_dtypes(include=['object']).columns)
label_encoder_feat = {}
for i, feature in enumerate(categorical_features):
    label_encoder_feat[feature] = LabelEncoder()
    df[feature] = label_encoder_feat[feature].fit_transform(df[feature])

df.head()
df.head()
# Splitting the data set into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(df[['age', 'workclass', 'education','marital.status', 'occupation', 'relationship', 'race',
       'capital.gain', 'capital.loss', 'hours.per.week', 'native.country']],df['income'],test_size=0.3)
#Showing train and test size
print ("Train data set size : ", X_train.shape)
print ("Test data set size : ", X_test.shape)
# Plotting the feature importances using the Boosted Gradient Descent
from xgboost import XGBClassifier
from xgboost import plot_importance

# Training the model
model = XGBClassifier()
model_importance = model.fit(X_train, y_train)

# Plotting the Feature importance bar graph
plt.rcParams['figure.figsize'] = [14,12]
sns.set(style = 'darkgrid')
plot_importance(model_importance);
# Importing the required libraries
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import warnings; warnings.simplefilter('ignore')
# Training the model 1
NB = BernoulliNB(alpha = 0.3)
model_1 = NB.fit(X_train, y_train)

# Predictions
pred_1 = model_1.predict(X_test)

print("Accuracy for BerNoulliNB Model: %.2f" % (accuracy_score(y_test, pred_1) * 100))

# Training the model_2
nbc = GaussianNB()
model_2 = nbc.fit(X_train, y_train)

# Predictions
pred_2 = model_2.predict(X_test)
print("Accuracy for Naive Bayes Model: %.2f" % (accuracy_score(y_test, pred_2) * 100))

# Training the model_3
clf5 = MLPClassifier()
model_3 = clf5.fit(X_train, y_train)

# Predictions
pred_3 = model_3.predict(X_test)
print("Accuracy for ANN Model: %.2f" % (accuracy_score(y_test, pred_3) * 100))

# Training the model_4
logistic = LogisticRegression(C = 0.5, max_iter = 500)
model_4 = logistic.fit(X_train, y_train)

# Predictions
pred_4 = model_4.predict(X_test)
print("Accuracy for Logistic Regression Model: %.2f" % (accuracy_score(y_test, pred_4) * 100))

# Training the model_5
drugTree = DecisionTreeClassifier(criterion="gini")
model_5 = drugTree.fit(X_train, y_train)

# Predictions
pred_5 = model_5.predict(X_test)
print("Accuracy for Decision Tree Model: %.2f" % (accuracy_score(y_test, pred_5) * 100))

# Training the model_6
R_forest = RandomForestClassifier(n_estimators = 200)
model_6 = R_forest.fit(X_train, y_train)

# Predictions
pred_6 = model_6.predict(X_test)
print("Accuracy for Random Forest Model: %.2f" % (accuracy_score(y_test, pred_6) * 100))

# Training the model 7
boosted_gd = xgb.XGBClassifier(learning_rate = 0.35, n_estimator = 200)
model_7 = boosted_gd.fit(X_train, y_train)

# Predictions
pred_7 = model_7.predict(X_test)

print("Accuracy for XGB Model: %.2f" % (accuracy_score(y_test, pred_7) * 100))

list_pred = [pred_1, pred_2, pred_3, pred_4, pred_5, pred_6, pred_7]
model_names = [ "Bernoulli NB","Naive Bayes","ANN","Logistic Regression","Decision Tree" ,"Random Forest Classifier", "Boosted Gradient Descent"]

for i, predictions in enumerate(list_pred) :
    print ("Classification Report of ", model_names[i])
    print ()
    print (classification_report(y_test, predictions, target_names = ["<=50K", ">50K"]))
for i, pred in enumerate(list_pred) :
    print ("The Confusion Matrix of : ", model_names[i])
    print (pd.DataFrame(confusion_matrix(y_test, pred)))
    print ()
# ROC Curve for the classification models

from sklearn.metrics import roc_auc_score, roc_curve
models = [model_1, model_2, model_3, model_4, model_5, model_6, model_7]

# Setting the parameters for the ROC Curve
plt.rcParams['figure.figsize'] = [10,8]
plt.style.use("bmh")

color = ['red', 'blue', 'green', 'fuchsia', 'cyan','yellow','brown']
plt.title("ROC CURVE", fontsize = 15)
plt.xlabel("Specificity", fontsize = 15)
plt.ylabel("Sensitivity", fontsize = 15)
i = 1

for i, model in enumerate(models) :
    prob = model.predict_proba(X_test)
    prob_positive = prob[:,1]
    fpr, tpr, threshold = roc_curve(y_test, prob_positive)
    plt.plot(fpr, tpr, color = color[i])
    plt.gca().legend(model_names, loc = 'lower right', frameon = True)

plt.plot([0,1],[0,1], linestyle = '--', color = 'black')
plt.show()