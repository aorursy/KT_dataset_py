# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split



##models

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB 



## Model evaluators

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import precision_score, recall_score, f1_score



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
## Importing train dataset

df_train = pd.read_csv("/kaggle/input/titanic/train.csv")
df_train.head()
## Let's have a look at bottom five rows

df_train.tail()
## Checking for  the number of rows and columns  in the dataset

print(f"Number of rows :{df_train.shape[0]} \nNumber of columns:{df_train.shape[1]}")
df_train = df_train.drop(["Name", "Ticket", "Cabin"], axis=1)
df_train.info()
## CHecking the dtypes of the columns in the dataset

df_train.dtypes
## Checking for any null values in the dataset

df_train.isna().sum()
## Replacing age nan values with its mean

df_train["Age"]= df_train["Age"].fillna(df_train["Age"].mean())

df_train["Age"].isna().sum()
## Checking for any null values in the dataset

df_train.isna().sum()
df_train.hist(figsize=(16,8));
survived = df_train[df_train.Survived==1].count()[0]

not_survived = df_train[df_train.Survived==0].count()[0]

text = ["survived","not survived"]

label = [survived,not_survived]

plt.style.use('seaborn')

plt.figure(figsize=(8,6),dpi=100)

for bar in range(0,2):

    plt.bar(text[bar],label[bar])

    plt.text(text[bar],label[bar],str(label[bar]),fontsize=16, fontweight='bold')

plt.title("Number of people survived vs not survived")

plt.xlabel("Survived vs not survived")

plt.ylabel("Number of people")

plt.show()
df_train.Age = df_train.Age.astype(int)
## Checking the survival rate according to age

ages = df_train[df_train.Survived==1]["Age"].sort_values()

dc =  {}

for age in ages:

    if age not in dc.keys():

        dc[age] = 1

    else:

        dc[age] +=1

plt.figure(figsize=(30,10))

key = list(dc.keys())

value = list(dc.values())

for index in range(len(key)):

    plt.bar(key[index],value[index],color ='maroon')

    plt.text(key[index],value[index],str(value[index]),color="green")

plt.xticks(np.arange(len(key)),key)

plt.title("Different Ages of number of people survived")

plt.xlabel("Age")

plt.ylabel("Number of people")

plt.show()
dc = {0: 7, 1: 5, 2: 3, 3: 5, 4: 7, 5: 4, 6: 2, 7: 1, 8: 2, 9: 2, 11: 1, 12: 1, 13: 2, 14: 3, 15: 4, 16: 6, 17: 6, 18: 9, 19: 9, 20: 3, 21: 5, 22: 11, 23: 5, 24: 15, 25: 6, 26: 6, 27: 11, 28: 7, 29: 60, 30: 10, 31: 8, 32: 10, 33: 6, 34: 6, 35: 11, 36: 11, 37: 1, 38: 5, 39: 5, 40: 6, 41: 2, 42: 6, 43: 1, 44: 3, 45: 5, 47: 1, 48: 6, 49: 4, 50: 5, 51: 2, 52: 3, 53: 1, 54: 3, 55: 1, 56: 2, 58: 3, 60: 2, 62: 2, 63: 2, 80: 1}

dc_sorted = sorted(dc.items(), key=lambda x: x[1], reverse=True)

key_10 = [dc_sorted[i][0] for i in range(len(dc_sorted))][:10]

value_10= [dc_sorted[i][1] for i in range(len(dc_sorted))][:10]

plt.bar(np.arange(len(key_10)),value_10, color ='blue')

for index in range(len(key_10)):

    plt.text(index,value_10[index],str(value_10[index]),color="green")

plt.xticks(np.arange(len(key_10)),key_10)

plt.title("Top 10 Ages of Survived People")

plt.xlabel("Age")

plt.ylabel("Number of people")

plt.show()
df_train[df_train.Survived==1]["Age"].hist()

plt.title("Distribution of Age(Survived People)")

plt.xlabel("Age")

plt.ylabel("Number of people")

plt.show()
## Replacing Sex "male":0 and "female":1

print(df_train.Sex[:5])

df_train["Sex"]= df_train["Sex"].replace({"female":0, "male":1})

df_train.Sex.head()
## Comparision between number of mails and feamals survivied

males = df_train[(df_train["Survived"]==1) & (df_train.Sex==1)]["Sex"].count()

female = df_train[(df_train["Survived"]==1) & (df_train.Sex==0)]["Sex"].count()

value = [males,female]

labels = ["males","Females"]

plt.bar(np.arange(len(value)),value);

for index in range(len(value)):

    plt.text(index,value[index],str(value[index]),color="green")

plt.xticks(np.arange(len(labels)),labels)

plt.title("Comparison between number of males and females survived")

plt.xlabel("Sex")

plt.ylabel("Count")

plt.style.use("ggplot")

plt.show()
class_1 = df_train[df_train.Pclass==1].count()[0]

class_2 = df_train[df_train.Pclass==2].count()[0]

class_3 = df_train[df_train.Pclass==3].count()[0]

classs = [class_1,class_2,class_3]

plt.bar(df_train.Pclass.unique(),classs)

plt.xticks(np.arange(5),["","class 1","class 2","class 3",""])

plt.title("number of Survived people based on the class")

plt.xlabel("Class")

plt.ylabel("Numper of people")

plt.show()
df_train.dropna(inplace=True)
print(df_train.Embarked.unique())

df_train.Embarked = df_train.Embarked.replace({"S":0, "C":1,"Q":2})

df_train.Embarked.isnull().sum()

print(df_train.shape)
# Everything except target variable

print(df_train.iloc[:,2:-1].head())

X = df_train.iloc[:,2:-1].values



# Target variable

y = df_train.Survived.values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
X_train[:5]
X_test[:5]


# Put models in a dictionary

models = {"KNN": KNeighborsClassifier(),

          "Logistic Regression": LogisticRegression(), 

          "Random Forest": RandomForestClassifier(),

         "SVM": SVC(),

         "Naive bayses":GaussianNB(),

         "Decision Tree":DecisionTreeClassifier()}



# Create function to fit and score models

def fit_and_score(models, X_train,y_train,X_test,y_test):

    # Random seed for reproducible results

    np.random.seed(42)

    # Make a list to keep model scores

    model_scores = {}

    # Loop through models

    for name, model in models.items():

        # Fit the model to the data

        model.fit(X_train, y_train)

        # Evaluate the model and append its score to model_scores

        model_scores[name] = model.score(X_test, y_test)

    return model_scores

model_scores = fit_and_score(models=models,

                             X_train=X_train,

                             y_train=y_train,

                             X_test=X_test,

                             y_test=y_test)

model_scores
model_compare = pd.DataFrame(model_scores, index=['accuracy'])

model_compare.T.plot.bar();
from sklearn.ensemble import GradientBoostingClassifier

gradboost= GradientBoostingClassifier(n_estimators=300, random_state=0).fit(X_train, y_train)

preds= gradboost.predict(X_test)

sns.heatmap(confusion_matrix(y_test,preds), annot=True,cbar=False, fmt='g')

plt.xlabel("True label")

plt.ylabel("Predicted label");

print(gradboost.score(X_test,y_test))
print(classification_report(y_test, preds))
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

df_test = df_test.drop(["Name", "Ticket", "Cabin"], axis=1)

df_test["Sex"]= df_test["Sex"].replace({"female":0, "male":1})

df_test["Age"]= df_test["Age"].fillna(df_test["Age"].mean())

df_test.Age = df_test.Age.astype(int)

df_test.Embarked = df_test.Embarked.replace({"S":0, "C":1,"Q":2,"nan":3})

df_test["Fare"]= df_test["Fare"].fillna(df_test["Fare"].median())

test_x = df_test.iloc[:,1:-1].values

print(test_x)
data = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

test_y = data["Survived"].values

test_y
y_pred = gradboost.predict(test_x)

print(y_pred)

print(len(y_pred))

print(len(df_test["PassengerId"]))
preds_df= pd.DataFrame(df_test, columns=['PassengerId'])

preds_df['Survived']=y_pred

preds_df.head()
data = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

data.head()
preds_df.shape
preds_df.to_csv('/kaggle/working/Titanic_Submission.csv', index=False)