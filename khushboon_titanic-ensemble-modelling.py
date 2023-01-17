# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno



from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

import xgboost as xgb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
titanic_train = pd.read_csv("/kaggle/input/titanic/train.csv")

titanic_test = pd.read_csv("/kaggle/input/titanic/test.csv")
titanic_train.head(5)
titanic_train.hist(bins=20, figsize=(20,9), color='#91992C')

plt.tight_layout()
titanic_train.describe()
titanic_train['Survived'].value_counts()
plt.figure(figsize=(10,5))

                         

color = ('#F5934B', '#C7F54B')

plt.pie(titanic_train["Survived"].value_counts(), data = titanic_train, explode=[0.08,0], labels=("Not Survived", "Survived"), 

        autopct="%1.1f%%", colors=color, shadow=True, startangle=400, radius=1.6, textprops = {"fontsize":20})

plt.title("Survival Rate", loc='right',fontsize=30)

plt.legend()

plt.show();
fig, axes = plt.subplots(1, 3, figsize=(25,7))



#Survival by Sex

sns.countplot(x='Sex', hue='Survived', data=titanic_train, ax=axes[0], palette= 'twilight_shifted')

axes[0].set_title('Survival by sex')

axes[0].set_ylabel('')



#Survival by Pclass

sns.countplot(x='Pclass', hue='Survived', data=titanic_train, ax=axes[1], palette='spring')

axes[1].set_title('Survival by Pclass')

axes[1].set_ylabel('')





#Survival by Embarkation

sns.countplot(x='Embarked', hue='Survived', data=titanic_train, ax=axes[2], palette= 'viridis')

axes[2].set_title('Survival by Embarked')

axes[2].set_ylabel('')





plt.show()



# 0 = Not Survived

# 1 = Survived
# Look at survival rate by sex

titanic_train.groupby('Sex')[['Survived']].mean()
# Survival rate by sex and class

titanic_train.pivot_table('Survived', index = 'Sex', columns = 'Pclass')
# sns.barplot(y='Pclass', x='Sex', hue= 'Survived', data=titanic_train, palette="seismic")

fig, axes = plt.subplots(1, 2, figsize=(15,5))



sns.barplot(x='Pclass', y='Survived', data = titanic_train, palette='Blues_r', ax=axes[0])

axes[0].set_title('Survival rate by Class')

axes[0].set_ylabel('')



sns.barplot(x='Embarked', y='Survived', data = titanic_train, palette='icefire', ax=axes[1])

axes[1].set_title('Survival rate by Embarktion')

axes[1].set_ylabel('')

# Survival rate bt age, class & sex

age = pd.cut(titanic_train['Age'], [0,18,30,80])

t = titanic_train.pivot_table('Survived', ('Sex', age), 'Pclass')

t.style.background_gradient(cmap='Set2')
# Prices paid by each class

plt.figure(figsize=(10,10))

plt.scatter(titanic_train['Fare'], titanic_train['Pclass'], label='Passenger Paid', color='#ED710A')

plt.ylabel('Class')

plt.xlabel('Fare')

plt.legend()

plt.show
#Try uncommenting the below code, we can see that the functions that we were using till now like, info(). describe(), isnull().sum(), etc, all can be viewed at once with the below function.

#PS: It uses a lot of RAM



# import pandas_profiling as pp

# pp.ProfileReport(titanic_train)
# Train set

titanic_train = titanic_train.drop(["PassengerId", "Ticket"], axis=1)



# For submission

submission = pd.DataFrame(columns=["PassengerId", "Survived"])

submission["PassengerId"] = titanic_test["PassengerId"]



# Test set

titanic_test = titanic_test.drop(["PassengerId", "Ticket"], axis=1)
titanic_train.head(3)
fig, axes = plt.subplots(1, 2, figsize=(20,7), sharey=True)

msno.bar(titanic_train, ax=axes[0], color='#E3ED0A')

axes[0].set_title("Training Set")



msno.bar(titanic_test, ax=axes[1], color='#0AEDEB')

axes[1].set_title("Test Set")



plt.show()
# Checking for outliers, helps to decide what should be used, Mean or Media to fill NAN values.



fig, axes = plt.subplots(1, 2, figsize=(18,5))

sns.boxplot(y="Age",data=titanic_train, orient="h", color='#ED590A', ax=axes[0])

sns.set_color_codes(palette="colorblind")

sns.distplot(titanic_train['Age'],color='#BA0AED', ax=axes[1])

plt.grid()



# As we can see there are few outliers in Age attribute & also it is positive skewed, we shall use median to fill the missing values.
fig, axes = plt.subplots(1, 2, figsize=(18,5))

sns.boxplot(y="Age",data=titanic_test, orient="h", color='#ED590A', ax=axes[0])

sns.set_color_codes(palette="colorblind")

sns.distplot(titanic_test['Age'],color='#BA0AED', ax=axes[1])

plt.grid()



# We can see there are few outliers in the test data set also, so we shall fill this with ``Median``
titanic_train["Age"].fillna(titanic_train["Age"].median(), inplace=True)

titanic_test["Age"].fillna(titanic_test["Age"].median(), inplace=True)
# As we saw earlier also in the graph

titanic_train["Embarked"].value_counts()
# Fill NAN of Embarked in training set with 'S'

titanic_train["Embarked"].fillna("C", inplace = True)
# Fill Missing Values for Cabin in training set with 0

titanic_train["Cabin"] = titanic_train["Cabin"].apply(lambda x: str(x)[0])

titanic_train.groupby(["Cabin", "Pclass"])["Pclass"].count()
titanic_train["Cabin"] = titanic_train["Cabin"].replace("n", 0)

titanic_train["Cabin"] = titanic_train["Cabin"].replace(["A", "B", "C", "D", "E", "T"], 1)

titanic_train["Cabin"] = titanic_train["Cabin"].replace("F", 2)

titanic_train["Cabin"] = titanic_train["Cabin"].replace("G", 3)
titanic_test["Cabin"] = titanic_test["Cabin"].apply(lambda x: str(x)[0])

titanic_test.groupby(["Cabin", "Pclass"])["Pclass"].count()
titanic_test["Cabin"] = titanic_test["Cabin"].replace("n", 0)

titanic_test["Cabin"] = titanic_test["Cabin"].replace(["A", "B", "C", "D", "E"], 1)

titanic_test["Cabin"] = titanic_test["Cabin"].replace("F", 2)

titanic_test["Cabin"] = titanic_test["Cabin"].replace("G", 3)
# Train Set

titanic_train["Family"] = titanic_train["SibSp"]+titanic_train["Parch"]



#Test Set

titanic_test["Family"] = titanic_test["SibSp"]+titanic_test["Parch"]
# 1 If alone & 0 if it has family members

titanic_train["Alone"] = titanic_train["Family"].apply(lambda x:1 if x==0 else 0)

titanic_test["Alone"] = titanic_test["Family"].apply(lambda x:1 if x==0 else 0)
titanic_test.head(3)
# Checkin the row were there is misisng value for Fare

titanic_test[titanic_test["Fare"].isnull()]
# Considering the other features, filling the NAN value of Fare accordingly

m_fare = titanic_test[(titanic_test["Pclass"] == 3) & (titanic_test["Embarked"] == "S") & (titanic_test["Alone"] == 1)]["Fare"].mean()

m_fare
titanic_test["Fare"] = titanic_test["Fare"].fillna(m_fare)
def title(name):

    for string in name.split():

        if "." in string:

            return string[:-1]



titanic_train["Title"] = titanic_train["Name"].apply(lambda x: title(x))

titanic_test["Title"] = titanic_test["Name"].apply(lambda x: title(x))



print(titanic_train["Title"].value_counts())

print(titanic_test["Title"].value_counts())
for titanic in [titanic_train, titanic_test]:

    titanic["Title"] = titanic["Title"].replace(["Dr", "Rev", "Major", "Col", "Capt", "Lady", "Jonkheer", "Sir", "Don", "Countess", "Dona"], "Others")

    titanic["Title"] = titanic["Title"].replace("Mlle", "Miss")

    titanic["Title"] = titanic["Title"].replace("Ms", "Miss")

    titanic["Title"] = titanic["Title"].replace("Mme", "Mr")
# Remove few more columns



titanic_train = titanic_train.drop(["Name", "SibSp", "Parch"], axis=1)

titanic_test = titanic_test.drop(["Name", "SibSp", "Parch"], axis=1)
titanic_train.head(3)
titanic_test.head(3)
# Print the unique values of the categorical columns

print(titanic_train['Sex'].unique())

print(titanic_train['Embarked'].unique())

print(titanic_train['Title'].unique())
label_encode = LabelEncoder()

var_mod = ['Sex','Embarked','Title']

for i in var_mod:

    titanic_train[i] = label_encode.fit_transform(titanic_train[i])

    titanic_test[i] = label_encode.fit_transform(titanic_test[i])
titanic_train = pd.get_dummies(titanic_train, columns =['Sex','Embarked','Cabin', 'Pclass', 'Title'])

titanic_test = pd.get_dummies(titanic_test, columns =['Sex','Embarked', 'Cabin', 'Pclass', 'Title'])
titanic_train.columns
titanic_test.head()
# Split the titanic_train data set into features ``x`` & label ``y``

x = titanic_train.iloc[:,1:22].values

y = titanic_train.iloc[:,0].values
# Splitting the data set into 80% Training & 20% Testing

train_x, test_x, train_y, test_y = train_test_split(x,y, test_size = 0.2, random_state = 42)

train_x.shape, test_x.shape, train_y.shape, test_y.shape
feature_scale = StandardScaler()

train_x = feature_scale.fit_transform(train_x)

test_x = feature_scale.transform(test_x)
# Scaling titanic_test data set as well

scale_test_data = feature_scale.fit_transform(titanic_test)
scale_test_data.shape
def models(train_x, train_y):

    

    #Logistic Regression

    log_reg = LogisticRegression(solver = 'lbfgs', random_state = 42)

    log_reg.fit(train_x,train_y)

    

    #KNN

    knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')

    knn.fit(train_x, train_y)

    

    #SVC Linear

    svc_lin = SVC(kernel = 'linear', random_state=42)

    svc_lin.fit(train_x, train_y)

    

    #SVC RBF

    svc_rbf = SVC(kernel = 'rbf', random_state=42)

    svc_rbf.fit(train_x, train_y)

    

    #Decision Tree Classifier

    dec_tree = DecisionTreeClassifier(criterion='entropy', random_state=42)

    dec_tree.fit(train_x, train_y)

    

    #Random Forest Classifier

    rf = RandomForestClassifier(n_estimators = 500, criterion='entropy', max_depth = 6,random_state=42)

    rf.fit(train_x, train_y)

    

    #XGBoost

    xg = xgb.XGBClassifier(n_estimators=1000, max_depth = 4, learning_rate=0.01, reg_lambda=0.22, gamma=0.2, random_state=42)

    xg.fit(train_x, train_y)



    #Printing accuracy for every model

    print('[0] Logistic Regression training accuracy: ', log_reg.score(train_x, train_y))

    print('[1] KNN training accuracy: ', knn.score(train_x, train_y))

    print('[2] SVC_Linear training accuracy: ', svc_lin.score(train_x, train_y))

    print('[3] SVC_RBF training accuracy: ', svc_rbf.score(train_x, train_y))

    print('[4] Decision Tree training accuracy: ', dec_tree.score(train_x, train_y))

    print('[5] Random Forest training accuracy: ', rf.score(train_x, train_y))

    print('[6] XGBoost training accuracy: ', xg.score(train_x, train_y))

        

    return log_reg, knn, svc_lin, svc_rbf, dec_tree, rf, xg
# Get and Train all the models

model = models(train_x, train_y)
# Creating confusion matrix and see the accuracy for all the models for test data



for i in range( len(model) ):

    cm  = confusion_matrix(test_y, model[i].predict(test_x))

    

    # Extract the confusion matrix parameters

    TN, FP, FN, TP = confusion_matrix(test_y, model[i].predict(test_x)).ravel()

    

    test_score = (TP+TN) / (TP+TN+FP+FN)

    

    print(cm)

    print('Model[{}] Testing Accuracy ="{}"'.format(i, test_score))

    print()



# Get Important Features for Random Forest

xg = model[5]

importance = pd.DataFrame({'Features': titanic_train.iloc[:,1:22].columns, 'Importance' : np.round(xg.feature_importances_,3)})

importance = importance.sort_values('Importance', ascending = False).set_index('Features')

importance
# Visualize the important features

importance.plot.bar(color='b')

plt.show()
# Printing the prediction of Random Forest

pred = model[6].predict(test_x)

print(pred)



print()



# Printing the actual values

print(test_y)
pred_rand_for = xg.predict(scale_test_data)

submission["Survived"] = pred_rand_for
submission.head(6)
submission.to_csv("Submission_Khushboo_XGB.csv", index=False)