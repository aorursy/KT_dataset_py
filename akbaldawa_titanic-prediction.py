# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
train = pd.read_csv('../input/titanic/train.csv')
train.info()
test = pd.read_csv('../input/titanic/test.csv')
test.info()
test.isna().sum()
# Outlier detection 

def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n )

    return multiple_outliers   

# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])
train.loc[Outliers_to_drop]
# Drop outliers
train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
train['source'] = 'Train'
test['source'] = 'Test'
data = pd.concat([train,test],ignore_index=True,sort= False)
data.info()
data['Family'] = data['SibSp'] + data['Parch'] + 1
#data['IsAlone'] = data[data['Family'] == 1]
data.head()
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in data['Name']]
data["Title"] = pd.Series(dataset_title)
data["Title"].head()
data.isna().sum()
# Convert to categorical values Title 
data["Title"] = data["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
data["Title"] = data["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
data["Title"] = data["Title"].astype(int)
# Create new feature of family size
data['Single'] = data['Family'].map(lambda s: 1 if s == 1 else 0)
data['SmallF'] = data['Family'].map(lambda s: 1 if  s == 2  else 0)
data['MedF'] = data['Family'].map(lambda s: 1 if 3 <= s <= 4 else 0)
data['LargeF'] = data['Family'].map(lambda s: 1 if s >= 5 else 0)
# convert to indicator values Title and Embarked 
data = pd.get_dummies(data, columns = ["Title"])
data['Embarked'].fillna('S',inplace=True)
data = pd.get_dummies(data, columns = ["Embarked"], prefix="Em")
# Create categorical values for Pclass
data["Pclass"] = data["Pclass"].astype("category")
data = pd.get_dummies(data, columns = ["Pclass"],prefix="Pc")
data.head()
#Filter categorical variables
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']
categorical_columns
data.head()
#Print frequency of categories
for col in categorical_columns:
    print('\nFrequency of Categories for varible %s'%col)
    print(data[col].value_counts())
data['Cabin'].unique()
data.drop(['Cabin'],axis=1,inplace=True)
data.info()
data['Age'].describe()
data['Age'].fillna((data['Age'].mean()),inplace=True)
data['Fare'].fillna((data['Fare'].mean()),inplace=True)
import seaborn as sns
sns.boxplot(x=data['Fare'])
sns.boxplot(x=test['Fare'])
sns.boxplot(x=train['Fare'])
data.head()
data.drop(['Name','Ticket','PassengerId','SibSp','Parch'],axis=1,inplace=True)
Sex_Val = {"male": 0,"female": 1}
data['Sex'] = data['Sex'].map(Sex_Val)
data['Sex'] = data['Sex'].astype("int64")
#Divide into test and train:
train_data = data.loc[data['source']=="Train"]
test_data = data.loc[data['source']=="Test"]
train_data["Survived"] = train_data["Survived"].astype("int64")
train_data.drop('source',axis=1,inplace=True)
test_data.drop(['source','Survived'],axis=1,inplace=True)
from sklearn.model_selection import train_test_split

independent_var = train_data.drop(["Survived"], axis=1)
dependent_var = train_data["Survived"]
x_train, x_test, y_train, y_test = train_test_split(independent_var, dependent_var, 
                                                  test_size = 20, random_state = 0)
## Machine learning tools.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
models = []
models.append(SVC())
models.append(LinearSVC())
models.append(Perceptron())
models.append(GaussianNB())
models.append(SGDClassifier())
models.append(LogisticRegression())
models.append(KNeighborsClassifier())
models.append(RandomForestClassifier())
models.append(DecisionTreeClassifier())
models.append(GradientBoostingClassifier())

accuracy_list = []
for model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = (accuracy_score(y_pred, y_test, normalize=True)*100)
    accuracy_list.append(accuracy)


model_name_list = ["SVM","Linear SVC","Perceptron","Gaussian NB","SGD Classifier","Logistic Regression",
                   "K-Neighbors Classifier","Random Forest Classifier","Decision Tree","Gradient Boosting"]

best_model = pd.DataFrame({"Model": model_name_list, "Score": accuracy_list})
best_model.sort_values(by="Score", ascending=False)
DT = DecisionTreeClassifier()
DT.fit(x_train, y_train)

LR = LogisticRegression()
LR.fit(x_train, y_train)

GB = GradientBoostingClassifier()
GB.fit(x_train, y_train)

passenger_id = test["PassengerId"]
pred = GB.predict(test_data)
predictions = pd.DataFrame({ "PassengerId" : passenger_id, "Survived": pred })

## predictions.to_csv("submission.csv", index=False)
predictions.to_csv("Titanic_submission_GB.csv", index=False)