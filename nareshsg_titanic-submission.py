# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np



import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline



#Some default display settings.

import seaborn as sns

sns.set(style = "white",color_codes=True)

sns.set(font_scale=1.5)

from IPython.display import display

pd.options.display.max_columns = None





from sklearn.model_selection import GridSearchCV #to fine tune Hyperparamters using Grid search

from sklearn.model_selection import RandomizedSearchCV# to seelect the best combination(advance ver of Grid Search)



# importing some ML Algorithms 

from sklearn.linear_model import LogisticRegression # y=mx+c

from sklearn.tree import DecisionTreeRegressor # Entropy(impurities),Gain. 

from sklearn.ensemble import RandomForestRegressor # Average of Many DT's

from sklearn.ensemble import RandomForestClassifier



# Testing Libraries - Scipy Stats Models

from scipy.stats import shapiro # Normality Test 1

from scipy.stats import normaltest # Normality Test 2

from scipy.stats import anderson # Normality Test 3

from statsmodels.graphics.gofplots import qqplot # plotting the Distribution of Y with a Line of dot on a 45 degree Line.



# Model Varification/Validation Libraries

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import ShuffleSplit





# Matrices and Reporting Libraries

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error,r2_score

from sklearn.metrics import make_scorer

from statsmodels.tools.eval_measures import rmse

from sklearn.model_selection import learning_curve




train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()



test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
train_data.shape , test_data.shape
train_data.columns , test_data.columns
sns.countplot(x= 'Sex',data=train_data)
sns.countplot(x='Survived',hue='Sex',data=train_data)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)

women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women*100)

print("% of men who survived:", rate_men*100)
sns.countplot(x='Survived',hue='Pclass',data=train_data)
sns.distplot(train_data["Age"],bins=50)
train_data["Fare"].plot.hist(bins=40,figsize=(10,6))

plt.xlabel("Fare")
sns.barplot(x= 'Sex',y='Survived',data=train_data)
plt.figure(figsize=(7,6))

sns.heatmap(train_data.isnull(),yticklabels= False,cmap='viridis')
plt.figure(figsize=(7,6))

sns.heatmap(test_data.isnull(),yticklabels= False,cmap='viridis')
# Training Data

sns.boxplot(y="Age",x="Pclass",data=train_data)

def train_age(col):

    Age= col[0]

    Pclass=col[1]

    

    if pd.isnull(Age):

        if Pclass==1:

            return 37

        elif Pclass==2:

            return 31

        else:

            return 27

    else:

        return Age

        
train_data["Age"] = train_data[["Age","Pclass"]].apply(train_age,axis=1)
train_data.drop("Cabin",axis=1,inplace=True)

train_data.head()
plt.figure(figsize=(7,6))

sns.heatmap(train_data.isnull(),yticklabels= False,cmap='viridis')
test_data.isnull().sum()
plt.figure(figsize=(7,6))

sns.heatmap(test_data.isnull(),yticklabels= False,cmap='viridis')
test_data.isnull().sum()
#Testing Data

sns.boxplot(y="Age",x="Pclass",data=test_data)
def test_age(col):

    age = col[0]

    pclass=col[1]

    

    if pd.isnull(age):

        if pclass==1:

            return 42

        elif pclass==2:

            return 28

        else:

            return 25

    else:

        return age
test_data['Age']= test_data[['Age','Pclass']].apply(test_age,axis=1)

test_data.head()
plt.figure(figsize=(7,6))

sns.heatmap(test_data.isnull(),yticklabels= False,cmap='viridis')
test_data.drop("Cabin",axis=1,inplace=True)

test_data.dropna(inplace=True)

test_data.head()
plt.figure(figsize=(7,6))

sns.heatmap(test_data.isnull(),yticklabels= False,cmap='viridis')
test_data.isnull().sum()
combined_data = train_data.append(test_data,ignore_index=True)

print(combined_data.head())

print(combined_data.shape)

combined_data.shape
combined_data.drop(["Name","Ticket"],axis=1)
train_data.shape , test_data.shape , combined_data.shape
features = ["PassengerId","Survived","Pclass", "Sex","Age", "SibSp", "Parch","Fare","Embarked"]

ds = pd.get_dummies(combined_data[features],drop_first=True)

ds
train,test = ds[0:len(train_data)],ds[len(test_data):]
train.shape , test.shape , ds.shape
x_feature = train.drop('Survived',axis=1)

y_target = train['Survived']
X_train,X_test,y_train,y_test = train_test_split(x_feature,y_target,test_size= 0.30 , random_state=4)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
lr = LogisticRegression()

lr.fit(X_train,y_train)
rmc = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

rmc.fit(X_train,y_train)

predictions = rmc.predict(X_test)

output = pd.DataFrame({'PassengerId': X_test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")

print(cross_val_score(rmc,x_feature,y_target,scoring='accuracy'))



results=cross_val_score(rmc,x_feature,y_target,scoring='accuracy')

print(results.mean()*100)