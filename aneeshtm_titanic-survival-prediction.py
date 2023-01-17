# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from collections import Counter

from sklearn.ensemble import RandomForestClassifier, VotingClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
gender = pd.read_csv('../input/titanic/gender_submission.csv')

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

train.select_dtypes(include=[np.number])
print('train data have {0} rows and {1} columns'.format(train.shape[0],train.shape[1]))
train.columns[train.isnull().any()]
train.Age.fillna((train.Age.median()),inplace=True)

train.Age
sns.distplot(train.Age)
feature_columns = ['Age','SibSp','Parch','Fare']

train.Fare = train.Fare.map(lambda i:np.log(i) if i > 0 else 0)

sns.distplot(train.Fare)

#mean = train.Age.mean()

#train.Age.fillna('mean', inplace=True)
train.SibSp.skew()
sns.scatterplot(x='SibSp', y='Survived',data=train)
train.drop(train[train.SibSp >= 8].index, inplace=True)

train.SibSp.skew()
train.Parch.skew()
sns.scatterplot(x='Parch',y='Survived',data=train)
X = pd.get_dummies(train[feature_columns])

Y = train['Survived']

X_test = pd.get_dummies(test[feature_columns])

X_test.head()
sns.scatterplot(x='Fare', y='Survived',data=train)
train.drop(train[train.Fare > 250].index, inplace = True)

train.Fare.skew()
def detect_outliers(df,n,features):

    outlier_indices = []

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col],25)

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

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    return multiple_outliers   

# detect outliers from Age, SibSp , Parch and Fare

Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])

train.loc[Outliers_to_drop] # Show the outliers rows

    
#train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)

#Dropping above outliers is not giving
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X,Y)

#predict = logreg.predict(X_test)
X
from sklearn.ensemble import RandomForestClassifier



y = train["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train[features])

X_test = pd.get_dummies(test[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission_random.csv', index=False)

print("Your submission was successfully saved!")
y.shape
X_lg = pd.DataFrame(train, columns=features)

X_lg = pd.get_dummies(X_lg)

X_lg

logreg.fit(X,y)

predictions = logreg.predict(X_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission_logistic.csv', index=False)
from sklearn import linear_model

lm = linear_model.LinearRegression()

lm.fit(X_lg,y)

predictions1 = lm.predict(X_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions1})

output.to_csv('my_submission_linear.csv', index=False)

from sklearn.svm import SVC

sv = SVC(kernel='rbf')

sv.fit(X_lg,y)

predict_svm = sv.predict(X_test)

output = pd.DataFrame({'PassengerID': test.PassengerId, 'Survived': predict_svm})

output.to_csv('my_submission_svm.csv',index=False)


VotingPredictor = VotingClassifier(estimators=[('ExtC', ExtC_best), ('GBC',GBC_best),

('SVMC', SVMC_best), ('random_forest', random_forest_best)], voting='soft', n_jobs=4)

VotingPredictor = VotingPredictor.fit(X_lg, y)

VotingPredictor_predictions = VotingPredictor.predict(X_test)

test_Survived = pd.Series(VotingPredictor_predictions, name="Survived")

Submission3 = pd.concat([PassengerId,test_Survived],axis=1)

output.to_csv('my_submission3.csv',index=False)