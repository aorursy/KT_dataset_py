# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from collections import Counter



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.describe()
test_df.describe()
train_df.head(4)
test_df.head(3)
train_df['Sex'].unique()

train_df['Embarked'].unique()
train_df['SibSp'].unique()
train_df['Parch'].unique()
train_df[['Age', 'Fare', 'Pclass']].info()
def extractTitleFromName(name):

    return name.split(',').pop(1).split('.').pop(0).strip()





def addTitlesToSource(source):

    titleDict = {

      'Mr': 'Mr',

      'Mrs': 'Mrs',

      'Miss': 'Miss', 

      'Master': 'Master',

      'Don': 'Sir',

      'Rev': 'Sir',

      'Dr' : 'Dr',

      'Mme': 'Miss',

      'Ms' : 'Miss',       

      'Major': 'Off', 

      'Lady' : 'Lady',

      'Sir': 'Sir',

      'Mlle': 'Miss', 

      'Col': 'Off',

      'Capt': 'Off' ,

      'the Countess': 'Lady',

      'Jonkheer': 'Sir',

      'Dona': 'Lady'

    }

    source['Title'] = source['Name'].apply(lambda x: titleDict[extractTitleFromName(x)])

    return source

    
train_df = addTitlesToSource(train_df)

test_df = addTitlesToSource(test_df)
#Lets remove outliers from the training data, thanks to 

#https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling



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

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   
Outliers_to_drop = detect_outliers(train_df,2,["Age","SibSp","Parch","Fare"])

train_df = train_df.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
#Lets look at the average fare per class

mean_fare_per_class = train_df.groupby('Pclass')['Fare'].mean().to_dict()

train_df['ImputedFare'] = train_df[['Fare', 'Pclass']].apply(lambda x: x['Fare'] if np.isfinite(x['Fare']) else mean_fare_per_class[x['Pclass']], axis = 1)

test_df['ImputedFare'] = test_df[['Fare', 'Pclass']].apply(lambda x: x['Fare'] if np.isfinite(x['Fare']) else mean_fare_per_class[x['Pclass']], axis = 1)



ageImputationDict = train_df.groupby('Title')['Age'].median().to_dict()

#Age has a lot of nans, need to impute those values using the ageImputationDict

train_df['ImputedAge'] = train_df[['Age', 'Title']].apply(lambda x: x['Age'] if np.isfinite(x['Age']) else ageImputationDict[x['Title']], axis = 1)

test_df['ImputedAge'] = test_df[['Age', 'Title']].apply(lambda x: x['Age'] if np.isfinite(x['Age']) else ageImputationDict[x['Title']], axis = 1)

test_df[['ImputedAge', 'Age']].info()
train_df[['ImputedAge', 'Age']].info()


def prepareInputDataFrame(source):

   df = pd.DataFrame()

   df['IsFemale'] = source['Sex'].apply(lambda x: 1 if x == 'female' else 0)

   df['IsClass1'] = source['Pclass'].apply(lambda x: 1 if x == 1 else 0)

   df['IsClass2'] = source['Pclass'].apply(lambda x: 1 if x == 2 else 0)

   df['IsClass3'] = source['Pclass'].apply(lambda x: 1 if x == 3 else 0)

   maxSibSp = source['SibSp'].max()

   #maxSibSp = 1

   df['SibSp']  = source['SibSp'].apply(lambda x: x / maxSibSp)

   maxParch = source['Parch'].max()

   #maxParch = 1

   df['Parch']  = source['Parch'].apply(lambda x: x / maxParch)

   #Standardize Age and Fare

   maxFare = source['ImputedFare'].max()

   #maxFare = 1

   #Standardize Fare   

   df['Fare'] = source['ImputedFare'].apply(lambda x: x / maxFare)

   maxAge = source['ImputedAge'].max()

   #maxAge = 1

   #Removing Age gives better accuracy on test set

   df['Age'] = source['ImputedAge'].apply(lambda x: x / maxAge) 

   

   return df
train_data = prepareInputDataFrame(train_df)
train_data.head(2)
X_train, X_test, y_train, y_test = train_test_split(train_data, train_df['Survived'], test_size=0.2, random_state=0)



from sklearn.neural_network import MLPClassifier

nn_model = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(10, 4), random_state=1, max_iter = 500)

nn_model = nn_model.fit(X_train, y_train)

nn_score = round(nn_model.score(X_test, y_test) * 100.0, 2)

train_score = round(nn_model.score(X_train, y_train) * 100.0, 2)

(nn_score, train_score)
from sklearn.svm import SVC

svm_model = SVC()

svm_model.fit(X_train, y_train)

svm_score = round(svm_model.score(X_test, y_test) * 100.0, 2)

train_score = round(svm_model.score(X_train, y_train) * 100.0, 2)

(svm_score, train_score)

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier()

dt_model.fit(X_train, y_train)

dt_score = round(dt_model.score(X_test, y_test) * 100.0, 2)

train_score = round(dt_model.score(X_train, y_train) * 100.0, 2)

(dt_score, train_score)
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()

lr_model.fit(X_train, y_train)

lr_score = round(lr_model.score(X_test, y_test) * 100.0, 2)

train_score = round(lr_model.score(X_train, y_train) * 100.0, 2)

(lr_score, train_score)
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators = 150)

rf_model.fit(X_train, y_train)

rf_score = round(rf_model.score(X_test, y_test) * 100.0, 2)

train_score = round(rf_model.score(X_train, y_train) * 100.0, 2)

(rf_score, train_score)

test_data = prepareInputDataFrame(test_df)
res = list(zip(test_df['PassengerId'].as_matrix(), nn_model.predict(test_data)))

print('PassengerId,Survived')

for x, y in res:

    print('{},{}'.format(x, y))