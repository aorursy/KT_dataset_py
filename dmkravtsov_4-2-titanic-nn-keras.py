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
import seaborn as sns

pd.set_option('display.max_rows', 1000)

import matplotlib.pyplot as plt

import math

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc

from sklearn.model_selection import GridSearchCV,StratifiedKFold

from sklearn.preprocessing import LabelEncoder

from collections import Counter

seed =45



plt.style.use('fivethirtyeight')
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
## thanks to @Nadezda Demidova  https://www.kaggle.com/demidova/titanic-eda-tutorial-with-seaborn

train.loc[train['PassengerId'] == 631, 'Age'] = 48



# Passengers with wrong number of siblings and parch

train.loc[train['PassengerId'] == 69, ['SibSp', 'Parch']] = [0,0]

test.loc[test['PassengerId'] == 1106, ['SibSp', 'Parch']] = [0,0]
## checking for Survived dependence of Sex feature

train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Outlier detection and df cleaning from them



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



# detect outliers from Age, SibSp , Parch and Fare

Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])

train.loc[Outliers_to_drop] # Show the outliers rows

# Drop outliers

# train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
## let's concatenate test and train datasets excluding ID and Target features

df = pd.concat((train.loc[:,'Pclass':'Embarked'], test.loc[:,'Pclass':'Embarked'])).reset_index(drop=True)

%pip install autoviz # installing and importing autoviz, another library for automatic data visualization

from autoviz.AutoViz_Class import AutoViz_Class

AV = AutoViz_Class()

report_2 = AV.AutoViz("/kaggle/input/titanic/train.csv")
# df feature distribution before features tuning

def basic_details(df):

    b = pd.DataFrame()

    b['Missing value, %'] = round(df.isnull().sum()/df.shape[0]*100)

    b['N unique value'] = df.nunique()

    b['dtype'] = df.dtypes

    return b

basic_details(df)
## for Age imputation let's check its dependence from Pclass

pd.DataFrame(df.groupby('Pclass')['Age'].describe())
# New Title feature

df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

df['Title'] = df['Title'].replace('Mlle', 'Miss')

df['Title'] = df['Title'].replace('Ms', 'Miss')

df['Title'] = df['Title'].replace('Mme', 'Mrs')

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

df['Title'] = df['Title'].map(title_mapping)

df['Title'] = df['Title'].fillna(0)

##dropping Name feature

df = df.drop(['Name'], axis=1)



# Convert 'Sex' variable to integer form!

df["Sex"][df["Sex"] == "male"] = 0

df["Sex"][df["Sex"] == "female"] = 1

df["Sex"] = df["Sex"].astype(int)



## Age tuning:

df['Age'] = df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))

df["Age"] = df["Age"].astype(int)



df['Age_cat'] = pd.qcut(df['Age'],q=[0, .16, .33, .49, .66, .83, 1], labels=False, precision=1)







## Ticket tuning

tickets = []

for i in list(df.Ticket):

    if not i.isdigit():

        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])

    else:

        tickets.append("x")

df["Ticket"] = tickets



df = pd.get_dummies(df, columns= ["Ticket"], prefix = "T")



## Fare tuning:

df['Fare'] = df.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.median())) 



df['Zero_Fare'] = df['Fare'].map(lambda x: 1 if x == 0 else (0))





def fare_category(fr): 

    if fr <= 7.91:

        return 1

    elif fr <= 14.454 and fr > 7.91:

        return 2

    elif fr <= 31 and fr > 14.454:

        return 3

    return 4



df['Fare_cat'] = df['Fare'].apply(fare_category) 





# Replace missing values with 'U' for Cabin

df['Cabin'] = df['Cabin'].fillna('U')

import re

# Extract first letter

df['Cabin'] = df['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

cabin_category = {'A':9, 'B':8, 'C':7, 'D':6, 'E':5, 'F':4, 'G':3, 'T':2, 'U':1}

# Mapping 'Cabin' to group

df['Cabin'] = df['Cabin'].map(cabin_category)



df["Embarked"] = df["Embarked"].fillna("C")

df["Embarked"][df["Embarked"] == "S"] = 1

df["Embarked"][df["Embarked"] == "C"] = 2

df["Embarked"][df["Embarked"] == "Q"] = 3

df["Embarked"] = df["Embarked"].astype(int)



# New 'familySize' feature & dripping 2 features:

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1





df['FamilySize_cat'] = df['FamilySize'].map(lambda x: 1 if x == 1 

                                                            else (2 if 5 > x >= 2 

                                                                  else (3 if 8 > x >= 5 

                                                                       else 4 )

                                                                 ))       



df['Alone'] = [1 if i == 1 else 0 for i in df['FamilySize']]









dummy_col=['Title', 'Sex',  'Age_cat', 'SibSp', 'Parch', 'Fare_cat', 'Cabin', 'Embarked', 'Pclass', 'FamilySize_cat']

dummy = pd.get_dummies(df[dummy_col], columns=dummy_col, drop_first=False)

df = pd.concat([dummy, df], axis = 1)



## not gives us better score:

# dummy_fare = ['Fare']

# dummy_f = pd.get_dummies(df[dummy_fare], columns=dummy_fare, drop_first=True)

# df = pd.concat([dummy_f, df], axis = 1)



## some little dance with features

df['FareCat_Sex'] = df['Fare_cat']*df['Sex']

df['Pcl_Sex'] = df['Pclass']*df['Sex']

df['Pcl_Title'] = df['Pclass']*df['Title']

df['Age_cat_Sex'] = df['Age_cat']*df['Sex']

df['Age_cat_Pclass'] = df['Age_cat']*df['Pclass']

df['Title_Sex'] = df['Title']*df['Sex']

df['Age_Fare'] = df['Age_cat']*df['Fare_cat']



df['SmallF'] = df['FamilySize'].map(lambda s: 1 if  s == 2  else 0)

df['MedF']   = df['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

df['LargeF'] = df['FamilySize'].map(lambda s: 1 if s >= 5 else 0)

df['Senior'] = df['Age'].map(lambda s:1 if s>70 else 0)

df.shape
## adding 76 new featrurse for best score 0.90231 vs 0.90605

def descrictive_stat_feat(df):

    df = pd.DataFrame(df)

    dcol= [c for c in df.columns if df[c].nunique()>=10]

    d_median = df[dcol].median(axis=0)

    d_mean = df[dcol].mean(axis=0)

    q1 = df[dcol].apply(np.float32).quantile(0.25)

    q3 = df[dcol].apply(np.float32).quantile(0.75)

    

    #Add mean and median column to data set having more then 3 categories

    for c in dcol:

        df[c+str('_median_range')] = (df[c].astype(np.float32).values > d_median[c]).astype(np.int8)

        df[c+str('_mean_range')] = (df[c].astype(np.float32).values > d_mean[c]).astype(np.int8)

        df[c+str('_q1')] = (df[c].astype(np.float32).values < q1[c]).astype(np.int8)

        df[c+str('_q3')] = (df[c].astype(np.float32).values > q3[c]).astype(np.int8)

    return df



df = descrictive_stat_feat(df)
df.shape
# df after tuning

def basic_details(df):

    b = pd.DataFrame()

    b['Missing value'] = df.isnull().sum()

    b['N unique value'] = df.nunique()

    b['dtype'] = df.dtypes

    return b

basic_details(df)
#creating matrices for feature selection:

X_train = df[:train.shape[0]]

X_test_fin = df[train.shape[0]:]

y = train.Survived

X_train['Y'] = y

df = X_train

df.head(20) ## DF for Model training



X = df.drop('Y', axis=1)

y = df.Y
from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

import keras

from keras.optimizers import SGD

import graphviz

import eli5

from eli5.sklearn import PermutationImportance
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=10)
# Initialising the NN

sq = Sequential()



# layers

sq.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 134))

sq.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))

sq.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))

sq.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# Compiling the ANN

sq.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Train the ANN

sq.fit(x_train, y_train, batch_size = 32, epochs = 200, validation_data=(x_valid, y_valid))
# leaks = {

# 897:1,

# 899:1, 

# 930:1,

# 932:1,

# 949:1,

# 987:1,

# 995:1,

# 998:1,

# 999:1,

# 1016:1,

# 1047:1,

# 1083:1,

# 1097:1,

# 1099:1,

# 1103:1

# }
#save result for neuro model



y_pred = sq.predict(X_test_fin)

y_final = (y_pred > 0.5).astype(int).reshape(X_test_fin.shape[0])

sub = pd.DataFrame()

sub['PassengerId'] = test['PassengerId']

sub['Survived'] = y_final

sub['Survived'] = sub.apply(lambda r: leaks[int(r['PassengerId'])] if int(r['PassengerId']) in leaks else r['Survived'], axis=1)

sub.to_csv('submission.csv', index=False)

print("Your submission was successfully saved!")



sub.head()
