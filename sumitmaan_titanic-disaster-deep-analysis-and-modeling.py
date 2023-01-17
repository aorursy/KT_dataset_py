import warnings

warnings.filterwarnings("ignore")

import pandas as pd

import numpy as np

from scipy import stats

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report

from sklearn.metrics import make_scorer
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")

print(df_train.columns)

df_train.describe(include='all')
df_train.info()
null_data = (df_train.isnull().sum()/df_train.isnull().count())*100

null_data.sort_values(ascending=False).head(4)
null_data = (df_test.isnull().sum()/df_test.isnull().count())*100

null_data.sort_values(ascending=False).head(4)
print(df_train.columns.tolist())

df_train.hist(figsize=(12,8))

plt.show()
print(df_test.columns.tolist())

df_test.hist(figsize=(12,8))

plt.show()
sns.catplot(x="Embarked", y="Survived",data=df_train)

sns.catplot(x="Sex", y="Survived",data=df_train);
sns.boxplot(x='SibSp', data=df_train)
sns.boxplot(x='Fare', data=df_train)
sns.boxplot(x='Fare', data=df_test)
# z = np.abs(stats.zscore(df_train['SibSp']))

# df_train.iloc[np.where(z>3)]['SibSp']
# def removing_outliers(df)

# df_num = df_train.select_dtypes(include=[np.number])    ## selecting dataframe columns with numerical values

# z = np.abs(stats.zscore(df_num))                      ## Getting Z score for the numerical values columns 

# outlier_rows = np.where(z>3)[0]

# df_train.drop(outlier_rows, axis=0, inplace= True)

# df_train.reset_index(drop=True, inplace=True)
df_train["Embarked"].fillna(df_train["Embarked"].mode()[0], inplace=True)

df_test["Fare"].fillna(df_test["Fare"].median(), inplace=True)



def age_transform(df):

    df["Age"].fillna(df['Age'].median(), inplace=True)

    bins = (0, 5, 12, 18, 25, 35, 60, 120)

    group_names = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    df["Age"] = pd.cut(df["Age"], bins, labels=group_names) 

    return df



def fare_transform(df):

    df['Fare'] = pd.cut(df['Fare'], bins=[0, 7.91, 14.45, 31, 120, 200], 

                        labels=['Low_fare', 'median_fare', 'Average_fare','high_fare', 'v_high_fare'])

    

    return df



def name_transform(df):

    df["Lname"] = df["Name"].apply(lambda x: x.split(',')[0])

    df["prefix"] = df["Name"].apply(lambda x: x.split(' ')[1])

    df['prefix'] =  df['prefix'].apply(lambda x: x.split('.')[0])

    df['prefix'] = df['prefix'].apply(lambda x: x if x in ['Mr', 'Mrs', 'Miss', 'Master']  else 'Misc')

    return df



def feature_transform(df):

    df = name_transform(df)

    df = age_transform(df)

    df = fare_transform(df)

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    df.drop(["PassengerId","Name", "Ticket", "Cabin", "SibSp", "Parch"], axis=1, inplace=True)

    return df
df_train = feature_transform(df_train)



df_test = feature_transform(df_test)
def get_dummy_data(df):

    cols = ['Age', 'Embarked', 'prefix', 'Sex', 'Fare']

    df = pd.get_dummies(df, columns=cols, prefix=cols)

    return df

df_train = get_dummy_data(df_train)

df_test = get_dummy_data(df_test)
sns.heatmap(df_train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix

fig=plt.gcf()

fig.set_size_inches(20,12)

plt.show()
X = df_train.iloc[:, 1:]

y = df_train.iloc[:, 0]

X_test =  df_test
X.head()
def features_encode(df):

    features = df.columns

    for feature in features:

        le = LabelEncoder()

        le = le.fit(df[feature])

        df[feature] = le.transform(df[feature])

    return df



X= features_encode(X)

X_test =  features_encode(X_test)
scaler = StandardScaler()

X = scaler.fit_transform(X)

X_test = scaler.fit_transform(X_test)
X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold



kfold = KFold(n_splits=10, random_state=22)

model = XGBClassifier()

param_grid = {'n_estimators' : [100,210,300,400],

              'gamma': [0.1, 0.05, 0.01,0.001],

              'max_depth': [4,6, 8]

              }



modelf = GridSearchCV(model, param_grid = param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



modelf.fit(X_train,y_train)



# Best score

modelf.best_score_



# Best Estimator

modelf.best_estimator_
modelf.best_score_
clf = XGBClassifier(n_estimators=210, max_depth=6,gamma=0.05)

# clf = RandomForestClassifier(random_state=42)

clf.fit(X_train, y_train)
xgb_pred = clf.predict(X_cv)

accuracy = accuracy_score(y_cv, xgb_pred)

clf_report = classification_report(y_cv, xgb_pred)

print("XGB Classifier Accuracy: ", accuracy*100, "%", "\n\n",  "XGB Classification Report: \n ", clf_report)
clf = XGBClassifier(n_estimators=210, max_depth=6,gamma=0.05)

clf.fit(X, y)
y_pred = modelf.predict(X_test)

df = pd.read_csv("../input/gender_submission.csv")

df["Survived"] = y_pred

df.set_index(['PassengerId', 'Survived'], drop=True, inplace=True, )

df.to_csv("titanic_disaster.csv")
print('cheers !')
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=None, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=1,

            oob_score=False, random_state=None, verbose=0,

            warm_start=False)

random_forest.fit(X_train, y_train)

Y_pred_rf = random_forest.predict(X_test)

random_forest.score(X_cv,y_cv)

acc_random_forest = round(random_forest.score(X_cv, y_cv) * 100, 2)



print("Important features")

# pd.Series(random_forest.feature_importances_,df_train.columns).sort_values(ascending=True).plot.barh(width=0.8)

print('__'*30)

print(acc_random_forest)
df = pd.read_csv("../input/gender_submission.csv")

df["Survived"] = Y_pred_rf

df.set_index(['PassengerId', 'Survived'], drop=True, inplace=True, )

df.to_csv("titanic_disaster.csv")