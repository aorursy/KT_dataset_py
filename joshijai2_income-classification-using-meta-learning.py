#importing the required libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')



from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split



from numpy import mean, std
#reading the dataset and converting it to dataframe

df = pd.read_csv("../input/adult-census-income/adult.csv")
#Viewing the top 5 rows of our dataset

df.head()
sns.countplot(df.income)
sns.distplot(df[df.income=='<=50K'].age, color='g')

sns.distplot(df[df.income=='>50K'].age, color='r')
plt.xticks(rotation=90)

sns.countplot(df.workclass, hue=df.income, palette='tab10')
sns.distplot(df[df.income=='<=50K'].fnlwgt, color='r')

sns.distplot(df[df.income=='>50K'].fnlwgt, color='g')
plt.xticks(rotation=90)

sns.countplot(df.education, hue=df.income, palette='muted')
sns.countplot(df["education.num"], hue=df.income)
plt.xticks(rotation=90)

sns.countplot(df['marital.status'], hue=df.income)
plt.xticks(rotation=90)

sns.countplot(df.occupation, hue=df.income, palette='rocket')
plt.xticks(rotation=90)

sns.countplot(df.relationship, hue=df.income, palette='muted')
plt.xticks(rotation=90)

sns.countplot(df.race, hue=df.income, palette='Set2')
plt.xticks(rotation=90)

sns.countplot(df.sex, hue=df.income)
df['capital.gain'].value_counts()
df['capital.loss'].value_counts()
sns.distplot(df[df.income=='<=50K']['hours.per.week'], color='b')

sns.distplot(df[df.income=='>50K']['hours.per.week'], color='r')
df['native.country'].value_counts()
df[df.select_dtypes("object") =="?"] = np.nan

nans = df.isnull().sum()

if len(nans[nans>0]):

    print("Missing values detected.\n")

    print(nans[nans>0])

else:

    print("No missing values. You are good to go.")
#majority of the values are "Private". Lets fill the missing values as "Private".

df.workclass.fillna("Private", inplace=True)



df.occupation.fillna(method='bfill', inplace=True)



#majority of the values are "United-States". Lets fill the missing values as "United-States".

df['native.country'].fillna("United-States", inplace=True)



print("Handled missing values successfully.")
from sklearn.preprocessing import LabelEncoder

from sklearn.utils import column_or_1d



class MyLabelEncoder(LabelEncoder):



    def fit(self, y, arr=[]):

        y = column_or_1d(y, warn=True)

        if arr == []:

            arr=y

        self.classes_ = pd.Series(arr).unique()

        return self



le = MyLabelEncoder()
# age_enc = pd.cut(df.age, bins=(0,25,45,65,100), labels=(0,1,2,3))

df['age_enc'] = df.age.apply(lambda x: 1 if x > 30 else 0)



def prep_workclass(x):

    if x == 'Never-worked' or x == 'Without-pay':

        return 0

    elif x == 'Private':

        return 1

    elif x == 'State-gov' or x == 'Local-gov' or x == 'Federal-gov':

        return 2

    elif x == 'Self-emp-not-inc':

        return 3

    else:

        return 4



df['workclass_enc'] = df.workclass.apply(prep_workclass)



df['fnlwgt_enc'] = df.fnlwgt.apply(lambda x: 0 if x>200000 else 1)



le.fit(df.education, arr=['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th','10th', '11th', '12th', 

                                             'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', 'Some-college', 'Bachelors', 'Masters', 'Doctorate'])

df['education_enc'] = le.transform(df.education)





df['education.num_enc'] = df['education.num'].apply(lambda x: 1 if x>=9 else 0)



df['marital.status_enc'] = df['marital.status'].apply(lambda x: 1 if x=='Married-civ-spouse' or x == 'Married-AF-spouse' else 0)



def prep_occupation(x):

    if x in ['Prof-specialty', 'Exec-managerial', 'Tech-support', 'Protective-serv']:

        return 2

    elif x in ['Sales', 'Craft-repair']:

        return 1

    else:

        return 0



df['occupation_enc'] = df.occupation.apply(prep_occupation)



df['relationship_enc'] = df.relationship.apply(lambda x: 1 if x in ['Husband', 'Wife'] else 0)



df['race_enc'] = df.race.apply(lambda x: 1 if x=='White' else 0)



df['sex_enc'] = df.sex.apply(lambda x: 1 if x=='Male' else 0)



df['capital.gain_enc'] = pd.cut(df["capital.gain"], 

                                bins=[-1,0,df[df["capital.gain"]>0]["capital.gain"].median(), df["capital.gain"].max()], labels=(0,1,2)).astype('int64')



df['capital.loss_enc'] = pd.cut(df["capital.loss"], 

                                bins=[-1,0,df[df["capital.loss"]>0]["capital.loss"].median(), df["capital.loss"].max()], labels=(0,1,2)).astype('int64')



# hpw_enc = pd.cut(df['hours.per.week'], bins= (0,30,40,53,168), labels=(0,1,2,3))

df['hours.per.week_enc'] = pd.qcut(df['hours.per.week'], q=5, labels=(0,1,2,3), duplicates='drop').astype('int64')



df['native.country_enc'] = df['native.country'].apply(lambda x: 1 if x=='United-States' else 0)



df['income_enc'] = df.income.apply(lambda x: 1 if x==">50K" else 0)



print("Encoding complete.")
df.select_dtypes("object").info()
#dropping encoded columns - education, sex, income

df.drop(['education', 'sex', 'income'], 1, inplace=True)
for feature in df.select_dtypes("object").columns:

    df[feature]=le.fit_transform(df[feature])
df.info()
#Visualizing the pearson correlation with the target class

pcorr = df.drop('income_enc',1).corrwith(df.income_enc)

plt.figure(figsize=(10,6))

plt.title("Pearson Correlation of Features with Income")

plt.xlabel("Features")

plt.ylabel("Correlation Coeff")

plt.xticks(rotation=90)

plt.bar(pcorr.index, list(map(abs,pcorr.values)))
df.drop(['workclass', 'fnlwgt','occupation', 'race', 'native.country', 'fnlwgt_enc', 'race_enc', 'native.country_enc'], 1, inplace=True)
sns.heatmap(df.corr().apply(abs))
df.drop(['age', 'education.num_enc', 'education_enc', 'marital.status_enc', 'capital.gain', 'capital.loss', 'hours.per.week'], 1, inplace = True)
df.info()
X = df.drop('income_enc', 1)

y = df.income_enc
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print("No. of rows in training data:",X_train.shape[0])

print("No. of rows in testing data:",X_test.shape[0])
oversample = RandomOverSampler(sampling_strategy='minority') #100% oversampling

X_over, y_over = oversample.fit_resample(X_train, y_train)
y_over.value_counts()
#Model Imports

from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.metrics import classification_report, confusion_matrix



from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, StackingClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.model_selection import GridSearchCV
seed= 42
models = {

    'LR':LogisticRegression(random_state=seed),

    'SVC':SVC(random_state=seed),

    'AB':AdaBoostClassifier(random_state=seed),

    'ET':ExtraTreesClassifier(random_state=seed),

    'GB':GradientBoostingClassifier(random_state=seed),

    'RF':RandomForestClassifier(random_state=seed),

    'XGB':XGBClassifier(random_state=seed),

    'LGBM':LGBMClassifier(random_state=seed)

    }
# evaluate a give model using cross-validation

def evaluate_model(model, xtrain, ytrain):

    cv = StratifiedKFold(shuffle=True, random_state=seed)

    scores = cross_val_score(model, xtrain, ytrain, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

    return scores
# evaluate the models and store results for 100% oversampled minority class

results, names = list(), list()

for name, model in models.items():

    scores = evaluate_model(model, X_train, y_train) 

    results.append(scores) 

    names.append(name) 

    print('*%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
plt.boxplot(results, labels=names, showmeans=True)

plt.show() 