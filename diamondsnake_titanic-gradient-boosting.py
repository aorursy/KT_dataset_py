# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_raw = pd.read_csv('../input/titanic/train.csv')

test_raw = pd.read_csv('../input/titanic/test.csv')
#train.isna().sum()

#test.isna().sum()
# [Pclass, Title, Sex, Age, SibSp, Parch, Fare, Cabin_letter, Cabin_number, Embarked] [survived]

# +[title, cabin_letter, cabin_number, no_cabins]

# ~[Age, Cabin_letter, Cabin_number, Embarked, Fare]
def find_title(name):

    finish = name.find(".")

    start = name.rfind(" ", 0, finish)

    title = name[start+1:finish+1]

    if title in ['Master.', 'Miss.', 'Mrs.', 'Mr.']:

        return title

    else:

        return 'Other'

    

def count_cabins(cabin):

    if pd.isnull(cabin):

        cabins = 0 

    else:

        cabins = cabin.count(" ")+1

    return cabins
def data_prep(dataframe):

    dataframe['title'] = dataframe['Name'].apply(find_title)

    dataframe['no_cabins'] = dataframe['Cabin'].apply(count_cabins)



    med_ages = dataframe.groupby('Pclass').agg(Age_ave=pd.NamedAgg(column='Age', aggfunc=np.median))

    med_fares = dataframe.groupby('Pclass').agg(Fare_ave=pd.NamedAgg(column='Fare', aggfunc=np.median))



    dataframe = dataframe.merge(med_ages, how='left', on='Pclass')

    dataframe = dataframe.merge(med_fares, how='left', on='Pclass')



    dataframe['age_filled'] = dataframe.apply(lambda row: row['Age_ave'] if np.isnan(row['Age']) else row['Age'], axis=1)

    dataframe['fare_filled'] = dataframe.apply(lambda row: row['Fare_ave'] if np.isnan(row['Fare']) else row['Fare'], axis=1)

    dataframe['embarked_filled'] = dataframe['Embarked'].fillna(dataframe['Embarked'].mode()[0])

    dataframe['FamilySize'] = dataframe['SibSp'] + dataframe['Parch'] + 1

    dataframe['IsAlone'] = 1 #initialize to yes/1 is alone

    dataframe['IsAlone'].loc[dataframe['FamilySize'] > 1] = 0

    dataframe['FareBin'] = pd.qcut(dataframe['fare_filled'], 4)

    dataframe['AgeBin'] = pd.cut(dataframe['age_filled'].astype(int), 5)



    dataframe = dataframe.drop(["Name", "Age", "Ticket", "Fare", "Cabin", "Embarked", "Age_ave", "Fare_ave", "fare_filled", "age_filled"], axis=1)

    

    return dataframe

    
train = data_prep(train_raw)

test = data_prep(test_raw)
import matplotlib.pyplot as plt

for field in train.columns.drop('PassengerId'):

    agg = train.groupby(field).Survived.agg({'survival': lambda x: x.sum()/x.count(), "count": lambda x: x.count()})

    agg.reset_index(level=0, inplace=True)

    

    fig, ax1 = plt.subplots()

    ax1.set_xlabel(field)

    ax1.set_ylabel('survival rate')

    ax1.bar(agg[field].astype(str), agg['survival'], color='b')

    ax2 = ax1.twinx()

    ax2.set_ylabel('count')

    ax2.plot(agg[field].astype(str), agg['count'], color='r')

    plt.show()
import seaborn as sns

def correlation_heatmap(df):

    _ , ax = plt.subplots(figsize =(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        df.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)



#correlation_heatmap(train.drop(['PassengerId'], axis=1))
train = pd.get_dummies(train)

test = pd.get_dummies(test)
# Create training and validation sets

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import accuracy_score

y = train.Survived.values

train_features = train.drop(['PassengerId', 'Survived'], axis=1)

x = train_features.values

x, x_test, y, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(x, y)

print(model)



# make predictions for test data

y_pred = model.predict(x_test)

predictions = [round(value) for value in y_pred]

# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))



# Create a submission

ids = test['PassengerId'].values

test.drop('PassengerId', inplace=True, axis=1)

x_out = test.values

y_out = model.predict(x_out).round().astype(int)

output = pd.DataFrame({'PassengerId': ids, 'Survived': y_out})

output.to_csv("submission.csv", index=False)
import lightgbm



# Create the LightGBM data containers

train_data = lightgbm.Dataset(x, label=y)

test_data = lightgbm.Dataset(x_test, label=y_test)



# Train the model



parameters = {

    'application': 'binary',

    'objective': 'binary',

    'metric': 'auc',

    'is_unbalance': 'true',

    'boosting': 'gbdt',

    'num_leaves': 32,

    'feature_fraction': 0.5,

    'bagging_fraction': 0.5,

    'bagging_freq': 20,

    'learning_rate': 0.05,

    'verbose': 0

}



model = lightgbm.train(parameters,

                       train_data,

                       valid_sets=test_data,

                       num_boost_round=5000,

                       early_stopping_rounds=100)



# Create a submission



ids = test['PassengerId'].values

test.drop('PassengerId', inplace=True, axis=1)



x_out = test.values

y_out = model.predict(x_out).round().astype(int)



output = pd.DataFrame({'PassengerId': ids, 'Survived': y_out})

#output.to_csv("submission.csv", index=False)