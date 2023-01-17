# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



train_data = pd.read_csv("../input/titanic/train.csv")

test_data = pd.read_csv("../input/titanic/test.csv")

genderSub = pd.read_csv("../input/titanic/gender_submission.csv")

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
passengerid = test_data.PassengerId





train_data.info()



test_data.info()
def missing_percentage(df):

    

    total = df.isnull().sum().sort_values(ascending=False)

    percent = round(df.isnull().sum().sort_values(ascending=False) / len(df) * 100)

    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    

missing_percentage(train_data)

missing_percentage(test_data)
train_data[train_data.Embarked.isnull()]



# Fares = 80,  Pclass = 1,    Try to guess Embarked
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

fig, ax = plt.subplots(figsize=(16,12),ncols=2)

ax1 = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=train_data, ax = ax[0]);

ax2 = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=test_data, ax = ax[1]);

ax1.set_title("Training Set", fontsize = 18)

ax2.set_title('Test Set',  fontsize = 18)



fig.show()
train_data.Embarked.fillna("C", inplace=True)
# CABIN HAS 77-778 % MISSING VALUE



survivers = train_data.Survived



train_data.drop(["Survived"], axis=1, inplace=True)



all_data = pd.concat([train_data, test_data], ignore_index=False)



#Assigning N to null(NaN) values



all_data.Cabin.fillna("N", inplace=True)
def percent_value_counts(df, feature):

    

    percent = pd.DataFrame(round(df.loc[:,feature].value_counts(dropna=False, normalize=True)*100,2))

    ## creating a df with th

    total = pd.DataFrame(df.loc[:,feature].value_counts(dropna=False))

    ## concating percent and total dataframe



    total.columns = ["Total"]

    percent.columns = ['Percent']

    return pd.concat([total, percent], axis = 1)
all_data.Cabin = [i[0] for i in all_data.Cabin]



percent_value_counts(all_data, "Cabin")

all_data.groupby("Cabin")['Fare'].mean().sort_values()
def cabin_estimator(i):

    

    a = 0

    if i<16:

        a = "G"

    elif i>=16 and i<27:

        a = "F"

    elif i>=27 and i<38:

        a = "T"

    elif i>=38 and i<47:

        a = "A"

    elif i>= 47 and i<53:

        a = "E"

    elif i>= 53 and i<54:

        a = "D"

    elif i>=54 and i<116:

        a = 'C'

    else:

        a = "B"

    return a
with_N = all_data[all_data.Cabin == "N"]

without_N = all_data[all_data.Cabin != "N"]
# Estimating cabins based on the Passenger Fares(Ticket Prices)

with_N["Cabin"] = with_N.Fare.apply(lambda x: cabin_estimator(x))



all_data = pd.concat([with_N, without_N], axis=0)



all_data.sort_values(by= 'PassengerId', inplace=True)



#We were taken train and test into all_data and we are seperating them now

train_data = all_data[:891]



test_data = all_data[891:]
train_data['Survived'] = survivers
test_data[test_data.Fare.isnull()]
missing_value_fare = test_data[(test_data.Pclass == 3) & 

                               (test_data.Embarked == "S") &

                               (test_data.Sex == "male")].Fare.mean()



test_data.Fare.fillna(missing_value_fare, inplace=True)
# 20 % of Age is missing value
train_data['Sex'] = train_data.Sex.apply(lambda x: 0 if x == "female" else 1)

test_data['Sex'] = test_data.Sex.apply(lambda x: 0 if x == "female" else 1)

train_data.head()
train_data["title"] = [i.split('.')[0] for i in train_data.Name]

train_data["title"] = [i.split(',')[1] for i in train_data.title]

#Getting the spaces

train_data.title = train_data.title.apply(lambda x: x.strip())

test_data['title'] = [i.split('.')[0].split(',')[1].strip() for i in test_data.Name]

def fuse_title(feature):

    

    result = ''

    if feature in ['the Countess','Capt','Lady','Sir','Jonkheer','Don','Major','Col', 'Rev', 'Dona', 'Dr']:

        result = 'rare'

    elif feature in ['Ms', 'Mlle']:

        result = 'Miss'

    elif feature == 'Mme':

        result = 'Mrs'

    else:

        result = feature

    return result



test_data.title = test_data.title.map(fuse_title)

train_data.title = train_data.title.map(fuse_title)
train_data['family_size'] = train_data.SibSp + train_data.Parch + 1

test_data['family_size'] = test_data.SibSp + test_data.Parch + 1
train_data['is_alone'] = [1 if i<2 else 0 for i in train_data.family_size]

test_data['is_alone'] = [1 if i<2 else 0 for i in test_data.family_size]
train_data.drop(['Ticket'], axis=1, inplace=True)

test_data.drop(['Ticket'], axis=1, inplace=True)
train_data['calculated_fare'] = train_data.Fare / train_data.family_size

test_data['calculated_fare'] = test_data.Fare / test_data.family_size
def fare_group(fare):

    

    a= ''

    if fare <= 4:

        a = 'Very_low'

    elif fare <= 10:

        a = 'low'

    elif fare <= 20:

        a = 'mid'

    elif fare <= 45:

        a = 'high'

    else:

        a = "very_high"

    return a



train_data['fare_group'] = train_data['calculated_fare'].map(fare_group)

test_data['fare_group'] = test_data['calculated_fare'].map(fare_group)

passengerids = test_data['PassengerId']

train_data.drop(['PassengerId'], axis=1, inplace=True)

test_data.drop(['PassengerId'], axis=1, inplace=True)



train_data.drop(['Embarked'], axis=1, inplace=True)

test_data.drop(['Embarked'], axis=1, inplace=True)
women = train_data.loc[train_data.Sex == 0]["Survived"]



rate_women = sum(women) / len(women)

print("% of women who survived: ", rate_women)





men = train_data.loc[train_data.Sex == 1]["Survived"]

rate_men = sum(men) / len(men)

print("% of men who survived: ", rate_men)
train_data

train_data = pd.get_dummies(train_data, columns=['title', "Pclass",'Cabin', 'fare_group'], drop_first=False)

test_data = pd.get_dummies(test_data, columns=['title', "Pclass", 'Cabin', 'fare_group'], drop_first=False)





train_data.drop(['Name', 'Fare'], axis=1, inplace=True)

test_data.drop(['Name', 'Fare'], axis=1, inplace=True)

train_data.columns
train_data = pd.concat([train_data[["Survived", "Age", "Sex", "SibSp", "Parch"]], train_data.loc[:, "family_size":]], axis=1)

test_data = pd.concat([test_data[["Age", "Sex"]], test_data.loc[: , "SibSp":]], axis=1)

train_data
from sklearn.ensemble import RandomForestRegressor



## writing a function that takes a dataframe with missing values and outputs it by filling the missing values. 

def completing_age(df):

    ## gettting all the features except survived

    age_df = df.loc[:,"Age":] 



    temp_train = age_df.loc[age_df.Age.notnull()] ## df with age values

    temp_test = age_df.loc[age_df.Age.isnull()] ## df without age values

    

    y = temp_train.Age.values ## setting target variables(age) in y 

    x = temp_train.loc[:, "Sex":].values

    

    rfr = RandomForestRegressor(n_estimators=1500, n_jobs=-1)

    rfr.fit(x, y)

    

    

    predicted_age = rfr.predict(temp_test.loc[:, "Sex":])

    

    df.loc[df.Age.isnull(), "Age"] = predicted_age

    



    return df



## Implementing the completing_age function in both train and test dataset. 

completing_age(train_data)

completing_age(test_data)

## Let's look at the his

plt.subplots(figsize = (22,10),)

sns.distplot(train_data.Age, bins = 100, kde = True, rug = False, norm_hist=False);


## create bins for age

def age_group_fun(age):



    a = ''

    if age <= 1:

        a = 'infant'

    elif age <= 4: 

        a = 'toddler'

    elif age <= 13:

        a = 'child'

    elif age <= 18:

        a = 'teenager'

    elif age <= 35:

        a = 'Young_Adult'

    elif age <= 45:

        a = 'adult'

    elif age <= 55:

        a = 'middle_aged'

    elif age <= 65:

        a = 'senior_citizen'

    else:

        a = 'old'

    return a

        

## Applying "age_group_fun" function to the "Age" column.

train_data['age_group'] = train_data['Age'].map(age_group_fun)

test_data['age_group'] = test_data['Age'].map(age_group_fun)



## Creating dummies for "age_group" feature. 

train_data = pd.get_dummies(train_data,columns=['age_group'], drop_first=True)

test_data = pd.get_dummies(test_data,columns=['age_group'], drop_first=True);

# separating our independent and dependent variable

X = train_data.drop(['Survived'], axis = 1)

y = train_data["Survived"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.33, random_state=42)



headers = X_train.columns
from sklearn.preprocessing import StandardScaler

std_scale = StandardScaler()



## transforming "train_x"

X_train_scaled = std_scale.fit_transform(X_train)

## transforming "test_x"

X_test_scaled = std_scale.transform(X_test)
from sklearn.preprocessing import StandardScaler

std_scale = StandardScaler()



## transforming "train_x"

X_scaled = std_scale.fit_transform(X)

## transforming "test_x"

test_data_scaled = std_scale.transform(test_data)



"""

from sklearn.ensemble import RandomForestClassifier



model =RandomForestClassifier(criterion='gini',

                              n_estimators=1100,

                              max_depth=5,

                              min_samples_split=4,

                              min_samples_leaf=5,

                              max_features='auto',

                              oob_score=True,

                              random_state=42,

                              n_jobs=-1,

                              verbose=1)



model.fit(X_scaled, y)



predictions = model.predict(test_data_scaled)







output = pd.DataFrame({'PassengerId': passengerid, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")

"""
from sklearn.model_selection import cross_val_score,GridSearchCV,RepeatedStratifiedKFold

from sklearn.neighbors import KNeighborsClassifier



random_state = 7



## Define grid params

param_grid = {"n_neighbors":[3,4,5,6,7],\

             "weights":["uniform","distance"],\

             "p":[1,2]}



## Define Kfold

kfold = RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=random_state)



## Define model

model = KNeighborsClassifier()



## Define and execute Grid Search

grid_search = GridSearchCV(model, param_grid=param_grid,\

                                       scoring="accuracy",cv=kfold)

grid_result = grid_search.fit(X_scaled, y)



## Get and print results

best_acc_score=grid_result.best_score_

best_params=grid_result.best_params_

best_model=grid_result.best_estimator_



print("Accuracy: {:6f}".format(best_acc_score))

print("Params: {}".format(best_params))
## Fit best model on all training data set

best_model.fit(X_scaled, y)



## Predict using the best model from training

predictions = best_model.predict(test_data_scaled)



## Save file to submit

output = pd.DataFrame({'PassengerId': passengerid, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")