# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  #plotting! 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import cross_val_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')
train.head()
#Pclass is an integer that needs to be categorized
#Sex is a category
#Age could be bucketed
#SibSp Number of siblims or spouses can be bucketed or normalized
#Parch  Can be bucketed or combined with the previous feature
test.describe()
#Age is not allways populated
test["Fare"].mean()
sex_pivot = train.pivot_table(index = "Sex", values = "Survived")
sex_pivot.plot.bar()
plt.show()
#the pivot table aggregates groups and applies a function to those
#The proportion of women that survived is 70% compared to aprox 20% for males
#TO DO: Include in the bar the number o people in each group
class_pivot = train.pivot_table(index="Pclass",values="Survived")
class_pivot.plot.bar()
plt.show()
#the proportion of people in first class that survived is much higher
#TO DO Include the number of people on each group that survived
siblins_plot = train.pivot_table(index="SibSp",values="Survived")
siblins_plot.plot.bar()
plt.show()
#people with 1 sibling or spouse had a higher survival rate than the rest.
survived = train[train["Survived"]==1]
died = train[train["Survived"]==0]
survived["Age"].plot.hist(alpha = 0.5,color = 'green',bins = 50)
died["Age"].plot.hist(alpha = 0.5,color = 'black',bins = 50)
plt.legend(['Survived','Died'])
plt.show()
#Shows the distributions of of population by the age and if they survived and Died
#Bucketing the population into segments
#This allows to make a story 
def process_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

cut_points = [-1,0,5,12,18,35,60,100]
label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

train = process_age(train,cut_points,label_names)
test = process_age(test,cut_points,label_names)
train.head()
age_category_plot = train.pivot_table(index="Age_categories",values='Survived')
age_category_plot.plot.bar()
plt.show()
#Infants had the highest survival rate
train["Pclass"].value_counts()
#Interesting that there were more people in 1st class than in second
#Converting this integer into a categorical value
#Converts categorical values into new colums with boolean values 0 or 1 for each category
def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis = 1)
    return df

for column in ["Pclass","Sex","Age_categories"]:
    train = create_dummies(train,column)
    test = create_dummies(test,column)    
    
train.head()
#Filling NANs and creating dummie columns
train['Embarked'] = train['Embarked'].fillna('S')
test['Embarked'] = test['Embarked'].fillna('S')
train = create_dummies(train,'Embarked')
test = create_dummies(test,'Embarked')

train.head()
#Filling the non populated Fare in the test set with the mean
test["Fare"] = test["Fare"].fillna(test["Fare"].mean())
test["Fare"].mean()
#train['Embarked'] = train['Embarked'].fillna('S')
#Scales Numeric Columns
#TOTO: Create a function that gets columns to scale and returns new df
columns_to_scale = ["Fare","SibSp","Parch"]
scaled_columns = ["Fare_scaled","SibSp_scaled","Parch_scaled"]
scaled_df_train = pd.DataFrame(minmax_scale(train[columns_to_scale]), columns = scaled_columns)
train  =pd.concat([train,scaled_df_train], axis = 1)
scaled_df_test = pd.DataFrame(minmax_scale(test[columns_to_scale]), columns = scaled_columns)
test  =pd.concat([test,scaled_df_test], axis = 1)
test.head()
#Preparing the dataframe to train it
list(train)
#Getting only engineered columns
model_columns = [ 'Pclass_1',
 'Pclass_2',
 'Pclass_3',
 'Sex_female',
 'Sex_male',
 'Age_categories_Missing',
 'Age_categories_Infant',
 'Age_categories_Child',
 'Age_categories_Teenager',
 'Age_categories_Young Adult',
 'Age_categories_Adult',
 'Age_categories_Senior',
 'Embarked_C',
 'Embarked_Q',
 'Embarked_S',
 'Fare_scaled',
 'SibSp_scaled',
 'Parch_scaled']
#Dividing now the train datafrain between train and test set (Since the given test df doesn't have the target column)
all_X = train[model_columns]
all_y = train['Survived']
train_X, test_X, train_y, test_y = train_test_split(all_X,all_y, test_size = 0.20, random_state = 0)
test_X.describe()
#training the data in the training set and predicting in the test set
lr = LogisticRegression()
lr.fit(train_X,train_y)
#predictions = lr.predict(test_X)
coefficients = lr.coef_
print(coefficients)
coefficients[0]
#Graph sowing the feature importances
feature_importance = pd.Series(coefficients[0], index = train_X.columns)
feature_importance.plot.barh()
plt.show()
#The most important features tohave survived the titanic
feature_importance_ordered = feature_importance.sort_values()
feature_importance_ordered.plot.barh()
plt.show()
#Now the most important features for determining if someone survived or died
abs_feature_importance_ordered = feature_importance.abs().sort_values()
abs_feature_importance_ordered.plot.barh()
plt.show()
abs_feature_importance_ordered = abs_feature_importance_ordered.sort_values(ascending=False)
features_to_model = []
for index,value in abs_feature_importance_ordered.items():
       if  len(features_to_model) < 8:
            features_to_model.append(index)
            print(index)
print(features_to_model)
#Performs Cross Validation on different training and test sets
lr = LogisticRegression()
scores = cross_val_score(lr,all_X[features_to_model],all_y,cv = 10)
scores.sort()
accuracy = scores.mean()
print(scores)
print(accuracy)

all_X[features_to_model].head()
lr.fit(all_X[features_to_model],all_y)
holdout_predictions = lr.predict(test[features_to_model])
print(holdout_predictions)
submission_df = {"PassengerId": test["PassengerId"],
                              "Survived":holdout_predictions}
submission = pd.DataFrame(submission_df)
submission.head()
submission.to_csv("submission_2.csv",index=False)
