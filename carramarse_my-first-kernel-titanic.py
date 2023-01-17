# Import basic libraries
import pandas as pd
import numpy as np

#load data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#check variables
print(train.columns)
print(test.columns)

    #1st: set PassengerId as index
train.set_index(['PassengerId'], inplace = True)
test.set_index(['PassengerId'], inplace = True)
idx_train = train.index.values              # This will be used at the end to separate the data
idx_test = test.index.values                # This will be used at the end to separate the data

    #2nd: concat datasets
data = pd.concat([train.drop(['Survived'], axis = 1),test])
for x in data:
    print('{} : {}'.format(x,data[x].count()/data[x].size) )
Age_availability_class = data.Age.groupby(data.Pclass).count()/data.Age.groupby(data.Pclass).size()
print(Age_availability_class)
Cabin_availability_class = data.Cabin.groupby(data.Pclass).count()/data.Cabin.groupby(data.Pclass).size()
print(Cabin_availability_class)
#create a function to separate titles, names and surnames:
def separate_title(data):
    Title = []
    Surname = []
    FirstName = []
    
    for names in data['Name']:
        name_split = names.split(',')
        Surname.append(name_split[0])
        title_and_name = name_split[1]
        Title.append(title_and_name.split('.')[0].strip())
        FirstName.append(title_and_name.split('.')[-1].strip())
    
    return Title, FirstName, Surname

#apply this function
Title, FirstName, Surname = separate_title(data)

#analyze Titles
title_count = pd.Series()
for titles in set(Title):
    title_count[titles] = sum([x == titles for x in Title])
print(title_count)
#Standarize titles:
    
titles_dic = {'Miss':'Miss', 'Mr':'Mr','Mme':'Mrs','Mrs':'Mrs','Ms':'Mrs',
              'Don':'Mr','Mlle':'Miss', 'Master':'Master'}    

def norm_title(Title, titles_dic, fill = 'Other'):
    Title_norm = [titles_dic[x] if x in titles_dic else fill for x in Title ]
    Title_dummies = pd.get_dummies(pd.DataFrame(Title_norm), prefix = 'Title')
    return Title_dummies

Title_dummies = norm_title(Title, titles_dic)
Cabin_letter = data.apply(lambda x: x.Cabin[0] if not pd.isnull(x.Cabin) 
else 'F' if x.Pclass == 2 else 'G' if x.Pclass == 3 else np.nan, axis = 1)
Cabin_dummies = pd.get_dummies(Cabin_letter, prefix = 'Cabin')
Cabin_dummies.drop(['Cabin_T'], inplace = True, axis = 1)  # we drop Cabin T since it only has 1 element
Sex_dummies = pd.get_dummies(data.Sex)['female']  # We only need 1 column
Embarked_dummies = pd.get_dummies(data.Embarked, prefix = 'Embarked')

new_data = data.drop(['Name','Sex','Ticket','Cabin','Embarked'], axis = 1)
new_data = (new_data.join(Title_dummies).join(Cabin_dummies).
            join(Sex_dummies).join(Embarked_dummies))
# Create the train datasets from those rows with notnull Age
y_train = new_data.Age[new_data.Age.notnull()]
X_train = (new_data.drop(['Age'], axis = 1)[new_data.Age.notnull()])

# Import the needed libraries:
from xgboost import XGBRegressor
from sklearn.preprocessing import Imputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

# And model the Age:
model_age = XGBRegressor(n_estimators=1000, learning_rate=0.05,
                         early_stopping_rounds=5) # Parameters copied from the tutorial, a deeper analysis shoul be made

pipe_age = make_pipeline(Imputer(), model_age)
pipe_age.fit(X_train,y_train)

# evaluate
scores_age = -cross_val_score(pipe_age, X_train, y_train, scoring='neg_mean_absolute_error')
print('Expected MAE of imputed age between {0:.2f} and {1:.2f}'
      .format(np.min(scores_age),np.max(scores_age)))

# Impute values    
X = (new_data.drop(['Age'], axis = 1)[new_data.Age.isnull()])
X_imputed = Imputer().fit_transform(X)
y = pipe_age.predict(X_imputed)

# We create a variable to indicate the cases with imputed ages
Age_imputed = pd.Series(np.zeros(data.shape[0]))
Age_imputed.name = 'AgeImp'
Age_imputed.index = data.index
Age_imputed[new_data.Age.isnull()] = 1

Age_filled = new_data.Age.copy()
Age_filled.loc[new_data.Age.isnull()] = y.copy()

# update data
new_data['Age'] = Age_filled
new_data['AgeImp'] = Age_imputed
from xgboost import XGBClassifier

X_train = new_data.loc[idx_train]
y_train = train.Survived
X_test = new_data.loc[idx_test]

model = XGBClassifier(n_estimators=1000, learning_rate=0.05,
                         early_stopping_rounds=5) # Parameters copied from the tutorial, a deeper analysis shoul be made
pipe = make_pipeline(Imputer(), model)
pipe.fit(X_train,y_train)
survived = pipe.predict(X_test)

scores = cross_val_score(pipe, X_train, y_train, scoring='accuracy')
print('Expected accuracy between {0:.2f} and {1:.2f}'.format(np.min(scores),np.max(scores)))

# prepare data for submission
submission = pd.DataFrame(survived, columns = ['Survived'], index = X_test.index)