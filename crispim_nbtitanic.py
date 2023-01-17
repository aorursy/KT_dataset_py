import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot

from sklearn.model_selection import train_test_split



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Step 1 - Load data files

df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")
# Step 2 - Understanding the data



#df_train.head()

#df_test.head()

#df_train.describe()



women = df_train.loc[df_train.Sex=='female']["Survived"]

rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

#women.describe()



men = df_train.loc[df_train.Sex=='male']["Survived"]

rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)

#men.describe()





#y = df_train.loc[np.isnan(df_train.Age)==1]

#y.head(20)
# Step 2 - ...continue to understand, visually



#serie_survived = df_train["Survived"]

#serie_pclass = df_train["Pclass"]

#serie_age = df_train["Age"]

#serie_fare = df_train["Fare"] # morreram igual

#serie_sibsp = df_train["SibSp"]

#serie_parch = df_train["Parch"]



##serie_pclass.head()

#pyplot.scatter(serie_parch, serie_survived)

#pyplot.show()
#Step 3 - Cleaning the data (putting bias)



#df_train = df_train.dropna(axis=0, subset=['Age'])

#df_train = df_train.fillna(df_train.mean().to_dict())

df_train[['Age']] = df_train[['Age']].fillna(df_train.mean().to_dict())

df_train.describe()



# É correto alterar o dataframe de teste? #

# df_test = df_test.dropna(axis=0, subset=['Age'])

#df_test = df_test.fillna(df_test.mean().to_dict())

df_test[['Age']] = df_test[['Age']].fillna(df_test.mean().to_dict())

#df_test.describe()
# Step 4 - Selecting features, and target

features = ["Pclass", "Age", "Sex", "SibSp", "Parch"]

y = df_train["Survived"]

X = pd.get_dummies(df_train[features])

X_test = pd.get_dummies(df_test[features])



# Step 4 - Spliting trainning and test

df_train_X, df_val_X, df_train_y, df_val_y = train_test_split(X, y, random_state = 0)
# Step 5 - Testing models and defining

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeRegressor



def get_dtr_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)



def get_rfc_mae(max_depth, max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = RandomForestClassifier(n_estimators=100, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)



# max_depth, max_leaf_nodes

candidates_matrix = [

    [5,10],  [5,20],  [5,30],  [5,15],  [5,25], 

    [6,10],  [6,20],  [6,30],  [6,15],  [6,25], 

    [7,10],  [7,20],  [7,30],  [7,15],  [7,25], 

    [8,10],  [8,20],  [8,30],  [8,15],  [8,25], 

    [9,10],  [9,20],  [9,30],  [9,15],  [9,25]

]



best_depth_value = 0

best_leaf_value = 0

best_mae_value = 9999999

for couple in candidates_matrix:

    current_mae = get_rfc_mae(couple[0], couple[1], df_train_X, df_val_X, df_train_y, df_val_y)

    print("For max_depth of %s, max_leaf_nodes of %s the mean returned is %s" % (couple[0], couple[1], current_mae))

    if(current_mae < best_mae_value):

        best_mae_value = current_mae

        best_depth_value = couple[0]

        best_leaf_value = couple[1]



print("WINNERs - max_depth: %s, max_leaf_nodes: %s, mean returned: %s" % (best_depth_value, best_leaf_value, best_mae_value))





## RandomForestClassifier

## Define the model

#model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

## Fit

#model.fit(df_train_X, df_train_y)

## Predict

#predictions = model.predict(df_val_X)

## Evaluate

#print(mean_absolute_error(df_val_y, predictions))

##output = pd.DataFrame({ 'PassengerId': df_test.PassengerId,  'Survived': predictions})





# DecisionTreeRegressor #

#model = DecisionTreeRegressor(random_state=7)

#model.fit(X,y)

#predictions = model.predict(X_test)

#output = pd.DataFrame({ 'PassengerId': df_test.PassengerId,  'Survived': predictions.astype(int) })

# Code to sumbit to competition

model = RandomForestClassifier(n_estimators=100, max_depth=9, max_leaf_nodes=25, random_state=0)

model.fit(X, y)

predictions = model.predict(X_test)

output = pd.DataFrame({ 'PassengerId': df_test.PassengerId,  'Survived': predictions})
# Code to sumbit to competition

model.fit(X, y)

predictions = model.predict(X_test)

output = pd.DataFrame({ 'PassengerId': df_test.PassengerId,  'Survived': predictions})



output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
