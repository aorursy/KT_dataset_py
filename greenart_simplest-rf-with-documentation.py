# imporing libraries

import pandas as pd # data processing

from sklearn.ensemble import RandomForestClassifier #modelling
# reading training file

train = pd.read_csv('../input/learn-together/train.csv', index_col="Id")



# defining a target 

y_train = train.Cover_Type



# defining a training set

X_train = train.drop(['Cover_Type'], axis=1)



# defining the model

model = RandomForestClassifier(n_estimators=3000)



# fitting the model

model.fit(X_train, y_train)



print('model fitted on test set')
# reading test file data to variable 'test' (excluding 'Id' column)

test = pd.read_csv('../input/learn-together/test.csv')



# applying model to the test data, getting predictions

test_pred = model.predict(test.drop('Id', axis=1))



# making a dataframe with a result set

output = pd.DataFrame({'ID': test.Id,

                       'Cover_Type': test_pred})



# exporting result dataframe to csv

output.to_csv('submission.csv', index=False)



print('Done')