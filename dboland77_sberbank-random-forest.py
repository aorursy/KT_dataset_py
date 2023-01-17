import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestRegressor # import the random forest model

from sklearn import  preprocessing # used for label encoding and imputing NaNs
train_df = pd.read_csv('../input/train.csv',)

test_df = pd.read_csv('../input/test.csv')

macro_df = pd.read_csv('../input/macro.csv')
id_test = test_df.id

y_train = train_df["price_doc"]

x_train = train_df.drop(["id", "timestamp", "price_doc"], axis=1)

x_test = test_df.drop(["id", "timestamp"], axis=1)
#from sklearn.cross_validation import cross_val_score # We also need the cross validation functionality

#scores = list()

#scores_std = list()



#print('Start learning...')

#n_trees = [10, 50, 75]

#for n_tree in n_trees:

#        print(n_tree)

#        recognizer = RandomForestRegressor(n_tree)

#        score = cross_val_score(recognizer, x_train, y_train)

#        scores.append(np.mean(score))

#        scores_std.append(np.std(score))



#sc_array = np.array(scores)

#std_array = np.array(scores_std)

#print('Score: ', sc_array)

#print('Std  : ', std_array)





#plt.plot(n_trees, scores)

#plt.plot(n_trees, sc_array + std_array, 'b--')

#plt.plot(n_trees, sc_array - std_array, 'b--')

#plt.ylabel('CV score')

#plt.xlabel('# of trees')

#plt.savefig('cv_trees.png')
for c in x_train.columns:

    if x_train[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(x_train[c].values)) 

        x_train[c] = lbl.transform(list(x_train[c].values))

        

for c in x_test.columns:

    if x_test[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(x_test[c].values)) 

        x_test[c] = lbl.transform(list(x_test[c].values))  
imputer = preprocessing.Imputer(missing_values='NaN', strategy = 'mean', axis = 0)

x_train = imputer.fit_transform(x_train)

x_test = imputer.fit_transform(x_test)
Model = RandomForestRegressor(3)

Model.fit(x_train, y_train)

y_predict = Model.predict(x_test)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})



output.to_csv('RandomForest.csv', index=False)