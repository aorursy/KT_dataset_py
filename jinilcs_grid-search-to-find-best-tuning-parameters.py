#import required libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, GridSearchCV



from sklearn.preprocessing import MinMaxScaler



import seaborn as sns
#load the input data using pandas

df = pd.read_csv('../input/data.csv')

df.head(3)
#Split the data into training and test set

train, test = train_test_split(df, random_state=42)

X_train = train[train.columns[2:-1]]

y_train = train['diagnosis']

X_test = test[test.columns[2:-1]] 

y_test = test['diagnosis']
#Lets build a simple Linear SVC model and test it

#SVC Models are good when the data is scaled. Lets scale the data and build the model

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)



svc_model = SVC(random_state=0).fit(X_train_scaled,y_train)

print("train score - " + str(svc_model.score(X_train_scaled, y_train)))

print("test score - " + str(svc_model.score(X_test_scaled, y_test)))
#Lets see the default parameters used

print(svc_model)
#We can use a grid search to find the best parameters for this model. Lets try



#Define a list of parameters for the models

params = {'C': [0.001, 0.01, 0.1, 1, 10, 100],

               'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}



#We can build Grid Search model using the above parameters. 

#cv=5 means cross validation with 5 folds

grid_search = GridSearchCV(SVC(random_state=0), params, cv=5, n_jobs=-1)

grid_search.fit(X_train_scaled, y_train)



print("train score - " + str(grid_search.score(X_train_scaled, y_train)))

print("test score - " + str(grid_search.score(X_test_scaled, y_test)))
#We got a better score now. Lets check the best parameters the model used.

print(grid_search.best_params_)
#We can visualize the parameter dependency with the models

results_df = pd.DataFrame(grid_search.cv_results_)

scores = np.array(results_df.mean_test_score).reshape(6, 6)
sns.heatmap(scores, annot=True, 

            xticklabels=params['gamma'], yticklabels=params['C'])