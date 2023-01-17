import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



#loading train data 

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
#Creating X and y for train data

X_train = train.drop(['id','species'], axis = 1)

y_train = train['species']



#Creating X for test data

X_test = test.drop(['id'], axis = 1)



#scaling X for train data---> all data in unit variance

scaler_train = StandardScaler().fit(X_train)

scaled_train = scaler_train.transform(X_train)

X_train = pd.DataFrame(scaled_train,columns = X_train.columns)



#scaling X for test data---> all data in unit variance

scaler_test = StandardScaler().fit(X_test)

scaled_test = scaler_test.transform(X_test)

X_test = pd.DataFrame(scaled_test,columns = X_test.columns)
# initializing Random Forest 

RFmodel =  RandomForestClassifier(n_estimators = 500, criterion = 'gini',\

                                  max_features = 'sqrt' , max_depth = 50, \

                                  n_jobs= -1, random_state=3)



# initializing Stratified K-Fold

SKF= StratifiedKFold(n_splits=5, shuffle=False, random_state=3)
# SKF.split() splits data into train/test sets and returns train/test indices of each set 

# We're going to loop through all the train/test sets and run random forest

# We'll print out the accuracy of each train/test set





for train_skf_index,test_skf_index in SKF.split(X_train.values, y_train.values):

    

    #data which we will use to fit the model

    X_cv_fit = X_train.values[train_skf_index]

    y_cv_fit = y_train.values[train_skf_index]

    

    #data which we will use to score the model

    X_cv_score = X_train.values[test_skf_index]

    y_cv_score = y_train.values[test_skf_index]

    

    #printing score

    print (RFmodel.fit(X_cv_fit,y_cv_fit).score(X_cv_score,y_cv_score))
# Finding hyperparameters for Random Forest using GridSearchCV



#setting the parameters we want to test

params = { 'n_estimators' : [50,100,200,500,800],'max_depth' : [20,50,100,200] }



#finding best hyperparameters with cross-validation

gsearch = GridSearchCV(estimator =  RFmodel, param_grid = params, scoring='neg_log_loss', n_jobs=1, refit=True, cv=5)

gsearch.fit(X_train,y_train)

gsearch.best_score_,gsearch.best_params_
#predicting the probabilities for each class

pred = pd.DataFrame(RFmodel.fit(X_train,y_train).predict_proba(X_test), columns = RFmodel.classes_)



#saving file for submission

pd.DataFrame(pd.read_csv('../input/test.csv')['id']).join(pred).to_csv('submission.csv', index= False)