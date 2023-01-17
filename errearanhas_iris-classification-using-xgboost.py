from sklearn import datasets



iris = datasets.load_iris()

X = iris.data

y = iris.target
#Splitting the arrays into random train and test subsets (80% training, 20% testing)



from sklearn.cross_validation import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("Train data length:",len(X_train));

print("Test data length:",len(X_test));
#Creating the Xgboost DMatrix data format (from the arrays already obtained)



import xgboost as xgb



dtrain = xgb.DMatrix(X_train, label=y_train)

dtest = xgb.DMatrix(X_test, label=y_test)
# Setting some parameters



parameters = {

    'eta': 0.3,  

    'silent': True,  # option for logging

    'objective': 'multi:softprob',  # error evaluation for multiclass tasks

    'num_class': 3,  # number of classes to predic

    'max_depth': 3  # depth of the trees in the boosting process

    }  

num_round = 20  # the number of training iterations
#training the model

bst = xgb.train(parameters, dtrain, num_round)
#resut

preds = bst.predict(dtest)
preds[:5]
'''

Selecting the column that represents the highest probability 

(note that, for each line, there is 3 columns, indicating the probability for each class)

'''



import numpy as np



best_preds = np.asarray([np.argmax(line) for line in preds])
best_preds
#calculating the precision



from sklearn.metrics import precision_score



print(precision_score(y_test, best_preds, average='macro'))