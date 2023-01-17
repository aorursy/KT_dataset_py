import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier
#import and split the data
diabetes = pd.read_csv("../input/diabetes/diabetes.csv")
df = diabetes.copy()
df = df.dropna()
y = df["Outcome"]
X = df.drop(['Outcome'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)
# scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
#set and fit the model
mlpc=MLPClassifier().fit(X_train_scaled,y_train)
y_pred = mlpc.predict(X_test_scaled)
accuracy_score(y_test, y_pred)    #We didn't tune any hiperparameters so this is primitive test error. 
#Model Tuning
mlpc
#Important hiperparameters
#hidden_layer_sizes
#solver
#learning_rate( through methods)
#alpha
mlpc_params = {"alpha": [0.1, 0.01, 0.02, 0.005, 0.0001,0.00001],
              "hidden_layer_sizes": [(10,10,10),
                                     (100,100,100),
                                     (100,100),
                                     (3,5), 
                                     (5, 3)],
              "solver" : ["lbfgs","adam","sgd"],
              "activation": ["relu","logistic"]}
mlpc_cv_model=GridSearchCV(mlpc,mlpc_params,cv=10, n_jobs=-1,verbose=2)
mlpc_cv_model.fit(X_train_scaled,y_train)
    
mlpc_cv_model.best_params_
mlpc_final_model= MLPClassifier(activation='logistic', alpha=0.1, hidden_layer_sizes= (100, 100, 100), solver= 'adam')
mlpc_final_model.fit(X_train_scaled,y_train)
y_pred=mlpc_final_model.predict(X_test_scaled)
accuracy_score(y_pred,y_test)
# We found 0.774 by Logistic Regression
#          0.775 by Naive Bayes 
#          0.731 by KNN
#          0.744 by Linear SVC
#          0.735 by Nonlinear SVC Steps
#And now,  0.735  by ANN
