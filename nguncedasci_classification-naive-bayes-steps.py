import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
#import and split the data
diabetes = pd.read_csv("../input/diabetes/diabetes.csv")
df = diabetes.copy()
df = df.dropna()
y = df["Outcome"]
X = df.drop(['Outcome'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)
#set and fit the model
from sklearn.naive_bayes import GaussianNB
nb_model =  GaussianNB().fit(X_train, y_train)
nb_model
#prediction
nb_model.predict(X_test)[0:5]
nb_model.predict_proba(X_test)[0:5]
#Model Tuning
#NOTE: There isn't ANY hiperparameter(external parameter) for Naive Bayes. Therefore, we couldn't apply model tuning, 
# we can only verify our model..
# test error
y_pred=nb_model.predict(X_test)
accuracy_score(y_test,y_pred)
# verified test error
cross_val_score(nb_model, X_test,y_test,cv=10).mean()
# We found 0.766 by Logistic Regression
# And now, 0.775 by Naive Bayes 
