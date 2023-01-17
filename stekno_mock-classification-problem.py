# Load libraries

import pandas as pd

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
insure = pd.read_csv('../input/claims_data.csv')
#split dataset in features and target variable



feature_cols = ['age', 'sex', 'bmi', 'steps','children','smoker','region']

X = insure[feature_cols] # Features

y = insure.insurance_claim  # Target variable
X_transformed = pd.get_dummies(X, drop_first=True)
# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=1) # 70% training and 30% test
# Create Decision Tree classifer object

insure_clf = DecisionTreeClassifier()



# Train Decision Tree Classifer

insure_clf = insure_clf.fit(X_train,y_train)



#Predict the response for test dataset

y_pred = insure_clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
confusion_matrix(y_test, y_pred)
labels = ['No claim', 'Claim']



pd.DataFrame(data=confusion_matrix(y_test, y_pred), index=labels, columns=labels)
print('Classification Report')

print(classification_report(y_test, y_pred, target_names=['No claim', 'Claim']))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))