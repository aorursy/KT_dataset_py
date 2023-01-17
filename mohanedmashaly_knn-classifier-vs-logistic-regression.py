import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer 
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
from seaborn import lineplot
import seaborn as sns
df = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
#cleaning and processing the data
features = df.drop("Outcome", axis=1)
pregnancy_column = df['Pregnancies']
target   = df["Outcome"]
imputer = SimpleImputer(missing_values = 0, strategy='mean')
imputer.fit(features)
features = imputer.transform(features)
data_frame = pd.DataFrame(features, columns = ['Pregnancies', 'Glucose', 'BloodPressure', 
                                              'SkinThickness', 'Insulin', 'BMI',
                                              'DiabetesPedigreeFunction', 'Age'])
data_frame.insert(0 , 'Pregnancies',pregnancy_column, True)
X_train, X_test, y_train, y_test = train_test_split(data_frame, target, test_size = 0.2, random_state = 42)
training_scaler = MinMaxScaler().fit(X_train)
training_set = training_scaler.transform(X_train)
testing_scaler = MinMaxScaler().fit(X_test)
testing_set = testing_scaler.transform(X_test)
KNN_classifier = KNeighborsClassifier(n_neighbors = 8)
KNN_classifier.fit(training_set, y_train)
y_predict = KNN_classifier.predict(testing_set)
#Calculating accuracy of the model using classification_report and auc_Roc using KNN_classifier 
print("Accuracy:",metrics.accuracy_score(y_test, y_predict))
print(classification_report(y_test, y_predict))
print(roc_auc_score(y_test, y_predict))
#Calculating accuracy of the model using classification_report and auc_Roc using Logistic Regression
LogisticReg = LogisticRegression(max_iter = 300, random_state = 0)
LogisticReg.fit(X_train, y_train)
y_prediction = LogisticReg.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_prediction))
print(classification_report(y_test, y_prediction))
print(roc_auc_score(y_test, y_prediction))
#Visualizing with seaborn 
sns.pairplot(data=df, y_vars = ['Outcome'], x_vars = ['Pregnancies','Glucose','BloodPressure',
                                                      'SkinThickness','Insulin',
                                                      'BMI','DiabetesPedigreeFunction','Age'])
pyplot.show()