import pandas as pd



import matplotlib.pyplot as plt
diabetes_data = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')



diabetes_data.head(10)
diabetes_data.shape
diabetes_data.describe()
pd.crosstab(diabetes_data['Pregnancies'], diabetes_data['Outcome'])
plt.figure(figsize=(8, 8))



plt.scatter(diabetes_data['Glucose'], diabetes_data['Outcome'], c='g')



plt.xlabel('Glucose')

plt.ylabel('Test')



plt.show()
plt.figure(figsize=(8, 8))



plt.scatter(diabetes_data['Insulin'], diabetes_data['Outcome'], c='g')



plt.xlabel('Insulin')

plt.ylabel('Test')



plt.show()
plt.figure(figsize=(8, 8))



plt.scatter(diabetes_data['Age'], diabetes_data['Insulin'], c='g')



plt.xlabel('age')

plt.ylabel('insulin')



plt.show()
diabetes_data_correlation = diabetes_data.corr()



diabetes_data_correlation
import seaborn as sns



fig, ax = plt.subplots(figsize=(8, 8))



sns.heatmap(diabetes_data_correlation, annot=True)
features = diabetes_data.drop('Outcome', axis=1)



features.head()
features.describe()
from sklearn import preprocessing



standard_scaler = preprocessing.StandardScaler()
features_scaled = standard_scaler.fit_transform(features)



features_scaled.shape
features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)



features_scaled_df.head()
features_scaled_df.describe()
diabetes_data = pd.concat([features_scaled_df, diabetes_data['Outcome']], axis=1).reset_index(drop=True)



diabetes_data.head()
from sklearn.model_selection import train_test_split



X = diabetes_data.drop('Outcome', axis=1)

Y = diabetes_data['Outcome']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
x_train.shape, y_train.shape
x_test.shape, y_test.shape
from sklearn.linear_model import LogisticRegression



classifier = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')



classifier.fit(x_train, y_train)
# For recording: First do the Logistic regression, then just paste this in and shift-enter the rest

# from sklearn.tree import DecisionTreeClassifier



# classifier = DecisionTreeClassifier(max_depth=4)



# classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)



y_pred
pred_results = pd.DataFrame({'y_test': y_test,

                             'y_pred': y_pred})



pred_results.head(10)
from sklearn.metrics import accuracy_score, precision_score, recall_score
model_accuracy = accuracy_score(y_test, y_pred)

model_precision = precision_score(y_test, y_pred)

model_recall = recall_score(y_test, y_pred)



print("Accuracy of the model is {}% " .format( model_accuracy * 100))

print("Precision of the model is {}% " .format(model_precision * 100))

print("Recall of the model is {}% " .format(model_recall * 100))
diabetes_crosstab = pd.crosstab(pred_results.y_pred, pred_results.y_test)



diabetes_crosstab
TP = diabetes_crosstab[1][1]

TN = diabetes_crosstab[0][0]

FP = diabetes_crosstab[0][1]

FN = diabetes_crosstab[1][0]
accuracy_score_verified = (TP + TN) / (TP + FP + TN + FN)



accuracy_score_verified
precision_score_survived = TP / (TP + FP)



precision_score_survived
recall_score_survived = TP / (TP + FN)



recall_score_survived