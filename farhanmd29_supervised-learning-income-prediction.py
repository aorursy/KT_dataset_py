import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
# Load the census income dataset
data = pd.read_csv("../input/adult.csv")
data.head()
# TODO: Total number of records
n_records = len(data)

# TODO: Number of records where individual's income is more than $50,000
n_greater_50k = len(data.query('income == ">50K"'))

# TODO: Number of records where individual's income is at most $50,000
n_at_most_50k = len(data.query('income == "<=50K"'))

# TODO: Percentage of individuals whose income is more than $50,000
greater_percent = (float(n_greater_50k) / n_records * 100)

# Print the results
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent))
# Drop this column since it is not a significant feature to build our model prediction
data.drop(labels='fnlwgt',axis=1,inplace=True)
data[data.isnull()].count()
X = data.iloc[:,:-1]
Y = data.iloc[:,-1]
# Check out the head of input features in X
X.head()
# Check out the head of output target in Y
Y.head()
# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

X = pd.DataFrame(data = X)
X[numerical] = scaler.fit_transform(X[numerical])
# Show an example of a record with scaling applied
display(X.head(n = 5))
# TODO: One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
X_final = pd.get_dummies(X)

# TODO: Encode the 'income_raw' data to numerical values
Y_final = [1 if x == '>50K' else 0 for x in Y ]

# Print the number of features after one-hot encoding
encoded = list(X_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))
# Print the list of features which were encoded
print(encoded)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y_final, test_size=0.20, random_state=0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))
# TODO: Calculate accuracy
accuracy = float(n_greater_50k) / (n_greater_50k + n_at_most_50k) 

# TODO: Calculate F-score using the formula above for beta = 0.5
fscore = 1.25 * (accuracy) / ( 0.25 * accuracy + 1)
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Params for Random Forest
num_trees = 100
max_features = 3

#Spot Check 5 Algorithms (LR, LDA, KNN, CART, GNB, SVM)
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier(n_estimators=num_trees, max_features=max_features)))
#models.append(('SVM', SVC()))
# evalutate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=0)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# Model Comparison
fig = plt.figure()
fig.suptitle('Algorith Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
random_forest = RandomForestClassifier(n_estimators=250,max_features=5)
random_forest.fit(X_train, Y_train)
predictions = random_forest.predict(X_test)
print("Accuracy: %s%%" % (100*accuracy_score(Y_test, predictions)))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))