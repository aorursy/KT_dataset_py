import warnings
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from yellowbrick.classifier import ROCAUC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve, accuracy_score,r2_score, confusion_matrix
warnings.filterwarnings('ignore')
%matplotlib inline
plt.style.use('seaborn')
file_name = '../input/mushrooms.csv'
mushroom_df = pd.read_csv(file_name)
mushroom_df.head()
print("The total Number of Observations: {}".format(mushroom_df.shape[0]))
print("Total Number of Edible Mushroom in the data set: {}".format(mushroom_df['class'].value_counts()[0]))
print("Total Number of Poisinous Mushroom in the data set: {}".format(mushroom_df['class'].value_counts()[1]))
mushroom_df['class'].value_counts().plot(kind='bar')
plt.title('Distribution of Target Class')
plt.ylabel("Number of Observation")
plt.xlabel("Frequency of Edible and Poisinious Mushroom")
for col in mushroom_df.columns:
    print(col)
    mushroom_df[col].value_counts().plot(kind='bar')
    plt.title('Distribution {}'.format(col))
    plt.ylabel("Number of Observation")
    plt.xlabel("Frequency of {}".format(col))
    plt.show()
for col in mushroom_df.columns:
    print(col)
    print(mushroom_df[col].value_counts())
X_data = mushroom_df.drop(columns=['class'],axis=1)
y_data = mushroom_df['class']
X_data_dummy = pd.get_dummies(X_data, prefix=X_data.columns, drop_first=True)
X_data_dummy.head()
y_data = y_data.map({'p':0,'e':1}).values
X_data = X_data_dummy.values
X_train,X_test,y_train,y_test = train_test_split(X_data, y_data, test_size = 0.2,random_state=0)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
# Instantiate the visualizer with the classification model
visualizer = ROCAUC(LogisticRegression(), classes=[1,0])

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data
accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
# Transform to df for easier plotting
cm_df = pd.DataFrame(cm,
                     index = ['Poisinous','Edible'], 
                     columns = ['Poisinous','Edible'])

plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df, annot=True)
plt.title('Logistic Regression \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
clf = SVC(kernel = 'linear').fit(X_train,y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
# Transform to df for easier plotting
cm_df = pd.DataFrame(cm,
                     index = ['Poisinous','Edible'], 
                     columns = ['Poisinous','Edible'])

plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df, annot=True)
plt.title('Logistic Regression \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
clf = SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X_data, y_data, cv=5)
scores 
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(classifier, X_data, y_data, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))