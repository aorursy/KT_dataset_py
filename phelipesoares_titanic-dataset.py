import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing


from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import plot_confusion_matrix # to draw a confusion matrix
from sklearn.metrics import confusion_matrix # to create a confusion matrix
from sklearn.model_selection import train_test_split # to split the data into training and testing sets
from sklearn.model_selection import cross_val_score

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsTransformer, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier # To build a classifier tree
from sklearn.tree import plot_tree # to draw a classification tree
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv(r"C:\Users\Phelipe\OneDrive\Data Science\Datasets\Titanic.csv")
df.head()
dfm = df[['Age', 'Fare', 'Sex', 'sibsp', 'Parch', 'Pclass', 'Embarked', '2urvived']]
dfm.head()
dfm.describe()
dfm.info()
dfm_nn = dfm.dropna(axis=0)
dfm_nn.describe()
dfm_nn.head()
ax = sns.boxplot(x='2urvived', y='Age', data=dfm_nn)
ax = sns.boxplot(y='2urvived', x='Fare', data=dfm_nn, orient='h')
dfm_nn.hist(column='Sex', by='2urvived', figsize=(10,5))
dfm_nn.hist(column='Pclass', by='2urvived', figsize=(10,5))
dfm_nn.hist(column='Parch', by='2urvived', figsize=(10,5))
dfm_nn.hist(column='sibsp', by='2urvived', figsize=(10,5))
dfm_nn.hist(column='Embarked', by='2urvived', figsize=(10,5))
dfm_nn.hist(figsize=(15,10))
# Correlation Matrix
corr = dfm_nn.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
x = dfm_nn.drop(columns='2urvived')
X_dummed = pd.get_dummies(x, columns=['Embarked', 'Sex', 'sibsp', 'Parch', 'Pclass'])
y = dfm_nn['2urvived']
scores = cross_val_score(LinearSVC(class_weight='balanced'), X_dummed, y, cv=10)
scores
print("Accuracy Linear SVC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(RandomForestClassifier(class_weight='balanced'), X_dummed, y, cv=10)
scores
print("Accuracy Random Forest: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(LogisticRegression(class_weight='balanced', max_iter=10000), X_dummed, y, cv=10)
scores
print("Accuracy Logistic Regression: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(svm.SVC(class_weight='balanced'), X_dummed, y, cv=10)
scores
print("Accuracy SVM Classifier: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(KNeighborsClassifier(), X_dummed, y, cv=10)
scores
print("Accuracy KNeighbors: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(DecisionTreeClassifier(class_weight='balanced'), X_dummed, y, cv=10)
scores
print("Accuracy DecisionTree: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
X_train, x_test, y_train, y_test = train_test_split(X_dummed, y, random_state=42)
KN = KNeighborsClassifier()

kn = KN.fit(X_train, y_train)
predict = kn.predict(x_test)

acc = accuracy_score(y_test, predict)
# Function responsible for checking our model's performance on the test data
def testSetResultsClassifier(kn, x_test, y_test):
    predictions = kn.predict(x_test)
    
    results = []
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    
    results.append(f1)
    results.append(precision)
    results.append(recall)
    results.append(roc_auc)
    results.append(accuracy)
    
    print("\n\n#---------------- Test set results (Best Classifier) ----------------#\n")
    print("F1 score, Precision, Recall, ROC_AUC score, Accuracy:")
    print(results)
    
    return results
    f1 = f1_score(y_test, predict)
print(f1)

