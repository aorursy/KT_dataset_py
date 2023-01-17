# Algebra
import numpy as np

# Data processing
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

# Model building
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Cross validation
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split

# Evaluation
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, \
                             recall_score, f1_score, precision_recall_curve, \
                             roc_curve, roc_auc_score, matthews_corrcoef)

# Warnings
import warnings
warnings.filterwarnings('ignore')
# Datasets
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
print('Shape of train data: {}'.format(train_data.shape))
train_data.head()
print('Shape of test data: {}'.format(test_data.shape))
test_data.head()
# Describe data
train_data.describe(include='all')
print('Overall Survivability(%): {}'.format(round(100*train_data.Survived.mean(),2)))

# Plot count by survival
ax = sns.countplot(x='Survived', data=train_data).set_title('Passenger Count by Survived')
print('Survivability(%) by {}'.format(round(100*train_data.groupby('Sex').mean().Survived,2)))

# Plot count and survivability of passengers by sex
fig, axes = plt.subplots(1, 2, figsize=(15,5))
ax = sns.countplot(x='Sex', data=train_data, ax=axes[0]).set_title('Passenger Count by Sex')
ax = sns.barplot(x='Sex', y='Survived', data=train_data,  ax=axes[1]).set_title('Passenger Survivability by Sex')
print('Survivability(%) by {}'.format(round(100*train_data.groupby('Pclass').mean().Survived,2)))

# Plot count and survivability of passengers by class
fig, axes = plt.subplots(1, 2, figsize=(15,5))
ax = sns.countplot(x='Pclass', data=train_data, ax=axes[0]).set_title('Passenger Count by Pclass')
ax = sns.barplot(x='Pclass', y='Survived', data=train_data, ax=axes[1]).set_title('Passenger Survivability by Pclass')
print('Survivability(%) by {}'.format(round(100*train_data.groupby('Embarked').mean().Survived,2)))

# Plot count and survivability of passengers by port of embarkation
fig, axes = plt.subplots(1, 2, figsize=(15,5))
ax = sns.countplot(x='Embarked', data=train_data, ax=axes[0]).set_title('Passenger Count by Embarked')
ax = sns.barplot(x='Embarked', y='Survived', data=train_data, ax=axes[1]).set_title('Passenger Survivability by Embarked')
# Plot count and survivability of passengers by class, sex and port of embarkation
fig, axes = plt.subplots(2, 3, figsize=(20,10))
ax = sns.countplot(x='Pclass', hue='Sex', hue_order=['male','female'], data=train_data[train_data.Embarked=='S'], ax=axes[0][0]).set_title("Passenger Count Embarked=S")
ax = sns.countplot(x='Pclass', hue='Sex', hue_order=['male','female'], data=train_data[train_data.Embarked=='C'], ax=axes[0][1]).set_title("Passenger Count Embarked=C")
ax = sns.countplot(x='Pclass', hue='Sex', hue_order=['male','female'], data=train_data[train_data.Embarked=='Q'], ax=axes[0][2]).set_title("Passenger Count Embarked=Q")
ax = sns.pointplot(x='Pclass', y='Survived', hue_order=['male','female'], hue='Sex', data=train_data[train_data.Embarked=='S'], ax=axes[1][0]).set_title("Passenger Survivability for Embarked=S")
ax = sns.pointplot(x='Pclass', y='Survived', hue_order=['male','female'], hue='Sex', data=train_data[train_data.Embarked=='C'], ax=axes[1][1]).set_title("Passenger Survivability for Embarked=C")
ax = sns.pointplot(x='Pclass', y='Survived', hue_order=['male','female'], hue='Sex', data=train_data[train_data.Embarked=='Q'], ax=axes[1][2]).set_title("Passenger Survivability for Embarked=Q")
# Plot total passenger and survivors count by age
fig, ax = plt.subplots(figsize=(15,5))
ax = sns.distplot(train_data.Age.dropna(), bins=32, kde=False, label='Total') \
    .set_title('Total Passengers and Survivors Count by Age')
ax = sns.distplot(train_data.Age.dropna()[train_data['Survived']==1], bins=32, kde=False, label='Survived').legend()

# Plot likelihood of surviving/not surviving by age
ax = sns.FacetGrid(train_data, hue="Survived", aspect=4).set(xlim=(0, 80)).map(sns.kdeplot,'Age', shade=True).add_legend() \
    .fig.suptitle('Survivability by Age')
# Group ages in 5 age bands
train_data['Ageband'] = pd.cut(train_data.Age.dropna(), 5, labels=[0,1,2,3,4])
print('Survival(%) by {}'.format(round(100*train_data.groupby('Ageband').Survived.mean(),2)))

# Plot count and survivability of passengers by ageband, sex and class
fig, axes = plt.subplots(2, 3, figsize=(20,10))
ax = sns.countplot(x='Ageband', hue='Sex', hue_order=['male','female'], data=train_data[train_data.Pclass==1], ax=axes[0][0]).set_title("Passenger Count Pclass=1")
ax = sns.countplot(x='Ageband', hue='Sex', hue_order=['male','female'], data=train_data[train_data.Pclass==2], ax=axes[0][1]).set_title("Passenger Count Pclass=2")
ax = sns.countplot(x='Ageband', hue='Sex', hue_order=['male','female'], data=train_data[train_data.Pclass==3], ax=axes[0][2]).set_title("Passenger Count Pclass=3")
ax = sns.pointplot(x='Ageband', y='Survived', hue_order=['male','female'], hue='Sex', data=train_data[train_data.Pclass==1], ax=axes[1][0]).set_title("Passenger Survivability for Pclass=1")
ax = sns.pointplot(x='Ageband', y='Survived', hue_order=['male','female'], hue='Sex', data=train_data[train_data.Pclass==2], ax=axes[1][1]).set_title("Passenger Survivability for Pclass=2")
ax = sns.pointplot(x='Ageband', y='Survived', hue_order=['male','female'], hue='Sex', data=train_data[train_data.Pclass==3], ax=axes[1][2]).set_title("Passenger Survivability for Pclass=3")
# Plot total passenger and survivors count by age
fig, ax = plt.subplots(figsize=(15,5))
ax = sns.distplot(train_data.Fare, bins=50, kde=False).set_title('Total Passengers and Survivors Count by Fare')
ax = sns.distplot(train_data.Fare[train_data['Survived']==1], bins=50, kde=False)

# Plot likelihood of surviving/not surviving by age
ax = sns.FacetGrid(train_data, hue="Survived", aspect=4).set(xlim=(0, 500)).set(ylim=(0, 0.1)) \
    .map(sns.kdeplot,'Fare', shade=True).add_legend().fig.suptitle('Survivability by Age')
# Plot distribution of fares by class
ax = sns.FacetGrid(train_data, hue='Pclass', aspect=4).set(xlim=(0,100)).map(sns.kdeplot, 'Fare', shade=True).add_legend() \
    .fig.suptitle('Passenger Fare by Pclass')
# Plot count and survivability of passengers by possession or not of a cabin
fig, axes = plt.subplots(1, 2, figsize=(15,5))
ax = sns.countplot(x=train_data.Cabin.notnull().astype(int), ax=axes[0]).set_title('Passenger Count by Cabin Ownership')
ax = sns.barplot(x=train_data.Cabin.notnull().astype(int), y=train_data.Survived, ax=axes[1]) \
    .set_title('Passenger Survivability by Cabin Ownership')
# Plot count and survivability of passengers by class, sex and port of embarkation
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
ax = sns.countplot(x='SibSp', data=train_data, ax=axes[0][0]).set_title('Passenger Count by number of Siblings/Spouses')
ax = sns.countplot(x='Parch', data=train_data, ax=axes[0][1]).set_title('Passenger Count by number of Parents/Children')
ax = sns.pointplot(x='SibSp', y='Survived', data=train_data, ax=axes[1][0]) \
    .set_title('Passenger Survivability by number of Siblings/Spouses')
ax = sns.pointplot(x='Parch', y='Survived', data=train_data, ax=axes[1][1]) \
    .set_title('Passenger Survivability by number of Parents/Children')
# Combine SibSp and Parch into one feature
train_data['Relatives'] = train_data.SibSp+train_data.Parch

# Plot count and survivability of passengers by possession or not of a cabin
fig, axes = plt.subplots(1, 2, figsize=(15,5))
ax = sns.countplot(x='Relatives', data=train_data, ax=axes[0]).set_title('Passenger Count by number of Relatives')
ax = sns.pointplot(x='Relatives', y='Survived', data=train_data, ax=axes[1]) \
    .set_title('Passenger Survivability by number of Relatives')
# Combine train and test sets into a single dataset for data processing
data = pd.concat([train_data, test_data], join='inner', ignore_index=True)
# Show passengers with missing Embarked values
data[data.Embarked.isnull()]
# Fill missing values in Embarked with 'S'
data.Embarked.fillna('S', inplace=True)
# For each different Sex and Pclass combination subgroup, fill NaN values with its median
for sex in set(data.Sex):
    for pclass in set(data.Pclass):
        age = data[(data.Sex==sex) & (data.Pclass==pclass)].Age.median()
        data.loc[(data.Age.isnull()) & (data.Sex==sex) & (data.Pclass==pclass), 'Age'] = age

# Median values of each subgroup for reference
data.groupby(['Pclass','Sex']).Age.median().unstack()
# Group ages in 5 age bands
data['Ageband'] = pd.cut(data.Age, 5, labels=[0,1,2,3,4])
# Show passenger with missing Fare value
data[data.Fare.isnull()]
# Fill missing value with median of 3rd class passengers boarding from Southampton
data.loc[1043,'Fare'] = data[(data.Pclass==3) & (data.Embarked=='S')].Fare.median()
# Group fares in 5 fare bands
data['Fareband'] = pd.qcut(data.Fare, 5, labels=False)
# Change Cabin feature to have a value of 1 if the passenger owned a cabin and 0 otherwise
data['Cabin'] = data.Cabin.notnull().astype(int)
# Create new feature that shows if the passenger has a small family onboard
data['Small_Family'] = pd.Series(data.SibSp+data.Parch).apply(lambda x: 1 if x>0 and x<4 else 0)
# Extract Title from Name and count the sum of occurences for each distinct value
data['Title'] = data.Name.apply(lambda x: x.split(',')[1].strip().split('.')[0].strip())
data.Title.value_counts()
# Replace rare occurences in Title with 'Other'
data.Title = data.Title.replace(['Dr', 'Rev', 'Col', 'Major', 'Sir', 'Capt', 'Lady', 'Jonkheer', \
                                   'Dona', 'Don', 'the Countess'], 'Other', regex=True)

# Replace french honorifics with their english equivalents
data.Title = data.Title.replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'}, regex=True)
# Change 'female'/'male' into 0/1
data.Sex = data.Sex.map({'female':0, 'male':1}).astype(int)
# Create dummy variables for categorical features
data = pd.get_dummies(data, columns=['Embarked', 'Title'], drop_first=True)
# Discard unnecessary features
data.drop(['PassengerId', 'Ticket', 'Name', 'Age', 'SibSp', 'Parch', 'Fare'], axis=1, inplace=True)
# Split data back into train set, test set and target variable
X = data[:891]
test = data[891:].reset_index(drop=True)
y = train_data.Survived
X.head()
corr = data.corr()

# Plot correlation heatmap
fig, size = plt.subplots(figsize=(15,10))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
ax = sns.heatmap(corr, cmap=sns.diverging_palette(250, 10, as_cmap=True), annot=True, mask=mask, square=True, \
                 xticklabels=data.corr().columns, yticklabels=data.corr().columns).set_title('Correlation of Features')
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)
# List of algorithms to be used
models = [('Logistic Regression', LogisticRegression()),
          ('Perceptron', Perceptron()),
          ('Naive Bayes', GaussianNB()),
          ('Random Forest', RandomForestClassifier(random_state=7)),
          ('K Nearest Neighbors', KNeighborsClassifier()),
          ('Linear Support Vector', LinearSVC()),
          ('Support Vector', SVC()),
          ('Gradient Boosting', GradientBoostingClassifier())]

# Train each model and evaluate its accuracy
scores = []
for name, model in models:
    model.fit(X_train,y_train)
    score = round(100*model.score(X_test,y_test), 2)
    scores.append((name, score))
scores.sort(reverse=True, key=lambda x:x[1])
    
# Show performances of all models
model_scores = pd.DataFrame([x[1] for x in scores], index=[x[0] for x in scores], columns=['Accuracy(%)'])
model_scores
# Train model with optimal hyperparameters
random_forest = RandomForestClassifier(max_depth=5, criterion='entropy', min_samples_leaf=2, n_estimators=50, random_state=7)

# Calculate cross validated accuracy
model_score = format(100*np.mean(cross_val_score(random_forest, X, y, cv=10)), '0.2f')
print('Random Forest Accuracy: {}'.format(model_score))
# Fit model to all training data
random_forest.fit(X,y)

# Create features importance dataframe
features = pd.DataFrame()
features['feature'] = X.columns
features['importance'] = random_forest.feature_importances_
features.sort_values(by=['importance'], ascending=False, inplace=True)

# Plot feature importances
fig, ax = plt.subplots(figsize=(10,5))
ax = sns.barplot(x='importance', y='feature', color='lightblue', data=features).set_title('Feature Importances')
# Predict survivors
predictions = cross_val_predict(random_forest, X, y, cv=10)

# Create confusion matrix
matrix = pd.DataFrame(confusion_matrix(y, predictions), index=['Actual Negatives', 'Actual Positives'], \
                      columns=['Predicted Negatives', 'Predicted Positives'])
matrix
# Calculate precision, recall, f1-score
print('Precision: {}'.format(round(precision_score(y, predictions), 4)))
print('Recall: {}'.format(round(recall_score(y, predictions), 4)))
print('F1-Score: {}'.format(round(f1_score(y, predictions), 4)))
# Calculate mathews correlation coefficient
print('Matthews Correlation Coefficient: {}'.format(round(matthews_corrcoef(y, predictions), 4)))
# Calculate probabilities of survival predictions
y_scores = random_forest.predict_proba(X)[:,1]

# Calculate true positive rates, false positive rates, thresholds
fpr, tpr, thresholds = roc_curve(y, y_scores)

# Plot ROC curve
fig, ax = plt.subplots(figsize=(15,4))
ax = plt.plot(fpr, tpr)
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
ax = plt.plot([0,1],[0,1])
auc_score = roc_auc_score(y, y_scores)
print("AUC-Score: {}".format(round(auc_score,4)))
# Calculate optimal threshold
youden_j = np.argmax(tpr-fpr)
threshold = thresholds[youden_j]
print('Optimal threshold: {}'.format(round(threshold,4)))
# Calculate new predictions
predictions = [1 if x>threshold else 0 for x in y_scores]

# Re-calculate metrics
print('Precision: {}'.format(round(precision_score(y, predictions), 4)))
print('Recall: {}'.format(round(recall_score(y, predictions), 4)))
print('F1-Score: {}'.format(round(f1_score(y, predictions), 4)))
print('Matthews Correlation Coefficient: {}'.format(round(matthews_corrcoef(y, predictions), 4)))
# Predict survivors
y_submit = random_forest.predict_proba(test)[:,1]
y_submit = [1 if x>threshold else 0 for x in y_submit]

# Create submission dataframe and save to csv
submission = pd.DataFrame([test_data.PassengerId, pd.Series(y_submit, name='Survived')]).T
submission.to_csv('submission.csv', index=False)