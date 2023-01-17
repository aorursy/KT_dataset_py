import numpy as np # For Linear Algebra
import pandas as pd # Data

# For visualization
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white") #white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)

# Model building
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, cross_val_score
# Read CSV train data file into DataFrame
train_df = pd.read_csv("../input/train.csv")

# Read CSV test data file into DataFrame
test_df = pd.read_csv("../input/test.csv")

# preview train data
train_df.head()
print(f'The number of records in the train data is {train_df.shape[0]}.')
print(f'The number of records in the test data is {test_df.shape[0]}.')
# preview test data
test_df.head()
# check missing values in train data
train_df.isnull().sum()
# percent of missing "Age" 
print('Percent of missing "Age" records is %.2f%%' %((train_df['Age'].isnull().sum()/
                                                      train_df.shape[0])*100))
# How the Age column looks,
%matplotlib inline
ax = train_df["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
train_df["Age"].plot(kind='density', color='teal')
ax.set(xlabel='Age')
plt.xlim(-10,85)
train_df["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)
test_df["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)
# percent of missing "Embarked" 
print('Percent of missing "Embarked" records is %.2f%%' %((train_df['Embarked'].isnull().sum()/
                                                           train_df.shape[0])*100))
print('Passengers by Port of Embarkation: ')
print(train_df['Embarked'].value_counts())
sns.countplot(x='Embarked', data=train_df, palette='rainbow')
plt.show()
train_df["Embarked"].fillna(train_df['Embarked'].value_counts().idxmax(), inplace=True)
test_df["Embarked"].fillna(train_df['Embarked'].value_counts().idxmax(), inplace=True)
# percent of missing "Cabin" 
print('Percent of missing "Cabin" records is %.2f%%' %((train_df['Cabin'].isnull().sum()/
                                                        train_df.shape[0])*100))
train_df.drop('Cabin', axis=1, inplace=True)
test_df.drop('Cabin', axis=1, inplace=True)
train_df.drop('Fare', axis=1, inplace=True)
test_df.drop('Fare', axis=1, inplace=True)
# check missing values in adjusted train data
train_df.isnull().sum()
## Create categorical variable for traveling alone
train_df['TravelAlone']=np.where((train_df["SibSp"]+train_df["Parch"])>0, 0, 1)
train_df.drop('SibSp', axis=1, inplace=True)
train_df.drop('Parch', axis=1, inplace=True)

test_df['TravelAlone']=np.where((test_df["SibSp"]+test_df["Parch"])>0, 0, 1)
test_df.drop('SibSp', axis=1, inplace=True)
test_df.drop('Parch', axis=1, inplace=True)
#create categorical variables and drop some variables
training=pd.get_dummies(train_df, columns=["Pclass","Embarked","Sex"])
training.drop('Sex_female', axis=1, inplace=True)
training.drop('PassengerId', axis=1, inplace=True)
training.drop('Name', axis=1, inplace=True)
training.drop('Ticket', axis=1, inplace=True)

final_train = training
final_train.head()

testing=pd.get_dummies(test_df, columns=["Pclass","Embarked","Sex"])
testing.drop('Sex_female', axis=1, inplace=True)
testing.drop('PassengerId', axis=1, inplace=True)
testing.drop('Name', axis=1, inplace=True)
testing.drop('Ticket', axis=1, inplace=True)

final_test = testing
plt.figure(figsize=(15,8))
ax = sns.kdeplot(final_train["Age"][final_train.Survived == 1], color="darkturquoise", shade=True)
sns.kdeplot(final_train["Age"][final_train.Survived == 0], color="lightcoral", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Age for Surviving Population and Deceased Population')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()
final_train['IsMinor']=np.where(final_train['Age']<=16, 1, 0)
final_test['IsMinor']=np.where(final_test['Age']<=16, 1, 0)
sns.barplot('Pclass', 'Survived', data=train_df, color="orange")
plt.show()
sns.barplot('Embarked', 'Survived', data=train_df, color="teal")
plt.show()
sns.barplot('Sex', 'Survived', data=train_df, color="green")
plt.show()
import warnings
warnings.filterwarnings('ignore')

X = training.drop('Survived', axis=1) # Independent varaibles
y = training['Survived'] # Dependent variables

# Let's choose Logistic Regression
rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10, scoring='accuracy')
rfecv.fit(X, y)

print(f'Number of optimal features: {rfecv.n_features_}')
print(f'Selected optimal features: {list(X.columns[rfecv.support_])}')
# Splitting the data into train and test to evaluate our model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Logistic Regression
lg = LogisticRegression(n_jobs=-1)

# Training (Finding the optimal weights)
lg.fit(X_train, y_train)

# Predictions
y_pred = lg.predict(X_test)
# Review our predictions
y_pred
# Evaluation
print(f'Accuracy: {metrics.accuracy_score(y_test, y_pred)}')
print(f'ROC AUC Score: {metrics.roc_auc_score(y_test, y_pred)}')
print(f'Classification Report:\n{metrics.classification_report(y_test, y_pred)}')
predictions = lg.predict(testing)
ID = pd.read_csv('../input/test.csv').PassengerId
submit_df = pd.DataFrame()
submit_df['PassengerId'] = ID
submit_df['Survived'] = predictions

submit_df.head()
# Saving the file,
submit_df.to_csv('submission.csv', index=False)