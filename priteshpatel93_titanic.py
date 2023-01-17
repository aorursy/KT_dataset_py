import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline

train_df = pd.read_csv("../input/titanic/train.csv")
test_df = pd.read_csv("../input/titanic/test.csv")
train_df.info()
test_df.isnull().sum()
train_df.describe()
train_df.head(10)
train_df.isnull().sum()
def preprocess(df):
    # Fill in missing ages with average
    df['Age'] = train_df['Age'].fillna(df['Age'].mean())
    
    # Convert to ints
    df['Age'] = df['Age'].astype(int)

    # Derrive titles
    df['Title'] = df['Name'].str.extract(r'(?<=\, )(.*?)(?=\.)', expand=True)

    # Embarked fill with most common
    df['Embarked'] = df['Embarked'].fillna('S')

    # Family Group Size
    df['FamilyGroupSize'] = df['SibSp'] + df['Parch']

    # Drop useless columns
    df = df.drop(['Cabin', 'Name', 'Ticket'], axis=1)

    # Convert features to numerics
    df = pd.get_dummies(df, drop_first=True)
    
    return df

train_df = preprocess(train_df)
train_df = train_df.drop('PassengerId', axis=1)
test_df = preprocess(test_df)

train_df.info()

ax = sns.countplot(x = 'Pclass', hue = 'Survived', palette = 'Set1', data = train_df)
ax.set(title = 'Passenger status (Survived/Died) against Passenger Class', 
       xlabel = 'Passenger Class', ylabel = 'Total')
plt.show()
ax = sns.countplot(x = 'Sex_male', hue = 'Survived', palette = 'Set1', data = train_df)
ax.set(title = 'Total Survivors According to Sex (0 = female, 1 = male)', xlabel = 'Sex', ylabel='Total')
plt.show()
plt.figure(figsize=(15,8))
ax = sns.kdeplot(train_df["Age"][train_df['Survived'] == 1], color="darkturquoise", shade=True)
sns.kdeplot(train_df["Age"][train_df['Survived'] == 0], color="lightcoral", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Age of surivors')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()
X_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']

logreg = LogisticRegression(max_iter=1000)
rfe_cv = RFECV(logreg, step=1, cv=5, scoring='accuracy')
rfe_cv = rfe_cv.fit(X_train, y_train)

Selected_features = list(X_train.columns[rfe_cv.support_])
print("Features Selected: {}".format(Selected_features))
                                                  
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfe_cv.grid_scores_) + 1), rfe_cv.grid_scores_)
plt.show()
plt.subplots(figsize=(40,20))
sns.heatmap(X_train[Selected_features].corr(), annot=True, cmap="RdYlGn")
plt.show()
from sklearn.model_selection import train_test_split

X = X_train[Selected_features]
y = y_train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print("accuracy:", accuracy_score(y_test, y_pred))

from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score

y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.legend(loc=4)
plt.show()
test_df[['Title_Major', 'Title_Jonkheer', 'Title_Lady', 'Title_the Countess', 'Title_Mlle', 'Title_Don', 'Title_Mme', 'Title_Col', 'Title_Sir']] = 0
test_df['Survived'] = logreg.predict(test_df[Selected_features])

submit = test_df[['PassengerId','Survived']]
submit.to_csv("submit.csv", index=False)

submit.head()
