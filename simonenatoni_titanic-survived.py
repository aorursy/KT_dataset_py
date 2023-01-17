import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
file = "train.csv"
train = pd.read_csv(file)

file = "test.csv"
test = pd.read_csv(file)
train.head()
test.head()
train.info()
col_drop = ["PassengerId","Name","Ticket","Cabin"]
train = train.drop(columns=col_drop)
test = test.drop(columns=col_drop)
train.columns.tolist()
train.isnull().sum()
train.Age.fillna(train.Age.median(), inplace=True)
from statistics import mode
train.Embarked=train.Embarked.fillna(mode(train.Embarked))
test.Age.fillna(test.Age.median(), inplace=True)
test.Fare.fillna(test.Fare.mean(), inplace=True)
test.isnull().sum()
train.isnull().sum()
print("Sopravvissuti in valori assoluti:")
print(train.Survived.value_counts())
print("\nSopravvissuti in valori percentuali")
print(train.Survived.value_counts(normalize=True)*100)
total = float(len(train)) 
ax = sns.countplot(x="Sex", data=train) 
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center")
plt.show()
ax = sns.countplot(x="Pclass", data=train) 
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center")
plt.show()
ax = sns.countplot(x="Pclass",hue="Sex", data=train) 
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center")
plt.show()
round(pd.crosstab(index=train.Pclass, columns=train.Sex, normalize='index')*100)
# Tabella di contingenza
pd.crosstab(index=train.Pclass, columns=train.Sex, normalize='index').plot(kind='bar')
plt.title("relationship between sex and class");
ax = sns.countplot(x="Embarked", data=train) 
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center")
plt.show()
pd.crosstab(index=train.Embarked, columns=train.Pclass, normalize='index')*100
# Tabella di contingenza
pd.crosstab(index=train.Embarked, columns=train.Pclass, normalize='index').plot(kind='bar')
plt.title("relationship between embarked and class");
pd.crosstab(index=train.Survived, columns=train.Pclass, normalize='columns')*100
# Tabella di contingenza
pd.crosstab(index=train.Survived, columns=train.Pclass, normalize='columns').plot(kind='bar')
plt.title("relationship between class and survivor");
pd.crosstab(index=train.Survived, columns=train.Embarked, normalize='columns')*100
# Tabella di contingenza
pd.crosstab(index=train.Survived, columns=train.Embarked, normalize='columns').plot(kind='bar')
plt.title("relationship between embarked and survivor");
# analisi multivariata
tab=pd.crosstab(train.Survived, [train.Embarked,train.Pclass], rownames=['Survived'], colnames=['Embarked','Pclass'],normalize='columns')*100
tab
pd.crosstab(index=train.Survived, columns=train.Sex, normalize='columns')*100
# Tabella di contingenza
pd.crosstab(index=train.Survived, columns=train.Sex, normalize='columns').plot(kind='bar')
plt.title("relationship between sex and survivor");
pd.crosstab(index=train.Embarked, columns=train.Sex, normalize='index')*100
# Tabella di contingenza
pd.crosstab(index=train.Embarked, columns=train.Sex, normalize='index').plot(kind='bar')
plt.title("relationship between sex and Embarked");
sns.catplot(x="Survived", y="Age", kind="box", hue="Sex",data=train);
plt.title("Age distribution by survived and sex");
train.groupby("Survived")['Age'].mean()
categorial = train.select_dtypes(exclude=["int64", "float"]).columns.tolist()
from sklearn.preprocessing import LabelEncoder
for col in categorial:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
X = train.drop("Survived", axis=1).values
#X = train[["Sex","Pclass","Age","Fare"]]

Y = train["Survived"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

lr = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=5) # k=5
tree = DecisionTreeClassifier(criterion="gini", max_depth=8, random_state=1)
forest = RandomForestClassifier(criterion='gini', n_estimators=100, max_depth=8, random_state=1)
svc = LinearSVC()


for clf in [lr, knn, tree, forest, svc]:
    clf.fit(X_train,Y_train)
    y_pred_train = clf.predict(X_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__+" ACCURACY TRAIN: %.4f TEST: %.4f"%(accuracy_score(Y_train,y_pred_train),accuracy_score(Y_test,y_pred)))
X = train.drop("Survived", axis=1).values

Y = train["Survived"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
from sklearn.model_selection import cross_val_score
import numpy as np


estimator_range = range(10, 210, 10)
accuracy_score_train = []
accuracy_score_test = []

for estimator in estimator_range:
    forest = RandomForestClassifier(n_estimators = estimator, random_state=1)
    forest.fit(X_train, Y_train)
    y_pred_train = forest.predict(X_train)
    y_pred = forest.predict(X_test)
    accuracy_score_train.append(accuracy_score(Y_train, y_pred_train))
    accuracy_score_test.append(accuracy_score(Y_test, y_pred))
    
# grafico di n_estimators rispetto all'accuracy

plt.plot(estimator_range, accuracy_score_train, label="Train")
plt.plot(estimator_range, accuracy_score_test,label="Test")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.title("Ottimazione n_estimators")
plt.legend()
plt.show()
accuracy_score_test
max_depth_range = range(1, 21)
accuracy_score_train = []
accuracy_score_test = []

for depth in max_depth_range:
    forest = RandomForestClassifier(n_estimators = 80, max_depth=depth, random_state=1)
    forest.fit(X_train, Y_train)
    y_pred_train = forest.predict(X_train)
    y_pred = forest.predict(X_test)
    accuracy_score_train.append(accuracy_score(Y_train, y_pred_train))
    accuracy_score_test.append(accuracy_score(Y_test, y_pred))
# grafico di max_depth rispetto all'accuracy

plt.plot(max_depth_range, accuracy_score_train, label="Train")
plt.plot(max_depth_range, accuracy_score_test,label="Test")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Ottimazione max_depth")
plt.legend()
plt.show()
accuracy_score_test
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(criterion='gini', n_estimators=80, max_depth=8, random_state=1)

forest.fit(X_train, Y_train)

y_pred_train = forest.predict(X_train)
y_pred = forest.predict(X_test)
accuracy_train = accuracy_score(Y_train, y_pred_train)
accuracy_test = accuracy_score(Y_test, y_pred)

print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train,accuracy_test))

feature_cols = test.columns.tolist()
#feature_cols = ["Sex","Pclass","Age","Fare"]
# importanza delle caratteristiche
feature_importances = pd.DataFrame({'feature':feature_cols, 'importance':forest.feature_importances_}).sort_values('importance',ascending=False)
feature_importances
col_drop = ["SibSp","Parch"]
train = train.drop(columns=col_drop)
test = test.drop(columns=col_drop)
X = train.drop("Survived",axis=1)

Y = train["Survived"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(criterion='gini', n_estimators=80, max_depth=8, random_state=1)

forest.fit(X_train, Y_train)

y_pred_train = forest.predict(X_train)
y_pred = forest.predict(X_test)


accuracy_train = accuracy_score(Y_train, y_pred_train)
accuracy_test = accuracy_score(Y_test, y_pred)

print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train,accuracy_test))
y_pred = forest.predict(test)

y_pred
#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':y_pred})

#Visualize the first 5 rows
submission.head()
#Convert DataFrame to a csv file that can be uploaded
#This is saved in the same directory as your notebook
filename = 'Passenger Survived.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)

