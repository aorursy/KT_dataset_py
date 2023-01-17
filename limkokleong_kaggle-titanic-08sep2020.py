import pandas as pd 

train_df = pd.read_csv('../input/titanic/train.csv')
train_df.head()
train_df.describe()

# Looking at the mean, 25%, 50% and 75% quantile of Age, it seems like most passengers are young. But at 1909, life expectancy is not very long.
# In Britain at 1900, life expectancy is about 47 for man, 50 for women. 

# For Fare, it seems like there are people who did not pay any ticket fare, having the minimum as $0. 
# Checking for null values in train_df
print(train_df.isnull().sum())

# 177 null values for Age, 687 null values for Cabin, 2 null values for Embarked
print("---Survived---")
print(train_df.Survived.value_counts())

# 549 did not survive
# 342 survived

print("---Pclass---")
print(train_df.Pclass.value_counts(sort=False))

# 3 (Lower Class) 491
# 2 (Middle Class) 184
# 1 (Upper Class) 216

print("---Sex---")
print(train_df.Sex.value_counts())

# Male 577
# Female 314
print("---Age---")
print(train_df.Age.value_counts(bins=10,sort=False))

# More than half was below 49 years old. This confirms our observation above.

print("---SibSp---")
print(train_df.SibSp.value_counts())

# Many did not have their spouse or sibling with them. Some had either their sibling or spouse with them.
print("---Parch---")
print(train_df.Parch.value_counts())

# Similar to SibSp, many only had either no parent and children with them, or just one of them.

print("---Fare---")
print(train_df.Fare.value_counts(bins=8))
print("$0 Fare", (train_df.Fare == 0).sum())
# Most of the people got their fare below $64.
print("--Cabin--")
print(train_df.Cabin.value_counts())

# It seems like those who have cabins are split into different letters.


print("---Embarked--")
print(train_df.Embarked.value_counts(dropna=False))


# Split the honorifics from column "Name"

train_df["Honorific"] = train_df["Name"].apply(lambda x: x.split(',')[1].split('.')[0].strip())
train_df["Honorific"].value_counts()

# Master is an English honorific for boys and young men
# Rev is Reverend (Christen ministers)
# Dr is Doctor 
# Major, Capt and Col is army Rank 
# Mme is Miss 
# The countess is wives of knights 
# Lady for members of nobility 
# Don is for European nobles
# Sir is english nobile/knights 
# Jonkheer nobility 
# To tidy up the honorifics

def title(x):
    if x in ["Mr","Miss","Mrs","Master"]:
        return x
    elif x in ["Ms","Mme"]:
        return "Miss"
    else:
        return "Nobles"
    
train_df["Title"] = train_df["Honorific"].apply(title)
print(train_df["Title"].value_counts())
train_df.head()
train_df_cleaned = train_df.drop(columns=["Name","Ticket","Cabin","Honorific"])
train_df_cleaned.dropna(inplace=True) 
print("Length of Original dataset (train_df)", len(train_df), "\n"
     "Length of Cleaned dataset (train_df_cleaned)",len(train_df_cleaned))
train_df_cleaned.head()
# Checking for multicollinearity
train_df_cleaned.corr()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Feature Selection
X_train_df = train_df_cleaned.drop(columns=["Survived","PassengerId"]).reset_index(drop=True)
y_train_df = train_df_cleaned["Survived"]

# One Hot Encoding
enc = OneHotEncoder().fit(X_train_df[["Sex","Embarked","Title"]])
enc_df = pd.DataFrame(enc.transform(X_train_df[["Sex","Embarked","Title"]]).toarray(),
                      columns=enc.get_feature_names(["Sex","Embarked","Title"]))
X_train_df_encoded = X_train_df.join(enc_df).drop(columns=["Sex","Embarked","Title"])

# Train test split 
X_train, X_test, y_train, y_test = train_test_split(X_train_df_encoded, y_train_df,random_state=0)

# Scaling 
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_df_encoded
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

logreg = LogisticRegression()
logreg.fit(X_train_scaled,y_train)
plot_confusion_matrix(logreg,X_test_scaled,y_test,values_format='.0f',cmap="Blues")
plt.title("Logistic Regression Confusion Matrix - Test Set")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")

print("Accuracy score (train set) = {:.3f}".format(accuracy_score(y_train,logreg.predict(X_train_scaled))))
print("Accuracy score (test set) = {:.3f}".format(accuracy_score(y_test,logreg.predict(X_test_scaled))))
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree

full_tree = DecisionTreeClassifier()
full_tree.fit(X_train_scaled,y_train)
print("Depth of Full Tree:", full_tree.tree_.max_depth)
print("Full tree accuracy:", full_tree.score(X_test_scaled,y_test))

plt.figure(figsize=(20,8))
tree.plot_tree(full_tree, filled=True, impurity=True)
plt.show()
# Pruned Tree
pruned_acc = 0

for c in range(1,full_tree.tree_.max_depth):
    pruned_tree = DecisionTreeClassifier(max_depth=c,random_state=0)
    pruned_tree.fit(X_train_scaled,y_train)
    if pruned_tree.score(X_test_scaled,y_test)>pruned_acc:
        pruned_acc = pruned_tree.score(X_test_scaled,y_test)
        best_depth = c
print("For best pruned tree, depth =", best_depth,",pruned accuracy:", pruned_acc)
from sklearn.metrics import plot_confusion_matrix

# Best Pruned Tree

best_pruned_tree = DecisionTreeClassifier(max_depth=best_depth,random_state=0)
best_pruned_tree.fit(X_train_scaled,y_train)


plot_confusion_matrix(best_pruned_tree,X_test_scaled,y_test,cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Pruned Tree with highest accuracy")
print("Accuracy:",best_pruned_tree.score(X_test_scaled,y_test))
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix

rf = RandomForestClassifier(n_estimators=20,random_state=0)
rf.fit(X_train_scaled,y_train)

plot_confusion_matrix(rf,X_test_scaled,y_test,cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Random Forest")

print("Accuracy:", rf.score(X_test_scaled,y_test))
from sklearn.ensemble import GradientBoostingClassifier

xgb = GradientBoostingClassifier(random_state=0)
xgb.fit(X_train_scaled,y_train)

plot_confusion_matrix(xgb,X_test_scaled,y_test,cmap=plt.cm.Blues)
plt.title("Confusion Matrix - XG Boost")

print("Accuracy", xgb.score(X_test_scaled,y_test))
test_df = pd.read_csv("../input/titanic/test.csv")
test_df
test_df["Honorific"] = test_df["Name"].apply(lambda x: x.split(',')[1].split('.')[0].strip())
test_df["Title"] = test_df["Honorific"].apply(title)
test_df_cleaned = test_df.drop(columns=["Name","Ticket","Cabin","Honorific"])

# Feature Selection
test_df_cleaned = test_df_cleaned.drop(columns=["PassengerId"]).reset_index(drop=True)
test_df_cleaned.fillna(test_df_cleaned["Age"].mean(),inplace=True)

# One Hot Encoding
encoded_df = pd.DataFrame(enc.transform(test_df_cleaned[["Sex","Embarked","Title"]]).toarray(),
                      columns=enc.get_feature_names(["Sex","Embarked","Title"]))
test_df_encoded = test_df_cleaned.join(encoded_df).drop(columns=["Sex","Embarked","Title"])

# Scaling
test_df_scaled = scaler.transform(test_df_encoded)

test_ypred = xgb.predict(test_df_scaled)
test_ypred
test_ypred_df = pd.DataFrame(test_ypred,columns=["Survived"])
test_ypred_df
titanic_results = test_df[["PassengerId"]].join(test_ypred_df)
titanic_results
titanic_results.to_csv("titanic_results.csv",index=False)
