#ייבוא המודולים מהספרייה

import pandas as pd

from sklearn.tree import DecisionTreeClassifier
#המרת קבצי הנתונים

train = pd.read_csv("train.csv") 

test = pd.read_csv("test.csv") 
#יצירת עותק קשיח של הקבצים

tr = train.copy(deep = True)

te = test.copy(deep = True)
#הצגת 5 הרשומו העליונות בקובץ הנתונים

tr.head()
#הצגת 5 הרשומות העליונות בקובץ המבחן

te.head()
#'Name', 'Ticket', 'Cabin' הורדת השדות



tr = tr.drop(["Name", "Ticket", "Cabin"], axis=1)

te = te.drop(["Name", "Ticket", "Cabin"], axis=1)
#הצגת קובץ הנתונים לאחר הורדת העמודות שביצענו

tr.head()
#הצגת קובץ המבחן לאחר הורדת העמודות שביצענו

te.head()


# בקובץ הנתונים'Sex','Embarked' לתכונות One Hot Encoding ביצוע

dummies1= pd.get_dummies(tr.Embarked)

dummies2=pd.get_dummies(tr.Sex)

merge=pd.concat([tr,dummies1,dummies2],axis='columns')

new_tr=merge.drop(["Embarked","Sex"],axis='columns')

new_tr.head()



# בקובץ המבחן'Sex','Embarked' לתכונות One Hot Encoding ביצוע

dummies= pd.get_dummies(te.Embarked)

dummies3=pd.get_dummies(te.Sex)

merged=pd.concat([te,dummies,dummies3],axis='columns')

new_te=merged.drop(["Embarked","Sex"],axis='columns')

new_te.head()
#תיאור פרמטרים אודות הנתונים בקובץ האימון

new_tr.describe()
# בקובץ הנתונים החדש null בדיקת ערכי 

new_tr.isnull().sum().sort_values(ascending=True)
#נציב בערכי הגיל הריקים את ממוצע הגילאים

new_tr["Age"].fillna(new_tr["Age"].mean(), inplace=True)

new_te["Age"].fillna(new_te["Age"].mean(), inplace=True)
# תיאור פרמטרים אודות הנתונים בקובץ המבחן

new_te.describe()

#בקובץ המבחן החדש null בדיקת ערכי 

new_te.isnull().sum().sort_values(ascending=True)
#נציב בערכי ההפלגה הריקים את ממוצע ערכע ההפלגה

new_te["Fare"].fillna(new_te["Fare"].mean(), inplace=True)
#הכנת משתנה ההחלטה לקראת השימוש בעץ ההחלטה

X = new_tr.drop("Survived", axis=1)

y = new_tr["Survived"]
#שימוש במודל עץ ההחלטה

tree = DecisionTreeClassifier(max_depth = 10, random_state = 0)

tree.fit(X, y)
#tree.score(X, y)
#ייבוא המודלים המתאימים להמשך העבודה

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split 
#קובץ האימון החדש ללא עמודת משתנה המטרה

X.head()
#נתוני המבחן

Xtest = new_te

Xtest.head()
Xtrain, Xvalidation, Ytrain, Yvalidation = train_test_split(X, y, test_size=0.2, random_state=True)
#שימוש במודל

model = RandomForestClassifier(n_estimators=100,

                               max_leaf_nodes=12,

                               max_depth=12,

                               random_state=0)

model.fit(Xtrain, Ytrain)

#חיזוי התוצאה הצפויה

from sklearn.metrics import accuracy_score

Yprediction = model.predict(Xvalidation)

accuracy_score(Yvalidation, Yprediction)
#הכנת קובץ ההגשה

submission = pd.DataFrame()



submission["PassengerId"] = Xtest["PassengerId"]

submission["Survived"] = model.predict(Xtest)



#CSV-שמירת הקובץ כ

submission.to_csv("submission_9_June.csv", index=False)
submission.head(10)