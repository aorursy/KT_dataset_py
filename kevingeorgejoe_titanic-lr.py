import pandas as pd
from sklearn.preprocessing import LabelEncoder
# get titanic training file as a DataFrame
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.shape
# preview the data
train.head()
train.Ticket.nunique()
colsToDrop = ["Name","Ticket", "PassengerID"]
train.describe()
train.info()
embarkedEncoder = LabelEncoder()
sexEncoder = LabelEncoder()
train["Embarked"] = embarkedEncoder.fit_transform(train["Embarked"].fillna("Nan"))
train["Sex"] = sexEncoder.fit_transform(train["Sex"].fillna("Nan"))
train.head()
y = train["Survived"] # copy “y” column values out
X = train.drop(['Survived'], axis=1) # then, drop y column
X.drop(['Ticket'], axis=1, inplace=True) 
X.drop(['Name'], axis=1, inplace=True) 
X.drop(['PassengerId'], axis=1, inplace=True)
X.drop(['Cabin'], axis=1, inplace=True)
X.info()
X.isnull().values.any()
X.isnull().any()
X["Age"].fillna(X.Age.mean(), inplace=True)  # replace NaN with average age
X.isnull().any()
from sklearn.model_selection import train_test_split
  # 80 % go into the training test, 20% in the validation test
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_train, y_train)
model.score(X_valid, y_valid)
model.intercept_ # the fitted intercept
model.coef_  # the fitted coefficients