import pandas as pd

#gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test_df = pd.read_csv("../input/titanic/test.csv")

train_df = pd.read_csv("../input/titanic/train.csv")
frames = [train_df, test_df]
combined_df=pd.concat(frames, ignore_index=True, sort = False)
combined_df.head()
combined_df.columns
train_df.isnull().sum()
print(test_df.isnull().sum())
combined_df.isnull().sum()
combined_df.dtypes
combined_df[['Age','Fare','SibSp','Parch']].describe()
train_df[['Age','Fare','SibSp','Parch']].describe()
test_df[['Age','Fare','SibSp','Parch']].describe()
#combined_df[['PassengerId','Survived','Pclass','Name','Sex','Ticket','Cabin','Embarked']].count()
#combined_df[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']].nunique(dropna=True)
cat_combined_df=combined_df[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']].astype('category')
cat_combined_df.describe()
cat_train_df=train_df[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']].astype('category')
cat_train_df.describe()
cat_test_df=test_df[['Pclass','Sex','Ticket','Cabin','Embarked']].astype('category')
cat_test_df.describe()