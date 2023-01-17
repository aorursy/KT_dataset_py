import pandas as pd
df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

df.head()
x = df.drop('Survived', axis=1)

y = df.Survived
pathOfSubmissionFile = ""

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })



submission.to_csv(pathToSubmission, index=False)