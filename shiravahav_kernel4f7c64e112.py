# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv") #store my data in the variable "train_data"

train_data.head() #show only the head of the data
test_data = pd.read_csv("/kaggle/input/titanic/test.csv") #store my data in the variable "test_data"

test_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X_test = pd.get_dummies(test_data[features])#הפוך את כל הערכים ל-0 ואחד. המרה ממחרוזת או מספר רציף. מסדר אותם

X = pd.get_dummies(train_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)#100 עצים 

#עומק עד 5 כי יש 4 מאפיינים ועוד הערך שרוצים לחזות. 

#הערך האחרון הוא איך רוצים לפצל את העצים- קבוע או אקראי



model.fit(X, y)#מאמן את המודל למספר קבוע של איטרציות



#מאמן את המודל עם הנתונים של סט האימון ומאפיין ה"מטרה" אותו נרצה לחזות



predictions = model.predict(X_test)#לחזות את המודל על סט הבדיקה(משלים לבד ערכים של שרד/לא- 0/1)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})#הפלט זה הערכים ממסגרת הדאטה שלנו

output.to_csv('my_submission.csv', index=False)#המר את הקובץ ל CSV

print("Your submission was successfully saved!")