# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Import things that I need
df = pd.read_csv("../input/titanic/train.csv")
display(df)

# Read train file
null_value = df.isnull().sum()
print(null_value)

# Check null values I can see some null values in this data
df_drop = df.dropna(axis=0)
print(df_drop.isnull().sum())
display(df_drop)

# In this case, I just use dropna for drop all rows that have null values. Than I can check there are no null values after I did dropna process
drop_val = ["PassengerId","Name","Ticket"]

df_1 = df_drop.drop(drop_val,axis=1)
display(df_1)

# I think those things are not affect to result so I drop those columns
def encoder(x):
    if "female" in x:
        return 0
    else:
        return 1

# For doing machine learning, we should encode all char values to num values so I defined function that encode male, female in to 0 and 1.
# In this case I define "Female" as 0 and "Male" as 1
sex_num = df_1.Sex.apply(lambda x : encoder(x)).to_frame()
df_2 = df_1.drop(labels="Sex",axis=1)
df_2["Sex"] = sex_num
display(df_2)

# I apply encoder function to "Sex" column.
embark_encoded = pd.get_dummies(df_2.Embarked)
display(embark_encoded)
df_3 = pd.concat([df_2,embark_encoded],axis=1)
df_3.drop(labels="Embarked",axis=1,inplace=True)
display(df_3)

# When we see that Embarked columns there are three values. C,Q,S. So I used "pd.get_dummies" to encode char to num.
#  After I made "embark_encoded", I use concat to sum up with df_2 and defined new dataframe called df_3
cabin_encoded = pd.get_dummies(df_3.Cabin)
display(cabin_encoded)

# I also did same thing on "cabin" column. 
df_4 = pd.concat([df_3,cabin_encoded],axis=1)
df_4.drop(labels="Cabin",axis=1,inplace=True)
display(df_4)
scaler = MinMaxScaler()
df_5 = scaler.fit_transform(df_4)
df_5 = pd.DataFrame(df_5,columns=df_4.columns)
display(df_5)

# OK we made all columns char to num. After this process we should see whether some values are too high than other values.
# We can check that "Age" and "Fare" columns have high values than other columns. So we should do some normalization.
# In this case I use "MinMaxScaler" and made new dataframe df_5.
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# OK we finished preprocessing. We should import model that we can use. I use "RandomForestClassifier", "AdaBoostClassifier" and "SVC"
# And for scoring, I import "accuracy_score" and "f1_score"
X = df_5.iloc[:,1:]
y = df_5.iloc[:,0]

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=62)

model1 = RandomForestClassifier()
model1.fit(X_train,y_train)
model1_preds = model1.predict(X_test)
accuracy1 = accuracy_score(y_test,model1_preds)
f1_1 = f1_score(y_test,model1_preds)
print("Accuracy score of RFC : {} F1 score : {}".format(accuracy1,f1_1))

# First, I made RandomForestClassifier as model1 and check accuracy and f1 score. 
model2 = AdaBoostClassifier()
model2.fit(X_train,y_train)
model2_preds = model2.predict(X_test)
accuracy2 = accuracy_score(y_test,model2_preds)
f1_2 = f1_score(y_test,model2_preds)
print("Accuracy score of AdaBoost : {} F1 score : {}".format(accuracy2,f1_2))

# Second, I made AdaBoostClassifier as model2. It has higher accuracy score and f1 score than model1
model3 = SVC()
model3.fit(X_train,y_train)
model3_preds = model3.predict(X_test)
accuracy3 = accuracy_score(y_test,model3_preds)
f1_3 = f1_score(y_test,model3_preds)
print("Accuracy score of SVC : {} F1 score : {}".format(accuracy3,f1_3))

# Third, I made SVC model as model3 and check accuracy and f1 score. It is lower than model2.
#I select Model2(AdaBoost)

from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import make_scorer
import time

# As a result, I choosed model2 as final model and I will do GridSearch to find best hyperparameters.
parameters = {
    "learning_rate" : np.arange(0.1,1,0.01)
}

scorer = make_scorer(f1_score)

gs = GridSearchCV(estimator=model2,param_grid=parameters,scoring=scorer,n_jobs=6)
start = time.time()
gs.fit(X_train,y_train)
end = time.time()
print("Search Time : {} seconds".format(end-start))

# I defined scoring method as f1 score and my cpu has 6 cpu core so n_jobs is 6. 
# I did gridsearch with "n_estimators" but with those things, f1 score become lower than default value of AdaBoostClassifier. 
# So I just defined learning_rate in parameters.
# Check time and it just took 23 seconds.
model4 = gs.best_estimator_
model4_preds = model4.predict(X_test)
accuracy4 = accuracy_score(y_test,model4_preds)
f1_4 = f1_score(y_test,model4_preds)
print("Best estimator accuracy score : {} f1 score : {}".format(accuracy4,f1_4))