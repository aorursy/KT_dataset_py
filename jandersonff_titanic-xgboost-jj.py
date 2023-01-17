import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from subprocess import check_output

%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
t = train[(train.Survived==1) & (train.Age>40)]
t.AgePlus10 = t.Age * 10
t.shape
# Correct the age colums

age_mean = train.Age.mean()

age_mean2 = np.mean(train[train.Age.notnull()].Age)

age_mean, age_mean2

test

plt.figure(figsize=(7,4))

fig, axes = plt.subplots(nrows=1, ncols=3)

# plots a bar graph of those who surived vs those who did not.               

#plt.title("Sobreviventes sem acompanhantes")    

train.Survived[train.Parch==0].value_counts().plot(kind='bar', ax=axes[0], title="sem acomp.")



#plt.title("Sobreviventes com acompanhantes")    

train.Survived[train.Parch>0].value_counts().plot(kind='bar' , ax=axes[1],  title="com acomp.")

train.withpart = train.Parch>0

train.withpart[train.Survived==1].value_counts().plot(kind='bar' , ax=axes[2],  title="sobre. com/sem parc.")

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder



features = ['Pclass', 'Sex', 'Age', 'Fare']

convert_str = ['Sex']





joined_df = train[features].append(test[features])



# convert str features to Number 

#le = LabelEncoder()

#for f in convert_str:

#    print(joined_df[f], le.fit_transform(joined_df[f]))

#    joined_df[f] = le.fit_transform(joined_df[f])

joined_df['Sex'] = joined_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



#joined_df = joined_df.dropna(axis=0)

train_X = joined_df[0:train.shape[0]].as_matrix()

test_X = joined_df[train.shape[0]::].as_matrix()



train_y = train['Survived']
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)



predictions = gbm.predict(test_X)



submission = pd.DataFrame({ 'PassengerId': test['PassengerId'],

                            'Survived': predictions })

submission.to_csv("../input/submission.csv", index=False)

print(check_output(["ls", ".."]).decode("utf8"))