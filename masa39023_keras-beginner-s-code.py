import pandas as pd
import csv as csv

train_df = pd.read_csv("../input/train.csv", header=0)

train_df["Gender"] = train_df["Sex"].map({"female": 0, "male": 1}).astype(int)
train_df.head(3)

median_age = train_df["Age"].dropna().median()
if len(train_df.Age[train_df.Age.isnull()]) > 0:
    train_df.loc[(train_df.Age.isnull()), "Age"] = median_age

# sib+parch+1
train_df['family_size'] = train_df['SibSp'] + train_df['Parch'] + 1

# isAlone
train_df['isAlone'] = train_df['family_size'][train_df['family_size'] == 1]
train_df.loc[(train_df.isAlone.isnull()), 'isAlone'] = 0

# familyflag
def name_classifier(name_df):
    name_class_df = pd.DataFrame(columns={'miss', 'mrs', 'master', 'mr'})
    for name in name_df:
        if 'Miss' in name:
            df = pd.DataFrame([[1,0,0,0]], columns={'miss', 'mrs', 'master', 'mr'})
        elif 'Mrs' in name:
            df = pd.DataFrame([[0,1,0,0]], columns={'miss', 'mrs', 'master', 'mr'})
        elif 'Master' in name:
            df = pd.DataFrame([[0,0,1,0]], columns={'miss', 'mrs', 'master', 'mr'})
        elif 'Mr' in name:
            df = pd.DataFrame([[0,0,0,1]], columns={'miss', 'mrs', 'master', 'mr'})
        else:
            df = pd.DataFrame([[0,0,0,0]], columns={'miss', 'mrs', 'master', 'mr'})
        name_class_df = name_class_df.append(df, ignore_index=True)
    return name_class_df

train_df = pd.concat((train_df,name_classifier(train_df['Name'])), axis=1)

train_df = train_df.drop(["Name", "Ticket", "Sex", "SibSp", "Parch", "Cabin","Embarked", "PassengerId"], axis=1)

#######################################################################

test_df = pd.read_csv("../input/test.csv", header=0)
test_df["Gender"] = test_df["Sex"].map({"female": 0, "male": 1}).astype(int)

median_age = test_df["Age"].dropna().median()
if len(test_df.Age[test_df.Age.isnull()]) > 0:
    test_df.loc[(test_df.Age.isnull()), "Age"] = median_age

# sib+parch+1
test_df['family_size'] = test_df['SibSp'] + test_df['Parch'] + 1

# isAlone
test_df['isAlone'] = test_df['family_size'][test_df['family_size'] == 1]
test_df.loc[(test_df.isAlone.isnull()), 'isAlone'] = 0

# familyflag
test_df = pd.concat((test_df,name_classifier(test_df['Name'])), axis=1)

ids = test_df["PassengerId"].values
test_df = test_df.drop(["Name", "Ticket", "Sex", "SibSp", "Parch", "Cabin", "Embarked", "PassengerId"], axis=1)
########################################################################

train_data = train_df.values
test_data = test_df.values

# keras
import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(9, activation='relu', input_dim=10))
model.add(Dense(9, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model.fit(train_data[:,1:], train_data[:,0], batch_size=32, epochs=200)

y_pred = model.predict(test_data)
y_final = (y_pred > 0.5).astype(int).reshape(test_data.shape[0])
output = y_final

submit_file = open("./submit.csv", "w")
file_object = csv.writer(submit_file)
file_object.writerow(["PassengerId", "Survived"])
file_object.writerows(zip(ids, output))
submit_file.close()
train_df.head(5)