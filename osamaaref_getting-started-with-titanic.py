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
train_data = pd.read_csv("../input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("../input/titanic/test.csv")

test_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)


y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])

from sklearn.ensemble import RandomForestClassifier



def random_forest():

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

    model.fit(X, y)

    return model



model = random_forest()

from tensorflow import keras



model = keras.Sequential([keras.layers.Dense(28, activation="relu", input_shape=(X.shape[-1],)),

                          keras.layers.Dense(28, activation="relu"),

                          keras.layers.Dense(2, activation='sigmoid')])

model.summary()



model.compile(

    optimizer="adam", loss="binary_crossentropy", metrics="accuracy"

)



x_con = X.to_numpy(dtype="float32")

y_cat = keras.utils.to_categorical(y, num_classes=2, dtype="float32")



print(x_con.shape)

print(y_cat.shape)

#print(x_con[0])

print(y_cat[0:5])

print(y[0:5].to_list())



model.fit(x_con,

    y_cat,

    #validation_split=0.2,

    batch_size=32,

    epochs=10

    )
X.shape[-1]
results = model.predict(X_test.to_numpy(dtype="float32"))



predictions=[]

for row in results:

    if(row[0] > row[1]):

        predictions.append(0)

    else:

        predictions.append(1)

        



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")