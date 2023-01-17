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
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler



# To preprocess all our data. We can add to this later

def preprocess(df):

    df = df[['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']] # We're now interested in a lot of things

    

    def normalize_age(data): # Neural networks work MUCH better when all the data is close to 0, so here we scale it to be from 0 to 1

        data["Age"] = MinMaxScaler().fit_transform(data["Age"].values.reshape(-1,1))

        data["Fare"] = MinMaxScaler().fit_transform(data["Fare"].values.reshape(-1,1))

        return data

    

    df = normalize_age(df)

    

    df = pd.get_dummies(df, columns=['Sex', 'Pclass']) # One hot encode (we create binary colums for male and female)

    

    df = df.dropna() # Drop rows with no data

    

    x_train, x_test, y_train, y_test = train_test_split(df[['Pclass_1', 'Pclass_2', 'Pclass_3','Sex_female', 'Sex_male', 'Age', 'SibSp', 'Parch', 'Fare']], df['Survived'], test_size=0.33, random_state=42) # Our testing set is 1/3 of the original, and we set a random seed so it is the same every time

    

    return x_train, x_test, y_train, y_test



train_df = pd.read_csv("/kaggle/input/titanic/train.csv")



x_train, x_test, y_train, y_test = preprocess(train_df)
x_train.describe() # Our inputs, also called features
y_train.describe() # Our output
import tensorflow as tf



model = tf.keras.models.Sequential([

    tf.keras.layers.Dense(64, activation='relu'),    # relu activation is a piecewise function that is 0 when x < 0 and x when x > 0

                                                    # activation helps a neuron "decide when to fire". Otherwise our output is a whole lot of numbers

                                                    # also makes it a lot faster to calculate. just a general good idea. use relu for 99% of the time

    tf.keras.layers.Dropout(0.2), # Randomly sets input to 0, prevents overfitting (when your model is too tied to your training data)

    tf.keras.layers.Dense(32, activation='relu'), # Our second layer

    tf.keras.layers.Dense(16, activation='relu'), # Our second layer

    tf.keras.layers.Dense(1, activation='sigmoid'), # Our last layer. Notice activation=sigmoid. This gives us the probability of the output from 0 to 1. Our layer is also 1 wide, because we only want to output, if it survived or not

])                                                  # If you want multiple outputs with probabilities, use softmax



model.compile(optimizer='adam', # The algorithm to update our model during training. Just keep this

              loss='binary_crossentropy', # How the model measures how good it is. This is good for binary classification

              metrics=['accuracy']) # How we can monitor its progress



model.fit(np.array(x_train), np.array(y_train), epochs=50) # Epochs is basically how long it runs
model.evaluate(np.array(x_test), np.array(y_test))