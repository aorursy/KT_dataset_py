#import codecademylib3_seaborn

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
df= pd.read_csv("../input/final-psngrs/titanic_train_duplicate.csv")
df.shape
#Update sex column to numerical

df["sex"] = df["Sex"].map({"male": 0, "female":1})

print(df["sex"].head())
# Fill the nan values in the age column

df["Age"].fillna(value = round(df["Age"].mean()), inplace = True )

print(df["Age"])
# Create a first class column

df["FirstClass"] = df["Pclass"].apply(lambda p: 1 if p==1 else 0)



# Create a second class column

df["SecondClass"] = df["Pclass"].apply(lambda p: 1 if p==2 else 0)
print(df.head(10))
# Select the desired features

features = df[["sex", "Age", 'FirstClass', "SecondClass"]]

survival = df["Survived"]
#print(survival.head(5))

# Perform train, test, split

train_features, test_features , train_labels, test_labels = train_test_split(features,survival, test_size = 0.2,  )

# Scale the feature data so it has mean = 0 and standard deviation = 1

scaler = StandardScaler()

train_features = scaler.fit_transform(train_features)

test_features = scaler.transform(test_features)
# Create and train the model

model = LogisticRegression()

model.fit(train_features,train_labels)
# Score the model on the train data

print(model.score(train_features, train_labels))

# Score the model on the test data

print(model.score(test_features, test_labels))

# Analyze the coefficients

print(features.columns)

print(model.coef_)

print(model.intercept_)
# Sample passenger features

Jack = np.array([0.0,20.0,0.0,0.0])

Rose = np.array([1.0,17.0,1.0,0.0])

You  = np.array([0.0,24.0,0.0,1.0])

your_gf = np.array([1.0,24.0,1.0,0.0])

# Combine passenger arrays

sample_passengers = np.array([Jack, Rose, You, your_gf])
# Scale the sample passenger features

sample_passengers = scaler.transform(sample_passengers)
print(sample_passengers)

# Make survival predictions!

print(model.predict(sample_passengers))