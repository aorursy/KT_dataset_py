!pip install dabl
import pandas as pd
import dabl
import seaborn as sns
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
# load data
train = pd.read_csv("/kaggle/input/titanic/train.csv")
pred = pd.read_csv("/kaggle/input/titanic/test.csv")

train_count = len(train)
print("Train Samples : ",len(train))
print("Evaluation Samples :", len(pred))
# drop cabin and ticket
train = train.drop(["PassengerId", "Cabin", "Ticket"], axis=1)
pred_id = pred["PassengerId"]
pred = pred.drop(["PassengerId", "Cabin", "Ticket"], axis=1)

# combine train and pred for feature engineering 
df = pd.concat([train, pred]).reset_index(drop=True)
print("Total Samples :", len(df))
df.head()
# analyze feature types
types = dabl.detect_types(df)

continuous_features = types[types["continuous"]].index.values.tolist()
categorical_features = types[types["categorical"]].index.values.tolist()
low_card_features = types[types["low_card_int"]].index.values.tolist()
types
# move low cardinality features - SibSp and Parch to continuous features
continuous_features += low_card_features
print("Continuous Features :", continuous_features)
print("Categorical Features :", categorical_features)
# remove outlier using interquartile range
outlier_indices = []

for feature in continuous_features:
    # 1st & 3rd quartile and IQR range
    Q1, Q3 = np.percentile(df[feature], [25, 75])
    IQR = Q3 - Q1

    # finding index of outliers
    outlier_step = 1.5 * IQR
    outlier_list_col = df[(df[feature] < Q1 - outlier_step) | (df[feature] > Q3 + outlier_step )].index
    outlier_indices.extend(outlier_list_col)

# observations with more than 2 outliersR
outlier_indices = Counter(outlier_indices)
multiple_outliers = list( k for k, v in outlier_indices.items() if v > 2 )
print("Outliers : ", len(multiple_outliers))

# drop outliers
df = df.drop(multiple_outliers, axis=0).reset_index(drop=True)
# fill missing data
print(df.isnull().sum())

# fill age by median age grouped over Pclass & Sex
df['Age'] = df.groupby(
    ['Pclass', 'Sex'])['Age'].apply(lambda x: x.fillna(x.median()))

# fill embarked with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# fill fare by median fare grouped over class
df['Fare'] = df.groupby(
    ['Pclass', 'Embarked'])['Fare'].apply(lambda x: x.fillna(x.median()))
df.isnull().sum()
sns.distplot(df["Age"])
np.quantile(df["Age"], [0, .25, .50, .75])
sns.distplot(df["Fare"])
np.quantile(df["Fare"], [0, .25, .50, .75])
# Feature Engineering
# Adding Title Column
title_dictionary = {
    'Capt': 'Dr/Clergy/Mil', 'Col': 'Dr/Clergy/Mil', 'Major': 'Dr/Clergy/Mil',
    'Jonkheer': 'Honorific', 'Don': 'Honorific', 'Dona': 'Honorific', 
    'Sir': 'Honorific', 'Dr': 'Dr/Clergy/Mil', 'Rev': 'Dr/Clergy/Mil',
    'the Countess': 'Honorific', 'Mme': 'Mrs', 'Mlle': 'Miss',
    'Ms': 'Mrs', 'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss',
    'Master': 'Master', 'Lady': 'Honorific'
}

# Extract and Map
df['Title'] = df['Name'].map(
    lambda name: name.split(',')[1].split('.')[0].strip())
df['Title'] = df['Title'].map(title_dictionary)


# Adding Age Bins
names = ['<5', '5-18', '18-35', '35-65', '>65']
df['AgeBin'] = pd.qcut(df['Age'], q=5, labels=names)

# Adding Fare Bins
bins = np.quantile(df["Fare"], [0, .25, .50, .75]).tolist()
bins.append(200)
names = ["fare_00", "fare_25", "fare_50", "fare_75"]
df['FareBin'] = pd.cut(df['Fare'], bins,
                             labels=names)

# Adding Alone & Family Size Column
df['FamilySize'] = df['Parch'] + df['SibSp'] + 1
df['Alone'] = df['FamilySize'].map(lambda s: "alone" if s == 1 else "not alone")

df.head(20)
# drop unnecessary columns
survived = df["Survived"]
df = df.drop(["Survived", "Name", "Age", "SibSp", "Parch", "Fare"], axis=1)
# df = df.drop(["Survived", "Name", "SibSp", "Parch"], axis=1)
df.head(10)
# analyze feature types
types = dabl.detect_types(df)

continuous_features = types[types["continuous"]].index.values.tolist()
categorical_features = types[types["categorical"]].index.values.tolist()
low_card_features = types[types["low_card_int"]].index.values.tolist()
continuous_features += low_card_features
types
# one hot encoding
df = pd.get_dummies(df, columns=categorical_features, prefix=categorical_features)
# final shapes

# final train test and pred set
X = df.iloc[:train_count]
y = survived.iloc[:train_count]
X_pred = df.iloc[train_count:]

print("Final Data")
print("X:", X.shape, " y:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=69)
print("X_train:", X_train.shape, " y_train:", y_train.shape)
print("X_test:", X_train.shape, " y_test:", y_train.shape)
print("X_pred:", X_pred.shape)
# network
model = Sequential()

model.add(Dense(256, activation="relu", input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(16, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(1, activation="sigmoid"))
model.summary()
# training with low learning rate and high epochs
model.compile(
    optimizer=Adam(lr=0.00015),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


history = model.fit(
    X_train, y_train, 
    epochs=50,
    validation_data=(X_test, y_test)
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# predict for submission

pred_survived = model.predict(X_pred)
pred_survived = np.rint(pred_survived).reshape((-1)).astype(int)

# create csv
submission = pd.DataFrame(zip(pred_id, pred_survived), columns=['PassengerId', 'Survived'])
submission.to_csv('submission.csv', index=False)
submission.head(20)
