import numpy as np

import pandas as pd

import tensorflow as tf

from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing

import matplotlib.pyplot as plt



from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout



import seaborn as sns

%matplotlib inline
# fix random seed for reproducibility

seed = 7

np.random.seed(seed)
df = pd.read_csv('../input/train.csv')

df_subm = pd.read_csv('../input/test.csv')

df.describe(include = 'all')
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=df,

              palette={"male": "blue", "female": "red"});
def feat_sex(data, data_subm):

    le = preprocessing.LabelEncoder()

    le.fit(data.Sex)

    data['sex_normalized'] = le.transform(data.Sex)

    data_subm['sex_normalized'] = le.transform(data_subm.Sex)

    return data, data_subm
def feat_ages(data, data_subm):

#    data.loc[(data['Name'].str.contains('Mr.')) & (data.Age.isnull()), 'Age'] = 30

#    data.loc[(data['Name'].str.contains('Mrs.')) & (data.Age.isnull()), 'Age'] = 35

#    data.loc[(data['Name'].str.contains('Miss.')) & (data.Age.isnull()), 'Age'] = 23

#    data.loc[data.Age.isnull(), 'Age'] = 30

    

#    data_subm.loc[(data_subm['Name'].str.contains('Mr.')) & (data_subm.Age.isnull()), 'Age'] = 30

#    data_subm.loc[(data_subm['Name'].str.contains('Mrs.')) & (data_subm.Age.isnull()), 'Age'] = 35

#    data_subm.loc[(data_subm['Name'].str.contains('Miss.')) & (data_subm.Age.isnull()), 'Age'] = 23

#    data_subm.loc[(data_subm.Age.isnull()), 'Age'] = 30

    data['Age'] = data.Age.fillna(30)

    data_subm['Age'] = data_subm.Age.fillna(30)

    bins = (0, 5, 12, 18, 25, 35, 60, 120)

    group_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']



    le = preprocessing.LabelEncoder()

    categories = pd.cut(data.Age, bins, labels=group_names)

    le.fit(categories)

    data['age_bins'] = le.transform(categories)

    categories = pd.cut(data_subm.Age, bins, labels=group_names)

    data_subm['age_bins'] = le.transform(categories)

    return data, data_subm
def feat_name(data, data_subm):

    le = preprocessing.LabelEncoder()

    tmp = data.Name.apply(lambda x: x.split(' ')[0]).values

    tmp = np.append(tmp, data_subm.Name.apply(lambda x: x.split(' ')[0]).values)

    le.fit(tmp)

    data['last_name'] = le.transform(data.Name.apply(lambda x: x.split(' ')[0]))

    data_subm['last_name'] = le.transform(data_subm.Name.apply(lambda x: x.split(' ')[0]))



    le = preprocessing.LabelEncoder()

    tmp = data.Name.apply(lambda x: x.split(' ')[1])

    tmp = np.append(tmp, data_subm.Name.apply(lambda x: x.split(' ')[1]))

    le.fit(tmp)

    data['name_prefix'] = le.transform(data.Name.apply(lambda x: x.split(' ')[1]))

    data_subm['name_prefix'] = le.transform(data_subm.Name.apply(lambda x: x.split(' ')[1]))

    return data, data_subm
def feat_fares(data, data_subm):

    data.Fare = data.Fare.fillna(-0.5)

    data_subm.Fare = data_subm.Fare.fillna(-0.5)

    bins = (-1, 0, 8, 15, 31, 1000)

    group_names = ['Unknown', 'A', 'B', 'C', 'D']



    le = preprocessing.LabelEncoder()

    categories = pd.cut(data.Fare, bins, labels=group_names)

    le.fit(categories)

    data['fare_bins'] = le.transform(categories)

    categories = pd.cut(data_subm.Fare, bins, labels=group_names)

    data_subm['fare_bins'] = le.transform(categories)

    return data, data_subm
df, df_subm = feat_sex(df, df_subm)

df, df_subm = feat_ages(df, df_subm)

df, df_subm = feat_name(df, df_subm)

df, df_subm = feat_fares(df, df_subm)



df_train = df.iloc[:int(len(df) * 0.7), :]



# removed 'last_name', 'Age', 'Fare'

features = ['Pclass', 'sex_normalized', 'Parch', 'SibSp', 'age_bins', 'fare_bins', 'name_prefix']



y_train = df_train.Survived.values

y_train_onehot = pd.get_dummies(df_train.Survived).values
print(y_train[0:3])

y_train_onehot[0:3]
df_train[features].head()
scaler = StandardScaler()

tmp = df_train[features].values

print(tmp[0])

X_train = scaler.fit_transform(np.nan_to_num(tmp))

X_train[0]
tmp = df_subm[features].values

print(tmp[0])

X_subm = scaler.fit_transform(np.nan_to_num(tmp))

X_subm[0]
df_test = df.iloc[int(len(df) * 0.7):, :]



tmp = df_test[features].values

print(tmp[0])

X_test = scaler.transform(np.nan_to_num(tmp))

y_test = df_test.Survived.values

X_test[0]
model = Sequential()

model.add(Dense(input_dim=len(X_train[0]), units=100))

#model.add(Dropout(0.2))

model.add(Dense(units=2))

model.add(Activation("softmax"))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



model.fit(X_train, y_train_onehot, epochs=10, verbose=1)
predicted = model.predict_classes(X_subm)



# Write data

sol = pd.DataFrame()

sol['PassengerId'] = df_subm['PassengerId']

sol['Survived'] = pd.Series(predicted.reshape(-1)).map({True:1, False:0})

sol.to_csv('mfa-keras-2-solution.csv', index=False)