import numpy as np

import pandas as pd

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn.model_selection import train_test_split
data = pd.read_csv('/kaggle/input/titanic/train.csv')

data = data[['Sex','Cabin','Embarked','Survived']]

data.head(3)
#checking missing values

data.isnull().sum()
#non imputed data

df_nn = data.copy()

df = data.drop(columns=['Survived'])

y = data['Survived']



# We will fill the missing values

imp = SimpleImputer(strategy="most_frequent")

imp.fit(df)

df[df.columns] = imp.transform(df)

# train test split



X_train,X_test,y_train,y_test = train_test_split(df,y)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
dff = df.copy()

for feat in df.columns:

    dff[feat] = dff[feat].astype('category')

    dff[feat] = dff[feat].cat.codes

dff.head()
dff = df.copy()

for feat in dff.columns:

    lb = LabelEncoder()

    lb.fit(dff[feat])

    dff[feat] = lb.transform(dff[feat])



print(dff.shape)

dff.head()
df_train = X_train.copy()

df_test = X_test.copy()
df_train = pd.get_dummies(df_train)

df_test = pd.get_dummies(df_test)



# aliging the dataframes

df_train, df_test = df_train.align(df_test, join='left', axis=1) 



# checking df_test

df_test.head(3)
df_train = df_train.fillna(0)

df_test = df_test.fillna(0)



print(df_train.shape)

print(df_test.shape)
df_train.head(3)
df_test.head(3)
# df_nn is the dataframe containing NaN values

dff = df_nn.copy()

dff = pd.get_dummies(dff,dummy_na=True)



print(dff.shape)

dff = df.copy()

dff = pd.get_dummies(dff,sparse=True)

print(dff.shape)
df_train = X_train.copy()

df_test = X_test.copy()
one = OneHotEncoder(handle_unknown ='ignore')

one.fit(df_train)

train_vec = one.transform(df_train)

test_vec = one.transform(df_test)
feats = one.get_feature_names()

# sparce to dense

train_vec = train_vec.toarray()

test_vec = test_vec.toarray()





train_df = pd.DataFrame(train_vec,columns=feats)

test_df = pd.DataFrame(test_vec,columns=feats)



print(train_df.shape)

print(test_df.shape)
train_df.head(3)
test_df.head(3)
dff = df.copy()

one = OneHotEncoder(sparse=False)

one.fit(dff)

dff_ = one.transform(dff)

feats = one.get_feature_names()

dff_ = pd.DataFrame(dff_,columns=feats)

print(dff_.shape)
dff = df.copy()

one = OneHotEncoder(drop='first',sparse=False)

one.fit(dff)

dff_ = one.transform(dff)

feats = one.get_feature_names()

dff_ = pd.DataFrame(dff_,columns=feats)

print(dff_.shape)

df_train = X_train.copy()

df_test = X_test.copy()
df['Cabin'].value_counts()


df_train['Cabin_encoded'] = df_train['Cabin'].apply(lambda x: 1 if x=='B96 B98' else 0)

df_test['Cabin_encoded'] = df_test['Cabin'].apply(lambda x: 1 if x=='B96 B98' else 0)

df_train['Cabin_encoded']