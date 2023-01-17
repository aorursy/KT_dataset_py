import pandas as pd

import numpy as np



path = "../input/adult-pmr3508/"

df = pd.read_csv(f"{path}train_data.csv", index_col="Id", na_values="?")



df.head(10)
df.drop(columns=['education'], inplace=True)

df.info()
df.describe()
df.describe(exclude=[np.number])
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

df['income'] = enc.fit_transform(df['income'])



import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(20,20))

sns.heatmap(df.corr(), annot=True, cmap='rocket_r', vmin=-1)
df.drop(columns=['fnlwgt', 'native.country'], inplace=True)
df.isnull().sum()
for col in df.columns:

    df[col] = df[col].fillna(df[col].mode()[0])
# Lista as features de cada tipo

categorical = list(df.select_dtypes(exclude=[np.number]))

numerical = list(df.select_dtypes(include=[np.number]))

print(categorical)

print(numerical)
numerical.remove('income')

numerical.remove('capital.gain')

numerical.remove('capital.loss')
from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder()



df[categorical] = enc.fit_transform(df[categorical])
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



df[numerical + categorical] = scaler.fit_transform(df[numerical + categorical])

df.head()
Y_train = df.pop('income')

X_train = df
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score



min_k = 2

max_k = 30

best_k = 0

best_acc = 0
for k in range(min_k, max_k):

    knn = KNeighborsClassifier(k)

    score = cross_val_score(knn, X_train, Y_train, cv=5, n_jobs=-1, scoring='accuracy')

    print(f"{k}: {score.mean()} +/- {score.std()}")

    if score.mean() > best_acc:

        best_acc = score.mean()

        best_k = k
knn = KNeighborsClassifier(best_k)

knn.fit(X_train, Y_train)
test_df = pd.read_csv(f"{path}test_data.csv", index_col="Id", na_values="?")

test_df.head()
# Remove as features ignoradas

test_df.drop(columns=['fnlwgt', 'education', 'native.country'], inplace=True)



# Preenche valores não existentes com a moda de cada coluna

for col in test_df.columns:

    test_df[col] = test_df[col].fillna(test_df[col].mode()[0])



# Codifica variáveis categóricas

categorical = list(test_df.select_dtypes(exclude=[np.number]))

test_df[categorical] = enc.fit_transform(test_df[categorical])



# Normaliza as features desejadas

numerical = list(test_df.select_dtypes(include=[np.number]))

numerical.remove('capital.gain')

numerical.remove('capital.loss')

test_df[numerical + categorical] = scaler.fit_transform(test_df[numerical + categorical])



test_df.head()
predictions = knn.predict(test_df)
predictions_class = np.array([{0: '<=50K', 1: '>50K'}[i] for i in predictions], dtype=object)

print(predictions_class)
submission = pd.DataFrame()

submission[0] = test_df.index

submission[1] = predictions_class

submission.columns = ['Id', 'income']



submission.head()
submission.to_csv('submission.csv', index=False)