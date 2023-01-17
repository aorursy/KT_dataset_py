import numpy as np
import pandas as pd
from sklearn import tree
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
# Visual tabel dari train, 8 baris teratas
train.head(8)
# Info jumlah data terisi dan type data dari masing-masing kolom di dataset train
train.info()
# Impute the Embarked variable
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

# Impute the Age variable
train['Age'] = train['Age'].fillna(train['Age'].mean())

# Impute the cabin variable
train['Cabin'] = train['Cabin'].fillna(train['Cabin'].mode()[0])
# Cek perubahannya
train.info()
# convert the category of male and female to integer form
train['Sex'][train['Sex'] == 'male'] = 0
train['Sex'][train['Sex'] == 'female'] = 1
# Convert the Embarked classes to integer form
# S = 0
# C = 1
# Q = 2
train['Embarked'][train['Embarked'] == 'S'] = 0
train['Embarked'][train['Embarked'] == 'C'] = 1
train['Embarked'][train['Embarked'] == 'Q'] = 2

train.head(8)
# Analisa apakah gender berpengaruh terhadap prediksi survival?
# Penumpang yang hidup vs mati
print(train['Survived'].value_counts(normalize = True))

# Penumpang kelas 1 yang hidup vs mati
print(train['Survived'][train['Pclass'] == 1].value_counts(normalize = True))

# Penumpang kelas 2 yang hidup vs mati
print(train['Survived'][train['Pclass'] == 2].value_counts(normalize = True))

# Penumpang kelas 3 yang hidup vs mati
print(train['Survived'][train['Pclass'] == 3].value_counts(normalize = True))
# Penumpang laki-laki yang hidup vs mati
print(train['Survived'][train['Sex'] == 0].value_counts(normalize = True))

# Penumpang perempuan yang hidup vs mati
print(train['Survived'][train['Sex'] == 1].value_counts(normalize = True))
# Buat kolom baru dengan nama Child, type data float
train['Child'] = float('NaN')

# masukan kategori sesuai penjelasan diatas
train['Child'][train['Age'] < 18] = 1
train['Child'][train['Age'] >= 18] = 0

train.head(8)
# Penumpang anak-anak hidup vs mati
print(train['Survived'][train['Child'] == 1].value_counts(normalize = True))

# Penumpang dewasa hidup vs mati
print(train['Survived'][train['Child'] == 0].value_counts(normalize = True))
target = train['Survived'].values
features_one = train[['Pclass', 'Sex', 'Age', 'Fare']].values # dari artikel lain ada yang memasukan fare tetapi belum nemu penjelasan logisnya
# (atau saya aja yg belum paham) dan fare ini dimasukan tanpa kategori seperti pada kolom umur. kita lihat aja dulu.
# My first decision tree
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)
# Cek feature importance dari 4 predictor, dengan urutan: Pclass, Sex, Age lalu Fare
print(my_tree_one.feature_importances_)
# Cek score dari data training pada train menggunakan model Decision Tree
my_tree_one.score(features_one, target)
# Persiapkan data test nya, seperti pada data train
test['Sex'][test['Sex'] == 'female'] = 1
test['Sex'][test['Sex'] == 'male'] = 0
test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Cabin'] = test['Cabin'].fillna(test['Cabin'].mode()[0])
test.Fare[152] = test['Fare'].median()
test_features = test[['Pclass', 'Sex', 'Age', 'Fare']].values # Features yang sama dengan training
test.info()
# Prediksi, hasil dari training diterapkan pada data test. my_tree_one adalah hasil dari training, menggunakan atribut predict dengan argumen test_features
# yang berisi data dari test.csv
my_prediction = my_tree_one.predict(test_features)
# Karena kaggle menentukan yang di upload hanya dua kolom: PassengerId dan Survived, maka kita bikin dataframe baru yang berisikan hasil prediksi
PassengerId = np.array(test['PassengerId']).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns= ['Survived'])
# lakukan konversi ke format .csv
my_solution.to_csv('my_solution_one.csv', index_label=['PassengerId'])