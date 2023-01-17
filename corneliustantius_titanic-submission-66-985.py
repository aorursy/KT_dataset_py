import pandas as pd

#best practice to import our data is using _filepath

train_filepath = "../input/titanic/train.csv"
test_filepath = "../input/titanic/test.csv"
train_data = pd.read_csv(train_filepath)
test_data = pd.read_csv(test_filepath)
# jadi kita bakal intip dulu data yang mau kita kasi makan ke modelnya (fitting)
# dan data yang mau kita test. fit=train_data , test=test_data

#jalanin 1 1 ya
train_data.head() 
test_data.head() 
# selanjutnya kita cek nih, kalau kita liat
# pas datanya di describe, kita bsa liat kalo ada
# data yang NaN/NULL di kolom age
# yang lain 891 data tapi age cuma 714

train_data.describe()

# bisa diliat di index 5 itu ada yang NaN ya
train_data['Age'].head(10)
# untuk mengatasi masalah di atas, kita pakai
# fillna aja, dan kita akan fill dengan nilai tengah Age nya aja

train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
train_data.head(10)


# kita juga harus fillna data yang mau di uji ya
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
from sklearn.tree import DecisionTreeRegressor

# jadi kita bakal coba predict data sendiri dulu ya 
# buat testing key yang kita ambil itu bener apa gak
# jadi si Y ini gunanya menandakan data yang mau kita predict
train_y = train_data['Survived']

# lalu si X ini, bakal jadi dasar, jadi kita mw predict si Y itu berdasar apa
# nah di kasus ini aku pake 2 kolom yaitu Pclass sama Age(yang barusan kita fillna juga)
train_feature = ['Pclass','Age','SibSp','Parch']
train_x = train_data[train_feature]

# sekarang kita bisa bikin model tree regressor nya
train_model = DecisionTreeRegressor(random_state = 1)

# kemudian stlh kita buat modelnya, kita kasi makan data nya
train_model.fit(train_x, train_y)

# baru kita coba force print predict
# NB: itu akurat krna kita predict data sendiri
print(train_model.predict(train_x.head()).tolist())
print(train_y.head().tolist())
from sklearn.tree import DecisionTreeRegressor

# nah sekarang kita bakal coba predict data nya ya langsung menuju test data
test_feature = ['Pclass','Age','SibSp','Parch']
test_x = test_data[test_feature]

# kalau km notice, itu data yang keluar bakalan float, nah itu gak sesuai
# dengan yang diminta soal karna soal minta 1/0. 
# itu kan tandanya masih belum akurat ya data hasil predict kita,
print(train_model.predict(test_x))

# jadi kita bakal pakai DecisionTreeClassifier buat dapat nilai antara 

from sklearn.tree import DecisionTreeClassifier
import pandas as pd

classifier_train_model = DecisionTreeClassifier(random_state = 1)
classifier_train_model.fit(train_x, train_y)
prediction = classifier_train_model.predict(test_x)

output = pd.DataFrame({'PassengerId':test_data['PassengerId'] , 'Survived':prediction})
output

# lalu di output
output.to_csv('ans1.csv',index=False)