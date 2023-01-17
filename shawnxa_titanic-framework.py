import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
og_test = pd.read_csv('../input/test.csv')
print(train.shape, test.shape)
print(train.columns)

print(train.dtypes)
print(train.describe())


survived_f = train.Survived[train.Sex == 'female'].value_counts()
survived_m = train.Survived[train.Sex == 'male'].value_counts()
df = pd.DataFrame({'female': survived_f, 'male': survived_m})
df.plot(kind= 'bar', stacked = True)
plt.show()
train.Age.hist()
plt.title('all passengers age distribution')
plt.show()

train.Age[train.Survived == 0].hist()
plt.title('age distribution among victims')
plt.show()

train.Age[train.Survived == 1].hist()
plt.title('age distribution among survivers')
plt.show()
train.Fare.hist()
plt.title('overall fare distribution')
plt.show()

train.Fare[train.Survived == 0].hist()
plt.title('fare distribution among victims')
plt.show()

train.Fare[train.Survived == 1].hist()
plt.title('fare distribution among survivers')
plt.show()
train.Pclass.hist()
plt.title('class distribution among all')
plt.show()

survived_p1 = train.Survived[train.Pclass == 1].value_counts()
survived_p2 = train.Survived[train.Pclass == 2].value_counts()
survived_p3 = train.Survived[train.Pclass == 3].value_counts()

df = pd.DataFrame({'p1': survived_p1, 'p2': survived_p2, 'p3': survived_p3})
df.plot(kind = 'bar', stacked = True)
plt.show()

survived_s = train.Survived[train.Embarked == 'S'].value_counts()
survived_c = train.Survived[train.Embarked == 'C'].value_counts()
survived_q = train.Survived[train.Embarked == 'Q'].value_counts()

df = pd.DataFrame({'s': survived_s, 'c': survived_c, 'q': survived_q})
df.plot(kind = 'bar', stacked = True)
plt.show()
label = train.Survived
cols = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
data = train[cols]
test = test[cols]

print(data.columns)
def fill_nan(data):
    data_copy = data.copy(deep=True)
    data_copy.Age= data_copy.Age.fillna(data_copy.Age.median())
    data_copy.Fare = data_copy.Fare.fillna(data_copy.Fare.median())
    data_copy.Pclass = data_copy.Pclass.fillna(data_copy.Pclass.median())
    data_copy.Sex = data_copy.Sex.fillna('male')
    data_copy.Embarked = data_copy.Embarked.fillna('S')
    return data_copy
data = fill_nan(data)
test = fill_nan(test)
print(data.isnull().values.any(), test.isnull().values.any())
def trans_sex(data):
    data_copy = data.copy(deep = True)
    data_copy.loc[data_copy.Sex == 'female', 'Sex'] = 0
    data_copy.loc[data_copy.Sex == 'male', 'Sex'] = 1
    return data_copy
data = trans_sex(data)
test = trans_sex(test)

def trans_embark(d):
    dcopy = d.copy(deep = True)
    dcopy.loc[dcopy.Embarked == 'S', 'Embarked'] = 0
    dcopy.loc[dcopy.Embarked == 'C', 'Embarked'] = 1
    dcopy.loc[dcopy.Embarked == 'Q', 'Embarked'] = 2
    return dcopy
data = trans_embark(data)
test = trans_embark(test)
from sklearn.model_selection import train_test_split


train_data, vali_data, train_labels, vali_labels = train_test_split(data, label, random_state = 0, test_size = 0.2)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
k_range = range(1, 51)
acc_scores = []
rec_scores = []
prec_scores = []
for k in k_range:
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(train_data, train_labels)
    print('k=', k)
    predictions = model.predict(vali_data)
    acc = accuracy_score(vali_labels, predictions)
    acc_scores.append(acc)
    print('accuracy', acc)
    rec = recall_score(vali_labels, predictions)
    rec_scores.append(rec)
    print('recall score', rec)
    prec = precision_score(vali_labels, predictions)
    prec_scores.append(prec)
    print('precision score', prec)
plt.plot(k_range, acc_scores)
plt.title('accuracy')
plt.show()

plt.plot(k_range, rec_scores)
plt.title('recall')
plt.show()

plt.plot(k_range, prec_scores)
plt.title('precision')
plt.show()
print(np.array(acc_scores).argsort()) 
print(np.array(rec_scores).argsort()) # 34 最好， 我把前期的扔了
print(np.array(prec_scores).argsort())

k = (32 + 34 + 27) / 3  #这样求平均k值好吗？
print(k)


# 检测模型precision， recall 等各项指标

model = KNeighborsClassifier(n_neighbors = 31)
model.fit(data, label)
res = model.predict(test)
print(res)
# 预测

print(test.iloc[12], res[12])
df = pd.DataFrame({"PassengerId": og_test.PassengerId,"Survived": res})
df.to_csv('submission.csv',header=True, index=False)