import numpy as np 
import pandas as pd 
path = "../input"
train = pd.read_csv(path + "/train.csv")
test = pd.read_csv(path + "/test.csv")
print(train.columns)
train.head()
print(train.dtypes)
print(train.describe())
import matplotlib.pyplot as plt
survived_male = train.Survived[train.Sex == 'male'].value_counts()
survived_female = train.Survived[train.Sex == 'female'].value_counts()

df = pd.DataFrame({'male': survived_male, 'female': survived_female})
df.plot(kind = 'bar', stacked = True)
plt.title("survived by sex")
plt.xlabel("survived")
plt.ylabel("count")
plt.show()
train['Age'].hist()
plt.ylabel("Number")
plt.xlabel("Age")
plt.title("Age distribution")
plt.show()

# survived == 0
train[train.Survived == 0]['Age'].hist()
plt.ylabel("Number")
plt.xlabel("Age")
plt.title("Age distribution of people who didn't survived")
plt.show()

# survived == 1
train[train.Survived == 1]['Age'].hist()
plt.ylabel("Number")
plt.xlabel("Age")
plt.title("Age distribution of people who survived")
plt.show()

print(train[train.Survived == 1]['Age'].head())
train.Fare.hist()
plt.ylabel("Number")
plt.xlabel("Fare")
plt.title("Fare distribution")
plt.show()

train[train.Survived == 0].Fare.hist()
plt.ylabel("Number")
plt.xlabel("Fare")
plt.title("Fare distribution for survivers")
plt.show()

train[train.Survived == 1].Fare.hist()
plt.ylabel("Number")
plt.xlabel("Fare")
plt.title("Fare distribution for non-survivers")
plt.show()
train.Pclass.hist()
plt.show()
print(train.Pclass.isnull().values.any())

sur_cls1 = train.Survived[train.Pclass == 1].value_counts()
sur_cls2 = train.Survived[train.Pclass == 2].value_counts()
sur_cls3 = train.Survived[train.Pclass == 3].value_counts()

df = pd.DataFrame({"p1": sur_cls1, "p2": sur_cls2, "p3":sur_cls3})
print(df)
df.plot(kind = "bar", stacked = True)
plt.title("survived by pclass")
plt.xlabel("pclass")
plt.ylabel("count")
plt.show()
sur_cls1 = train.Survived[train.Embarked == 'S'].value_counts()
sur_cls2 = train.Survived[train.Embarked == 'C'].value_counts()
sur_cls3 = train.Survived[train.Embarked == 'Q'].value_counts()

df = pd.DataFrame({"S": sur_cls1, "C": sur_cls2, "Q": sur_cls3})
df.plot(kind = "bar", stacked = True)
plt.title("survived by embarking location")
plt.xlabel("embarked")
plt.ylabel("count")
plt.show()

train_select = train[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
test_select = test[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
label = train[['Survived']]
print(label.head())
print(train_select.shape)
def fill_NaN(data):
    data_copy = data.copy(deep= True)
    data_copy.loc[:,"Age"] = data_copy["Age"].fillna(data_copy["Age"].median())
    data_copy.loc[:,"Fare"] = data_copy["Fare"].fillna(data_copy["Fare"].median())
    data_copy.loc[:,"Pclass"] = data_copy["Pclass"].fillna(data_copy["Pclass"].median())
    data_copy.loc[:,"Sex"] = data_copy["Sex"].fillna("female")
    data_copy.loc[:,"Embarked"] = data_copy["Embarked"].fillna("S")
    return data_copy

train_no_nan = fill_NaN(train_select)
test_no_nan = fill_NaN(test_select)
print(train_no_nan.isnull().values.any())
print(train.isnull().values.any())
print(train_no_nan.shape, test_no_nan.shape)
print(train_no_nan["Sex"].isnull().values.any())
print(train_no_nan.Sex.value_counts())
def transfer_sex(data):
    data_copy = data.copy(deep = True)
    data_copy.loc[data_copy.Sex == 'female', 'Sex']= 0
    data_copy.loc[data_copy.Sex == 'male', 'Sex']= 1
    return data_copy

train_after_sex = transfer_sex(train_no_nan)
test_after_sex = transfer_sex(test_no_nan)
print(train_after_sex.shape, test_after_sex.shape)
print(train_after_sex.Sex.value_counts())
def transfer_embark(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy.Embarked == 'S', 'Embarked'] = 0
    data_copy.loc[data_copy.Embarked == 'C', 'Embarked'] = 1
    data_copy.loc[data_copy.Embarked == 'Q', 'Embarked'] = 2
    return data_copy

train_after_embarked = transfer_embark(train_after_sex)
test_after_embarked = transfer_embark(test_after_sex)
print(train_after_embarked.Embarked.value_counts())
train_now = train_after_embarked
test_now = test_after_embarked
from sklearn.model_selection import train_test_split
train_data,test_data,train_labels, test_labels = train_test_split(train_now, label, random_state=0, test_size = 0.2)
print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)
# 预测
print(label)
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score
k_range = range(1,51)
k_scores = []

for k in k_range:
    clf = KNN(n_neighbors = k)
    clf.fit(train_data,train_labels)
    print("K=", k)
    predictions = clf.predict(test_data)
    score = accuracy_score(test_labels,predictions)
    print(score)
    k_scores.append(score)
# 检测模型precision， recall 等各项指标
plt.plot(k_range,k_scores)
plt.xlabel("K for KNN")
plt.ylabel("Accuracy on validation set")
plt.show()
print(np.array(k_scores).argsort())
clf = KNN(n_neighbors = 33)
clf.fit(train_now, label)
result=clf.predict(test_now)
# 预测
print(result)
df = pd.DataFrame({'PassengerId': test['PassengerId'], "Survived": result})
print(df.shape)
print(df.head())

df.to_csv('submission.csv', header=True, index = False)