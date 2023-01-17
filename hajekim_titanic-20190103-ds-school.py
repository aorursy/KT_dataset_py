# 파이썬의 import를 활용해 데이터 분석용 패키지인 판다스(Pandas)를 읽어옵니다.
import pandas as pd

# train.csv 파일을 읽어옵니다. 여기서 PassengerId를 index로 지정해줍니다.
# (=이제 PassengerId를 통해 승객 정보를 읽어올 수 있습니다)
# 읽어온 데이터를 train이라는 이름의 변수에 할당합니다.
train = pd.read_csv("../input/train.csv", index_col="PassengerId")

# train 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# 출력은 (row, column) 으로 표시됩니다.
print(train.shape)

# head()로 train 데이터의 상위 5개를 띄웁니다.
train.head()
# train.csv 파일을 읽어온 방식과 동일하게 test.csv를 읽어옵니다.
# 이후 이 데이터를 test라는 이름의 변수에 저장합니다.
test = pd.read_csv("../input/test.csv", index_col="PassengerId")

# 마찬가지로 행렬(row, column) 사이즈를 출력하고
print(test.shape)

# 전체 test 데이터에서 상위 5개만 출력합니다.
test.head()
# matplotlib로 실행하는 모든 시각화를 자동으로 쥬피터 노트북에 띄웁니다.
# seaborn 도 결국에는 matplotlib를 기반으로 동작하기 때문에, seaborn으로 실행하는 모든 시각화도 마찬가지로 쥬피터 노트북에 자동적으로 띄워집니다.
%matplotlib inline

# 데이터 시각화 패키지 seaborn을 로딩합니다. 앞으로는 줄여서 sns라고 사용할 것입니다.
import seaborn as sns

# 데이터 시각화 패키지 matplotlib를 로딩합니다. 앞으로는 줄여서 plt라고 사용할 것입니다.
import matplotlib.pyplot as plt
# 타이타닉의 train 데이터를 바탕으로 성별 컬럼을 시각화합니다.
# 크게 1) 남성 생존자, 2) 남성 사망자, 3) 여성 생존자, 4) 여성 사망자 를 시각화합니다.
sns.countplot(data=train, x="Sex", hue="Survived")
# pivot_table을 통해 성별(Sex)에 따른 생존률을 출력합니다.
pd.pivot_table(train, index="Sex", values="Survived")
# 타이타닉의 train 데이터를 바탕으로 객실 등급(Pclass)을 시각화합니다.
# 크게 1) 1등급 생존자/사망자, 2) 2등급 생존자/사망자, 3) 3등급 생존자/사망자 의 총 인원 수를 알 수 있습니다.
sns.countplot(data=train, x="Pclass", hue="Survived")
# pivot_table을 통해 객실 등급(Pclass)에 따른 생존률을 출력합니다.
pd.pivot_table(train, index="Pclass", values="Survived")
# 타이타닉의 train 데이터를 바탕으로 선착장(Embarked)을 시각화합니다.
# C, S, Q에서 탑승한 승객의 생존자/사망자의 총 인원수를 알 수 있습니다.
sns.countplot(data=train, x="Embarked", hue="Survived")
# pivot_table을 통해 선착장(Embarked)에 따른 생존률을 출력합니다.
pd.pivot_table(train, index="Embarked", values="Survived")
# lmplot을 통해 나이(Age)와 운임요금(Fare)의 상관관계를 분석합니다.
# 생존자와 사망자의 차이를 보여주기 위해 hue="Survived" 옵션을 넣습니다.
# 또한 회귀(Regression)선은 일반적으로 잘 쓰이지 않기 때문에 fit_reg에 False를 넣습니다.
sns.lmplot(data=train, x="Age", y="Fare", hue="Survived", fit_reg=False)
# 판다스의 색인(indexing) 기능을 활용하여, 운임요금(Fare)이 500달러 미만인 데이터만 가져옵니다.
# 이를 low_fare라는 변수에 할당합니다.
low_fare = train[train["Fare"] < 500]

# train 데이터와 low_fare 데이터의 행렬 사이즈를 출력합니다.
# 출력은 (row, column) 으로 표시됩니다.
# low_fare 데이터가 train 데이터보다 3개 적은 것을 확인할 수 있는데 (891 > 888),
# 이는 low_fare에서는 $500달러 이상을 지불한 세 명의 승객을 제거했기 때문입니다.
train.shape, low_fare.shape
# lmplot을 통해 나이(Age)와 운임요금(Fare)의 상관관계를 분석합니다.
# 다만 이전과는 달리 이번에는 train 데이터가 아닌 low_fare 데이터를 시각화 합니다.
sns.lmplot(data=low_fare, x="Age", y="Fare", hue="Survived", fit_reg=False)
# 판다스의 색인(indexing) 기능을 활용하여, 운임요금(Fare)이 100달러 미만인 데이터만 가져옵니다.
# 이를 low_low_fare라는 변수에 할당합니다.
low_low_fare = train[train["Fare"] < 100]

# train, low_fare, low_low_fare 데이터의 행렬 사이즈를 출력합니다.
# 출력은 (row, column) 으로 표시됩니다.
# low_low_fare에서는 train 데이터에 비해 무려 53명의 승객이 더 적습니다. (891 > 838)
train.shape, low_fare.shape, low_low_fare.shape
# lmplot을 통해 나이(Age)와 운임요금(Fare)의 상관관계를 분석합니다.
# 이번에는 low_low_fare 데이터를 시각화 합니다.
sns.lmplot(data=low_low_fare, x="Age", y="Fare", hue="Survived", fit_reg=False)
# train 데이터의 SibSp와 Parch 컬럼을 더해서 FamilySize라는 새로운 컬럼을 만듭니다.
# 다만 가족 수를 셀 때는 언제나 나 자신도 포함하는데, 나 자신은 SibSp와 Parch 중 어디에도 포함되어 있지 않기 때문에,
# 무조건 1을 더해서 총 인원 수를 하나 늘려줍니다.
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1

# train 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# 출력은 (row, column) 으로 표시됩니다.
print(train.shape)

# train 데이터의 상위 5개를 띄우되, SibSp와 Parch, 그리고 FamilySize 컬럼만 출력합니다.
train[["SibSp", "Parch", "FamilySize"]].head()
# 타이타닉의 train 데이터를 바탕으로 가족 수(FamilySize)을 시각화합니다.
# 가족 수가 늘어날 때 마다 생존자/사망자의 변화를 볼 수 있습니다.
sns.countplot(data=train, x="FamilySize", hue="Survived")
# 가족 수(FamilSize)가 1인 승객을 가져와서, FamilyType 컬럼에 Single 이라는 값을 넣어줍니다.
train.loc[train["FamilySize"] == 1, "FamilyType"] = "Single"

# 가족 수(FamilSize)가 2 이상 5 미만인 승객을 가져와서, FamilyType 컬럼에 Nuclear(핵가족) 이라는 값을 넣어줍니다.
train.loc[(train["FamilySize"] > 1) & (train["FamilySize"] < 5), "FamilyType"] = "Nuclear"

# 가족 수(FamilSize)가 5 이상인 승객을 가져와서, FamilyType 컬럼에 Big(대가족) 이라는 값을 넣어줍니다.
train.loc[train["FamilySize"] >= 5, "FamilyType"] = "Big"

# train 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# 출력은 (row, column) 으로 표시됩니다.
print(train.shape)

# train 데이터의 상위 10개를 띄우되, FamilySize와 FamilyType 컬럼만 출력합니다.
train[["FamilySize", "FamilyType"]].head(10)
# 타이타닉의 train 데이터를 바탕으로 가족 형태(FamilyType)을 시각화합니다.
# 싱글(Single), 핵가족(Nuclear), 대가족(Big)의 생존자/사망자의 총 인원 수를 알 수 있습니다.
sns.countplot(data=train, x="FamilyType", hue="Survived")
# pivot_table을 통해 가족 형태(FamilyType)의 변화에 따른 생존률을 출력합니다.
pd.pivot_table(data=train, index="FamilyType", values="Survived")
# train 데이터의 승객 이름(Name) 컬럼의 상위 5개를 출력합니다.
# 앞서 설명한것과 동일한 패턴으로 이름이 출력되는 것을 확인할 수 있습니다.
train["Name"].head()
# get_title이라는 이름의 함수를 정의합니다. 이 함수는 name이라는 변수를 인자로 받습니다.
# 이 함수는 이름을 받았을 때 이름에서 타이틀을 반환해줍니다.
# 가령 name에 "Braund, Mr. Owen Harris"가 들어오면 최종 결과는 Mr를 반환해줍니다.
def get_title(name):
    # 먼저 name을 , 을 기준으로 쪼갭니다. 쪼갠 결과는 0) Braund와 1) Mr. Owen Harris가 됩니다.
    # 여기서 1)번을 가져온 뒤 다시 . 을 기준으로 쪼갭니다. 쪼갠 결과는 0) Mr와 1) Owen Harris가 됩니다.
    # 여기서 0)번을 반환합니다. 최종적으로는 Mr를 반환하게 됩니다.
    return name.split(", ")[1].split('. ')[0]

# 모든 Name 컬럼 데이터에 get_title 함수를 적용합니다.
# 그 결과에서 unique를 통해 중복된 값을 제거합니다.
train["Name"].apply(get_title).unique()
# 호칭을 저장하는 컬럼은 없으므로 "Title"이라는 새로운 컬럼을 만듭니다.
# Name에 "Mr"가 포함되어 있으면 Title 컬럼에 "Mr"이라는 값을 넣어줍니다.
train.loc[train["Name"].str.contains("Mr"), "Title"] = "Mr"

# Name에 "Miss"가 포함되어 있으면 Title 컬럼에 "Miss"이라는 값을 넣어줍니다.
train.loc[train["Name"].str.contains("Miss"), "Title"] = "Miss"

# Name에 "Mrs"가 포함되어 있으면 Title 컬럼에 "Mrs"이라는 값을 넣어줍니다.
train.loc[train["Name"].str.contains("Mrs"), "Title"] = "Mrs"

# Name에 "Master"가 포함되어 있으면 Title 컬럼에 "Master"이라는 값을 넣어줍니다.
train.loc[train["Name"].str.contains("Master"), "Title"] = "Master"

# train 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# 출력은 (row, column) 으로 표시됩니다.
print(train.shape)

# train 데이터의 상위 10개를 띄우되, Name과 Title 컬럼만 출력합니다.
train[["Name", "Title"]].head(10)
# 타이타닉의 train 데이터를 바탕으로 호칭(Title)을 시각화합니다.
# "Mr", "Mrs", "Miss", "Master"의 생존자/사망자의 총 인원 수를 알 수 있습니다.
sns.countplot(data=train, x="Title", hue="Survived")
# pivot_table을 통해 호칭(Title)에 따른 생존률을 출력합니다.
pd.pivot_table(train, index="Title", values="Survived")
# 성별(Sex) 값이 male인 경우 0으로, female인 경우 1로 수정합니다.
# 단 Sex 컬럼을 바로 수정하지 않고, Sex_encode라는 새로운 컬럼을 추가해서 여기에 값을 넣습니다.
# 전처리를 할 때는 언제나 이런 방식으로 원본을 유지하고 사본에다가 작업해주는게 좋습니다.
train.loc[train["Sex"] == "male", "Sex_encode"] = 0
train.loc[train["Sex"] == "female", "Sex_encode"] = 1

# train 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# 출력은 (row, column) 으로 표시됩니다.
print(train.shape)

# train 데이터에서 Sex 컬럼과 Sex_encode 컬럼에 대한 상위 5개의 정보를 띄웁니다.
train[["Sex", "Sex_encode"]].head()
# train 데이터의 성별(Sex) 컬럼을 전처리해준 방식과 동일하게 test 데이터도 전처리를 해줍니다.
# 즉, Sex_encode 컬럼에 male이면 0을, female을 1을 대입해줍니다.
test.loc[test["Sex"] == "male", "Sex_encode"] = 0
test.loc[test["Sex"] == "female", "Sex_encode"] = 1

# test 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# 출력은 (row, column) 으로 표시됩니다.
print(test.shape)

# test 데이터에서 Sex 컬럼과 Sex_encode 컬럼에 대한 상위 5개의 정보를 띄웁니다.
test[["Sex", "Sex_encode"]].head()
# train 데이터에서 운임요금(Fare)이 비어있는 승객을 검색합니다. 검색 결과 아무런 값도 나오지 않습니다.
# 즉, train 데이터에는 운임요금(Fare)이 비어있는 승객이 한 명도 없다는 것으로 이해할 수 있습니다.
train[train["Fare"].isnull()]
# test 데이터에서 운임요금(Fare)이 비어있는 승객을 검색합니다.
# 검색 결과 1044번 승객(Storey, Mr. Thomas)의 운임요금(Fare) 값이 비어있다는 것을(NaN) 확인할 수 있습니다.
test[test["Fare"].isnull()]
# 앞서 성별(Sex) 컬럼을 전처리한 것과 마찬가지로,
# 원본(Fare)을 고치지 않고 사본(Fare_fillin)을 만들어 그 곳에 빈 값을 집어넣겠습니다.
train["Fare_fillin"] = train["Fare"]

# train 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# 출력은 (row, column) 으로 표시됩니다.
print(train.shape)

# train 데이터에서 Fare 컬럼과 Fare_fillin 컬럼에 대한 상위 5개의 정보를 띄웁니다.
train[["Fare", "Fare_fillin"]].head()
# train 데이터를 다룬 것과 비슷한 방식하게
# test 데이터도 사본을 만듭니다.
test["Fare_fillin"] = test["Fare"]

# test 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# 출력은 (row, column) 으로 표시됩니다.
print(test.shape)

# test 데이터에서 Fare 컬럼과 Fare_fillin 컬럼에 대한 상위 5개의 정보를 띄웁니다.
test[["Fare", "Fare_fillin"]].head()
# test 데이터에서 운임요금(Fare) 값이 비어있는 승객을 검색한 뒤,
# 해당 승객의 운임요금(Fare_fillin) 값을 0으로 채워줍니다.
test.loc[test["Fare"].isnull(), "Fare_fillin"] = 0

# 이후 다시 한 번 test 데이터에서 운임요금(Fare)이 비어있는 승객을 검색합니다.
# 검색 결과 1044번 승객의 운임요금의 사본(Fare_fillin)에 비어있던 값이 0으로 채워졌다는 걸 알 수 있습니다.
test.loc[test["Fare"].isnull(), ["Fare", "Fare_fillin"]]
# 먼저 Embarked 컬럼이 C이면 True, C가 아니면 False인 색인 코드를 작성합니다.
# 그리고 여기서 나온 결과를 그대로 Embarked_C 라는 새로운 컬럼에 대입합니다.
# 이제 Embarked_C 컬럼은 승객이 C(Cherbourg)에서 탑승했으면 True, 그렇지 않으면 False가 나옵니다.
train["Embarked_C"] = train["Embarked"] == "C"

# 비슷한 방식으로 Embarked_S 컬럼을 추가합니다.
# 승객이 S(Southampton)에서 탑승했으면 True, 그렇지 않으면 False가 나옵니다.
train["Embarked_S"] = train["Embarked"] == "S"

# 비슷한 방식으로 Embarked_Q 컬럼을 추가합니다.
# 승객이 Q(Queenstown)에서 탑승했으면 True, 그렇지 않으면 False가 나옵니다.
train["Embarked_Q"] = train["Embarked"] == "Q"

# train 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# 출력은 (row, column) 으로 표시됩니다.
print(train.shape)

# train 데이터의 상위 5개를 띄우되, Embarked와 직접적으로 연관된 컬럼만 따로 출력합니다.
train[["Embarked", "Embarked_C", "Embarked_S", "Embarked_Q"]].head()
# test 데이터도 train 데이터와 동일한 방식으로
# Embarked_C, Embarked_S, Embarked_Q 컬럼을 추가합니다.
test["Embarked_C"] = test["Embarked"] == "C"
test["Embarked_S"] = test["Embarked"] == "S"
test["Embarked_Q"] = test["Embarked"] == "Q"

# test 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# 출력은 (row, column) 으로 표시됩니다.
print(test.shape)

# test 데이터의 상위 5개를 띄우되, Embarked와 직접적으로 연관된 컬럼만 따로 출력합니다.
test[["Embarked", "Embarked_C", "Embarked_S", "Embarked_Q"]].head()
# train 데이터에 Child라는 이름의 새로운 컬럼을 추가합니다.
# 이 컬럼은 나이가 15세 미만일 경우 어린아이라고 가정하고(True), 반대로 15세 이상일 경우 어른이라고 가정합니다(False).
train["Child"] = train["Age"] < 15

# train 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# 출력은 (row, column) 으로 표시됩니다.
print(train.shape)

# train 데이터의 상위 10개를 띄우되, Age 컬럼과 Child 컬럼만 출력합니다.
train[["Age", "Child"]].head(10)
# test 데이터에도 train 데이터와 동일한 방식으로 Child 컬럼을 추가합니다.
test["Child"] = test["Age"] < 15

# test 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# 출력은 (row, column) 으로 표시됩니다.
print(test.shape)

# test 데이터의 상위 10개를 띄우되, Age 컬럼과 Child 컬럼만 출력합니다.
test[["Age", "Child"]].head(10)
# train 데이터의 SibSp와 Parch 컬럼을 더해서 FamilySize라는 새로운 컬럼을 만듭니다.
# 다만 가족 수를 셀 때는 언제나 나 자신도 포함하는데, 나 자신은 SibSp와 Parch 중 어디에도 포함되어 있지 않기 때문에,
# 무조건 1을 더해서 총 인원 수를 하나 늘려줍니다.
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1

# train 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# 출력은 (row, column) 으로 표시됩니다.
print(train.shape)

# train 데이터의 상위 5개를 띄우되, SibSp와 Parch, 그리고 FamilySize 컬럼만 출력합니다.
train[["SibSp", "Parch", "FamilySize"]].head()
# test 데이터도 train 데이터와 동일한 방식으로 FamilySize 컬럼을 만듭니다.
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

# test 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# 출력은 (row, column) 으로 표시됩니다.
print(test.shape)

# test 데이터의 상위 5개를 띄우되, SibSp와 Parch, 그리고 FamilySize 컬럼만 출력합니다.
test[["SibSp", "Parch", "FamilySize"]].head()
# 가족 수(FamilySize) 컬럼을 활용해 세 가지 구역을 나타내는 세 개의 새로운 컬럼을 만듭니다.
# 먼저 가족 수가 1명일 경우 Single 컬럼의 값에 True를, 1명이 아닐 경우 False를 대입합니다.
train["Single"] = train["FamilySize"] == 1

# 이후 가족 수가 2에서 4명 사이일 경우 Nuclear 컬럼의 값에 True를, 그렇지 않을 경우 False를 대입합니다.
train["Nuclear"] = (train["FamilySize"] > 1) & (train["FamilySize"] < 5)

# 마지막으로 가족 수가 5명 이상일 경우 Big 컬럼의 값에 True를, 그렇지 않을 경우 False를 대입합니다.
train["Big"] = train["FamilySize"] >= 5

# train 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# 출력은 (row, column) 으로 표시됩니다.
print(train.shape)

# train 데이터의 상위 10개를 띄우되, FamilySize, Single, Nuclear, Big 컬럼만 출력합니다.
train[["FamilySize", "Single", "Nuclear", "Big"]].head(10)
# train 데이터를 다룬 것과 마찬가지로,
# test 데이터에도 FamilySize를 활용하여 Single, Nuclear, Big 컬럼을 새로 만듭니다.
test["Single"] = test["FamilySize"] == 1
test["Nuclear"] = (test["FamilySize"] > 1) & (test["FamilySize"] < 5)
test["Big"] = test["FamilySize"] >= 5

# test 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# 출력은 (row, column) 으로 표시됩니다.
print(test.shape)

# test 데이터의 상위 10개를 띄우되, FamilySize, Single, Nuclear, Big 컬럼만 출력합니다.
test[["FamilySize", "Single", "Nuclear", "Big"]].head(10)
# 승객 이름(Name) 컬럼을 활용해 "Master"라는 새로운 컬럼을 만듭니다.
# 승객 이름(Name)에 "Master"라는 단어가 포함되어있으면 True를, 그렇지 않을 경우 False를 대입합니다.
train["Master"] = train["Name"].str.contains("Master")

# train 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# 출력은 (row, column) 으로 표시됩니다.
print(train.shape)

# train 데이터의 상위 10개를 띄우되, Name, Master 컬럼만 출력합니다.
train[["Name", "Master"]].head(10)
# train 데이터를 다룬 것과 마찬가지로,
# test 데이터에도 Name을 활용하여 Master 컬럼을 새로 만듭니다.
test["Master"] = test["Name"].str.contains("Master")

# test 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# 출력은 (row, column) 으로 표시됩니다.
print(test.shape)

# train 데이터의 상위 10개를 띄우되, Name, Master 컬럼만 출력합니다.
test[["Name", "Master"]].head(10)
# 1) 티켓 등급(Pclass), 2) 성별(Sex_encode), 3) 운임요금(Fare_fillin), 4) 선착장(Embarked) 이렇게 네 가지를 feature로 지정합니다.
# 여기서 선착장(Embarked)은 one hot encoding하였기 때문에, Embarked 자체 컬럼이 아닌 Embarked_C, Embarked_S, Embarked_Q 세 개의 컬럼을 사용하겠습니다.
# 그러므로 feature에 해당하는 컬럼의 갯수는 총 여섯 개입니다. 이 여섯 개의 컬럼명을 feature_names라는 이름의 파이썬 리스트(list)로 만들어 변수에 할당합니다.
feature_names = ["Pclass", "Sex_encode", "Fare_fillin",
                 "Embarked_C", "Embarked_S", "Embarked_Q",
                 "Child", "Single", "Nuclear", "Big", "Master"]
feature_names
# 생존 여부(Survived)를 label로 지정합니다.
# Survived라는 이름의 컬럼을 label_name 이라는 이름의 변수에 할당합니다.
label_name = "Survived"
label_name
# feature_names를 활용해 train 데이터의 feature를 가져옵니다.
# 이를 X_train이라는 이름의 변수에 할당합니다.
X_train = train[feature_names]

# X_train 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# 출력은 (row, column) 으로 표시됩니다.
print(X_train.shape)

# X_train 데이터의 상위 5개를 띄웁니다.
X_train.head()
# feature_names를 활용해 test 데이터의 feature를 가져옵니다.
# 이를 X_test라는 이름의 변수에 할당합니다.
X_test = test[feature_names]

# X_test 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# 출력은 (row, column) 으로 표시됩니다.
print(X_test.shape)

# X_test 데이터의 상위 5개를 띄웁니다.
X_test.head()
# label_name을 활용해 train 데이터의 label을 가져옵니다.
# 이를 y_train이라는 이름의 변수에 할당합니다.
y_train = train[label_name]

# y_train 변수에 할당된 데이터의 사이즈를 출력합니다.
# 출력은 (row, column) 으로 표시되나, column이 없기 때문에 (row,) 형태로 표시될 것입니다.
print(y_train.shape)

# y_train 데이터의 상위 5개를 띄웁니다.
y_train.head()
# scikit-learn(줄여서 sklearn)의 tree 모듈에서 DecisionTreeClassifier를 가져옵니다.
from sklearn.tree import DecisionTreeClassifier

# DecisionTreeClassifier를 생성하는데, 트리의 최대 깊이(depth)를 7로 설정합니다.
# 이렇게 하면 트리의 가지(branch)가 8 depth 아래로 내려가지 않습니다.
# 또한 생성한 DecisionTreeClassifier를 model이라는 이름의 변수에 할당합니다.
model = DecisionTreeClassifier(max_depth=8, random_state=0)
model
# DecisionTreeClassifier를 학습(fitting)합니다.
# 학습에는 fit 이라는 기능을 사용하며, train 데이터의 feature(X_train)와 label(y_train)을 집어넣습니다.
model.fit(X_train, y_train)
# graphviz 모듈을 가져옵니다.
import graphviz

# scikit-learn(줄여서 sklearn)의 tree 모듈에서 Decision Tree를 시각화 할 수 있는 export_graphviz를 가져옵니다.
from sklearn.tree import export_graphviz

# export_graphviz로 Decision Tree를 시각화합니다. 시각화 할 때는 다음의 옵션이 들어갑니다.
# 1) model. 시각화할 트리(Decision Tree) 입니다.
# 2) feature_names. 트리를 만들 때 사용한 feature들의 이름입니다.
# 3) class_names. 살았을 경우 시각화에서 어떻게 표현할 것인지(Survived), 반대로 죽었을 경우 시각화에서 어떻게 표현할 것인지(Perish)를 알려줍니다.
# 4) out_file. 시각화 겨롸를 저장할 파일명입니다. 이번에는 파일로 저장하지 않고 바로 쥬피터 노트북에 띄울 생각이므로 None을 주면 됩니다.
# 마지막으로 시각화한 결과를 dot_tree라는 이름의 변수에 저장합니다.
dot_tree = export_graphviz(model,
                           feature_names=feature_names,
                           class_names=["Perish", "Survived"],
                           out_file=None)

# graphviz에서 Source라는 기능을 통해 Decision Tree를 시각화합니다.
graphviz.Source(dot_tree)
# fit이 끝났으면, predict라는 기능을 사용하여 생존 여부(Survived)를 예측합니다.
# predict의 실행이 끝나면 test 데이터의 생존 여부(Survived)를 반환하며, 이를 predictions라는 이름의 변수에 할당합니다.
predictions = model.predict(X_test)

# predictions 변수에 할당된 데이터의 사이즈를 출력합니다.
# 출력은 (row, column) 으로 표시되나, column이 없기 때문에 (row,) 형태로 표시될 것입니다.
print(predictions.shape)

# predictions 변수에 값이 너무 많기 때문에, 상위 10개만 출력합니다.
predictions[0:10]
# 캐글이 제공하는 제출 포멧(gender_submission.csv)을 읽어옵니다.
# PassengerId는 test 데이터와 동일하며, Survived는 남자일 경우 0, 여자는 1이 들어가 있습니다.
# 이를 submission 이라는 이름의 변수에 할당합니다.
submission = pd.read_csv("../input/gender_submission.csv", index_col="PassengerId")

# submission 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# 출력은 (row, column) 으로 표시됩니다.
print(submission.shape)

# submission 데이터의 상위 5개를 띄웁니다.
submission.head()
# 제출 포멧(submission)의 생존 여부(Survived) 컬럼에 우리의 예측값(predictions)를 집어넣습니다.
# 두 데이터 모두 길이가 418개로 동일하기 때문에, 등호(=)를 통해 쉽게 예측값을 넣을 수 있습니다.
submission["Survived"] = predictions

# submission 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# 출력은 (row, column) 으로 표시됩니다.
print(submission.shape)

# submission 데이터의 상위 5개를 띄웁니다.
submission.head()
# 마지막으로 submission 변수에 들어간 값을 csv 형식의 데이터로 저장합니다.
submission.to_csv("decision-tree_0.81818.csv")
