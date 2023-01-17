import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('seaborn')

sns.set(font_scale=2.5)



import missingno as msno



#ignore warnings

import warnings

warnings.filterwarnings('ignore') 



%matplotlib inline



# 추가 임포트 (sklearn)



from collections import Counter



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



# 트레인 테스트 데이터 나누기
# 추가셀: Outlier detection (IQR(튜키의 방법)을 이용한 함수를 지정하여 이상치를 탐색)

def detect_outliers(df, n, features):

    outlier_indices = []

    for col in features:

        Q1 = np.percentile(df[col], 25)

        Q3 = np.percentile(df[col], 75)

        IQR = Q3 - Q1

        

        outlier_step = 1.5 * IQR

        

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

        

    return multiple_outliers

        

Outliers_to_drop = detect_outliers(df_train, 2, ["Age", "SibSp", "Parch", "Fare"])
# 이상치가 발견된 행 확인 

df_train.loc[Outliers_to_drop]
# 이상치 제거

df_train = df_train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
df_train.head(10)



# 트레인 데이터 불러오기 

#()앞에서부터 n개의 트레인 데이터 
df_train.describe()

# train 데이터의 통계적 수치들 표현 

# Pclass가 891인데 Age는 714이다. 이는 NA데이터 때문
df_test.describe()

# test 데이터의 통계적 수치들 표현
df_train.columns

# 각 열의 속성을 보여주는 함수
for col in df_train.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))

    print(msg)

    

    # 각 column에 결측치가 몇 % 인지 확인하기 위한 과정

    # df_train[col].isnull().sum() : 해당 열의 결측치가 몇개인지 알 수 있게하는 문장 (TRUE=1(결측치), FALSE=0으로 계산된다.)

    # df_train[col].shape[0] : 해당 열의 차원 (열이 지정되어 있으므로 행의 갯수를 보여준다.)

    # 100 * (df_train[col].isnull().sum() / df_train[col].shape[0] : 위의 설명을 통해 %를 출력해주는 문장임을 알 수 있다.
for col in df_test.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_test[col].isnull().sum() / df_test[col].shape[0]))

    print(msg)

    

    # 위의 과정과 동일 
msno.matrix(df=df_train.iloc[:, :], figsize=(8,8), color=(0.1, 0.6, 0.8))



# msno.matrix는 밑에 보이는것과 같은 매트릭스를 만들어준다. figsize는 크기, color는 순서대로 RGB값 

# 빈칸이 NULL데이터 
msno.bar(df=df_train.iloc[:, :], figsize=(8,8), color=(0.1, 0.6, 0.8))



# 바 형태의 그래프로 만들어준다.
f, ax = plt.subplots(1,2, figsize = (18,8)) # 도화지를 준비(행,열,사이즈)



df_train['Survived'].value_counts().plot.pie(explode = [0, 0.1], autopct = '%1.1f%%', ax=ax[0], shadow = True)

# 시리즈 타입의 파이플랏을 그려준다. 퍼센트를 나타내준다. 도화지 첫번째 공간에, 그림자를 넣어서

ax[0].set_title('Pie plot - Survived')

# 첫번째 플랏의 제목 설정

ax[0].set_ylabel('')

# 첫번째 플랏의 y라벨 설정

sns.countplot('Survived', data = df_train, ax=ax[1])

# 카운트 플랏을 그린다. 트레인셋의, 두번째 공간에

ax[1].set_title('Count plot - Survived')

# 카운트 플랏의 제목 설정

plt.show()

# 보여주세요



# 트레인 셋의 생존 0과 1의 비율을 그래프로 보여준다.
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index = True).count()

# 꼭 리스트로 묶어서 해야한다 []

# groupby로 묶어주고 count는 몇개가 있는지 세준다.
pd.crosstab(df_train['Pclass'], df_train['Survived'], margins = True).style.background_gradient(cmap='Pastel1')

# margin은 total을 보여준다.
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index = True).mean().sort_values(by='Survived', ascending = False).plot.bar()



# (0*해당 클래스의 갯수 + 1*해당 클래스의 갯수)/ALL

# 이는 곧 생존률을 의미한다.
y_position = 1.02

f, ax = plt.subplots(1, 2, figsize= (18,8))

df_train["Pclass"].value_counts().plot.bar(color = ["#CD7F32", "#FFDF00", "#D3D3D3"], ax = ax[0])

ax[0].set_title("Number of passengers By Pclass")

ax[0].set_ylabel("Count")

sns.countplot("Pclass", hue = "Survived", data = df_train, ax = ax[1])

ax[1].set_title("Pclass: Survived vs Dead", y = y_position)

plt.show()

     

# Passenger Class에 따른 승객 수와, 생존비율을 알 수 있는 플랏. 3클래스(economy라고 생각)이 가장 많이 탑승하였고, FirstCLass 승객들 생존 비율이 가장 높다.
print("제일 나이가 많은 탑승객 : {:.1f} years".format(df_train["Age"].max()))

print("제일 어린 탑승객 : {:.1f} years".format(df_train["Age"].min()))

print("탑승객 나이의 평균 : {:.1f} years".format(df_train["Age"].mean()))

      
fix, ax = plt.subplots(1, 1, figsize = (9, 5))

sns.kdeplot(df_train[df_train["Survived"] == 1]["Age"], ax=ax)

sns.kdeplot(df_train[df_train["Survived"] == 0]["Age"], ax=ax)    # df_train[df_train["Survived"] == 0: 판다스에서 많이 사용하는 인덱싱 방법, 그것에 ["Age"]만 가져와서 플랏에 넣겠다는 코드

plt.legend(["Survived == 1", "Survived == 0"])

plt.show()



# 데이터들의 분포를 추정하기 위해 kdeplot을 사용 (밀도함수, 히스토그램을 부드럽게 그린것이다.)
fix, ax = plt.subplots(1, 1, figsize = (9, 7))

sns.kdeplot(df_train[df_train["Pclass"] == 1]["Age"], ax=ax)

sns.kdeplot(df_train[df_train["Pclass"] == 2]["Age"], ax=ax)

sns.kdeplot(df_train[df_train["Pclass"] == 3]["Age"], ax=ax)

plt.xlabel("Age")

plt.title("Age Distribution within classes")

plt.legend(["1st Class", "2nd Class", "3rd Class"])

plt.show()                       



# 이런 상황에서 히스토그램으로 하면 겹쳐서 안보이기 때문에 kde를 사용해주는 것

# Class에 따른 Age 분포를 알 수 있다.
fig, ax  = plt.subplots(1, 1, figsize = (9, 5))

sns.kdeplot(df_train[(df_train["Survived"] == 0) & (df_train["Pclass"] == 1)]["Age"], ax=ax)

sns.kdeplot(df_train[(df_train["Survived"] == 1) & (df_train["Pclass"] == 1)]["Age"], ax=ax)

plt.legend(["Survived == 0", "Survived == 1"])

plt.title("1st Class")

plt.show()



# 생존하지 않은 사람중에 Pclass가 1인 사람들의 Age 분포

# 생존한 사람중에 Pclass가 1인 사람들의 Age 분포
fig, ax  = plt.subplots(1, 1, figsize = (9, 5))

sns.kdeplot(df_train[(df_train["Survived"] == 0) & (df_train["Pclass"] == 2)]["Age"], ax=ax)

sns.kdeplot(df_train[(df_train["Survived"] == 1) & (df_train["Pclass"] == 2)]["Age"], ax=ax)

plt.legend(["Survived == 0", "Survived == 1"])

plt.title("2nd Class")

plt.show()



# 생존하지 않은 사람중에 Pclass가 2인 사람들의 Age 분포

# 생존한 사람중에 Pclass가 2인 사람들의 Age 분포
fig, ax  = plt.subplots(1, 1, figsize = (9, 5))

sns.kdeplot(df_train[(df_train["Survived"] == 0) & (df_train["Pclass"] == 3)]["Age"], ax=ax)

sns.kdeplot(df_train[(df_train["Survived"] == 1) & (df_train["Pclass"] == 3)]["Age"], ax=ax)

plt.legend(["Survived == 0", "Survived == 1"])

plt.title("3rd Class")

plt.show()



# 생존하지 않은 사람중에 Pclass가 3인 사람들의 Age 분포

# 생존한 사람중에 Pclass가 3인 사람들의 Age 분포
chage_age_range_survival_ratio = []

i = 80

for i in range(1,81):

    chage_age_range_survival_ratio.append(df_train[df_train["Age"] < i]["Survived"].sum()/len(df_train[df_train["Age"] < i]["Survived"])) # i보다 작은 나이의 사람들이 생존률



plt.figure(figsize = (7, 7))

plt.plot(chage_age_range_survival_ratio)

plt.title("Survival rate change depending on range of Age", y = 1.02)

plt.ylabel("Survival rate")

plt.xlabel("Range of Age(0-x)")

plt.show()

    

# 그래프를 보면 나이가 어릴수록 생존확률이 높고 많아질수록 적은것을 알 수 있다.
f, ax = plt.subplots(1, 2, figsize=(18, 8))

sns.violinplot("Pclass","Age", hue = "Survived", data = df_train, scale = "count", split = True, ax=ax[0])

ax[0].set_title("Pclass and Age vs Survived")

ax[0].set_yticks(range(0, 110, 10))



sns.violinplot("Sex", "Age", hue = "Survived", data = df_train, scale = "count", split = True, ax=ax[1])

ax[1].set_title("Sex and Age vs Survived")

ax[1].set_yticks(range(0, 110, 10))



plt.show()



# 나이를 기준으로 Pclass에 따른 생존률, 성별에 따른 생존률을 한눈에 볼 수 있다.

# 결과적으로 Pclass가 좋을수록 생존률이 높고, 여자가 남자보다 생존률이 높은것을 알 수 있다.
f, ax = plt.subplots(1, 1, figsize=(7,7))

df_train[["Embarked","Survived"]].groupby(["Embarked"], as_index=True).mean().sort_values(by="Survived",

                                                                                         ascending = False).plot.bar(ax=ax)

# ascending은 오름차순(True), 내림차순(False)

# sort_values, sort_index()에 대해서 더 찾아보기
f, ax = plt.subplots(2, 2, figsize=(20,15))

sns.countplot("Embarked", data = df_train, ax=ax[0,0])

ax[0,0].set_title("(1) No. of Passengers Boared")



sns.countplot("Embarked", hue = "Sex", data = df_train, ax=ax[0,1])

ax[0,1].set_title("(2) Male-Female split for Embarked")



sns.countplot("Embarked", hue = "Survived", data = df_train, ax=ax[1,0])

ax[1,0].set_title("(3) Embarked vs Survived")



sns.countplot("Embarked", hue = "Pclass", data = df_train, ax=ax[1,1])

ax[1,1].set_title("(4) Embarked vs Pclass")



plt.subplots_adjust(wspace = 0.4, hspace = 0.5) # 겹치지않게 하는 공백 만들어주기

plt.show()



# 결과적으로 C에서 탑승한 사람들이 Firstclass가 많고 여자가 많기 때문에 생존률이 높게 나온다.
df_train["FamilySize"] = df_train["SibSp"] + df_train["Parch"]+1

df_test["FamilySize"] = df_test["SibSp"] + df_test["Parch"]+1



# 판다스 끼리는 연산이 가능하다. 새로운 feature인 "FamilySize"를 만든다.
df_train["FamilySize"].head(5)
df_test["FamilySize"].head()
print("Maximum size of Family: ", df_train["FamilySize"].max())

print("Minimum size of Family: ", df_train["FamilySize"].min())
f, ax = plt.subplots(1, 3, figsize = (40, 10))

sns.countplot("FamilySize", data = df_train, ax = ax[0])

ax[0].set_title("(1) No. of Passenger Boarded", y = 1.02)



sns.countplot("FamilySize", hue = "Survived", data = df_train, ax = ax[1])

ax[1].set_title("(2) Survived countplot depending of FamilySize")



df_train[["FamilySize", "Survived"]].groupby(["FamilySize"], as_index = True).mean().sort_values(by = "Survived",

                                                                                                      ascending = False).plot.bar(ax = ax[2])

ax[2].set_title("(3) Survived rate depending on FamilySize", y = 1.02)



plt.subplots_adjust(wspace = 0.2, hspace = 0.5)

plt.show()



# 첫번째 플랏은 가족수 (1인 부터 11인까지)에 따른 탑승객 수, 두번째 플랏은 가족 수에 따른 생존자 수, 세번째 플랏은 가족수에 따른 생존률

# 가족 수 가 4명인 가족의 생존률이 제일 높다.
f, ax = plt.subplots(1, 1, figsize = (8,8))

g = sns.distplot(df_train["Fare"], color = "b", label="Skewness: {:2f}".format(df_train["Fare"].skew()), ax=ax)

g = g.legend(loc = "best")



# skewness는 분포가 얼마나 비대칭인가를 알려주는 것 (왜도: a=0이면 정규분포, a<0 이면 오른쪽으로 치우침, a>0이면 왼쪽으로 치우침)
# skewness를 없애주기 위해 로그를 취한다.



df_train["Fare"] = df_train["Fare"].map(lambda i:np.log(i) if i>0 else 0)
df_train["Fare"].head(5)
f, ax = plt.subplots(1, 1, figsize = (8,8))

g = sns.distplot(df_train["Fare"], color = "b", label="Skewness: {:2f}".format(df_train["Fare"].skew()), ax=ax)

g = g.legend(loc = "best")



# 로그를 취해준 그래프가 바뀐 모습을 볼 수 있다. (정규근사화) (간단한 feature engineering이라고 할 수 있다.)
# 가장 먼저 NULL데이터를 처리한다. (채워준다)



df_train["Age"].isnull().sum()

# Age에는 NULL 데이터가 177개 있다.
# 이름에 들어가는 호칭(mr, ms, mrs)등을 이용하여 그룹핑해준다. 정규표현식을 이용하여 추출한다.



df_train["Initial"] = df_train["Name"].str.extract("([A-Za-z]+)\.") # Initial로 호칭을 저장해준다.

df_test["Initial"] = df_test["Name"].str.extract("([A-Za-z]+)\.")
df_train.head()
df_test.head()
pd.crosstab(df_train["Initial"], df_train["Sex"]).T.style.background_gradient(cmap = "Pastel2")



# 성별로 호칭을 확인 crosstab을 이용하여 빈도표를 만든다.
# 여러개의 호칭을 간단하게 치환시켜준다. (replace() 사용)



df_train["Initial"].replace(["Mlle","Mme", "Ms", "Dr","Major","Lady","Countess", "Jonkheer", "Col", "Rev", "Capt", "Sir", "Don", "Dona"],

                           ["Miss", "Miss","Miss", "Mr", "Mr", "Mrs", "Mrs", "Other", "Other", "Other", "Mr", "Mr", "Mr", "Mr"], inplace = True)



df_test["Initial"].replace(["Mlle","Mme", "Ms", "Dr","Major","Lady","Countess", "Jonkheer", "Col", "Rev", "Capt", "Sir", "Don", "Dona"],

                           ["Miss", "Miss","Miss", "Mr", "Mr", "Mrs", "Mrs", "Other", "Other", "Other", "Mr", "Mr", "Mr", "Mr"], inplace = True)
df_train.groupby("Initial").mean()
df_train.groupby("Initial")["Survived"].mean().plot.bar()



# 역시 Miss나 Mrs 같은 여성은 생존률이 높고, Master의 경우 평균나이가 어린데 생존률이 높다. 또한 Mr는 생존률이 낮다.
# train, test 두개의 데이터 셋을 합쳐서 통계량을 확인한다. (concat 사용: 데이터셋에 데이터셋을 쌓는 함수)

df_all = pd.concat([df_train,df_test])

df_all.shape
df_all.groupby("Initial").mean()



# 여기서 Age의 평균을 이요하여 NULL 값들을 채워준다.
# loc 인덱서: 보고싶은 행이나 열의 데이터들을 조건에 맞게 반환해준다. 

# 지금은 NULL데이터를 가지는 인덱스를 반환하게 해주어야 한다.



df_train.loc[(df_train["Age"].isnull()) & (df_train["Initial"] == "Mr"), "Age"] = 33

df_train.loc[(df_train["Age"].isnull()) & (df_train["Initial"] == "Master"),"Age"] = 5

df_train.loc[(df_train["Age"].isnull()) & (df_train["Initial"] == "Miss"), "Age"] = 22

df_train.loc[(df_train["Age"].isnull()) & (df_train["Initial"] == "Mrs"), "Age"] = 37

df_train.loc[(df_train["Age"].isnull()) & (df_train["Initial"] == "Other"), "Age"] = 45



df_test.loc[(df_test["Age"].isnull()) & (df_test["Initial"] == "Mr"), "Age"] = 33

df_test.loc[(df_test["Age"].isnull()) & (df_test["Initial"] == "Master"),"Age"] = 5

df_test.loc[(df_test["Age"].isnull()) & (df_test["Initial"] == "Miss"), "Age"] = 22

df_test.loc[(df_test["Age"].isnull()) & (df_test["Initial"] == "Mrs"), "Age"] = 37

df_test.loc[(df_test["Age"].isnull()) & (df_test["Initial"] == "Other"), "Age"] = 45



# Age가 NULL, Initial이 Mr인것 반환, Age 컬럼만 뽑고 그걸 전부 33(위에서 본 Age의 평균)으로 채워준다.

# 전부 다 동일한 방법으로 NULL데이터 채워주기.
# 확인



df_train["Age"].isnull().sum()
df_test["Age"].isnull().sum()
df_train["Embarked"].isnull().sum()
df_train.shape



# 891개의 row 중에 2개밖에 결측치가 없으므로 최빈값으로 대체해준다. 
df_train["Embarked"].fillna("S", inplace = True)



# fillna는 결측치 값을 지정값으로 전부 채워준다. EDA 과정에서 S가 가장 많았으므로 대체해준다.
df_test["Embarked"].isnull().sum()
df_train["Age_Categ"] = 0

df_test["Age_Categ"] = 0



# 새로운 feature 생성
def category_age(x):

    if x < 10:

        return 0

    elif x < 20:

        return 1

    elif x < 30:

        return 2

    elif x < 40:

        return 3 

    elif x < 50:

        return 4

    elif x < 60: 

        return 5

    elif x < 70: 

        return 6

    else:

        return 7

    

# apply 사용을 위해 함수를 만들어 준다.    
df_train["Age_Categ"] = df_train["Age"].apply(category_age)

df_test["Age_Categ"] = df_test["Age"].apply(category_age)



# apply 함수를 이용하여 train, test 셋에 범주화 시킨 나이 열의 정보를 추가한다.
df_train["Age_Categ"].head(10)
df_test["Age_Categ"].head(10)
# 이런 식으로 하드코딩하여 train, test 셋에 범주화 시킨 나이 열의 정보를 추가하는 방법도 있다.



df_train.loc[df_train["Age"] < 10, "Age_Categ"] = 0

df_train.loc[(10 <= df_train["Age"]) & (df_train["Age"] < 20), "Age_Categ"] = 1

df_train.loc[(20 <= df_train["Age"]) & (df_train["Age"] < 30), "Age_Categ"] = 2

df_train.loc[(30 <= df_train["Age"]) & (df_train["Age"] < 40), "Age_Categ"] = 3

df_train.loc[(40 <= df_train["Age"]) & (df_train["Age"] < 50), "Age_Categ"] = 4

df_train.loc[(50 <= df_train["Age"]) & (df_train["Age"] < 60), "Age_Categ"] = 5

df_train.loc[(60 <= df_train["Age"]) & (df_train["Age"] < 70), "Age_Categ"] = 6

df_train.loc[(70 <= df_train["Age"]), "Age_Categ"] = 7



df_test.loc[df_test["Age"] < 10, "Age_Categ"] = 0

df_test.loc[(10 <= df_test["Age"]) & (df_test["Age"] < 20), "Age_Categ"] = 1

df_test.loc[(20 <= df_test["Age"]) & (df_test["Age"] < 30), "Age_Categ"] = 2

df_test.loc[(30 <= df_test["Age"]) & (df_test["Age"] < 40), "Age_Categ"] = 3

df_test.loc[(40 <= df_test["Age"]) & (df_test["Age"] < 50), "Age_Categ"] = 4

df_test.loc[(50 <= df_test["Age"]) & (df_test["Age"] < 60), "Age_Categ"] = 5

df_test.loc[(60 <= df_test["Age"]) & (df_test["Age"] < 70), "Age_Categ"] = 6

df_test.loc[(70 <= df_test["Age"]), "Age_Categ"] = 7
df_train.head()
df_test.head()
# Age를 범주화 시켰으므로 필요없어진 Age 컬럼을 삭제시켜준다.



df_train.drop(["Age"], axis = 1 ,inplace = True)

df_test.drop(["Age"], axis = 1, inplace = True)
df_train.head()
# Initial을 제대로 된 러닝을 위해 숫자로 바꿔주는 작업을 해준다.

df_train["Initial"].unique()
df_train["Initial"] = df_train["Initial"].map({"Master" : 0, "Miss" : 1, "Mr" : 2, "Mrs" : 3, "Other" : 4})

df_test["Initial"] = df_test["Initial"].map({"Master" : 0, "Miss" : 1, "Mr" : 2, "Mrs" : 3, "Other" : 4})
# 마찬가지로 Embarked도 숫자로 바꿔준다.

df_train["Embarked"].value_counts()
df_train["Embarked"] = df_train["Embarked"].map({"C" : 0, "Q" : 1, "S" : 2})

df_test["Embarked"] = df_test["Embarked"].map({"C" : 0, "Q" : 1, "S" : 2})
df_train.head()
df_test.head()
# Sex도 바꿔준다.

df_train["Sex"].unique()
df_train["Sex"] = df_train["Sex"].map({"female" : 0, "male" : 1})

df_test["Sex"] = df_test["Sex"].map({"female" : 0, "male" : 1})

heatmap_data = df_train[["Survived", "Pclass", "Sex", "Fare", "Embarked", "FamilySize", "Initial", "Age_Categ"]]
colormap = plt.cm.PuBu

plt.figure(figsize=(10, 8))

plt.title("Person Correlation of Features", y = 1.05, size = 15)

sns.heatmap(heatmap_data.astype(float).corr(), linewidths = 0.1, vmax = 1.0,

           square = True, cmap = colormap, linecolor = "white", annot = True, annot_kws = {"size" : 16})





# 상관계수 분석을 통해 겹치는 feature가 있는지, 어느 feature끼리 상관관계를 보이는지 알 수 있다.
# 모델의 성능을 높이기 위해서 카테고리화 시켰던 데이터의 정보를 잘 사용하기 위해 형태를 바꿔주는 작업이다.

# One Hot Encoding은 이것들을 벡터화 시키는 것이다. (Dummy)



df_train = pd.get_dummies(df_train, columns = ["Initial"], prefix = "Initial")

df_test = pd.get_dummies(df_test, columns = ["Initial"], prefix = "Initial")
# Initial과 같이 마찬가지로 카테고리화 시켰던 Embarked도 실행해준다.



df_train = pd.get_dummies(df_train, columns = ["Embarked"], prefix = "Embarked")

df_test = pd.get_dummies(df_test, columns = ["Embarked"], prefix = "Embarked")
# 쓰이지 않는 feature들을 삭제해준다. 

df_train.head(1)
df_train.drop(["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin"], axis = 1, inplace = True)

df_test.drop(["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin"], axis = 1, inplace = True)
df_train.head() 



# 깔끔하게 정리된 것 확인
df_test.head()
kfold = StratifiedKFold(n_splits=10)
df_train["Survived"] = df_train["Survived"].astype(int)



Y_train = df_train["Survived"]



X_train = df_train.drop(labels = ["Survived"],axis = 1)
# 여러 알고리즘으로 모델링 테스트

random_state = 2

classifiers = []

classifiers.append(SVC(random_state = random_state))

classifiers.append(DecisionTreeClassifier(random_state = random_state))

classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state = random_state), random_state = random_state, learning_rate = 0.1))

classifiers.append(RandomForestClassifier(random_state = random_state))

classifiers.append(ExtraTreesClassifier(random_state = random_state))

classifiers.append(GradientBoostingClassifier(random_state = random_state))

classifiers.append(MLPClassifier(random_state = random_state))

classifiers.append(KNeighborsClassifier())

classifiers.append(LogisticRegression(random_state = random_state))

classifiers.append(LinearDiscriminantAnalysis())



cv_results = []

for classifier in classifiers:

    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs = 4))

    

cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())

    

cv_res = pd.DataFrame({"CrossValMeans": cv_means, "CrossValerrors": cv_std,

                       "Algorithm": ["SVC", "DecisionTree", "AdaBoost", "RandomForest",

                                     "ExtraTrees", "GradientBoosting", "MultipleLayerPerceptron", "KNeighboors",

                                    "LogisticRegression", "LinearDiscriminantAnalysis"]})



g = sns.barplot("CrossValMeans", "Algorithm", data = cv_res, palette = "Set3",

               orient = "h", **{'xerr': cv_std})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")
# 5가지 모델에 대한 그리드 검색 최적화 수행

    

# Adaboost

DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state = 7)



ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],

                 "base_estimator__splitter": ["best", "random"],

                 "algorithm": ["SAMME", "SAMME.R"],

                 "n_estimators": [1,2],

                 "learning_rate": [0.0001,0.001, 0.01, 0.1, 0.2, 0.3, 1.5]}



gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv = kfold, scoring = "accuracy",

                       n_jobs = 4, verbose = 1)

gsadaDTC.fit(X_train, Y_train)

ada_best = gsadaDTC.best_estimator_



gsadaDTC.best_score_
# ExtraTrees



ExtC = ExtraTreesClassifier()



# 최적의 매개 변수를위한 검색 그리드



ex_param_grid = {"max_depth": [None],

                "max_features": [1,2,10],

                "min_samples_split": [2, 3, 10],

                "min_samples_leaf": [1,3,10],

                "bootstrap": [False],

                "n_estimators": [100, 300],

                "criterion": ["gini"]}



gsExtC = GridSearchCV(ExtC, param_grid = ex_param_grid, cv = kfold, scoring = "accuracy",

                     n_jobs = 4, verbose = 1)



gsExtC.fit(X_train, Y_train)

ExtC_best = gsExtC.best_estimator_



gsExtC.best_score_
# RandomForestClassifier

RFC = RandomForestClassifier()



# 최적의 매개 변수를위한 검색 그리드

rf_param_grid = {"max_depth": [None],

                "max_features": [1,3,10],

                "min_samples_split": [2,3,10],

                "min_samples_leaf": [1,2,10],

                "bootstrap": [False],

                "n_estimators": [100,300],

                "criterion": ["gini"]}



gsRFC = GridSearchCV(RFC, param_grid = rf_param_grid, cv=kfold, scoring = "accuracy", n_jobs = 4,

                    verbose = 1)



gsRFC.fit(X_train,Y_train)

RFC_best = gsRFC.best_estimator_



gsRFC.best_score_

# Gradient boosting tunning



GBC = GradientBoostingClassifier()

gb_param_grid = {"loss": ["deviance"],

                "n_estimators": [100,200,300],

                "learning_rate": [0.1, 0.05, 0.01],

                "max_depth": [4, 8],

                "min_samples_leaf": [100,150],

                "max_features": [0.3, 0.1]}



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv = kfold, scoring = "accuracy",

                    n_jobs = 4, verbose = 1)



gsGBC.fit(X_train, Y_train)

GBC_best = gsGBC.best_estimator_



gsGBC.best_score_
# SVC classifier

SVMC = SVC(probability=True)

svc_param_grid = {'kernel': ['rbf'], 

                  'gamma': [ 0.001, 0.01, 0.1, 1],

                  'C': [1, 10, 50, 100,200,300, 1000]}



gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsSVMC.fit(X_train,Y_train)



SVMC_best = gsSVMC.best_estimator_



# Best score

gsSVMC.best_score_
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    """Generate a simple plot of the test and training learning curve"""

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt



g = plot_learning_curve(gsRFC.best_estimator_,"RF mearning curves",X_train,Y_train,cv=kfold)

g = plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees learning curves",X_train,Y_train,cv=kfold)

g = plot_learning_curve(gsSVMC.best_estimator_,"SVC learning curves",X_train,Y_train,cv=kfold)

g = plot_learning_curve(gsadaDTC.best_estimator_,"AdaBoost learning curves",X_train,Y_train,cv=kfold)

g = plot_learning_curve(gsGBC.best_estimator_,"GradientBoosting learning curves",X_train,Y_train,cv=kfold)
votingC = VotingClassifier(estimators = [("rfc", RFC_best), ("extc", ExtC_best),

                                        ("svc", SVMC_best), ("adac", ada_best),

                                        ("gbc", GBC_best)], voting = "soft", n_jobs = 4)



votingC = votingC.fit(X_train, Y_train)
# 사이킷런 import (파이썬 라이브러리를 이용한 머신러닝 책 참고)

# 문제가 binary 분류 이므로 randomforestClassifier을 불러와준다.



#from sklearn.ensemble import RandomForestClassifier

#from sklearn import metrics

#from sklearn.model_selection import train_test_split
# test 하기전 validation 과정을 거쳐주어야 한다.



#X_train = df_train.drop("Survived", axis = 1).values

#target_label = df_train["Survived"].values

#X_test = df_test.values
#X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size = 0.3, random_state = 2019)



# 파라미터를 어떻게 튜닝하느냐에 따라 성능이 많이 차이난다. 경험이 중요하므로 많이 찾아보고 사용한다.

# train 데이터의 30%를 validation으로 주고 70%을 train으로 둔다
# 모델 세우고 train set에 학습시키기 (default setting)



#model = RandomForestClassifier()

#model.fit(X_tr, y_tr)
# validation을 가지고 예측하기 



#prediction = model.predict(X_vld)
#prediction
#print("총 {}명 중 {:.2f}%의 정확도로 생존 예측".format(y_vld.shape[0], 100*metrics.accuracy_score

                                           #(prediction, y_vld)))
#model.feature_importances_
#from pandas import Series
#feature_importance = model.feature_importances_

#Series_feat_imp = Series(feature_importance, index = df_test.columns)
#df_test.head()#
#plt.figure(figsize = (8,8))#

#Series_feat_imp.sort_values(ascending = True).plot.barh()

#plt.xlabel("Feature importance")

#plt.ylabel("Feature")

#plt.show()



# Feature Importance를 보고 중요하지 않은 feature를 selection할 수도 있다.
submission = pd.read_csv("../input/gender_submission.csv")
submission.head()
df_test["Fare"].fillna("35.6271", inplace = True)

X_test = df_test.values
prediction = votingC.predict(X_test)
submission["Survived"] = prediction
submission.to_csv("./The_first_submission.csv", index = False)