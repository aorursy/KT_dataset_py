import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")



pd.options.display.max_columns = 50

pd.options.display.max_rows = 500



import matplotlib

matplotlib.rc("font", family = "AppleGothic")

matplotlib.rc("axes", unicode_minus = "False")



from IPython.display import set_matplotlib_formats

set_matplotlib_formats("retina")
train=pd.read_csv("../input/train.csv",index_col="PassengerId")

test=pd.read_csv("../input/test.csv",index_col="PassengerId")

print(train.shape)

print(test.shape)
train.head()
# 이 타이타닉에서 sex는 매우 중요함으로 이를 숫자로 변환시켜서 넣는다



train.loc[train["Sex"]=="male","Sex2"]=0

train.loc[train["Sex"]=="female","Sex2"]=1

test.loc[test["Sex"]=="male","Sex2"]=0

test.loc[test["Sex"]=="female","Sex2"]=1

print(train.shape)

print(test.shape)
train.head(2)
test.head(2)
# Embarked는 3개 뿐이고 이 것이 영향을 주기 때문에 넣어주는 것이 맞다.



pd.pivot_table(index="Embarked",values="SibSp",data=test)
train["Embarked_C"]=train["Embarked"]=="C"

train["Embarked_Q"]=train["Embarked"]=="Q"

train["Embarked_S"]=train["Embarked"]=="S"

print(train.shape)

train.head(2)
test["Embarked_C"]=test["Embarked"]=="C"

test["Embarked_Q"]=test["Embarked"]=="Q"

test["Embarked_S"]=test["Embarked"]=="S"

print(test.shape)

test.head(2)
train.loc[train["Age"].isnull(),"Age"]=0

train.head(6)

train.loc[train["Age"].isnull()]
test.loc[test["Age"].isnull(),"Age"]=0

test.loc[test["Age"].isnull()]

print(test.shape)

test.head(6)

test.loc[test["Age"].isnull()]
# fare 가격에 따른 영향이 있을 것으로 보여서 nan을 제거해준다



test.loc[test["Fare"].isnull()]
test.loc[test["Fare"].isnull(),"Fare"]=0

test.loc[test["Fare"].isnull()]

test.head() 
# 남자가 죽은 숫자가 여성보다 월등히 많은 관계로 성비율을 매우 중요하다고 볼 수 있다

# 클래스별로 3번이 특히 남성의 0 횟수가 많고 / embarked의 경우 Q는 오히려 남성이 더 많이 살았음을 알 수 있다



# 이 모든 것들은 필요한 자료라고 판단되어 적용한다

fig=plt.figure(figsize=[15,10])

ax1=fig.add_subplot(2,3,1)

ax1=sns.countplot(x="Sex",hue="Survived",data=train)



ax1=fig.add_subplot(2,3,2)

ax1=sns.countplot(x="Pclass",hue="Survived",data=train)



ax1=fig.add_subplot(2,3,3)

ax1=sns.countplot(x="Embarked",hue="Survived",data=train)
# 집중적으로 살펴보면 생존율을 좀더 자세히 볼 수 있다

# 이 둘을 전부 적용하면 이 생존율에 맞춰서 적용이 될 수 있다고 본다

pd.pivot_table(index=["Sex","Pclass","Embarked"],values="Survived",data=train)
# 분산도를 조사해본결과 

# o~10세 중 20fare 이하인 사람들은 거의 다 1 - > 저요금을 이용한 고객들의 경우 아이들위주로 살렸을 가능성이 높다

# 60세 이후로는 대부분 0을 가르킨다 -> 대부분 탈출하지 못했을 거라는 의미임



train2=train.loc[train["Fare"]<100]

sns.lmplot(x="Age",y="Fare",hue="Survived",data=train2,fit_reg=None)
train["Family"]=train["SibSp"]+train["Parch"]+1

test["Family"]=test["SibSp"]+test["Parch"]+1
# 같이 탑승한 가족 및 본인 숫자를 더했을 때 어떤 결과가 나오는지 살펴보고자 한다 

# 현재 이와같은 결과를 보았을 때 1인가구(혼자탑승) 수가 많고 그 사람들은 대부분 0의 비율이 높다고 볼 수 있다

# 반면에 2,3,4,5는 오히려 산 사람이 많다 

# 이는 변수로 작용할 가능성이 있기 때문에 묶어서 적용해보기로 한다

sns.countplot(x="Family",hue="Survived",data=train)
train.loc[train["Family"]<12,"FamilySize"]="Big"

train.loc[train["Family"]<5,"FamilySize"]="Nuclear"

train.loc[train["Family"]<2,"FamilySize"]="Single"
train["Big"]=train["FamilySize"]=="Big"

train["Nuclear"]=train["FamilySize"]=="Nuclear"

train["Single"]=train["FamilySize"]=="Single"



train.head(2)
# Nuclear, 즉 2,3,4,5인 가구의 경우 생존확률이 57프로 정도 되기 때문에 이를 적용시켜줄 필요가 있다 

pd.pivot_table(index="FamilySize",values="Survived",data=train)
test.loc[test["Family"]<12,"FamilySize"]="Big"

test.loc[test["Family"]<5,"FamilySize"]="Nuclear"

test.loc[test["Family"]<2,"FamilySize"]="Single"
test["Big"]=test["FamilySize"]=="Big"

test["Nuclear"]=test["FamilySize"]=="Nuclear"

test["Single"]=test["FamilySize"]=="Single"



test.head()
# 이름이 뒤죽박죽인데 그 영향력이 있을 것이라고 판단하고 적용해본다 

# 먼저 unique를 통해 성별 및 직함을 파악해본다 

def newname(name):

    return name.split(",")[1].split(".")[0]



test["Name"].apply(newname).unique()
# 분석결과 나머지 이름은 그 숫자가 상대적으로 적기에 배제하고 나머지 5숫자만 분석해보기로 한다 

# 이 5개 숫자를 별도로 빼내서 적용해본다 

train.loc[train["Name"].str.contains("Mr"),"Profile"]="Mr"

train.loc[train["Name"].str.contains("Mrs"),"Profile"]="Mrs"

train.loc[train["Name"].str.contains("Miss"),"Profile"]="Miss"

train.loc[train["Name"].str.contains("Master"),"Profile"]="Master"

train.loc[train["Name"].str.contains("Ms"),"Profile"]="Ms"



train.head()
test.loc[test["Name"].str.contains("Mr"),"Profile"]="Mr"

test.loc[test["Name"].str.contains("Mrs"),"Profile"]="Mrs"

test.loc[test["Name"].str.contains("Miss"),"Profile"]="Miss"

test.loc[test["Name"].str.contains("Master"),"Profile"]="Master"

test.loc[test["Name"].str.contains("Ms"),"Profile"]="Ms"



test.head()
# 다른것은 별다른 특이사항이 없다 

# mr는 남자인데 이미 성별에서 0의 비율이 더 높았고 / 여성 역시 1이 더 높았기 때문에 mrs, miss 등도 별다른 이슈가 못된다



# 다만 master는 변수가 될 수 있는 자료라고 생각이 된다

# master는 보통 남자에게 붙여지는 칭호인데 남자면 0이 더 높아야 하는데 이것은 1이 더 높다 즉 master에서 생존할 남성확률이 50%가 넘는다는 것

# 그래서 이 master만 따로 적용해야할 필요성이 있다 

sns.countplot(x="Profile",hue="Survived",data=train)
train["Master"]=train["Profile"]=="Master"

test["Master"]=test["Profile"]=="Master"

# 적용후 확인해보았을 때 true부분에서 더 높음을 알 수 있다 

train3=train.loc[train["Fare"]<200]

sns.barplot(x="Master",y="Fare",hue="Survived",data=train3)
feature_names=["Sex2","Pclass","Embarked_C","Embarked_Q","Embarked_S","Fare","Nuclear","Big","Single","Master"]

label_names=["Survived"]



x_train=train[feature_names]

y_train=train[label_names]

x_test=test[feature_names]



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)
from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier(max_depth=6)

model.fit(x_train,y_train)
# DecisionTreeClassifier 모델에서 tree모델을 활성화해본다

import graphviz

from sklearn.tree import export_graphviz

mott=export_graphviz(model,feature_names=feature_names,class_names=["survived","perish"],out_file=None)

graphviz.Source(mott)