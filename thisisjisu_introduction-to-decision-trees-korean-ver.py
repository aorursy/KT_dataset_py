# 분석을 위해 필요한 라이브러리들 import

import numpy as np

import pandas as pd

import re

import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



#결정 트리 만드는데 필요한 라이브러리들 import

from sklearn import tree

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from IPython.display import Image as PImage

from subprocess import check_call

from PIL import Image, ImageDraw, ImageFont



# 데이터 가져오기

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# 결과값 넣을 test셋 준비

PassengerId = test['PassengerId']



# train 잠깐만 보면...

train.head(3)

#주의 : 데이터 처리 과정은 Sina, Anisotropic, Megan Risdal 방식을 참고하여 만들어졌다.

# 학습 데이터 셋 : original_train

original_train = train.copy() # copy()함수로 데이터셋 복제

full_data = [train, test]



# 데이터셋 형성 및 Null Dataset 정리



# Has_Cabin 데이터셋 정리

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)



# Sibsp와 Parch의 데이터셋을 통해 FamilySize라는 새로운 데이터셋 생성

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# FamilySize를 통해 IsAlone이라는 새로운 데이터셋 생성

for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    

# Embarked의 Null 정리하기

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    

# Fare의 Null 정리하기

for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())



# Age의 Null 정리하기

for dataset in full_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

# Mapping 작업



# Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Mapping titles

    title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] ;

# 의미있는 데이터(학습할 의향 의사가 표시된 데이터만 골라내기)

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

train = train.drop(drop_elements, axis = 1)

test  = test.drop(drop_elements, axis = 1)
train.head(3)
colormap = plt.cm.viridis

plt.figure(figsize=(12,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
# Title, Sex 각각 비교하기



train[['Title', 'Survived']].groupby(['Title'], as_index=False).agg(['mean', 'count', 'sum'])

# 여기서 MEAN : 생존 확률, COUNT : 전체 관측 숫자, SUM : 생존숫자

# title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5} 
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).agg(['mean', 'count', 'sum'])

# 여기서 MEAN : 생존 확률, COUNT : 전체 관측 숫자, SUM : 생존숫자

# sex_mapping = {{'female': 0, 'male': 1}
#기존에 존재하는 original_train 데이터 셋을 통해서 Title 별로 분류된 Sex 분포를 확인해보자.

title_and_sex = original_train.copy()[['Name', 'Sex']]

title_and_sex['Title'] = title_and_sex['Name'].apply(get_title) #Title 섹션 생성

title_and_sex['Sex'] = title_and_sex['Sex'].map( {'female': 0, 'male': 1} ).astype(int) #Mapping 과정



title_and_sex[['Title', 'Sex']].groupby(['Title'], as_index=False).agg(['mean', 'count', 'sum']) #분류하기

#지니계수 계산 함수 정의하기

def get_gini_impurity(survived_count, total_count):

    survival_prob = survived_count/total_count #현재 survived 계산 값(survived 자체가 0,1로 되어있어서 두개의 범주 사이에 생각한다.)

    not_survival_prob = (1- survival_prob)

    

    random_observation_survived_prob = survival_prob

    random_observation_not_survived_prob = (1 - random_observation_survived_prob)

    

    mislabelling_survided_prob = not_survival_prob * random_observation_survived_prob #survived에 속하였을 때의 I값

    mislabelling_not_survided_prob = survival_prob * random_observation_not_survived_prob#not survived에 속하였을 때의 I값

    

    gini_impurity = mislabelling_survided_prob + mislabelling_not_survided_prob #합계

    

    return gini_impurity
gini_impurity_starting_node = get_gini_impurity(342,891)

gini_impurity_starting_node
# 분리 후 Sex 특성에서의 각각의 지니계수 값을 본다면

gini_impurity_men = get_gini_impurity(109, 577)

print(gini_impurity_men)



gini_impurity_women = get_gini_impurity(233, 314)

print(gini_impurity_women)
# 분리 후 Title 특성에서의 각각의 지니계수 값을 본다면

gini_impurity_title_Mr = get_gini_impurity(81, 517)

print(gini_impurity_title_Mr)



gini_impurity_title_others = get_gini_impurity(261, 374)

print(gini_impurity_title_others)
# Sex의 분리 후 전체 지니계수값을 본다면

men_weight = 577/891

women_weight = 314/891

weighted_gini_impurity_sex_split = (gini_impurity_men * men_weight) + (gini_impurity_women * women_weight)

print(weighted_gini_impurity_sex_split)



#Title의 분리 후 전체 지니계수값을 본다면

title_1_weight = 517/891

title_others_weight = 374/891

weighted_gini_impurity_title_split = (gini_impurity_title_Mr * title_1_weight) + (gini_impurity_title_others * title_others_weight)

print(weighted_gini_impurity_title_split)
# 지니 계수 변화를 Sex, Title에서 각각 본다면

sex_gini_decrease = weighted_gini_impurity_sex_split - gini_impurity_starting_node

print(sex_gini_decrease)



title_gini_decrease = weighted_gini_impurity_title_split - gini_impurity_starting_node

print(title_gini_decrease)

cv = KFold(n_splits = 10) #cross validation에서 데이터 셋을 10개의 폴더로 나누어 계산

accuracies = list() # 정확도를 적을 리스트 생성



max_attributes = len(list(test)) #가지고 있는 특성  = 만들수 있는 최대 split 개수

depth_range = range(1, max_attributes + 1)



# Cross validation을 통한 최선의 max depth 결정

for depth in depth_range:

    fold_accuracy = []

    tree_model = tree.DecisionTreeClassifier(max_depth = depth)  # sklearn의 결정트리제조 부름

    

    # 각각에서 어떻게 할지 결정

    for train_fold, valid_fold in cv.split(train):

        

        # 일단 추출 작업

        f_train = train.loc[train_fold] # 전체에서 해당 학습 데이터 추출

        f_valid = train.loc[valid_fold] # 전체에서 해당 평가할 데이터 추출



        # 추출된 학습데이터 학습

        model = tree_model.fit(X = f_train.drop(['Survived'], axis=1), y = f_train["Survived"]) 

        # 학습된 모델을 바탕으로 평가(valid)

        valid_acc = model.score(X = f_valid.drop(['Survived'], axis=1), y = f_valid["Survived"])

        # 정확도 기록하기

        fold_accuracy.append(valid_acc)



    avg = sum(fold_accuracy)/len(fold_accuracy)

    accuracies.append(avg)

    

# 결과 정리 및 print하기

df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})

df = df[["Max Depth", "Average Accuracy"]]

print(df.to_string(index=False))
# train, test 정리

y_train = train['Survived']

x_train = train.drop(['Survived'], axis=1).values 

x_test = test.values



# 결정 트리 정의 및 학습

decision_tree = tree.DecisionTreeClassifier(max_depth = 3)

decision_tree.fit(x_train, y_train)



# 학습된 결정 트리 바탕으로 Test set 예측하기

y_pred = decision_tree.predict(x_test)

submission = pd.DataFrame({"PassengerId": PassengerId,"Survived": y_pred}) #pandas 라이브러리로 제출해야되는 것들만 뽑아내기

submission.to_csv('submission.csv', index=False) #csv 형식 전환하기



print(submission.head(4))
# dot형식의 결정 트리 모델 확인해보기

with open("tree1.dot", 'w') as f:  #sklearn의 tree.export_graphviz 이용

     f = tree.export_graphviz(decision_tree,  

                              out_file=f, #파일 형식 정의

                              max_depth = 3,

                              impurity = True,

                              feature_names = list(train.drop(['Survived'], axis=1)),

                              class_names = ['Died', 'Survived'],

                              #이미지 형식 소개

                              rounded = True,

                              filled= True )

        

#.dot을 .png로 변환

check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])





img = Image.open("tree1.png")

draw = ImageDraw.Draw(img)

font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf', 26)

draw.text((10, 0), # Drawing offset (position)

          '"Title <= 1.5" corresponds to "Mr." title', # Text to draw

          (0,0,255), # RGB desired color

          font=font) # ImageFont object with desired font

img.save('sample-out.png')

PImage("sample-out.png")
# train set에서의 정확도 측정

acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)

acc_decision_tree