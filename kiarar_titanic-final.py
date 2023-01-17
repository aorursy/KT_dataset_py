from IPython.display import Image
Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/5095eabce4b06cb305058603/5095eabce4b02d37bef4c24c/1352002236895/100_anniversary_titanic_sinking_by_esai8mellows-d4xbme8.jpg")
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set() # setting seaborn default for plots

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head(3)
train.shape
test.shape
train.info()
test.info
train.isnull().sum()
#age에 177개의 null 값 있음
#cabin에 687개의 null 값 있음
#embarked에 2개의 null 값 있음
test.isnull().sum()
#age에 86개의 null 값 있음
#cabin에 327개의 null 값 있음
#막대 그래프 그리는 함수 정의
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
#모든 attributes들간의 pair graph
data1 = train.copy(deep = True)

pp = sns.pairplot(data1, hue = 'Survived', palette = 'deep', size=1.8, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )
pp.set(xticklabels=[])
bar_chart('Sex')
#여성의 대부분이 생존
#남성의 대부분이 사망
bar_chart('Pclass')
#1등석 탑승객은 다수가 생존
#3등석 탑승객의 대부분이 사망
#그 이유는 타이타닉호가 후미부터 침몰했지때문(이미지)
bar_chart('SibSp')
#혼자온 사람들은 사망활 확률이 더 높음
bar_chart('Parch')
#혼자온 사람들은 사망활 확률이 더 높음
bar_chart('Embarked')
#탑승객 수: S > C > Q
#하지만 S에서 탑승객은 과반수가 사망
train.head(3)
#한번에 연산하기위해서 train이랑 test 합침
train_test_data = [train, test]

#Title이라는 attribute를 만들고 Name에서 호칭만 빼내서 저장
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train['Title'].value_counts()
test['Title'].value_counts()
#주의: test set에는 새로운 호칭인 Dona가 있음
#Domain Knowledge: 0: 남성, 1 : 젊은 여성, 2 : 중장년 여성, 3: 젊은 남성, 4: 중장년 남성
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 4, "Rev": 4, "Col": 4, "Major": 4, "Mlle": 1,"Countess": 3,
                 "Ms": 1, "Lady": 2, "Jonkheer": 1, "Don": 1, "Dona" : 2, "Mme": 1,"Capt": 4,"Sir": 4 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
bar_chart('Title')
#남성과 중장년 남성은 대부분이 사망
#Name은 더이상 필요없으니 삭제
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)
#남성: 0, 여성: 1
sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
# 해당 Title 평균나이를 빈 age에 넣음 (Mr, Mrs, Miss 등..)
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)
train.groupby("Title")["Age"].transform("median")
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
 
plt.show()
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(0, 20)
#영유아: 0, 청소년: 1, 성인: 2, 중장년: 3, 노인: 4
for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4
train.head(3)
bar_chart('Age')
Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))
#Q에서 탄 승객은 거의 모두 3등석을 탐
#C에서 탄 승객은 1등석을 많이 탐(부촌인가보다)
#S에서 탑승한 인원이 제일 많으니 빈곳은 S로 채움
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
#S : 0, C : 1, Q : 2
embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
#Fare에 null값은 Pclass 그룹의 평균값으로 넣음
#train에는 미싱값이 없는데 test에 1개있으니까 대충 이렇게하자
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
train.head(5)
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
 
plt.show()
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
plt.xlim(0, 20)
for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3
#cabin은 값도 제각각이고 missing values가 너무 많음
train.Cabin.value_counts()
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]
#앞에 알파벳만 건져내기
Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))
#A,B,C는 1등석에만 있다
#피셔 스케일링: 유클리디언 디스턴스로 숫자범위를 좁게 해서 중요도 낮춤
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
# 빈값은 Pclass의 평균값으로 넣음
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'FamilySize',shade= True)
facet.set(xlim=(0, train['FamilySize'].max()))
facet.add_legend()
plt.xlim(0)
#familysize는 중요하지 않다고 판단 -> 낮은 weight를 줌
family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)
#티켓, 형제, 부모 버리기
features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)
#train_data와 target 생성
train_data = train.drop('Survived', axis=1) 
target = train['Survived']

train_data.shape, target.shape
#preprocessing 결과
train.head(5)
#원본 저장
train_copy = train_data
train.info()
# 필요한 모듈 Import
from sklearn.neighbors import KNeighborsClassifier #for KNN
from sklearn.tree import DecisionTreeClassifier #for Decision Tree
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.naive_bayes import GaussianNB #for Naive Bayes
from sklearn.svm import SVC #for SVM
from sklearn.model_selection import KFold #for kfold
from sklearn.model_selection import cross_val_score
import math
import traceback
Image(url = "https://static.oschina.net/uploads/img/201609/26155106_OfXx.png", width=500)
#kfold 라이브러리로 구현
#891개이므로 10번 나눠서 계산
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
Image(url ="https://helloacm.com/wp-content/uploads/2016/03/2012-10-26-knn-concept.png", width=500)
#KNN 라이브러리로 구현
clf = KNeighborsClassifier(n_neighbors = 13)
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring='accuracy')
print(score)
print(round(np.mean(score)*100, 2))
train_data = train_copy
#KNN 직접 구현

# MyKnn 클래스 생성
class MyKNeighborsClassifier:
    def __init__(self, para):
        self.k = para.get('k')  # k 값 : 13
        self.method = para.get('method')  # 거리계산 값: L2(유클리안)

    def train(self, x, y):  # train function
        self.x = x
        self.y = y
        # get matrix information(row, col)
        self.trow, self.tcol = self.x.shape

    def predict(self, arr):
        self.prow, self.pcol = arr.shape
        dist = self.distance(arr)  # 디스턴스 구하기
        result = np.zeros(self.prow)
        num = int(max(self.y)) + 1
        for i in range(0, len(result)):  # voting을 통해 예상값 결정
            table = np.zeros(num)
            for j in range(0, self.k):
                tmp = np.argmin(dist[i])
                table[int(self.y[tmp])] += 1
                dist[i][tmp] = math.inf
            result[i] = np.argmax(table)
        return result

    def distance(self, arr):
        # call function by method
        ords = {
            'Euclidean': 2,
            'L2': 2,
            'L1': 1,
            'Manhattan': 1,
            'Maximum': math.inf
        }.get(self.method, 1)
        dist = self.norm(arr, ords)
        return dist

    # 거리계산 (Manhattan is L1, Euclidean is L2, Maximum is L(inf))
    def norm(self, arr, ords):
            dist = np.zeros((self.prow, self.trow))
            for i in range(0, self.prow):
                tmp = self.x - arr[i]
                if self.prow == 1:
                    tmp = self.x - arr
                if self.trow == 1:
                    dist[i] = np.linalg.norm(tmp, ord=ords)
                    continue
                if self.prow == 1:
                    tmp = self.x - arr
                dist[i] = np.linalg.norm(tmp, axis=1, ord=ords)
            return dist
        
    # Kfold 구현
    def fold(self, n):
        originx = self.x
        originy = self.y
        each = int(self.trow/n)
        tmp = np.zeros(n)
        for i in range(0, n):
            arr = originx[i*each: (i+1)*each]  # slicing
            target = originy[i*each: (i+1)*each]
            if i == 0:
                self.x = originx[(i+1)*each:]
                self.y = originy[(i+1)*each:]
            elif i != (n-1):
                self.x = np.concatenate((originx[(i + 1) * each:], originx[:i * each]), axis=0)
                self.y = np.concatenate((originy[(i + 1) * each:], originy[:i * each]), axis=0)
            else:  # i == (n-1):
                arr = originx[i * each:]
                target = originy[i * each:]
                self.x = originx[:i*each]
                self.y = originy[:i*each]
            self.train(self.x, self.y)  # reset train data
            pre = self.predict(arr)
            tmp[i] = (sum(target[a] == pre[a] for a in range(0, len(target))))/len(target)
        print(tmp)
        print('fold cv : %.3f' % (sum(tmp)*100/n))
        self.train(originx, originy)  # recover train data
knn = MyKNeighborsClassifier({'k': 13, 'method': 'L2'})
tx = np.array(train_data.values)
ty = np.array(target.values)
t = np.array(test.values)
knn.train(tx, ty)

knn.fold(10)
train_data = train_copy
#Decision Tree 라이브러리로 구현
clf = DecisionTreeClassifier(criterion = "gini", random_state = 90, max_depth=4, min_samples_leaf=20)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
round(np.mean(score)*100, 2)
#구현된 트리를 시각화
from sklearn import tree

clf.fit(train_data, target)
with open("decision_tree.txt", "w") as f:
    f = tree.export_graphviz(clf, out_file=f, class_names=list(train_data.columns.values))
train_data = train_copy
import sys
import math

class Node(object):
    def __init__(self, attribute, threshold):
        self.attr = attribute
        self.thres = threshold
        self.left = None
        self.right = None
        self.leaf = False
        self.predict = None

# threshold 고르기
def select_threshold(df, attribute, predict_attr):
    #dataframe -> list로 convert
    values = df[attribute].tolist()
    values = [ float(x) for x in values]
    #list -> set으로 convert: 중복된 value 제거
    values = set(values)
    values = list(values)
    values.sort() #sorting
    max_ig = float("-inf")
    thres_val = 0
    for i in range(0, len(values) - 1):
        thres = (values[i] + values[i+1])/2
        ig = info_gain(df, attribute, predict_attr, thres)
        if ig > max_ig:
            max_ig = ig
            thres_val = thres
    # information gain이 가장 높은 threshold를 return
    return thres_val

# 엔트로피 계산
def info_entropy(df, predict_attr):
    p_df = df[df[predict_attr] == 1] #p_df: 생존
    n_df = df[df[predict_attr] == 0] #n_df: 사망
    p = float(p_df.shape[0])
    n = float(n_df.shape[0])
    # 엔트로피 계산
    if p  == 0 or n == 0:
        I = 0
    else:
        I = ((-1*p)/(p + n))*math.log(p/(p+n), 2) + ((-1*n)/(p + n))*math.log(n/(p+n), 2)
    return I

def remainder(df, df_subsets, predict_attr):
    # test data 개수
    num_data = df.shape[0]
    remainder = float(0)
    for df_sub in df_subsets:
        if df_sub.shape[0] > 1:
            remainder += float(df_sub.shape[0]/num_data)*info_entropy(df_sub, predict_attr)
    return remainder

# threshold로 informaion gain 계산 
def info_gain(df, attribute, predict_attr, threshold):
    sub_1 = df[df[attribute] < threshold]
    sub_2 = df[df[attribute] > threshold]
    # information gain = entropy - remainder
    ig = info_entropy(df, predict_attr) - remainder(df, [sub_1, sub_2], predict_attr)
    return ig #계산값은 계속 달라질수O

# 생존자&사망자 명수 return
def num_class(df, predict_attr):
    p_df = df[df[predict_attr] == 1]
    n_df = df[df[predict_attr] == 0]
    return p_df.shape[0], n_df.shape[0]

# info gain 제일 높은 attribute랑 threshold 고르기
def choose_attr(df, attributes, predict_attr):
    max_info_gain = float("-inf")
    best_attr = None
    threshold = 0
    # Testing (attributes 중복 선택 가능)
    for attr in attributes:
        thres = select_threshold(df, attr, predict_attr)
        ig = info_gain(df, attr, predict_attr, thres) #information gain 계산
        if ig > max_info_gain:
            max_info_gain = ig
            best_attr = attr
            threshold = thres
    return best_attr, threshold

#decision tree 구축
def build_tree(df, cols, predict_attr):
    p, n = num_class(df, predict_attr) #p: 생존자(p_df) ,n: 사망자(n_df)
    if p == 0 or n == 0: #전부다 생존했거나 사망했으면 -> leaf에 도달한거
        # leaf node 생성
        leaf = Node(None,None)
        leaf.leaf = True
        if p > n:
            leaf.predict = 1
        else:
            leaf.predict = 0
        return leaf
    else:
        #informaion gain을 기준으로 attribute, threshold 결정
        best_attr, threshold = choose_attr(df, cols, predict_attr)
        #internal tree 생성
        tree = Node(best_attr, threshold)
        sub_1 = df[df[best_attr] < threshold]
        sub_2 = df[df[best_attr] > threshold]
        
        # 재귀적으로 subtree생성  
        tree.left = build_tree(sub_1, cols, predict_attr)
        tree.right = build_tree(sub_2, cols, predict_attr)
        return tree

def predict(node, row_df):
    if node.leaf: #leaf node이면,
        return node.predict #leaf의 prediciton
    if row_df[node.attr] <= node.thres: #threshold보다 작으면,
        return predict(node.left, row_df) #왼쪽 가지로
    elif row_df[node.attr] > node.thres: #threshold보다 크면,
        return predict(node.right, row_df) #오른쪽 가지로

# 만들어진 디씨젼트리를 이용해서 test값 예측한 결과 정확도 계산
def test_predictions(root, df):
    num_data = df.shape[0]
    num_correct = 0
    for index,row in df.iterrows():
        prediction = predict(root, row)
        if prediction == row['Survived']: #답과 예측값이 일치할경우
            num_correct += 1 #num_correct 1 증가
    return round(num_correct/num_data, 2) #백분율 계산

def main():
    #df_train: 91~891까지 df_test: 0~90까지
    df_train = train.loc[91:]
    df_test = train.loc[0:90]

    attributes = train.columns.values
    
if __name__ == '__main__':
    main()
#Random Forest 라이브러리로 구현
clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
train_data = train_copy
round(np.mean(score)*100, 2)
#Gaussian Naive Bayes 라이브러리로 구현
clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
train_data = train_copy
round(np.mean(score)*100, 2)
#SVM 라이브러리로 구현
clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
round(np.mean(score)*100,2)
#SVM이 가장 높게 나왔기때문에 SVM으로 제출
clf.fit(train_data, target)

test_data = test.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test_data)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')
submission.head(5)