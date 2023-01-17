# 데이터 전처리
import numpy as np
import pandas as pd

# 기계학습 모델 생성, 학습, 평가
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# 시각화
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('../input/UniversalBank.csv')
data.head()
data = data.drop(['ID','ZIP Code'], axis=1)
# Education: dummies
data = pd.get_dummies(data, columns=['Education'], drop_first=True)
data.head()
X = data.drop('Personal Loan', axis=1)
y = data['Personal Loan']
plt.figure(figsize=(4, 5))
sns.countplot(y)
plt.show()
random_state = 2020
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2/10, random_state=random_state, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=2/8,
                                                      random_state=random_state, stratify=y_train)
print('클래스별 데이터 개수: Train')
print(y_train.value_counts())

print('\n ----------------------- \n')
print('클래스별 데이터 개수: Validation')
print(y_valid.value_counts())

print('\n ----------------------- \n')
print('클래스별 데이터 개수: Test')
print(y_test.value_counts())
max_depths = list(range(1, 10)) + [None]
print(max_depths)
# 평가 지표 저장
acc_valid = []
f1_valid = []
for max_depth in max_depths:
                      
    # 모델 학습
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)
    
    # validation 예측
    y_valid_pred = model.predict(X_valid)
    
    # 모델 평가 결과 저장
    acc = accuracy_score(y_valid, y_valid_pred)
    f1 = f1_score(y_valid, y_valid_pred)
    
    acc_valid.append(acc)
    f1_valid.append(f1)
xticks = list(map(str, max_depths))
print(xticks)
fig, ax = plt.subplots(figsize=(15, 6))
#fig.subplots_adjust(right=0.75)

ax.plot(range(len(max_depths)), acc_valid, color='red', marker='o')
ax.set_ylabel('accuracy', color='red', fontsize=12)

ax2 = ax.twinx()
ax2.plot(range(len(max_depths)), f1_valid, color='blue', marker='s')
ax2.set_ylabel('f1', color='blue', fontsize=12)

plt.xticks(range(len(max_depths)), xticks)
plt.show()
model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)
# 에측 결과 산출
y_test_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
cm = pd.DataFrame(cm)

# Accuracy, F1-Score
acc = accuracy_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print('- Accuracy (Test) : {:.3}'.format(acc))
print('- F1 score (Test) : {:.3}'.format(f1))
# 시각화
plt.figure(figsize=(6, 5))
sns.heatmap(data=cm, annot=True, annot_kws={'size': 15}, fmt='d', cmap='Blues')
plt.title('Acc = {:.3f} & F1 = {:.3f}'.format(acc, f1))
plt.show()
plt.figure(figsize=(20, 10))
plot_tree(decision_tree=model, filled=True)
plt.show()
# 변수 중요도
importances = model.feature_importances_

# 내림차순으로 정렬하기 위한 index
index = np.argsort(importances)[::-1]
plt.figure(figsize=(8, 6))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]),
        importances[index],
        align='center')
plt.xticks(range(X.shape[1]), X.columns, rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()
plt.figure(figsize=(6, 6))
plt.scatter(X['Age'], X['Experience'], c=y, cmap='binary', alpha=0.7)
plt.show()
path = model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
ccp_alpha = ccp_alphas[4]
model_prune = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
model_prune.fit(X_train, y_train)
# 에측 결과 산출
y_test_pred = model_prune.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
cm = pd.DataFrame(cm)

# Accuracy, F1-Score
acc = accuracy_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print('- Accuracy (Test) : {:.3}'.format(acc))
print('- F1 score (Test) : {:.3}'.format(f1))
plt.figure(figsize=(12, 8))
plot_tree(decision_tree=model_prune, filled=True)
plt.show()

