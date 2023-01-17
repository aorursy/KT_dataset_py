import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from patsy import dmatrices # 可根据离散变量自动生成哑变量
from sklearn.linear_model import LogisticRegression # sk-learn库Logistic Regression模型
from sklearn.model_selection import train_test_split, cross_val_score # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库
data = pd.read_csv('../input/train.csv')
#embarked Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton
data = data.drop(['Ticket', 'Cabin'], axis = 1)
data = data.dropna()
len(data.index)
data.Survived.value_counts().plot(kind='bar')
plt.xlabel('Survived')
plt.show()
female = data.Survived[data.Sex == 'female'].value_counts().sort_index()
female.plot(kind='barh', color='blue', label='Female')
plt.show()
male = data.Survived[data.Sex == 'male'].value_counts().sort_index()
male.plot(kind='barh',label='Male', color='red')
plt.show()
highclass = data.Survived[data.Pclass != 3].value_counts().sort_index()
highclass.plot(kind='bar',label='Highclass', color='red', alpha=0.6)
plt.show()
lowclass = data.Survived[data.Pclass == 3].value_counts().sort_index()
lowclass.plot(kind='bar',label='Highclass', color='Blue', alpha=0.6)
plt.show()
y, X = dmatrices('Survived~ C(Pclass) + C(Sex) + Age + C(Embarked)', data = data, return_type='dataframe')
y = np.ravel(y)
model = LogisticRegression()
model.fit(X, y)
model.score(X, y)
1 - y.mean()
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
test_data = pd.read_csv('../input/test.csv')
test_data['Survived'] = 1
test_data.loc[np.isnan(test_data.Age), 'Age'] = np.mean(data['Age'])
ytest, Xtest = dmatrices('Survived~ C(Pclass) + C(Sex) + Age + C(Embarked)', data = test_data, return_type='dataframe')
pred = model.predict(Xtest).astype(int)
solution = pd.DataFrame(list(zip(test_data['PassengerId'], pred)), columns=['PassengerID', 'Survived'])
solution.to_csv('./my_prediction.csv', index = False)
precs = []
recalls = []
scores = model.predict_proba(X)
min_score = min([s[1] for s in scores])
max_score = max([s[1] for s in scores])
for threshold in np.arange(0, 1, 0.01):
    # prediction score above threshold is considered as positive
    if threshold < min_score or threshold > max_score:
        continue
    preds = [1 if s[1] >= threshold else 0 for s in scores]
    TP = sum([1 for gt, pred in zip(y, preds) if gt == 1 and pred == 1])
    FP = sum([1 for gt, pred in zip(y, preds) if gt == 0 and pred == 1])
    FN = sum([1 for gt, pred in zip(y, preds) if gt == 1 and pred == 0])
    if TP + FP == 0:
        precs.append(0.0)
    else:
        precs.append(TP / (TP + FP + 0.0))
    if TP + FN == 0:
        recalls.append(0.0)
    else:
        recalls.append(TP / (TP + FN + 0.0))
#plt.plot(recalls, precs, 'bs', ls='-')
plt.plot(recalls, precs, ls='-')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()