import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

%matplotlib inline
dataset_path = '../input/glass.csv'
data = pd.read_csv(dataset_path)
feature_names = data.columns[:-1].tolist()
print('数据形状：', data.shape)
print('共有{}条记录'.format(data.shape[0]))
print('共有'+ str(data.shape[0]) +'条记录')
print('共有{}个特征：{}'.format(len(feature_names), feature_names))

print()
print('数据预览：')
print(data.head())

print('属性类型：')
print(data.dtypes)
print('数据统计信息：')
data.describe()
data['Type'].value_counts()
for feature in feature_names:
    skew = data[feature].skew()
    sns.distplot(data[feature], label='skew = %.3f' %(skew))
    plt.legend(loc='best')
    plt.show()
sns.boxplot(data[feature_names])
plt.show()
plt.figure(figsize=(8, 8))
sns.pairplot(data[feature_names])
plt.show()
X = data[feature_names].values
y = data['Type'].values

# 随机数生成种子
seed = 5
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size , random_state = seed)
# 选择模型，交叉验证
k_range = range(1, 31)
cv_scores = []
print('交叉验证：')
for k in k_range:
    knn = KNeighborsClassifier(k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    score_mean = scores.mean()
    cv_scores.append(score_mean)
    print('%i: %.4f' % (k, score_mean))

best_k = np.argmax(cv_scores) + 1
print('最优K: ', best_k)

plt.plot(k_range, cv_scores)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()
# 训练模型
knn_model = KNeighborsClassifier(best_k)
knn_model.fit(X_train, y_train)
print('测试模型，准确率：', knn_model.score(X_test, y_test))