import pandas as pd
df = pd.read_csv('../input/watermelon2.0.csv')
df.head()
from sklearn.preprocessing import LabelEncoder
X = df.iloc[:,1:-1]

X = X.apply(LabelEncoder().fit_transform).values
y = LabelEncoder().fit_transform(df.iloc[:,-1])
print(X)
print(y)
import numpy as np
from sklearn.preprocessing import LabelBinarizer

class MultinomialNB():
    def __init__(self, alpha=1.0):
        # 拉普拉斯平滑值
        self.alpha = alpha

    def predict(self, X):
        jll = np.dot(X, self.feature_log_prob_.T) + self.class_log_prior_
        return self.classes_[np.argmax(jll, axis=1)]

    def fit(self, X, y):
        _, n_features = X.shape

        # 二值化标记
        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)
        
        # 把类别存起来
        self.classes_ = labelbin.classes_

        n_effective_classes = Y.shape[1]

        # 每个类别的样本数
        self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
        self.class_count_ += Y.sum(axis=0)

        # 计算每个类别、每个特征的样本数    结果为k*n的矩阵
        self.feature_count_ = np.zeros((n_effective_classes, n_features), dtype=np.float64)
        self.feature_count_ += np.dot(Y.T, X)
        
        # 类别的概率
        # 为了避免数值下溢,取对数,除法变减法
        self.class_log_prior_ = (np.log(self.class_count_) - np.log(self.class_count_.sum()))
        
        # 特征的概率
        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1)
        self.feature_log_prob_ = (np.log(smoothed_fc) - np.log(smoothed_cc.reshape(-1, 1)))
        return self

mbmodel = MultinomialNB()
mbmodel.fit(X,y)
mbmodel.predict(X)
class GaussianNB():
    def __init__(self):
        self.priors = None
        self.var_smoothing = 1e-9

    def predict(self, X):
        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])
            n_ij = - 0.5 * np.sum(np.log(2. * np.pi * self.sigma_[i, :]))
            n_ij -= 0.5 * np.sum(((X - self.theta_[i, :]) ** 2) /
                                 (self.sigma_[i, :]), 1)
            joint_log_likelihood.append(jointi + n_ij)

        jll = np.array(joint_log_likelihood).T
        return self.classes_[np.argmax(jll, axis=1)]

    def fit(self, X, y):
        self.classes_ = classes = np.unique(y)

        n_features = X.shape[1]
        n_classes = len(self.classes_)

        # 样本平均值
        self.theta_ = np.zeros((n_classes, n_features))
        # 样本方差
        self.sigma_ = np.zeros((n_classes, n_features))
        # 每个类别的样本数
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        # 先验概率
        self.class_prior_ = np.zeros(len(self.classes_), dtype=np.float64)

        ## 计算样本均值、方差、每个类别的样本数、
        for i, y_i in enumerate(classes):
            X_i = X[y == y_i, :]
            self.theta_[i, :] = np.mean(X_i, axis=0)
            self.sigma_[i, :] = np.var(X_i, axis=0)
            self.class_count_[i] += X_i.shape[0]

        # 如果维度之间的数据差异比例太小，那么会导致数字错误。
        # 为了解决这个问题，我们人为地通过epsilon提升方差。
        self.epsilon_ = self.var_smoothing * np.var(X, axis=0).max()
        self.sigma_[:, :] += self.epsilon_

        # 计算先验概率
        self.class_prior_ = self.class_count_ / self.class_count_.sum()

        return self
    
mbmodel = GaussianNB()
mbmodel.fit(X,y)
mbmodel.predict(X)