!apt-get install build-essential swig
!pip install auto-sklearn==0.6.0
# 测试安装

import autosklearn
import pandas as pd



df = pd.read_csv("../input/train.csv")

df.tail()
X = df.iloc[:, 1:]  # 特征

y = df['label']  # 目标值



X.shape, y.shape
import warnings

warnings.filterwarnings('ignore')  # 忽略代码警告
from autosklearn.classification import AutoSklearnClassifier



# 限制算法搜索最大时间，更快得到结果

auto_model = AutoSklearnClassifier() # 默认最长搜索时间为 60 分钟

auto_model
auto_model.fit(X, y)  # 训练
df_test = pd.read_csv("../input/test.csv")

df_pred = pd.read_csv("../input/sample_submission.csv")
preds = auto_model.predict(df_test)

df_pred['Label'] = preds

df_pred.to_csv("preds.csv", index=None) # 保存推理文件