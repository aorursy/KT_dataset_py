!apt-get install build-essential swig
!pip install auto-sklearn
# 测试安装

import autosklearn
!ls /kaggle/input/digit-recognizer/
import pandas as pd



df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

df.tail()
X = df.iloc[:, 1:]  # 特征

y = df['label']  # 目标值



X.shape, y.shape
import warnings

warnings.filterwarnings('ignore')  # 忽略代码警告
from autosklearn.classification import AutoSklearnClassifier



# 限制算法搜索最大时间，更快得到结果

auto_model = AutoSklearnClassifier(time_left_for_this_task = 360,per_run_time_limit = 120)  # 设置最长搜索时间为 10 分钟

auto_model
auto_model.fit(X, y) # 训练
df_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

df_pred = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
preds = auto_model.predict(df_test)

df_pred['Label'] = preds

df_pred.to_csv("preds.csv", index=None) # 保存推理文件