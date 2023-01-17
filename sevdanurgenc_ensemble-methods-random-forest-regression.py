import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
rnd_clf.fit(iris["data"], iris["target"])
for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)

#feature importance : aldigimiz feature'in model uzerine etkisi
rnd_clf.feature_importances_

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
import seaborn as sns
data_df = pd.read_csv('../input/Random-Forest-Regression-Data.csv')
data_df.head()
x = data_df.x.values.reshape(-1, 1)
y = data_df.y.values.reshape(-1, 1)
print('x\n', x, '\n')
print('y\n', y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
rf_reg = RandomForestRegressor()
rf_reg.fit(x_train, y_train)
y_pred = rf_reg.predict(x_test)
y_pred
print("Test Accuracy = ", rf_reg.score(x_test, y_test))
