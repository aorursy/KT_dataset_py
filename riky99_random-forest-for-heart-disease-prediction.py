# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install jcopml
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer



from jcopml.pipeline import num_pipe, cat_pipe

from jcopml.utils import save_model, load_model

from jcopml.plot import plot_missing_value

from jcopml.feature_importance import mean_score_decrease
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")

df
df.replace("?", np.nan, inplace=True)
plot_missing_value(df)
df.target = df.target.apply(lambda x: int(x>0))

df.head()
plt.figure(figsize=(7, 6))

sns.distplot(df.age[df.target ==0], bins=[0, 5, 12, 18, 40, 120], color="g", label="tidak ada indikasi")

sns.distplot(df.age[df.target ==1], bins=[0, 5, 12, 18, 40, 120], color="r", label="ada indikasi")

plt.legend();
cat_var = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]



fig, axes = plt.subplots(2, 4, figsize=(15,10))

for cat, ax in zip(cat_var, axes.flatten()):

    sns.countplot(cat, data=df, hue="target", ax=ax)
X = df.drop(columns="target")

y = df.target





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
X_train.head()
X_train.columns
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from jcopml.tuning import grid_search_params as gsp
preprocessor = ColumnTransformer([

    ('numeric', num_pipe(),  ["age", "trestbps", "chol", "thalach", "oldpeak"]),

    ('categoric', cat_pipe(encoder='onehot'), ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]),

])



pipeline = Pipeline([

    ('prep', preprocessor),

    ('algo', RandomForestClassifier(n_jobs=-1, random_state=42))

])







model = GridSearchCV(pipeline, gsp.rf_params, cv=3, n_jobs=-1, verbose=1)

model.fit(X_train, y_train)



print(model.best_params_)

print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))
df_imp = mean_score_decrease(X_train, y_train, model, plot=True)
from jcopml.utils import save_model
save_model(model.best_estimator_, "rf_heart.pkl")