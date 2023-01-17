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
import numpy as np

import pandas as pd

import datetime

import random



# Plots

import seaborn as sns

import matplotlib.pyplot as plt



# Models

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import Ridge, RidgeCV

from sklearn.linear_model import ElasticNet, ElasticNetCV

from sklearn.svm import SVR

from mlxtend.regressor import StackingCVRegressor

import lightgbm as lgb

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor



# Stats

from scipy.stats import skew, norm

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



# Misc

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.decomposition import PCA



pd.set_option('display.max_columns', None)



# Ignore useless warnings

import warnings

warnings.filterwarnings(action="ignore")
sample = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")
sample.head()
print(sample.shape)

print("-"*60)

print(sample.info())

print("-"*60)

print(sample.describe())
sns.set()



plt.figure(figsize=(15,7))

plt.subplot(2,2,1)

sns.countplot(x="gender", data=sample)



plt.subplot(2,2,2)

sns.countplot(x="race/ethnicity", data=sample)



plt.subplot(2,2,3)

sns.countplot(x="lunch", data=sample)



plt.subplot(2,2,4)

sns.countplot(x="test preparation course", data=sample)



plt.tight_layout()
sns.set()



plt.figure(figsize=(15,7))



sns.countplot(x="parental level of education", data=sample)
plt.figure(figsize=(15,6))



plt.subplot(1,3,1)

sns.distplot(sample["math score"], kde=True, label="math",kde_kws={'color':'red','label':'kde'})



plt.subplot(1,3,2)

sns.distplot(sample["reading score"], kde=True, label="reading",kde_kws={'color':'red','label':'kde'})



plt.subplot(1,3,3)

sns.distplot(sample["writing score"], kde=True, label="writing",kde_kws={'color':'red','label':'kde'})



plt.tight_layout()
#Distribution of scores by gender

sns.set()



plt.figure(figsize=(15,10))



plt.subplot(2,2,1)

sns.boxplot(x="gender", y = "math score", data=sample)

plt.subplot(2,2,2)

sns.boxplot(x="gender", y = "reading score", data=sample)

plt.subplot(2,2,3)

sns.boxplot(x="gender", y = "writing score", data=sample)



plt.tight_layout()
sns.set()



plt.figure(figsize=(15,10))



plt.subplot(2,2,1)

sns.boxplot(x="lunch", y = "math score", data=sample)

plt.subplot(2,2,2)

sns.boxplot(x="lunch", y = "reading score", data=sample)

plt.subplot(2,2,3)

sns.boxplot(x="lunch", y = "writing score", data=sample)



plt.tight_layout()
sns.set()



plt.figure(figsize=(15,10))



plt.subplot(2,2,1)

sns.boxplot(x="test preparation course", y = "math score", data=sample)

plt.subplot(2,2,2)

sns.boxplot(x="test preparation course", y = "reading score", data=sample)

plt.subplot(2,2,3)

sns.boxplot(x="test preparation course", y = "writing score", data=sample)



plt.tight_layout()
sns.set()



plt.figure(figsize=(15,10))



plt.subplot(2,2,1)

sns.boxplot(x="race/ethnicity", y = "math score", data=sample)

plt.subplot(2,2,2)

sns.boxplot(x="race/ethnicity", y = "reading score", data=sample)

plt.subplot(2,2,3)

sns.boxplot(x="race/ethnicity", y = "writing score", data=sample)



plt.tight_layout()
sns.set()



plt.figure(figsize=(16,10))



plt.subplot(2,2,1)

sns.boxplot(x="parental level of education", y = "math score", data=sample)

plt.xticks(rotation=90)

plt.subplot(2,2,2)

sns.boxplot(x="parental level of education", y = "reading score", data=sample)

plt.xticks(rotation=90)

plt.subplot(2,2,3)

sns.boxplot(x="parental level of education", y = "writing score", data=sample)

plt.xticks(rotation=90)



plt.tight_layout()
sns.set()



plt.figure(figsize=(15,7))

plt.subplot(2,2,1)

sns.countplot(x="race/ethnicity", hue="gender", data=sample)



plt.subplot(2,2,2)

sns.countplot(x="race/ethnicity", hue="lunch", data=sample)



plt.subplot(2,2,3)

sns.countplot(x="race/ethnicity", hue="test preparation course", data=sample)



plt.tight_layout()
group_a = sample.groupby("race/ethnicity").get_group("group A")

group_b = sample.groupby("race/ethnicity").get_group("group B")

group_c = sample.groupby("race/ethnicity").get_group("group C")

group_d = sample.groupby("race/ethnicity").get_group("group D")

group_e = sample.groupby("race/ethnicity").get_group("group E")



sample["parental level of education"].unique()
sns.set()



score = sample[["math score", "reading score", "writing score","test preparation course"]]





sns.pairplot(score, hue="test preparation course")
sample["Total_score"] = sample["math score"] + sample["reading score"] + sample["writing score"]

sample["Avg_score"] = sample["Total_score"] / 3



top_scores = sample.sort_values("Total_score", ascending=False).head(50)



bottom_scorer = sample.sort_values("Total_score", ascending=False).tail(50)
top_scores
sns.set()



plt.figure(figsize=(15,7))



sns.countplot(x="parental level of education", data=top_scores)
sns.set()



plt.figure(figsize=(15,7))

plt.subplot(2,2,1)

sns.countplot(x="gender", data=top_scores)



plt.subplot(2,2,2)

sns.countplot(x="race/ethnicity", data=top_scores)



plt.subplot(2,2,3)

sns.countplot(x="lunch", data=top_scores)



plt.subplot(2,2,4)

sns.countplot(x="test preparation course", data=top_scores)



plt.tight_layout()
plt.figure(figsize=(15,6))



plt.subplot(1,3,1)

sns.distplot(top_scores["math score"], kde=True, label="math",kde_kws={'color':'red','label':'kde'})



plt.subplot(1,3,2)

sns.distplot(top_scores["reading score"], kde=True, label="reading",kde_kws={'color':'red','label':'kde'})



plt.subplot(1,3,3)

sns.distplot(top_scores["writing score"], kde=True, label="writing",kde_kws={'color':'red','label':'kde'})



plt.tight_layout()
bottom_scorer
sns.set()



plt.figure(figsize=(15,7))



sns.countplot(x="parental level of education", data=bottom_scorer)
sns.set()



plt.figure(figsize=(15,7))

plt.subplot(2,2,1)

sns.countplot(x="gender", data=bottom_scorer)



plt.subplot(2,2,2)

sns.countplot(x="race/ethnicity", data=bottom_scorer)



plt.subplot(2,2,3)

sns.countplot(x="lunch", data=bottom_scorer)



plt.subplot(2,2,4)

sns.countplot(x="test preparation course", data=bottom_scorer)



plt.tight_layout()
plt.figure(figsize=(15,6))



plt.subplot(1,3,1)

sns.distplot(bottom_scorer["math score"], kde=True, label="math",kde_kws={'color':'red','label':'kde'})



plt.subplot(1,3,2)

sns.distplot(bottom_scorer["reading score"], kde=True, label="reading",kde_kws={'color':'red','label':'kde'})



plt.subplot(1,3,3)

sns.distplot(bottom_scorer["writing score"], kde=True, label="writing",kde_kws={'color':'red','label':'kde'})



plt.tight_layout()