import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px
df = pd.read_csv("/kaggle/input/csvfiles/train_data.csv")

df.head()
df.info()
df = df.drop("id", axis=1)
sns.set_context(context="notebook", font_scale=1.5)

sns.set_style(style="whitegrid")
def make_histgram(x):

    sns.distplot(df[x], kde=False)

    plt.title(x + " distribution", fontsize=23)
make_histgram("battery_power")
make_histgram("bluetooth")
make_histgram("clock_speed")
make_histgram("dual_sim")
make_histgram("fc")
make_histgram("four_g")
make_histgram("int_memory")
make_histgram("m_dep")
make_histgram("mobile_wt")
make_histgram("n_cores")
make_histgram("pc")
make_histgram("px_height")
make_histgram("px_width")
make_histgram("ram")
make_histgram("sc_h")
make_histgram("sc_w")
make_histgram("talk_time")
make_histgram("three_g")
make_histgram("touch_screen")
make_histgram("wifi")
make_histgram("price_range")
df_corr = df.corr().round(2)

df_corr
plt.figure(figsize=(12,10))

sns.heatmap(df_corr,

            cmap='bwr_r',

            annot=True, annot_kws={"fontsize": 9},

            vmin=-1, vmax=1)
px.histogram(df.sort_values(by="price_range"), x="ram", color="price_range",

             width=1000, title="Price range of RAM")
px.histogram(df.sort_values(by="price_range"), x="battery_power", color="price_range", width=1000, title="Price range of battery_power")
px.histogram(df.sort_values(by="price_range"), x="px_height", color="price_range", width=1000, title="Price range of px_height")
px.histogram(df.sort_values(by="price_range"), x="px_width", color="price_range", width=1000, title="Price range of px_width")
px.scatter(df, x="px_width", y="px_height", color="price_range", height=600, width=800)
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.33, random_state=123)

print("train data : " + str(train.shape))

print("test data : " + str(test.shape))
train.head()
train_X = train.drop("price_range", axis=1)

train_y = train["price_range"]

test_X = test.drop("price_range", axis=1)

test_y = test["price_range"]



print(train_X.shape)

print(train_y.shape)

print(test_X.shape)

print(test_y.shape)
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics
model1 = DecisionTreeClassifier(random_state=123)

model1.fit(train_X, train_y)

prediction = model1.predict(test_X)

print("Accuracy : " + str(metrics.accuracy_score(prediction, test_y).round(3)))

metrics.confusion_matrix(prediction, test_y)
importances = model1.feature_importances_

importances = pd.DataFrame(data=importances).round(4)

importances["Feature"] = df.iloc[:, :20].columns

importances.rename(columns={0: "importance"}, inplace=True)

importances
plt.figure(figsize=(10, 5))

sns.barplot(x='Feature', y="importance", data=importances.sort_values("importance", ascending=False))

plt.tick_params(rotation=90)
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
model2 = RandomForestClassifier(random_state=123)

model2.fit(train_X, train_y)

prediction = model2.predict(test_X)

print("Accuracy : " + str(metrics.accuracy_score(prediction, test_y).round(3)))

metrics.confusion_matrix(prediction, test_y)
importances = model2.feature_importances_

importances = pd.DataFrame(data=importances).round(4)

importances["Feature"] = df.iloc[:, :20].columns

importances.rename(columns={0: "importance"}, inplace=True)

importances
plt.figure(figsize=(10, 5))

sns.barplot(x='Feature', y="importance", data=importances.sort_values("importance", ascending=False))

plt.tick_params(rotation=90)
submission_sample = pd.read_csv("/kaggle/input/csvfiles/sample_submission.csv")

submission_sample.head()
test = pd.read_csv("/kaggle/input/csvfiles/test_data.csv")

test.head()
test_X = test.drop("id", axis=1)

prediction = model2.predict(test_X)

prediction
test["price_range"] = prediction

test.head()
submission = test[["id", "price_range"]]

submission
submission.to_csv('/kaggle/working/submission.csv', index=False)