import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) 
#import jupyter-notebook 
#import scikit-learn
import matplotlib.pyplot as plt
import matplotlib
import seaborn

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
 
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data.shape
train_data.head(10) 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) 
#import jupyter-notebook 
#import scikit-learn
import matplotlib.pyplot as plt
import matplotlib
import seaborn

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
 
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data.shape
#train_data.head(10)
#explore_data.describe().round(3)
train_data.describe().round(3)
train_data.hist(bins=20, figsize=(15, 10), layout=(-1, 2))
plt.plot()
not_survived = train_data.loc[train_data["Survived"]==0.0]
survived = train_data.loc[train_data["Survived"]==1.0]
for column in ["Age", "Parch", "SibSp", "Fare"]:
    #plt.figure(figsize=(15, 10)) #kich thuoc Hang, Cot
    not_survived_data = not_survived[column]
    survived_data = survived[column]
    plt.hist([not_survived_data, survived_data], bins=20, color=["r", "g"], label=["Not survived", "Survived"])
    plt.legend(loc="upper left")
    plt.title(column)
    plt.show()
for column in ["Age", "Parch", "SibSp", "Fare"]:
    print(column)
not_survived = train_data.loc[train_data["Survived"]==0.0] #loc = select(where)
survived = train_data.loc[train_data["Survived"]==1.0]
print((not_survived))
#print(len(survived))
#print(train_data["Survived"])
import seaborn as sns
for feature in ["Sex", "Pclass", "Embarked"]:
    feature_data = train_data.groupby(["Survived", feature])["PassengerId"].count().reset_index(name="Count")
    #plt.figure(figsize=(15, 10))
    sns.barplot(x=feature, y="Count", hue="Survived", data=feature_data)
    plt.title(feature)
    plt.show()
train_data[["Sex", "Survived"]].groupby(["Sex"], as_index=False).mean().sort_values(by="Survived", ascending=False)
train_data[["Pclass", "Survived"]].groupby(["Pclass"], as_index=False).mean().sort_values(by="Survived", ascending=False)
train_data[["Survived", "Embarked"]].groupby(["Embarked"], as_index=False).mean().sort_values(by="Survived", ascending=False)

corr = train_data.corr()
corr.style.background_gradient(cmap="binary", low=0, high=0.2).set_precision(2)
from pandas.plotting import scatter_matrix
color_map = {
    0: "red", # Not survived
    1: "blue" # Survived
}
colors = train_data["Survived"].map(lambda x: color_map.get(x))
scatter_matrix(train_data, figsize=(50, 50), alpha=0.6, color=colors, s=20*4)
#plt.legend()
#plt.savefig("scatter_matrix.png")
processed_explore_data = pd.get_dummies(train_data, columns=["Embarked", "Sex", "Pclass"], drop_first=True)
# Drop cabin for data in TSNE
processed_explore_data.drop(["Cabin", "Name", "Ticket"], axis=1, inplace=True)
# Replace with most frequent value
processed_explore_data = processed_explore_data.apply(lambda x:x.fillna(x.value_counts().index[0]))
processed_explore_data.head(10)
from sklearn.preprocessing import StandardScaler
# Before put data in TSNE, we must scale the columns
continuous_columns = ["Age", "Fare", "SibSp", "Parch"]
scaled_continous_columns = ["scaled_" + continous_column for continous_column in continuous_columns]
std_scaler = StandardScaler()
scaled_explore_data = processed_explore_data.copy()
scaled_column_data = std_scaler.fit_transform(processed_explore_data[continuous_columns]).transpose()
for scaled_continous_column, continuous_column, scaled_column in zip(scaled_continous_columns, continuous_columns, scaled_column_data):
    scaled_explore_data[scaled_continous_column] = scaled_column
scaled_explore_data.drop(continuous_columns, axis=1, inplace=True)
visual_columns = scaled_explore_data.columns.tolist()
visual_columns = visual_columns[1:]
visual_columns
scaled_explore_data.head(10) 
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30)
tsne_vectors = tsne.fit_transform(scaled_explore_data[visual_columns])
visual_data = processed_explore_data.copy()
visual_data["tsne_x"] = tsne_vectors[:, 0]
visual_data["tsne_y"] = tsne_vectors[:, 1]
survived_df = visual_data.loc[visual_data["Survived"]  == 1]
not_survived_df = visual_data.loc[visual_data["Survived"]  == 0]
df_list = [survived_df, not_survived_df]
df_labels = ["Survived", "Not Survived"]
df_colors = ["Blue", "Red"] 
plt.figure(figsize=(20, 15))
for df, label, color in zip(df_list, df_labels, df_colors):    
    plt.scatter(df["tsne_x"], df["tsne_y"], label=label, color=color, alpha=0.6)
plt.legend()
plt.show()