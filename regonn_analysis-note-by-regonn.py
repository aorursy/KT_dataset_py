import os # ファイル一覧とかに使う
import numpy as np # 科学計算には必須だね
import pandas as pd # 🐼 データフレームを扱える
import matplotlib.pyplot as plt # seabornが matplotlib をいい感じにラッパーしてくれる
import japanize_matplotlib # グラフで日本語が文字化けしないように
import seaborn as sns # そのままデータフレームを渡したらいい感じにプロットしてくれたりする
print(os.listdir("../input"))
breeds = pd.read_csv('../input/breed_labels.csv')
breeds.head()
breeds.size
colors = pd.read_csv('../input/color_labels.csv')
colors.head()
colors.size
states = pd.read_csv('../input/state_labels.csv')
states.head()
states.size
train = pd.read_csv('../input/train/train.csv')
test = pd.read_csv('../input/test/test.csv')
sub = pd.read_csv('../input/test/sample_submission.csv')

train['dataset_type'] = 'train'
test['dataset_type'] = 'test'
all_data = pd.concat([train, test])
train.head()
train.info()
plt.figure(figsize=(14, 6));
g = sns.countplot(x='AdoptionSpeed', data=all_data.loc[all_data['dataset_type'] == 'train'])
plt.title('AdoptionSpeed分布');
ax= g.axes
for p in ax.patches:
     ax.annotate(f"{p.get_height() * 100 / train.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
         ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),
         textcoords='offset points')
all_data['PetType'] = all_data['Type'].apply(lambda x: 'Dog' if x == 1 else 'Cat')
plt.figure(figsize=(10, 6));
sns.countplot(x='dataset_type', data=all_data, hue='PetType');
plt.title('犬と猫の割合');
ax.patches[0].get_height() 
dog_count = train[train['Type'] == 1].shape[0]
cat_count = train[train['Type'] == 2].shape[0]
plt.figure(figsize=(14, 6));
g = sns.countplot(x='PetType', data=all_data.loc[all_data['dataset_type'] == 'train'], hue='AdoptionSpeed')
plt.title('AdoptionSpeed分布');
ax = g.axes

for index, p in enumerate(ax.patches):
    ax.annotate(f"{p.get_height() * 100 / (cat_count if index % 2 == 0 else dog_count):.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10), textcoords='offset points')