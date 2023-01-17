import pandas as pd
train_data_org = pd.read_csv('../input/train.csv')

train_data_org.head(2)
len(train_data_org.columns)
train_data_org.describe()
train_data_org.isnull().sum()
df = train_data_org
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

le.fit(df["Sex"])
df["Sex"] = le.transform(df["Sex"])

le.fit(df["Ticket"])
df["Ticket"] = le.transform(df["Ticket"])

le.fit(df["Ticket"])
df["Ticket"] = le.transform(df["Ticket"])


#### グラフ化してみる
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

df_drop = df.dropna()
sns.pairplot(df_drop, hue = 'Survived', size =2) # hue:指定したデータで分割
corr = df_drop.corr()
corr
#### Survivedに相関関係が最も高い変数を探す
max_num = 0
max_key = ""
for key,item in corr["Survived"].iteritems() :
    if(item != 1.0 and max_num < item) :
        max_key = key
        max_num = item
print(max_key,max_num)

#### Fareに相関関係が最も低い変数を探す
m_num = 0
max_key = ""
for key,item in corr["Survived"].iteritems() :
    if(item != 1.0 and max_num < item) :
        max_key = key
        max_num = item
print(max_key,max_num)

# 欠損値データ確認
train_data_org.isnull().sum()

train_y = train_data['Survived']
# train_X["Age"]
plt.scatter(train_X['Age'],train_X['Fare'],vmin=0, vmax=1,c=train_y)
# カラーバーを表示
plt.colorbar()

