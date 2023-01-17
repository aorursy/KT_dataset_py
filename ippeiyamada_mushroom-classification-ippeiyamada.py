%matplotlib inline 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import display
from dateutil.parser import parse

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from IPython.display import display
#まずはデータの内容を確認する。
df_data = pd.read_csv("../1_data/mushrooms.csv")
print(df_data.columns)
display(df_data.head())
display(df_data.tail())
# coutn missing
pd.DataFrame(df_data.isnull().sum(), columns=["num of missing"])
#上記処理では欠損値は見つけられず。
# corss tab %
for col in df_data.columns:
    if col=="class":
        continue
    print(col)
    df_c = pd.crosstab(index=df_data["class"], columns=df_data[col],margins=True,normalize=True)
    display(df_c)
    df_cross_show = df_c.T.sort_values(by=['All'], ascending=False)
    df_cross_show.plot.bar()
    plt.show()

# corss tab % 相関が高い変数を再度表示し、その中から3つに絞る。

df_c = pd.crosstab(index=df_data["class"], columns=df_data["bruises"],margins=True,normalize=True)
display(df_c)
df_cross_show = df_c.T.sort_values(by=['All'], ascending=False)
df_cross_show.plot.bar()
plt.show()

df_c = pd.crosstab(index=df_data["class"], columns=df_data["odor"],margins=True,normalize=True)
display(df_c)
df_cross_show = df_c.T.sort_values(by=['All'], ascending=False)
df_cross_show.plot.bar()
plt.show()

df_c = pd.crosstab(index=df_data["class"], columns=df_data["gill-size"],margins=True,normalize=True)
display(df_c)
df_cross_show = df_c.T.sort_values(by=['All'], ascending=False)
df_cross_show.plot.bar()
plt.show()

df_c = pd.crosstab(index=df_data["class"], columns=df_data["gill-color"],margins=True,normalize=True)
display(df_c)
df_cross_show = df_c.T.sort_values(by=['All'], ascending=False)
df_cross_show.plot.bar()
plt.show()

df_c = pd.crosstab(index=df_data["class"], columns=df_data["stalk-root"],margins=True,normalize=True)
display(df_c)
df_cross_show = df_c.T.sort_values(by=['All'], ascending=False)
df_cross_show.plot.bar()
plt.show()

df_c = pd.crosstab(index=df_data["class"], columns=df_data["stalk-surface-above-ring"],margins=True,normalize=True)
display(df_c)
df_cross_show = df_c.T.sort_values(by=['All'], ascending=False)
df_cross_show.plot.bar()
plt.show()

df_c = pd.crosstab(index=df_data["class"], columns=df_data["ring-type"],margins=True,normalize=True)
display(df_c)
df_cross_show = df_c.T.sort_values(by=['All'], ascending=False)
df_cross_show.plot.bar()
plt.show()

#欠損値と異常値の確認
#stalk-rootの?をnにする。
df_data = df_data.replace('?', 'n')

#確認
df_c = pd.crosstab(index=df_data["class"], columns=df_data["stalk-root"],margins=True,normalize=True)
display(df_c)
#?はnになっていることを確認。
#これらに関してはonehot encodingを使うことにする。
df_en = pd.concat([df_data,pd.get_dummies(df_data['bruises'])],axis=1)
df_en = df_en.drop(['ring-type','spore-print-color','population','habitat'],axis=1)
df_en = df_en.drop(['bruises','gill-color','stalk-shape','stalk-root','cap-shape','cap-surface','cap-color','odor','gill-attachment','gill-spacing','gill-size','stalk-color-below-ring','veil-type','veil-color','ring-number'],axis=1)
df_en = df_en.drop(['stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring'],axis=1)

df_en = df_en.drop(['f'],axis=1)#bruises の有無をtで表す。

df_en['class'] = pd.concat([df_data,pd.get_dummies(df_data['class'])],axis=1)
df_en = df_en.drop(['class'],axis=1)
df_en['class_2'] = np.array(df_data['class'] != 0)*1

display(df_en)
y = ((df_data.bruises > 0) * 1).values

plt.plot(X.df_data,y,'o')

lr = LinearRegression()

lr.fit(X.bruises.reshape(-1,1),y)

plt.plot(range(X.bruises.min(),X.bruises.max()),lr.coef_*range(X.bruises.min(),X.bruises.max())+lr.intercept_)

plt.xlabel('bruises')
plt.ylabel('class')
df_data["class"].value_counts().sort_index().head()


