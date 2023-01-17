# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/"
data = {

    'mname':'John',

    'sex':'male',

    'age':22

}

john_s = pd.Series(data)

print(john_s)
# NumPy Arraysを使って作ることも可能

array = np.array([100, 200, 300])

pd.Series(array)
ndarray=np.random.randint(5,size=(5,4))

columns=['a','b','c','d']

index=np.arange(0,50,10)

pd.DataFrame(data=ndarray,index=index,columns=columns)


df = pd.read_csv("../input/titanic/train.csv")
# 統計量の確認

df.describe()
# .columnsでカラムのリストを表示する



df.columns
# h複数のカラムを抽出

df[['Age','Parch','Fare']].head()

# []を二重にすることに注意
np.isnan(df.iloc[888]['Age'])

# このNaNというのはnp.nanでありNoneではないことに注意です．
df.iloc[888]['Age'] is None
df.drop('Age', axis=1).head()

# dfにはdropは反映されてない
df
# 上書きしたいときは次の通りに上書きする

# df = df.drop(['Age', 'Cabin'], axis=1)
df
#　生存者をフィルタリングしたい

## 生存者のindexを求める

df['Survived'] == 1
df[df['Survived']==1]
df[df['Survived']==1].describe()
# 60才以上女性

df[(df['Age'] >= 60) & (df['Sex']=='female')]
# 男性のデータを表示

df[~(df['Sex']=='female')]
df
# 特定のカラムについてNanをdropする

df.dropna(subset=['Age']).head()
# NaNに特定のvalueを代入する

df.fillna('のー')
# Ageに平均を入れたいとき

df['Age'].fillna(df['Age'].mean())

# これをdf['Age']に代入する
#すべての

pd.isna(df)
df.groupby('Pclass').mean()
df.groupby('Pclass').describe()
# Pclassでグループ分けしたときのAgeの統計量を調べる

df.groupby('Pclass').describe()['Age']
results=[]

for i, group_df in df.groupby('Pclass'):

    sorted_group_df=group_df.sort_values('Fare')

    sorted_group_df['RankInPclass']=np.arange(len(sorted_group_df))

    results.append(sorted_group_df)

results_df=pd.concat(results)

results_df
# キーは同じでほかのカラムはそれぞれ別の値をもっている

df1 = pd.DataFrame({ 'Key': ['k0', 'k1', 'k2'],

        'A': ['a0', 'a1', 'a2'],

        'B': ['b0', 'b1', 'b2']})

 

df2 = pd.DataFrame({ 'Key': ['k0', 'k1', 'k2'],

        'C': ['c0', 'c1', 'c2'],

        'D': ['d0', 'd1', 'd2']})
print(df1)

print(df2)
# merge　横に結合

df1.merge(df2)



pd.concat([df1,df2],axis=0)
pd.concat([df1,df2],axis=1)
df1.merge(df2)
# キーが一つだけ違ってほかのカラムはそれぞれ別の値をもっている

df3 = pd.DataFrame({ 'Key': ['k0', 'k1', 'k2'],

                    'A': ['a0', 'a1', 'a2'],

                    'B': ['b0', 'b1', 'b2']})

 

df4 = pd.DataFrame({ 'Key': ['k0', 'k1', 'k3'],

                    'C': ['c0', 'c1', 'c3'],

                    'D': ['d0', 'd1', 'd3']})
# how=leftの時

df3.merge(df4,how='left')
df3.merge(df4,how='right')
# how=outerの時

df3.merge(df4,how='outer')
# how=inner　デフォルト

df3.merge(df4,how='inner')
df5 = pd.DataFrame({ 'Key': ['k0', 'k1', 'k2'],

                    'ID': ['aa', 'bb', 'cc'],

                    'A': ['a0', 'a1', 'a2'],

                    'B': ['b0', 'b1', 'b2']})



df6 = pd.DataFrame({ 'Key': ['k0', 'k1', 'k3'],

                    'ID': ['aa', 'bb', 'cc'],

                    'C': ['c0', 'c1', 'c3'],

                    'D': ['d0', 'd1', 'd3']})
df5.merge(df6,on='Key')
df5.merge(df6,on='ID')
df5.merge(df6,on='ID',suffixes=('_left','_right'))
df['Pclass'].unique()

df['Pclass'].nunique()
# それぞれの値に関していくつかのレコードがあるのかをSeries形式で返す

df['Pclass'].value_counts()
def get_age_group(age):

    return str(age)[0] + '0s'

 

get_age_group(45)
df = pd.DataFrame({ 'name': ['john', 'Mike', 'Emily'],

                    'age': ['23', '36', '42']})

df
df['age'].apply(get_age_group)
# 'age_group'カラムを新たに作り，結果を代入

df['age_group'] = df['age'].apply(get_age_group)

df
# lambda関数を使う方法

df['age_group'] = df['age'].apply(lambda x: str(x)[0] + '0s')

df
df=pd.read_csv('../input/titanic/train.csv')
# idxはindex, rowは各行のSeriesです



for idx,row in df.iterrows():

    if row['Age']>40 and row['Pclass']==3 and row['Sex']=='male' and row['Survived']==1:

          print('{} is very lucky guy...!'.format(row['Name']))
df.sort_values('Age')
data = {'Date':['Jan-1', 'Jan-1', 'Jan-1', 'Jan-2', 'Jan-2', 'Jan-2'], 

        'User':['Emily', 'John', 'Nick', 'Kevin', 'Emily', 'John'],

        'Method':['Card', 'Card', 'Cash', 'Card', 'Cash', 'Cash'],

        'Price':[100, 250, 200, 460, 200, 130]}

df = pd.DataFrame(data)

df
df.pivot_table(values='Price',index=['Date','User'],columns=['Method'])
pivot=df.pivot_table(values='Price', index=['Date', 'Method'], columns=['User'])

pivot
pivot.loc['Jan-1']
pivot.loc['Jan-1'].loc['Card']
pivot.xs('Card',level='Method')