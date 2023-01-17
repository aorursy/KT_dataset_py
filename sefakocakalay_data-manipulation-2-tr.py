import numpy as np
import pandas as pd
l = [1,4,577,343]
l
pd.DataFrame(l, columns = ['degisken_ismi'] )
df = pd.DataFrame(l, columns = ['degisken_ismi'] )
type(df)
df.axes
df.shape
df.ndim
df.size
df.values
df.head()
df.tail(2)
a = np.array([1,2,3,4,56])
pd.DataFrame(a, columns = ['degisken_ismi'])
m = np.arange(1,10).reshape((3,3))
m
pd.DataFrame(m, columns = ['var1','var2','var3'])
df = pd.DataFrame(m, columns = ['var1','var2','var3'])
df
df.columns = ('deg1','deg2','deg3')
df
pd.DataFrame(m, columns = ['var1','var2','var3'], index = ['a','b','c'])
pd.Series([1,2,3,4])
pd.DataFrame(pd.Series([1,2,3,4]), columns = ['degisken'])
bir = pd.Series([1,2,3,4])
iki = pd.Series([1,2,3,4])
pd.DataFrame({'degisken1': bir,
             'degisken2': iki})
sozluk = {"reg" : {"RMSE": 10,
                    "MSE": 11,
                    "SSE": 90},
          
           "loj" : {"RMSE": 89,
                    "MSE": 12,
                    "SSE": 45},
                    
          "cart": {"RMSE": 45,
                    "MSE": 22,
                    "SSE": 11}}
pd.DataFrame(sozluk)
s1 = np.random.randint(10, size = 5)
s2 = np.random.randint(10, size = 5)
s3 = np.random.randint(10, size = 5)

df = pd.DataFrame({"var1": s1, "var2": s2,"var3": s3})
df
df[0:1]
df.index = ['a','b','c','d','e'] 
df
df['c':'e']
df.drop('a', axis = 0)
df
df.drop('a', axis = 0, inplace = True)
df
l = ["b","c"]
l
df.drop(l, axis = 0)
df
'var1' in df
l = ['var1','var2', 'var4']
for i in l:
    print(i in df)
df['var1'] is df['var1']
df['var2']
df.var1
df[['var1', 'var2']]
l = ['var1', 'var2']
df[l]
df
df['var4'] = df['var1'] / df['var2']
df
df
df.drop('var4', axis = 1, inplace = True)
df
df.drop(l, axis = 1)
s1 = np.random.randint(10, size = 5)
s2 = np.random.randint(10, size = 5)
s3 = np.random.randint(10, size = 5)

df = pd.DataFrame({"var1": s1, "var2": s2,"var3": s3})
df
df.loc[0:3]
df.iloc[0:3]
df.iloc[0,0]
df.iloc[:3,:2]
df.iloc[2:6,1:3]
df.loc[0:3, 'var3']
df.iloc[0:3, 1:3]
df.index = ["a","b","c","d","e"]
df
df.loc["c":"d", "var2":"var3"]
df
df[df.var1 > 5]['var2']
df[(df.var1 > 5) & (df.var3 < 7)]
df.loc[df.var1 > 5, ['var2','var1']]
s1 = np.random.randint(10, size = 5)
s2 = np.random.randint(10, size = 5)
s3 = np.random.randint(10, size = 5)

df1 = pd.DataFrame({"var1": s1, "var2": s2,"var3": s3})
df1
df2 = df1 + 99
df2
import pandas as pd
pd.concat([df1,df2], axis = 1)
pd.concat([df1,df2])
pd.concat([df1,df2], ignore_index = True)
df2.columns = ["var1","var2","deg3"]
df2
df1
pd.concat([df1,df2])
pd.concat([df1,df2], join = 'inner')
pd.concat([df1,df2], join_axes = [df1.columns])
#merge join
df1 = pd.DataFrame({'calisanlar': ['Ali', 'Veli', 'Ayse', 'Fatma'],
                    'grup': ['Muhasebe', 'Muhendislik', 'Muhendislik', 'İK']})

df1
df2 = pd.DataFrame({'calisanlar': ['Ayse', 'Ali', 'Veli', 'Fatma'],
                    'ilk_giris': [2010, 2009, 2014, 2019]})

df2
pd.merge(df1, df2)
pd.merge(df1, df2, on = 'calisanlar')
df3 = pd.merge(df1,df2)
df3
df4 = pd.DataFrame({'grup': ['Muhasebe', 'Muhendislik', 'İK'],
                    'mudur': ['Caner', 'Mustafa', 'Berkcan']})

df4
pd.merge(df3,df4)
df5 = pd.DataFrame({'grup': ['Muhasebe', 'Muhasebe',
                              'Muhendislik', 'Muhendislik', 'İK', 'İK'],
                    'yetenekler': ['matematik', 'excel', 'kodlama', 'linux',
                               'excel', 'yonetim']})

df5
df1
pd.merge(df1,df5)
df3 = pd.DataFrame({'name': ['Ali', 'Veli', 'Ayse', 'Fatma'],
                    'maas': [70000, 80000, 120000, 90000]})

df3
df1
pd.merge(df1,df3, left_on = 'calisanlar', right_on = 'name')
pd.merge(df1,df3, left_on = 'calisanlar', right_on = 'name').drop('name', axis =1)
df1a = df1.set_index('calisanlar')
df1a
df2
df2a = df2.set_index('calisanlar')
df2a
pd.merge(df1a, df2a, left_index = True, right_index = True)
df1a.join(df2a)

dfa = pd.DataFrame({'calisanlar': ['Ali', 'Veli', 'Ayse', 'Fatma'],
                    'siralama': [1, 2, 3, 4]})

dfa
dfb = pd.DataFrame({'calisanlar': ['Ali', 'Veli', 'Ayse', 'Fatma'],
                    'siralama': [3, 1, 4, 2]})

dfb
pd.merge(dfa, dfb, on = 'calisanlar')
pd.merge(dfa, dfb, on = 'calisanlar', suffixes = ["_MAAS", "_DENEYIM"])
import seaborn as sns
df = sns.load_dataset('planets')
df.head()
df.shape
df.count()
df['mass'].count()
df.describe().T
df['mass'].describe()
df.mean()
df.dropna().describe().T
df = pd.DataFrame({'gruplar': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'veri': [10,11,52,23,43,55]}, columns=['gruplar', 'veri'])
df
df.groupby('gruplar')
df.groupby('gruplar').sum()
df.head()
df.groupby('method')['orbital_period']
df.groupby('method')['orbital_period'].median()
df.groupby('method')['orbital_period'].describe()
df = pd.DataFrame({'gruplar': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'degisken1': [10,23,33,22,11,99],
                   'degisken2': [100,253,333,262,111,969]},
                   columns = ['gruplar', 'degisken1', 'degisken2'])
df
df.groupby('gruplar').aggregate(['min', np.median, max])
df.groupby('gruplar').aggregate({'degisken1': 'min', 'degisken2': 'max'})
def filter_func(x):
    return x['degisken1'].std() > 9
df.groupby('gruplar').filter(filter_func)

df.groupby('gruplar').transform(lambda x: (x-x.mean()) / x.std())

df.groupby('gruplar').apply(np.sum)
df.groupby('gruplar').apply(np.mean)
df.groupby(df['gruplar']).sum()
df
L = [0,1,0,1,2,0]
df.groupby(L).sum()
df.groupby(df['gruplar']).sum()
import pandas as pd
import seaborn as sns
df = sns.load_dataset('titanic')

df.head()
df.groupby('sex')[['survived']].mean()
df.groupby(['sex','class'])['survived'].aggregate('mean').unstack()
df.pivot_table('survived', index = 'sex', columns = 'class')
age = pd.cut(df['age'], [0,18,90])
age.head()
df.pivot_table('survived', ['sex',age], 'class')
fare = pd.qcut(df['fare'],2)
fare.head()
df.pivot_table('survived', ['sex', age], [fare, 'class'])
df.pivot_table(index = 'sex', columns = 'class', 
              aggfunc = {'survived' : sum, 'fare': 'mean'})
df.pivot_table('survived', index = 'sex', columns = 'class', margins = True)
!pip install dfply
from dfply import *
df = diamonds.copy()

df.head()
df >> head()
#df >>= head()
df.head()
df >> select(X.carat, X.cut) >> head()
df >> select(1, X.price, ['x','y']) >> head(3)
df >> select(columns_between('cut', 'table')) >> head(3)
df >> drop(1, X.price, ['x','y']) >> head()
df >> select(~X.carat, ~X.color) >> head()
df >> row_slice([10])
df >> group_by('cut') >> row_slice(5)
df >> distinct(X.color)
df >> mask(X.cut == 'Ideal') >> head()
df >> filter_by(X.cut == 'Ideal', X.color == 'E', X.table < 55, X.price < 500) >> head()
df >> mutate(x_plus = X.x + X.y) >> select(columns_from('x')) >> head()
df >> mutate(x_plus = X.x + X.y, y_div_z = (X.y / X.z)) >> select(columns_from('x')) >> head()
df >> transmute(x_plus = X.x + X.y, y_div_z = (X.y / X.z)) >> head()
df >> head()
df >> group_by(X.cut) >> arrange(X.price) >> ungroup() >> mask(X.carat < 0.23)
df >> summarize(price_ortalama = X.price.mean(), price_ss =  X.price.std())
df >> group_by('cut') >> summarize(price_ortalama = X.price.mean(), price_ss =  X.price.std())
pwd
import pandas as pd
pd.read_csv('reading_data/ornekcsv.csv', sep = ";")
pd.read_csv('reading_data/duz_metin.txt')
pd.read_excel('reading_data/ornekx.xlsx')