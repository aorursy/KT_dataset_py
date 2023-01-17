import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("../input/Level of education of female in BD by residence of 5 years since 2008.csv")
print(dataset)
input_data = dataset.values
print(input_data)
dataset.head()
dataset.info()
print(dataset['National'].describe())
             #Rural VS National
print("-------------Scatter Plot: Rural vs National -----------")
x_Rural = input_data[:,2] 
y_National = input_data[:,1]
plt.scatter(x_Rural,y_National)
plt.show()
             #Urban VS National
print("-------------Scatter Plot: Urban vs National -----------")
x_Urban = input_data[:,3]
y_National = input_data[:,1]
plt.scatter(x_Urban,y_National)
plt.show()
dataset.boxplot()
dataset.hist()
df = pd.DataFrame({'Classpassed': ['National', 'Rural', 'Urban'], 'Schooling': [35.4, 38.1,26.5]})

df.Schooling.groupby(df.Classpassed).sum().plot(kind='pie')
plt.axis('equal')
plt.show()
df=dataset.copy()
w=df.groupby(['Class passed'])['National'].sum().sort_values(ascending=False).head(9).reset_index()
x=df.groupby(['Class passed'])['Rural'].sum().sort_values(ascending=False).head(9).reset_index()
y=df.groupby(['Class passed'])['Urban'].sum().sort_values(ascending=False).head(9).reset_index()
z=w.merge(x,on=['Class passed']).merge(y,on=['Class passed'])
z.plot(x='Class passed',y=['National','Rural','Urban'], kind="bar",figsize=(9,4))
import statsmodels.formula.api as sm
ad = pd.read_csv("../input/Level of education of female in BD by residence of 5 years since 2008.csv", index_col=0)
ad.corr()
modelAll = sm.ols('National ~ Rural + Urban', ad).fit()
modelAll.params