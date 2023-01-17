import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import mean_absolute_error
df=pd.read_csv('../input/diverse-algorithm-analysis-dataset-daad/Pokemon_categorical.csv')
df.columns
df2=df[['name', 'type1', 'type2', 'hp', 'attack', 'defense', 'sp_attack',

       'sp_defense', 'speed', 'generation', 'is_legendary']]
df2.head()
df2.columns
df2.info()
df2.describe()
df2.corr()
plt.figure(figsize=(15,7))

sns.heatmap(df2.corr(),annot=True)
# plotting all the data
df3=df2.loc[:,['attack','defense','speed']]

df3.plot(figsize=(15,7))
df3.plot(subplots=True)

plt.show()
df2.plot(kind = "hist",y = "defense",bins = 50,range= (0,250))
features=['hp', 'attack', 'defense', 'sp_attack',

       'sp_defense', 'speed', 'generation']

X=df2[features]

y=df2.is_legendary

train_x,test_x,train_y,test_y=train_test_split(X,y)
model=DecisionTreeClassifier(random_state=1)

model.fit(train_x,train_y)

pred=model.predict(test_x)
print("Mean absolute error: ",mean_absolute_error(test_y,pred))

print("Model score:",model.score(test_x,test_y))
my_submission=pd.DataFrame({'index':test_x.index,'isLegendary':pred})

my_submission.to_csv('submission.csv',index=False)