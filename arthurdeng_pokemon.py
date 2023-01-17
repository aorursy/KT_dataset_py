# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
pokemon = pd.read_csv('../input/pokemon_alopez247.csv')
pokemon.head()
pokemon.info()
pokemon.describe()
Type1 = list(pokemon['Type_1'].unique())
for type1 in Type1:
  print (type1,":", np.mean(pokemon[pokemon['Type_1'] == type1].Total))
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
x = np.array(pokemon.loc[:, 'Total']).reshape(-1,1)
y = np.array(pokemon.loc[:,'Catch_Rate']).reshape(-1,1)
reg = LinearRegression()
predict_space = np.linspace(min(x), max(x)).reshape(-1,1)
reg.fit(x,y)
predicted = reg.predict(predict_space)
print('R^2 score: ',reg.score(x, y))
plt.plot(predict_space, predicted, color='black', linewidth=3)
plt.scatter(x=x,y=y)
plt.xlabel('Total')
plt.ylabel('CatchRate')
plt.show()
#make sense but not significant
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
knn = KNeighborsClassifier(n_neighbors = 3)
x1 = pokemon.loc[:,['HP','Attack','Defense','Speed','Sp_Atk','Sp_Def']]
y1 = pokemon.loc[:,'Type_1']
x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size = 0.3)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print('With KNN (K=3) accuracy is: ',knn.score(x_test,y_test)) 
#not siginificant
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train, knn.predict(x_train))
df1 = pokemon.groupby('Type_1')['Type_1'].count().reset_index(name = 'Count')
df1 = df1.sort_values(by = 'Count')
import seaborn as sns
plt.figure(figsize=(15,10))
sns.barplot(x=df1['Type_1'], y= df1['Count'])
plt.xticks(rotation= 90)
plt.xlabel('Type')
plt.ylabel('Count')
plt.title('Type Distribution')
plt.figure(figsize=(15,10))
ax = sns.boxplot(x = 'Type_1', y = 'Total', data = pokemon )
#If generation and islegendary relate to ability
plt.figure(figsize=(15,10))
ax = sns.boxplot(x = 'Generation', y = 'Total', hue = 'isLegendary', data = pokemon )
#Correlation between different pokemon stats
pokemon_stats = pokemon[["HP","Attack","Defense","Sp_Atk","Sp_Def","Speed","Height_m","Weight_kg","Catch_Rate"]]
corr = pokemon_stats.corr()
sns.heatmap(corr,annot = True)
