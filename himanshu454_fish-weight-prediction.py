import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score , mean_squared_error
data = pd.read_csv("../input/fish-market/Fish.csv")
data.head(10)
data.shape
data.info()
data.describe()
data.rename(columns= {'Length1':'Vertical Length', 'Length2':'Diagonal Length', 'Length3':'Cross Length'}, inplace=True)
data.head()
print(data.isnull().sum())
sns.countplot(data["Species"])
plt.figure(figsize = (16,9))
sns.heatmap(data.corr(), annot=True, cmap='YlGnBu');
sns.pairplot(data, kind='scatter', hue='Species');
sns.boxplot(x=data['Weight'])

data["Weight"] = data["Weight"].rank()
sns.boxplot(x=data['Weight'])
data["Vertical Length"] = data["Vertical Length"].rank()
data["Diagonal Length"] = data["Diagonal Length"].rank()
data["Cross Length"] = data["Cross Length"].rank()
sns.boxplot(x=data['Vertical Length'])
 
sns.boxplot(x=data['Diagonal Length'])
sns.boxplot(x=data['Cross Length'])
data.head(10)
x = data.iloc[:,2:7].values
y = data["Weight"]
train_x , test_x , train_y , test_y = train_test_split(x,y,test_size = 0.2)
model = LinearRegression()
model.fit(train_x , train_y)
pred = model.predict(test_x)
print(r2_score(test_y , pred))
print(mean_squared_error(test_y , pred))