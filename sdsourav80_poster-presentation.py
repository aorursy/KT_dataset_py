import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

col_types = {'Overall': np.int32, 'Age': np.int32}
 
#read only the columns that we will use (name and photo are loaded only to visualize the results at the end)
df = pd.read_csv("../input/Football_Data.csv", usecols=['Name','Agility', 'Value', 'Overall', 'Age', 'Finishing','Nationality'], dtype=col_types)
df.head()
#remove € character, leave just numbers
df['Value'] = df['Value'].str.replace('€', '')
 
#parse string for millions and thousands to numeric values
def parseValue(strVal):
    if 'M' in strVal:
        return int(float(strVal.replace('M', '')) * 1000000)
    elif 'K' in strVal:
        return int(float(strVal.replace('K', '')) * 1000)
    else:
        return int(strVal)    
df['Value'] = df['Value'].apply(lambda x: parseValue(x))
import os
print(os.listdir("../input"))
df.head(10)
#check if there are null/missing values and how many in each column
df.isnull().sum()
#Nobody can have a value lower or equal than zero, so those values are bad entries and we need to remove them
df = df.loc[df.Value > 0]
df.describe()
df.Overall.describe()
df.nlargest(5, columns='Overall')
df.nsmallest(5, columns='Overall')
import matplotlib.pyplot as plt

plt.hist(df.Overall, bins=16, alpha=0.6, color='b')
plt.title("#Players per Overall")
plt.xlabel("Overall")
plt.ylabel("Count")

plt.show()
import seaborn as sns
fig = plt.figure(1, figsize=(300, 6))
ax = fig.add_subplot(111)
sns.boxplot(x=df['Nationality'],y=df['Overall'],data=df)
# pie chart
players = df[["Name", "Age", "Nationality"]].dropna()
players.groupby("Nationality").Name.count().sort_values(ascending=False).head(9).plot(kind="pie")
fig = plt.figure(figsize=(8,5))
axes= fig.add_axes([0,0,1,1])

plt.scatter(x=df['Overall'],y=df['Value'])
plt.xlabel('Overall')
plt.ylabel('Value')
plt.title('This scatter diagram compares Overall vs Value')
df = pd.read_csv("../input/Football_Data.csv", usecols=['Value', 'Overall'], dtype=col_types)
#remove € character, leave just numbers
df['Value'] = df['Value'].str.replace('€', '')
#parse string for millions and thousands to numeric values
def parseValue(strVal):
    if 'M' in strVal:
        return int(float(strVal.replace('M', '')) * 1000000)
    elif 'K' in strVal:
        return int(float(strVal.replace('K', '')) * 1000)
    else:
        return int(strVal)    
df['Value'] = df['Value'].apply(lambda x: parseValue(x))
df
X=df.iloc[:,-1].values
y=df.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=.2,random_state=0)
# Create linear regression object
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train.reshape(-1,1), y_train)
# Predicting the Test set results
y_pred = regressor.predict(X_test.reshape(-1,1))
# Visualising the Training set results
plt.scatter(X_train.reshape(-1,1), y_train, color = 'red')
plt.plot(X_train.reshape(-1,1), regressor.predict(X_train.reshape(-1,1)), color = 'blue')
plt.title('Value vs Overall (Training set)')
plt.xlabel('Value')
plt.ylabel('Overall')
plt.show()
# Visualising the Training set results
plt.scatter(X_test.reshape(-1,1), y_test, color = 'red')
plt.plot(X_train.reshape(-1,1), regressor.predict(X_train.reshape(-1,1)), color = 'blue')
plt.title('Value vs Overall (Training set)')
plt.xlabel('Value')
plt.ylabel('Overall')
plt.show()
