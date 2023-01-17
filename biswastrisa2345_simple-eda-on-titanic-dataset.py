#importing libraries 

import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_excel("../input/Titanic.xlsm",header=None)
#checking top 5 rows
dataset.head(5)
#Assigning column Names to your dataframe
Frame = pd.DataFrame(dataset.values, columns = ["PassengerId", "Survived", "Pclass", "Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"])
Frame.head(5)
#Check number of rows and columns in dataset
Frame.shape
#Check for percentage of null values in data
(Frame.isnull().sum()/Frame.shape[0]) * 100
#Handling null values
Frame = Frame.drop(columns=['Cabin'])
Frame = Frame.dropna()
#Check datatypes of the columns
Frame.dtypes
#Datatype conversions
for col in ['Pclass', 'Survived', 'Embarked','Sex']:
    Frame[col] = Frame[col].astype('category')
Frame['Fare'] = Frame['Fare'].astype('float')
for cols in ['SibSp', 'Age', 'Parch']:
    Frame[cols] = Frame[cols].astype('int')

Frame.dtypes
#Creating age buckets
agebins = [0,15,30,45,60,75,90]
Frame['AgeGroups'] = pd.cut(Frame['Age'], agebins,labels=['0-15', '15-30', '30-45','45-60','60-75','75-90'],right=False)
#Extracting 'initials' out of Name column
Frame['Title'] = Frame.Name.str.split(r'\s*,\s*|\s*\.\s*').str[1]
#Assign category datatype 
Frame['Title'] = Frame['Title'].astype('category')
#Checking number of people by their initials
Frame.groupby('Title').size()
#Creating a new column 'Family' based on total siblings and total parents
Frame['Family'] = Frame['SibSp'] + Frame['Parch']
#5 number summary
Frame.describe()
#Vizualisations
#How many people survived by their 'initials'
ax = Frame.groupby(['Title','Survived'])["Family"].sum().unstack('Survived').plot(kind='bar', title ="Title vs Survived", figsize=(10, 5), legend=True, fontsize=12)
ax.set_xlabel("Title", fontsize=12)
ax.set_ylabel("No.of Passengers", fontsize=12)
plt.show()
from matplotlib.lines import Line2D
x = Frame.groupby(['Survived','Sex'])["Family"].sum().unstack("Survived").plot(kind='bar', title ="Survived vs Sex", figsize=(10, 5), legend=True, fontsize=12)
ax.set_xlabel("Sex", fontsize=12)
ax.set_ylabel("No. of Passengers", fontsize=12)
plt.show()
Frame.groupby(['Survived','Pclass'])["Family"].sum().unstack("Survived").plot(kind='bar', title ="Survived vs Pclass", figsize=(10, 5), legend=True, fontsize=12)
ax.set_xlabel("Sex", fontsize=12)
ax.set_ylabel("No. of Passengers", fontsize=12)
plt.show()
p = Frame.groupby(['Survived','AgeGroups'])["Family"].sum().unstack("Survived").plot(kind='bar', title ="Survived vs AgeGroups", figsize=(10, 5), legend=True, fontsize=12)
p.set_xlabel("Sex", fontsize=12)
p.set_ylabel("No. of Passengers", fontsize=12)
plt.show()
