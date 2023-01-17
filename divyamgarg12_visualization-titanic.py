import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# Read the dataset
df = pd.read_csv("/kaggle/input/titanic/train.csv")
df.head()
# Checking the null values
df.isna().sum()
# There are 177 null values in age variable, 687 in cabin and 2 in Embarked
# Passenger survived in each class
survivors = df.groupby('Pclass')['Survived'].agg(sum)
survivors.head()
# Total number of passengers in each class
total_passengers = df.groupby('Pclass')['PassengerId'].count()
total_passengers
# Survivor percentage 
survivor_percentage = survivors/total_passengers
survivor_percentage
# Plotting total number of survivors
fig = plt.figure()
ax = fig.add_subplot(111)
rect = ax.bar(survivors.index.values.tolist(),  
          survivors, color='blue', width=0.5)
ax.set_ylabel('No. of survivors')
ax.set_title('Total number of survivors based on class')
xTickMarks = survivors.index.values.tolist()
ax.set_xticks(survivors.index.values.tolist())
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, fontsize = 20)
plt.show()
# Plotting percentage of survivors in each class
fig = plt.figure()
ax = fig.add_subplot(111)
rect = ax.bar(survivor_percentage.index.values.tolist(),  
          survivor_percentage, color='blue', width=0.5)
ax.set_ylabel('Survivor Percentage')
ax.set_title('Percentage of survivor based on class')
xTickMarks = survivor_percentage.index.values.tolist()
ax.set_xticks(survivor_percentage.index.values.tolist())
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, fontsize = 20)
plt.show()
# Male passenger survived in each class
Male_survived = df[df['Sex']=='male'].groupby('Pclass')['Survived'].agg(sum)
Male_survived
# Total number of male passenger in each class
Male_passenger = df[df['Sex']=='male'].groupby('Pclass')['PassengerId'].count()
Male_passenger
# Male_survived Percentage
Male_per_survived = Male_survived/Male_passenger
Male_per_survived
# Female passenger survived in each class
Female_survived = df[df['Sex']=='female'].groupby('Pclass')['Survived'].agg(sum)
Female_survived

# Total number of female passenger in each class
Female_passenger = df[df['Sex']=='female'].groupby('Pclass')['PassengerId'].count()
Female_passenger

# Female_survived Percentage
Female_per_survived = Female_survived/Female_passenger
Female_per_survived
# Plotting total number of passengers who survived based on gender
fig = plt.figure()
ax = fig.add_subplot(111)
index = np.arange(Male_survived.count())
bar_width = 0.35
rect1 = ax.bar(index, Male_survived, bar_width, color='blue', label = 'Men')
rect2 = ax.bar(index + bar_width, Female_survived, bar_width, color='red', label = 'Women')

ax.set_ylabel('Survivor Numbers')
ax.set_title('Male and Female survivors based on class')
xTickMarks = Male_survived.index.values.tolist()
ax.set_xticks(index + bar_width)
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, fontsize = 20)
plt.legend()
plt.tight_layout()
plt.show()
# Plotting percentage of passengers who survived based on gender
fig = plt.figure()
ax = fig.add_subplot(111)
index = np.arange(Male_survived.count())
bar_width = 0.35
rect1 = ax.bar(index, Male_per_survived, bar_width, color='blue', label = 'Men')
rect2 = ax.bar(index + bar_width, Female_per_survived, bar_width, color='red', label = 'Women')

ax.set_ylabel('Survivor Percentage')
ax.set_title('Percentage Male and Female survivors based on class')
xTickMarks = Male_per_survived.index.values.tolist()
ax.set_xticks(index + bar_width)
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, fontsize = 20)
plt.legend()
plt.tight_layout()
plt.show()
# Total Number of non-survivors in each class
non_survivors = df[(df['SibSp'] > 0) | (df['Parch'] > 0) &  
       (df['Survived'] == 0)].groupby('Pclass')['Survived'].agg('count')
non_survivors
#Total passengers in each class
total_passengers = df.groupby('Pclass')['PassengerId'].count()
non_survivor_percentage = non_survivors / total_passengers
non_survivor_percentage
#Total number of non survivors with family based on class
fig = plt.figure()
ax = fig.add_subplot(111)
rect = ax.bar(non_survivors.index.values.tolist(), non_survivors,  
       color='blue', width=0.5)
ax.set_ylabel('No. of non survivors')
ax.set_title('Total number of non survivors with family based on class')
xTickMarks = non_survivors.index.values.tolist()
ax.set_xticks(non_survivors.index.values.tolist())
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, fontsize=20)
plt.show()
# Percentage of non-survivors in each class

fig = plt.figure()
ax = fig.add_subplot(111)
rect = ax.bar(non_survivor_percentage.index.values.tolist(), non_survivor_percentage,  
       color='blue', width=0.5)
ax.set_ylabel('Non survivor Percentage')
ax.set_title('Percentage of non survivors with family based on class')
xTickMarks = non_survivor_percentage.index.values.tolist()
ax.set_xticks(non_survivor_percentage.index.values.tolist())
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, fontsize=20)
plt.show()
#Defining the age binning interval
age_bin = [0, 18, 25, 40, 60, 100]
#Creating the bins
df['AgeBin'] = pd.cut(df.Age, bins=age_bin)
#Removing the null rows
d_temp = df[np.isfinite(df['Age'])]  # removing all na instances
#Number of survivors based on Age bin
survivors = d_temp.groupby('AgeBin')['Survived'].agg(sum)
#Total passengers in each bin
total_passengers = d_temp.groupby('AgeBin')['Survived'].agg('count')
#Plotting the pie chart of total passengers in each bin
plt.pie(total_passengers,  
     labels=total_passengers.index.values.tolist(),
     autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Total Passengers in different age groups')
plt.show()
#Plotting the pie chart of percentage passengers in each bin
plt.pie(survivors, labels=survivors.index.values.tolist(),
     autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Survivors in different age groups')
plt.show()