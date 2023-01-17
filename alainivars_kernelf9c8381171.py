import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def figure_size_small():
    """Set the size of matplotlib figures"""
    plt.rc('figure', figsize=(10, 5))

def figure_size_big():
    """Set the global default size of matplotlib figures"""
    plt.rc('figure', figsize=(15, 10))

# Set the global default size of matplotlib figures
figure_size_small()

# graph type
kind_type = ['bar', 'barh', 'box', 'kde', 'area', 'scatter', 'hexbin', 'pie']

'''Read the data:'''
df_train = pd.read_csv('../input/train.csv')
'''View the data types of each column:'''
df_train.dtypes
df_train.count()
(df_train.count()['PassengerId'] - df_train.count()['Age'], "Age", 
df_train.count()['PassengerId'] - df_train.count()['Cabin'], "Cabin", 
df_train.count()['PassengerId'] - df_train.count()['Embarked'], "Embarked")
'''Head data (first 5 records)'''
df_train.head()
'''tail data (last 5 records)'''
df_train.tail()
'''replace Age = Nan by, Age = 130'''
df_train['Age'] = df_train['Age'].fillna(np.float64(130))
'''replace Cabin empty by, Cabin = Alpha'''
df_train['Cabin'] = df_train['Cabin'].fillna('Alpha')
df_train.count()
crosstab_pclass_survived = pd.crosstab(df_train['Pclass'], df_train['Survived'], normalize='index')
crosstab_pclass_survived
figure_size_small()
kind=kind_type[0]
crosstab_pclass_survived.plot(kind=kind, stacked=True)
plt.title('Survival Rate by Class')
plt.xlabel('Class')
plt.ylabel('Survival Rate')
plt.show()
kind=kind_type[0]
df_crosstab = pd.crosstab(df_train['Cabin'], df_train['Survived'])
df_crosstab
# df_crosstab.plot(kind=kind, stacked=True)
# plt.show()
kind=kind_type[3]
class_list = sorted(df_train['Pclass'].unique())
for pclass in class_list:
    df_train.Age[df_train.Pclass == pclass].plot(kind=kind)
plt.title('Age Density Plot by Passenger Class')
plt.xlabel('Age')
plt.legend(('1st Class', '2nd Class', '3rd Class'), loc='best')
plt.show()
kind=kind_type[4]
df_train_norm1 = pd.crosstab([df_train['Age'],df_train['Pclass']], df_train['Survived'], normalize='index')
df_train_norm1.plot(kind=kind, stacked=True)
plt.title('Survival Rate by Age/Class')
plt.xlabel('Age/Class')
plt.ylabel('Survival Rate')
plt.show()
#kind_type = ['bar', 'barh', 'box', 'kde', 'area', 'scatter', 'hexbin', 'pie']
kind=kind_type[4]
df_train_norm1 = pd.crosstab([df_train['Pclass'],df_train['Age']], df_train['Survived'], normalize='index')
df_train_norm1.plot(kind=kind, stacked=True)
plt.title('Survival Rate by Class/Age')
plt.xlabel('Class/Age')
plt.ylabel('Survival Rate')
plt.show()
kind=kind_type[4]
df_train_norm1 = pd.crosstab([df_train['Sex'],df_train['Pclass'],df_train['Age']], df_train['Survived'], normalize='index')
df_train_norm1.plot(kind=kind, stacked=True)
plt.title('Survival Rate by Sex/Class/Age')
plt.xlabel('Sex/Class/Age')
plt.ylabel('Survival Rate')
plt.show()
ranges_age = [
    0, 5, 10, 15,   # childs
    25, 35, 45, 55,  # Adult
    65, 70, 80, 120,  # Older
    600]  # unknow Age
group_by_age = pd.cut(df_train["Age"], ranges_age)
#group_by_age
kind=kind_type[0]
df_train_norm1 = pd.crosstab([df_train['Pclass'],df_train['Sex'],group_by_age], df_train['Survived'], normalize='index')
df_train_norm1.plot(kind=kind, stacked=False)
plt.title('Survival Rate by Class/Sex/Age')
plt.xlabel('Sex/Class/Age')
plt.ylabel('Survival Rate')
plt.show()
figure_size_big()
kind=kind_type[0]
df_train_norm1 = pd.crosstab([df_train['Sex'],df_train['Pclass'],group_by_age], df_train['Survived'], normalize='index')
df_train_norm1.plot(kind=kind, stacked=False)
plt.title('Survival Rate by Sex/Class/Age')
plt.xlabel('Sex/Class/Age')
plt.ylabel('Survival Rate')
plt.show()
pd.crosstab([df_train['Sex'],df_train['Pclass'],group_by_age], df_train['Survived'], normalize='index')
ranges_age = [
    0, 5, 10, 15,   # childs
    25, 35, 45, 55,  # Adult
    65, 70, 80, 120,  # Older
    600]  # unknow Age
group_by_age = pd.cut(df_train["Age"], ranges_age)
# group_by_age
# age_grouping = df_train.groupby(group_by_age).mean()
# age_grouping['Survived'].plot.bar()
# plt.show()
data_fit = pd.crosstab(
    index=[
        df_train['Sex'],
        df_train['Pclass'],
        group_by_age
    ],
    columns=df_train['Survived'],
    rownames=['Sex', 'Pclass', 'Age'],
    colnames=['Survived'],
    normalize='index'
)
print(data_fit)
df_train['Survived'].value_counts()
total = df_train['Survived'].value_counts()
dead = total[0] / (total[0] + total[1])
survived = total[1] / (total[0] + total[1])
survived, dead
