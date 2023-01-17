# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import re
import seaborn as sns
color = sns.color_palette()
sns.set(font_scale=1.6)
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
sub_df = pd.read_csv('../input/gender_submission.csv')
train_df.info()
print('---------------------------')
test_df.info()
train_df.head(2)
def has_cabin(data):
    data['Has_Cabin'] = data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

def sex_map(data):
    data['Sex'] = data['Sex'].map({'female':0, 'male':1}).astype(int)

def familysize(data):
    data['Family_Size'] = data['SibSp'] + data['Parch'] + 1

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

def title_name(data):
    data['Title'] = data['Name'].apply(get_title)
    #-------------------training dataset-------------------------
    #Mr          517
    #Miss        182
    #Mrs         125
    #Master       40
    #Dr_7, Rev_6, Col_2, Major_2, Mlle_2, Lady_1
    #Sir_1, Countess_1, Don_1, Capt_1, Mme_1, Ms_1, Jonkheer_1
    data['Title'] = data['Title'].replace(['Countess','Capt', 'Col','Don', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace(['Mlle', 'Lady', 'Ms'], 'Miss')
    data['Title'] = data['Title'].replace('Dr', 'Master')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    data['Title'] = data['Title'].replace('Sir', 'Mr')
    #Mapping title
    data['Title'] = data['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
    #Remove all NULLS in the Title column
    data['Title'] = data['Title'].fillna(0)

def embarked_map(data):
    # Mapping Embarked
    data['Embarked'] = data['Embarked'].map( {'S': 1, 'C': 2, 'Q': 3} )
    #Remove all NULLS in the Embarked column
    data['Embarked'] = data['Embarked'].fillna(0).astype(int)
def fare_map(data):
    # Mapping Fare
    data.loc[ data['Fare'] <= 7.8, 'Fare'] = 1
    data.loc[(data['Fare'] > 7.8) & (data['Fare'] <= 14.455 ), 'Fare'] = 2
    data.loc[(data['Fare'] > 14.455 ) & (data['Fare'] <= 54), 'Fare']   = 3
    data.loc[ data['Fare'] > 54, 'Fare'] = 4
    #Remove all NULLS in the Fare column
    data['Fare'] = data['Fare'].fillna(0)
    data['Fare'] = data['Fare'].astype(int)
def age_map(data):
    # Mapping age
    data.loc[ data['Age'] <= 17, 'Age'] = 1
    data.loc[(data['Age'] > 17) & (data['Age'] <= 28 ), 'Age'] = 2
    data.loc[(data['Age'] > 28 ) & (data['Age'] <= 45), 'Age']   = 3
    data.loc[ data['Age'] > 45, 'Age'] = 4
    #Remove all NULLS in the Age column
    data['Age'] = data['Age'].fillna(0)
    data['Age'] = data['Age'].astype(int)
def data_map(data):
    has_cabin(data)
    sex_map(data)
    familysize(data)
    title_name(data)
    embarked_map(data)
    fare_map(data)
    age_map(data)
    return data
#---------------------------------
train_data = data_map(train_df)
test_data = data_map(test_df)
train_data = train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
test_data = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
colormap = plt.cm.viridis
plt.figure(figsize=(16,16))
plt.title(' The Absolute Correlation Coefficient of Features', y=1.05, size=15)
sns.heatmap(abs(train_data.astype(float).corr()),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True, )
plt.show()
import gc
lis=list(train_data.columns)
lis.remove('Survived')
lis_1=list(train_data.columns)
lis_1.remove('Survived')
for i in lis:
    lis_1.remove(i)
    for j in lis_1:
        print('group by...')
        gp = train_data[[i,j,'Survived']].groupby(by=[i,j])
        gp = gp[['Survived']].count().reset_index().rename(index=str, columns={'Survived': '{0}_{1}'.format(i, j)})
        print('merge...')
        train_data = train_data.merge(gp, on=[j,i], how='left')
        del gp
        gc.collect()



plt.figure(figsize=(10,10))
plt.cm.viridis
plt.show()
lis=list(train_data.columns)
lis.remove('Survived')
k = int(len(lis)/5)
d1 = lis[:k]
d1.extend(['Survived'])
train_data1 = train_data[d1]
colormap = plt.cm.viridis
plt.figure(figsize=(16,16))
plt.title(' The Absolute Correlation Coefficient of Features', y=1.05, size=15)
sns.heatmap(abs(train_data1.astype(float).corr()),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True, )
plt.show()
d2 = lis[k:k+k]
d2.extend(['Survived'])
train_data2 = train_data[d2]
colormap = plt.cm.viridis
plt.figure(figsize=(16,16))
plt.title(' The Absolute Correlation Coefficient of Features', y=1.05, size=15)
sns.heatmap(abs(train_data2.astype(float).corr()),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True, )
plt.show()
d3 = lis[k*2:k*3]
d3.extend(['Survived'])
train_data3 = train_data[d3]
colormap = plt.cm.viridis
plt.figure(figsize=(16,16))
plt.title(' The Absolute Correlation Coefficient of Features', y=1.05, size=15)
sns.heatmap(abs(train_data3.astype(float).corr()),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True, )
plt.show()
d4 = lis[k*3:k*4]
d4.extend(['Survived'])
train_data4 = train_data[d4]
colormap = plt.cm.viridis
plt.figure(figsize=(16,16))
plt.title(' The Absolute Correlation Coefficient of Features', y=1.05, size=15)
sns.heatmap(abs(train_data4.astype(float).corr()),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True, )
plt.show()
d5 = lis[k*4:k*5]
d5.extend(['Survived'])
train_data5 = train_data[d5]
colormap = plt.cm.viridis
plt.figure(figsize=(16,16))
plt.title(' The Absolute Correlation Coefficient of Features', y=1.05, size=15)
sns.heatmap(abs(train_data5.astype(float).corr()),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True, )
plt.show()
def calc_iv(df, feature, target, pr=False):
    """
    Set pr=True to enable printing of output.
    
    Output: 
      * iv: float,
      * data: pandas.DataFrame
    """

    lst = []

    df[feature] = df[feature].fillna("NULL")

    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append([feature,                                                        # Variable
                    val,                                                            # Value
                    df[df[feature] == val].count()[feature],                        # All
                    df[(df[feature] == val) & (df[target] == 0)].count()[feature],  # Good (think: Fraud == 0)
                    df[(df[feature] == val) & (df[target] == 1)].count()[feature]]) # Bad (think: Fraud == 1)

    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])

    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])

    data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})

    data['IV'] = data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])
    data['abs_WoE'] = abs(data['WoE'])
    data = data.sort_values(by=['Variable', 'Value'], ascending=[True, True])
    data.index = range(len(data.index))

    if pr:
        print(data)
        print('IV = ', data['IV'].sum())
    return data
def loop_iv(lis_val, train_df):
    dic_val = {}
    for i in lis_val:
        data = calc_iv(train_df, i, 'Survived')
        dic_val[i] = data
    return dic_val
lis=list(train_data.columns)
lis.remove('Survived')
dic=loop_iv(lis, train_data)
sum_iV={}
for i in lis:
    sum_num = dic[i]['IV'].sum()
    sum_iV[i] = sum_num
    print("The sum of IV parameter in {0} : {1}".format(i, sum_num))
plt.figure(figsize = (32,8))
m_colors=[]
k_num=0
for num in range(len(lis)):
    if (num//len(color))>k_num:
        k_num+=1
    t = num - k_num*len(color)
    m_colors.append(color[t])
sns.barplot(list(sum_iV.keys()), list(sum_iV.values()), alpha=0.8, palette = m_colors)
plt.ylabel('IV value', fontsize = 12)
plt.xlabel('Parameters', fontsize = 12)
plt.title("Each parameter's IV", fontsize = 16)
plt.xticks(rotation='vertical')
plt.show()
v=list(sum_iV.values())
v_m = list(sum_iV.values())
v_n = list(sum_iV.keys())
v.sort()
kv = []
for i in range(len(v)):
    vv = v_m.index(v[i])
    kv.append(v_n[vv])
over40 = ['Has_Cabin_Title', 'Family_Size_Title', 'Embarked_Title', 'Fare_Title', 'Parch_Title', 'SibSp_Title', 'Age_Title', 'Sex_Title', 'Sex_Has_Cabin', 'Sex_Fare', 'Sex_Parch', 'Sex_Age', 'Pclass_Title', 'Pclass_Sex', 'Title', 'Sex']
for i in range(1,len(over40)):
    if kv[-i] in over40:
        print('{0} is important.'.format(kv[-i]))
