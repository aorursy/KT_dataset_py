# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sn

from sklearn.preprocessing import LabelEncoder

from sklearn.impute import SimpleImputer
# save filepath to variable for easier access

titanic_train_path = '/kaggle/input/titanic/train.csv'

titanic_test_path = '/kaggle/input/titanic/test.csv'



# read the data and store data in DataFrame 

train_data = pd.read_csv(titanic_train_path, index_col='PassengerId') 

X_test_full = pd.read_csv(titanic_test_path, index_col='PassengerId') 

#fill NA values



train_data['Age'] = train_data.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median())) 

#carbin number

train_data['Cabin_First_Char'] = train_data['Cabin'].str[:1].fillna('Z').apply(lambda x: 1 if x == 'Z' else 0)



#number of passengers per ticket

train_data['Ticket_Passengers'] = train_data.groupby('Ticket')['Ticket'].transform('count')



#family size

train_data['family_size'] = train_data['SibSp'] + train_data['Parch'] + 1



####################Important Features for Female#########################

#ticket passenger band

train_data['Ticket Band(4)'] = train_data['Ticket_Passengers'].apply(lambda x: '> 4' if x > 4 else '<=4')

#fare band

train_data['Fare Band(46.9)'] = train_data['Fare'].apply(lambda x: 'Expensive' if x >= 46.9 else 'Cheap'  )
def plot_figure(pivot_table, title, xpos,x,y):

    pivot_table.fillna(0, inplace=True)

    ax = pivot_table.plot.barh(stacked = True, figsize = (x,y))

    #add labels

    labels = []

    for j in pivot_table.columns:

        for i in pivot_table.index:

            if ((j == 0) and (pivot_table.loc[i][j] < 10) and (pivot_table.loc[i][j] < sum(pivot_table.loc[i]))):

                label = ""

            else:                                                                                                                       

                label = str(round((pivot_table.loc[i][j]/sum(pivot_table.loc[i]))*100,1)) + "% (" + str(pivot_table.loc[i][j].astype('int64')) + ")"

            labels.append(label)

    

    patches = ax.patches

    for label, rect in zip(labels, patches):

        width = rect.get_width()

        if width > 0:

            x = rect.get_x()

            y = rect.get_y()

            height = rect.get_height()



            if width>xpos: 

                ax.text(x + width/2, y + height/2., label, ha='center', va='center')

            else:

                ax.text(x + xpos, y + height/2., label, ha='center', va='center')

    plt.title(title)

    plt.show()
#Survival rate by sex



pivot_train_data = train_data.pivot_table(index = ['Sex'], columns = 'Survived', values = 'Age', aggfunc = 'count')

plot_figure(pivot_train_data, "Survival Analysis",10,8,8)
#turn Sex into a numerical field to get correlation between Sex and Survival

train_data['Sex_Num'] = train_data['Sex'].apply(lambda x: 1 if x == 'female' else 0)

#turn Carbin First Char into a numerical field to get correlation between Carbin loc and Survival

train_data['Sex_Num'] = train_data['Sex'].apply(lambda x: 1 if x == 'female' else 0)

#correlation between numerical dataset columns 

corrMatrix = train_data.corr()

# Set the width and height of the figure

plt.figure(figsize=(14,7))

# Set title

plt.title("Correlation Matrix - All")

# Heatmap showing correlation 

sn.heatmap(corrMatrix,annot=True)

#split into female and male

train_data_female = train_data[train_data['Sex'] == 'female']

train_data_male = train_data[train_data['Sex'] == 'male'  ]



#correlation between numerical dataset columns 

corrMatrixFemale = train_data_female.corr()

# Set the width and height of the figure

plt.figure(figsize=(14,7))

# Set the title

plt.title("Correlation Matrix - Female")

# Heatmap showing correlation 

sn.heatmap(corrMatrixFemale,annot=True)



#correlation between numerical dataset columns 

corrMatrixMale = train_data_male.corr()

# Set the width and height of the figure

plt.figure(figsize=(14,7))

# Set the title

plt.title("Correlation Matrix - Male")

# Heatmap showing correlation 

sn.heatmap(corrMatrixMale,annot=True)
#Visual 1 - Female Passengers - Survival by Pclass, Fare Band, Number of Passengers per Ticket

train_data_female['Family Size Band'] = train_data_female['family_size'].apply(lambda x: '>=5' if x>4 else('1' if x==1 else '2-4'))

train_data_female['Fare Band'] = train_data_female['Fare'].apply(lambda x: '>=45' if x>=45 else'<45')

pivot_female = train_data_female.pivot_table(index = ['Pclass','Family Size Band','Fare Band'], columns = 'Survived', values = 'Age', aggfunc = 'count')

plot_figure(pivot_female, "Female Survival Analysis",10,8,8)
#split into male - with cabin no and male - without cabin no

train_data_male_M = train_data_male[train_data_male['Cabin_First_Char'] == 1]

train_data_male_O = train_data_male[train_data_male['Cabin_First_Char'] == 0]

#correlation between numerical dataset columns 

corrMatrixMaleM = train_data_male_M.corr()

# Set the width and height of the figure

plt.figure(figsize=(14,7))

# Set the title

plt.title("Correlation Matrix - Male - without Cabin No")

# Heatmap showing correlation 

sn.heatmap(corrMatrixMaleM,annot=True)



#correlation between numerical dataset columns 

corrMatrixMaleO = train_data_male_O.corr()

# Set the width and height of the figure

plt.figure(figsize=(14,7))

# Set the title

plt.title("Correlation Matrix  Male - with Cabin No")

# Heatmap showing correlation 

sn.heatmap(corrMatrixMaleO,annot=True)
#Visual 2 - Male Passengers - Carbin is NULL - Survival by Age Band and PRCH Band

#train_data_male_M['Age Band'] = train_data_male_M['Age'].apply(lambda x: ">70" if x>70 else ("<10" if x<10  else "10-70") ) 

bins = [0,10,20,30,40,50,60,70,200]

train_data_male_M['Age Band'] = pd.cut(train_data_male_M['Age'], bins)

train_data_male_M['Parch Band'] = train_data_male_M['Parch'].apply(lambda x: ">=2" if x>=2 else "<2" ) 

train_data_male_M_pivot = train_data_male_M.pivot_table(index = [ 'Age Band','Parch Band'], columns = 'Survived', values = 'Age', aggfunc = 'count')

plot_figure(train_data_male_M_pivot, "Male Survival Analysis - Carbin None", 20,12,12)
#Visual 3 - Male Passengers - Carbin Not Null - Survival by Age Band and Ticket Band

train_data_male_O = train_data_male_O[train_data_male_O['Cabin_First_Char'] != 1]

bins = [0,10,20,40,60,70,200]

train_data_male_O['Age Band'] = pd.cut(train_data_male_O['Age'], bins)

bins1 =[0,3,5,20]

train_data_male_O['Ticket Band'] = pd.cut(train_data_male_O['Ticket_Passengers'], bins1)

train_data_male_O_pivot = train_data_male_O.pivot_table(index = [ 'Age Band','Ticket Band'], columns = 'Survived', values = 'Age', aggfunc = 'count')

plot_figure(train_data_male_O_pivot, "Male Survival Analysis - Carbin Not None",2,12,12)