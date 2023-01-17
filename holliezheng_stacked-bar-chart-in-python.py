# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
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
# save filepath to variable for easier access

titanic_train_path = '/kaggle/input/titanic/train.csv'

# read the data and store data in DataFrame 

train_data = pd.read_csv(titanic_train_path, index_col='PassengerId') 

#Survival rate by sex

pivot_train_data = train_data.pivot_table(index = ['Sex'], columns = 'Survived', values = 'Age', aggfunc = 'count')

plot_figure(pivot_train_data, "Survival Analysis",10,8,8)