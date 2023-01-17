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

train_set = pd.read_csv('/kaggle/input/titanic/train.csv')

df = pd.DataFrame(train_set)

a = df.describe() #a dataframe give out 'count' and 'mean'

b = df.info()     #a dateframe including data type

#print(a,b)

survived_df = df[df['Survived']==1]

dead_df = df[df['Survived']==0]



def drawing():

    #drawing a stacked bar chart 'gender vs survivors'

    labels = ['men','women']

    survived_male = survived_df[survived_df['Sex']=='male']

    survived = [len(survived_male),len(survived_df)-len(survived_male)]

    dead_male = dead_df[dead_df['Sex']=='male']

    dead = [len(dead_male),len(dead_df)-len(dead_male)]

    print(survived,dead)



    width = 0.2



    fig =plt.figure()

    ax = fig.add_subplot(221)



    ax.bar(labels,survived,width,label='Survived')

    ax.bar(labels,dead,width,label='Dead',bottom=survived)



    ax.set_ylabel('Number of people')

    ax.set_title('Gender vs Survival')





    #drawing stacker bar chart 'Pclass vs survivors'

    labels_2 = ['class_1','class_2','class_3']

    survived_class_1 = survived_df[survived_df['Pclass']==1]

    survived_class_2 = survived_df[survived_df['Pclass']==2]

    survived_class_3 = survived_df[survived_df['Pclass']==3]

    survived = [len(survived_class_1),len(survived_class_2),len(survived_class_3)]

    dead_class_1 = dead_df[dead_df['Pclass']==1]

    dead_class_2 = dead_df[dead_df['Pclass']==2]

    dead_class_3 = dead_df[dead_df['Pclass']==3]

    dead = [len(dead_class_1),len(dead_class_2),len(dead_class_3)]



    bx = fig.add_subplot(222)



    bx.bar(labels_2,survived,width,label='Survived')

    bx.bar(labels_2,dead,width,label='Dead',bottom=survived)



    bx.set_ylabel('Number of people')

    bx.set_title('Pclass vs Survival')



    #drawing stacker bar chart 'Embarkment vs survivors'

    labels_3 = ['Southampton','Cherbourg','Queenstown']

    survived_s = survived_df[survived_df['Embarked']=='S']

    survived_c = survived_df[survived_df['Embarked']=='C']

    survived_q = survived_df[survived_df['Embarked']=='Q']

    survived = [len(survived_s),len(survived_c),len(survived_q)]

    dead_s = dead_df[dead_df['Embarked']=='S']

    dead_c = dead_df[dead_df['Embarked']=='C']

    dead_q = dead_df[dead_df['Embarked']=='Q']

    dead = [len(dead_s),len(dead_c),len(dead_q)]



    cx = fig.add_subplot(223)



    cx.bar(labels_3,survived,width,label='Survived')

    cx.bar(labels_3,dead,width,label='Dead',bottom=survived)



    cx.set_ylabel('Number of people')

    cx.set_title('Embarked vs Survival')



    #drawing stacked bar chart 'families vs survivors'

    #make a new dataframe including a new column called 'families = SibSp + Parch'

    new = df.eval('Families = SibSp + Parch',inplace = False)

    labels_4 = ['0','1','2','3','4','5','6','7','8','9','10']

    survived_new = new[new['Survived']==1]

    dead_new = new[new['Survived']==0]

    survived_fam_0 = survived_new[survived_new['Families']==0]

    survived_fam_1 = survived_new[survived_new['Families']==1]

    survived_fam_2 = survived_new[survived_new['Families']==2]

    survived_fam_3 = survived_new[survived_new['Families']==3]

    survived_fam_4 = survived_new[survived_new['Families']==4]

    survived_fam_5 = survived_new[survived_new['Families']==5]

    survived_fam_6 = survived_new[survived_new['Families']==6]

    survived_fam_7 = survived_new[survived_new['Families']==7]

    survived_fam_8 = survived_new[survived_new['Families']==8]

    survived_fam_9 = survived_new[survived_new['Families']==9]

    survived_fam_10 = survived_new[survived_new['Families']==10]

    survived = [len(survived_fam_0),len(survived_fam_1),len(survived_fam_2),\

    len(survived_fam_3),len(survived_fam_4),len(survived_fam_5),len(survived_fam_6),\

    len(survived_fam_7),len(survived_fam_8),len(survived_fam_9),len(survived_fam_10)]

    print(survived)

    dead_fam_0 = dead_new[dead_new['Families']==0]

    dead_fam_1 = dead_new[dead_new['Families']==1]

    dead_fam_2 = dead_new[dead_new['Families']==2]

    dead_fam_3 = dead_new[dead_new['Families']==3]

    dead_fam_4 = dead_new[dead_new['Families']==4]

    dead_fam_5 = dead_new[dead_new['Families']==5]

    dead_fam_6 = dead_new[dead_new['Families']==6]

    dead_fam_7 = dead_new[dead_new['Families']==7]

    dead_fam_8 = dead_new[dead_new['Families']==8]

    dead_fam_9 = dead_new[dead_new['Families']==9]

    dead_fam_10 = dead_new[dead_new['Families']==10]

    dead = [len(dead_fam_0),len(dead_fam_1),len(dead_fam_2),\

    len(dead_fam_3),len(dead_fam_4),len(dead_fam_5),len(dead_fam_6),\

    len(dead_fam_7),len(dead_fam_8),len(dead_fam_9),len(dead_fam_10)]



    dx = fig.add_subplot(224)



    dx.bar(labels_4,survived,width,label='Survived')

    dx.bar(labels_4,dead,width,label='Dead',bottom=survived)



    dx.set_ylabel('Number of people')

    dx.set_title('Families vs Survival')



    ax.legend()

    bx.legend()

    cx.legend()

    dx.legend()

    fig.tight_layout() #automativally adjust the blank part to exclude interacting

    plt.show()

    

drawing()