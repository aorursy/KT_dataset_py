# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



cvRates = pd.read_csv('../input/conversionRates.csv', encoding="ISO-8859-1")

freeForm = pd.read_csv('../input/freeformResponses.csv', encoding="ISO-8859-1")

data = pd.read_csv('../input/multipleChoiceResponses.csv', encoding="ISO-8859-1")

schema = pd.read_csv('../input/schema.csv', encoding="ISO-8859-1")





women_data = data.loc[data['GenderSelect'] == 'Female']

men_data = data.loc[data['GenderSelect'] == 'Male']

print('Total Respondents = ', len(data.index))

print('Women Respondents = ', len(women_data.index))

print('Men Respondents = ', len(men_data.index))



women_data['FirstTrainingSelect'].head(3)

# Any results you write to the current directory are saved as output.


# import seaborn and alias it as sns

import seaborn as sns

import matplotlib.pyplot as plt



sns.set()

with sns.axes_style('white'):

    ax = sns.countplot( x="FirstTrainingSelect",  data=data, color='steelblue')

    ax.set_title("First Training Selection")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

    for key,spine in ax.spines.items():

        spine.set_visible(False)

plt.show()



import matplotlib.pyplot as plt



temp_women=women_data['FirstTrainingSelect'].value_counts()

temp_men=men_data['FirstTrainingSelect'].value_counts()

labels = temp_women.index

sizes = temp_women.values

data_gender = ['Women', 'Men']

colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']

explode = (0.1, 0, 0, 0, 0, 0)  # only "explode" the 1st slice

fig = plt.figure(figsize=(12, 12))

for sp in range(0,2):

    ax = fig.add_subplot(2, 1,sp+1)

    patches, texts, autotexts = ax.pie(sizes, explode=explode, colors=colors,labels=labels, autopct='%1.1f%%',shadow=True,labeldistance=1.05)

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    for pie_wedge in patches:

        pie_wedge.set_edgecolor('white')

    for text in texts:

        text.set_color('grey')

    for autotext in autotexts:

        autotext.set_color('grey')



    ax.set_title(data_gender[sp])

    ax.tick_params(bottom="off", top="off", left="off", right="off")

    labels = temp_men.index

    sizes = temp_men.values



plt.show()

def plotPie (labels, sizes, title):

    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']

    explode = (0.1, 0, 0, 0, 0, 0)  # only "explode" the 1st slice

    fig = plt.figure(figsize=(12, 12))

    for sp in range(0,1):

        ax = fig.add_subplot(1, 1,sp+1)

        #explsion

       # explode = (0.2,0.2,0.2,0.2)

        patches, texts, autotexts = ax.pie(sizes, colors = colors, labels=labels, explode=explode, autopct='%1.1f%%', pctdistance=0.85)

#        patches, texts, autotexts = ax.pie(sizes, explode=explode, colors=colors,labels=labels, autopct='%1.1f%%',shadow=True,labeldistance=1.05)

        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        for pie_wedge in patches:

            pie_wedge.set_edgecolor('white')

        for text in texts:

            text.set_color('grey')

        for autotext in autotexts:

            autotext.set_color('grey')

        ax.set_title(title[sp])

        #draw circle

        centre_circle = plt.Circle((0,0),0.70,fc='white')

        fig = plt.gcf()

        fig.gca().add_artist(centre_circle)

        # Equal aspect ratio ensures that pie is drawn as a circle

        ax.axis('equal')  

        ax.tick_params(bottom="off", top="off", left="off", right="off")

    plt.tight_layout()

    plt.show()

    plt.show()

    return;



def selectPlot(temp_data, temp_title):

    temp=temp_data['FirstTrainingSelect'].value_counts()

    labels = temp.index

    sizes = temp.values

    plotPie(labels, sizes, temp_title )

    

young_data = data[(data['Age']>=18) & (data['Age']<=29)]

mid_data = data[(data['Age'] > 29) & (data['Age']<=40)]

senior_data = data[(data['Age'] > 41) & (data['Age']<=50)]

old_data = data[(data['Age'] > 50)]





selectPlot(young_data, ['Young Learners'])

selectPlot(mid_data, ['Middle Age Learners'])

selectPlot(senior_data, ['Senior Age Learners'])

selectPlot(old_data, ['Old Age Learners'])                                        
