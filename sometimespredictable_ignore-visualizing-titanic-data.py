from collections import Counter, OrderedDict



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns
data = pd.read_csv("../input/train.csv")

data.corr()
corr = data.corr()

sns.heatmap(

    corr, 

    xticklabels=corr.columns.values,

    yticklabels=corr.columns.values

)
# Helper functions



def plot_survivors_percentage_barchart(data, column):

    """Plot survivor % for groups in given column.

    

    Example.

    >>> data = pd.read_csv("../input/train.csv")

    >>> plot_survivors_percentage_barchart(data, 'Pclass')

    

    Params

    ------

        data : <pandas.DataFrame>

            Data loaded from ../input/train.csv.

            The person is considered a survivor if data['Survived'] == 1.

        column : str

            Column name, e.g. 'Pclass', 'Parch'.



    Returns

    -------

        fig, ax : ...

    """

    fig, ax = plt.subplots()

    

    total_by_group = Counter(data[column])

    survivors_by_group = Counter(data[data['Survived'] == 1][column])

    

    survivors_percentage_by_group = dict([

        (group, 100.0 * survivors_by_group.get(group, 0) / total_by_group[group])

        for group in total_by_group.keys()

    ])

    survivors_percentage_by_group = OrderedDict(sorted(survivors_percentage_by_group.items()))

    

    sns.barplot(

        x=list(survivors_percentage_by_group.keys()),

        y=list(survivors_percentage_by_group.values()),

        color="indigo"

    )

    ax.set_ylim([0, 100])

    ax.set_ylabel('Percentage survived')

    yticks = [i*10 for i in range(11)]

    ax.set_yticks(yticks)

    yticklabels = [str(num) + '%' for num in yticks]

    ax.set_yticklabels(yticklabels)

    

    return fig, ax





def plot_survivors_and_total_barchart(data, column):

    """Plot total number of passengers and survivors in given column.

    

    Params

    ------

        data : <pandas.DataFrame>

            Data loaded from ../input/train.csv.

            The person is considered a survivor if data['Survived'] == 1.

        column : str

            Column name, e.g. 'Pclass', 'Parch'.



    Returns

    -------

        fig, ax : ...

    """

    fig, ax = plt.subplots()



    total_by_group = OrderedDict(sorted(Counter(data[column]).items()))

    survivors_by_group = OrderedDict(sorted(

        Counter(data[data['Survived'] == 1][column]).items()

    ))



    sns.barplot(

        x=list(total_by_group.keys()),

        y=list(total_by_group.values()),

        color="blueviolet",

        label="Total # of passengers"

    )

    sns.barplot(

        x=list(survivors_by_group.keys()),

        y=list(survivors_by_group.values()),

        color="indigo",

        label="Survivors"

    )

    

    ax.legend()

    

    return fig, ax
# For nicer labels on the X axis

mapping = {'1': '$1^{st}$', '2': '$2^{nd}$', '3': '$3^{rd}$'}



# Plot percentage

fig, ax = plot_survivors_percentage_barchart(data, 'Pclass');

ax.set_title('Survivors % by Ticket Class')

ax.set_xlabel('Ticket Class')

xticklabels = [mapping[label.get_text()] for label in ax.get_xticklabels()]

ax.set_xticklabels(xticklabels)



# Plot number of people

fig, ax = plot_survivors_and_total_barchart(data, 'Pclass');

ax.set_title('Total passengers / Survivors by Ticket Class')

ax.set_xlabel('Ticket Class')

xticklabels = [mapping[label.get_text()] for label in ax.get_xticklabels()]

ax.set_xticklabels(xticklabels);
# Plot percentage

fig, ax = plot_survivors_percentage_barchart(data, 'Sex');

ax.set_title('Survivors % by Gender')

ax.set_xlabel('Gender')

xticklabels = [label.get_text().capitalize() for label in ax.get_xticklabels()]

ax.set_xticklabels(xticklabels)



# Plot number of people

fig, ax = plot_survivors_and_total_barchart(data, 'Sex');

ax.set_title('Total passengers / Survivors by Gender')

ax.set_xlabel('Gender')

xticklabels = [label.get_text().capitalize() for label in ax.get_xticklabels()]

ax.set_xticklabels(xticklabels);
# TODO: Age ranges

#fig, ax = plot_survivors_percentage_barchart(data, 'Age');
fig, ax = plot_survivors_percentage_barchart(data, 'SibSp');
fig, ax = plot_survivors_percentage_barchart(data, 'Parch');
#fig, ax = plot_survivors_percentage_barchart(data, 'Ticket');
#fig, ax = plot_survivors_percentage_barchart(data, 'Fare');
#fig, ax = plot_survivors_percentage_barchart(data, 'Cabin');
#fig, ax = plot_survivors_percentage_barchart(data, 'Embarked');