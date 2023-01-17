import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import plotly.plotly as py

import plotly.graph_objs as go

%pylab inline



from collections import Counter



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/database.csv')

df.info()
with plt.style.context('ggplot'):

    plt.plot(df.groupby('Year')['Model'].apply(lambda x: len(set(x))), '.-')

    plt.title('Unique models over the years')

    plt.xlabel('Years')

    plt.ylabel('Unique Model Count')
def plotvar(consider, style='ggplot', size=(15, 7)):

    linestyles=['.-', 'o-', '^-', '-.']

    cycle_length=8

    with plt.style.context(style):

        plt.figure(figsize=size)

        count = 0

        for name, group in df.groupby(consider):

            temp = list(sorted(list(Counter(group['Year']).items()), key=lambda x: x[0]))

            x, y = [i[0] for i in temp], [i[1] for i in temp]

            linestyle = linestyles[int(count / cycle_length)]

            plt.plot(x, y, linestyle, label=name)

            count += 1

        plt.xlabel('Years')

        plt.ylabel('Count')

        plt.title('Evolution of "{}" over the years'.format(consider))

        plt.legend(bbox_to_anchor=(1.3, 1.))
plotvar('Drive')
def clean_class(x):

    x = x.replace('2WD', '')

    x = x.replace('4WD', '')

    x = x.replace(' - ', '')

    x = x.replace('/2wd', '')

    x = x.replace('/4wd', '')

    x = x.replace('Vehicles', 'Vehicle')

    if 'Vans' in x:

        x = 'Vans'

    return x.strip()

df['ClassClean'] = df.Class.apply(clean_class)

plotvar('ClassClean')
plotvar('Engine Cylinders')
plotvar('Fuel Type')