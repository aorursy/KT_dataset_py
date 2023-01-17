# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
maori_file_path = '../input/cusersmarildownloadsmaorizip/csv/Business operations rates, innovation, biennial.csv'

maori = pd.read_csv(maori_file_path)

maori.head(5)

maori1_file_path = '../input/cusersmarildownloadsmaorizip/csv/Agriculture livestock information for Maori farms, annual.csv'

maori1 = pd.read_csv(maori1_file_path)

maori1.head(5)
maori2_file_path = '../input/cusersmarildownloadsmaorizip/csv/Business operations rates, debt types, 2014 and 2018.csv'

maori2 = pd.read_csv(maori2_file_path)

maori2.head(5)
maori3_file_path = '../input/cusersmarildownloadsmaorizip/csv/Business operations rates, product development, biennial.csv'

maori3 = pd.read_csv(maori3_file_path)

maori3.head(5)
maori4_file_path = '../input/cusersmarildownloadsmaorizip/csv/Business operations rates, activities, annual.csv'

maori4 = pd.read_csv(maori4_file_path)

maori4.head(5)
maori5_file_path = '../input/cusersmarildownloadsmaorizip/csv/Business operations rates, international markets, 2015.csv'

maori5 = pd.read_csv(maori5_file_path)

maori5.head(5)
maori6_file_path = '../input/cusersmarildownloadsmaorizip/csv/Agriculture land-use information for Maori farms, annual.csv'

maori6 = pd.read_csv(maori6_file_path)

maori6.head(5)
maori7_file_path = '../input/cusersmarildownloadsmaorizip/csv/Business operations rates, key factors competing, 2015.csv'

maori7 = pd.read_csv(maori7_file_path)

maori7.head(5)
maori8_file_path = '../input/cusersmarildownloadsmaorizip/csv/LEED worker turnover rates, quarterly.csv'

maori8 = pd.read_csv(maori8_file_path)

maori8.head(5)
maori9_file_path = '../input/cusersmarildownloadsmaorizip/csv/Business operations rates, proportion internet sales, biennial.csv'

maori9 = pd.read_csv(maori9_file_path)

maori9.head(5)
maori10_file_path = '../input/cusersmarildownloadsmaorizip/csv/Business demography enterprises for Maori authorities, annual.csv'

maori10 = pd.read_csv(maori10_file_path)

maori10.head(5)
maori11_file_path = '../input/cusersmarildownloadsmaorizip/csv/Business demography enterprises for Maori SMEs, annual.csv'

maori11 = pd.read_csv(maori11_file_path)

maori11.head(5)
maori12_file_path = '../input/cusersmarildownloadsmaorizip/csv/Business operations rates, skills recruitment, 2014.csv'

maori12 = pd.read_csv(maori12_file_path)

maori12.head(5)
maori13_file_path = '../input/cusersmarildownloadsmaorizip/csv/Business operations rates, vacancies, 2013 and 2014.csv'

maori13 = pd.read_csv(maori13_file_path)

maori13.head(5)
maori14_file_path = '../input/cusersmarildownloadsmaorizip/csv/Agriculture horticulture information for Maori farms, annual.csv'

maori14 = pd.read_csv(maori14_file_path)

maori14.head(5)
maori15_file_path = '../input/cusersmarildownloadsmaorizip/csv/Business operations rates, credit facilities, 2014 and 2018.csv'

maori15 = pd.read_csv(maori15_file_path)

maori15.head(5)
maori16_file_path = '../input/cusersmarildownloadsmaorizip/csv/LEED estimates of filled jobs, quarterly.csv'

maori16 = pd.read_csv(maori16_file_path)

maori16.head(5)
maori17_file_path = '../input/cusersmarildownloadsmaorizip/csv/Business operations rates, innovation barriers, 2015 and 2017.csv'

maori17 = pd.read_csv(maori17_file_path)

maori17.head(5)
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()
plotPerColumnDistribution(maori17, 10, 5)
labels=maori14.Kiwifruit.value_counts().index

sizes=maori14.Kiwifruit.value_counts().values

plt.figure(figsize=(11,11))

plt.pie(sizes,labels=labels,autopct="%1.f%%")

plt.title("Kiwifruit",size=25)

plt.show()
# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()
plotScatterMatrix(maori16, 15, 10)
labels1=maori12.Skill_areas.value_counts().index

sizes1=maori12.Skill_areas.value_counts().values

plt.figure(figsize=(11,11))

plt.pie(sizes1,labels=labels1,autopct="%1.1f%%")

plt.title("Skill_areas",size=25)

plt.show()
print ("Skew is:", maori14.Avocados.skew())

plt.hist(maori14.Avocados, color='green')

plt.show()
target = np.log(maori14.Wine_grapes)

print ("Skew is:", target.skew())

plt.hist(target, color='palegreen')

plt.show()
#Define a function which can pivot and plot the intended aggregate function 

def pivotandplot(data,variable,onVariable,aggfunc):

    pivot_var = data.pivot_table(index=variable,

                                  values=onVariable, aggfunc=aggfunc)

    pivot_var.plot(kind='bar', color='orange')

    plt.xlabel(variable)

    plt.ylabel(onVariable)

    plt.xticks(rotation=0)

    plt.show()
pivotandplot(maori11,'Industry','Enterprises',np.median)
# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
# It is a continous variable and hence lets look at the relationship of Employeecount with Enterprises using a Regression plot



_ = sns.regplot(maori10['EmployeeCount'], maori10['Enterprises'])
maori10.plot(kind='scatter', x='EmployeeCount', y='Enterprises', alpha=0.5, color='purple', figsize = (12,9))

plt.title('EmployeeCount And Enterprises')

plt.xlabel("EmployeeCount")

plt.ylabel("Enterprises")

plt.show()
import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling as pp

import plotly.graph_objs as go

from plotly.offline import iplot

import plotly.express as px
trace1 = go.Box(

    y=maori8["Maori_tourism_worker_turnover_rate"],

    name = 'Maori_tourism_worker_turnover_rate',

    marker = dict(color = 'rgb(0,145,119)')

)

trace2 = go.Box(

    y=maori8["Maori_authority_worker_turnover_rate"],

    name = 'Maori_authority_worker_turnover_rate',

    marker = dict(color = 'rgb(5, 79, 174)')

)



data = [trace1, trace2]

layout = dict(autosize=False, width=700,height=500, title='time', paper_bgcolor='rgb(243, 243, 243)', 

              plot_bgcolor='rgb(243, 243, 243)', margin=dict(l=40,r=30,b=80,t=100,))

fig = dict(data=data, layout=layout)

iplot(fig)
import matplotlib.pyplot as plt

plt.style.use(style='ggplot')

plt.rcParams['figure.figsize'] = (10, 6)
maori4.plot(kind='scatter', x='Exports_percent', y='Tourism_percent', alpha=0.5, color='mediumorchid', figsize = (12,9))

plt.title('Exports_percent And Tourism_percent')

plt.xlabel("Exports_percent")

plt.ylabel("Tourism_percent")

plt.show()
# libraries

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

 

# Dataset

df=pd.DataFrame({'X': range(1,101), 'Y': np.random.randn(100)*15+range(1,101), 'Z': (np.random.randn(100)*15+range(1,101))*2 })

 

# plot

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(maori14['Onions'], maori14['Squash'], maori14['Wine_grapes'], c='palegreen', s=60)

ax.view_init(30, 185)

plt.show()
ax = sns.violinplot(x="EmployeeCount", y="Enterprises", data=maori10, 

                    inner=None, color=".8")

ax = sns.stripplot(x="EmployeeCount", y="Enterprises", data=maori10, 

                   jitter=True)

ax.set_title('EmployeeCount vs Enterprises')

ax.set_ylabel('EmployeeCount . Enterprises')
g = sns.jointplot(maori16.Maori_authority_filled_jobs, maori16.Maori_SME_filled_jobs, kind="kde", height=7)

plt.savefig('graph.png')

plt.show()