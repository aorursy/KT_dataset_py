import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Visualisation Libraries

import matplotlib.pyplot as plt

from matplotlib import cm

import seaborn as sns

import warnings

import re



pd.plotting.register_matplotlib_converters()

%matplotlib inline

plt.style.use('seaborn-darkgrid')

pd.set_option('display.max_columns', 50)

warnings.filterwarnings("ignore")

pd.options.display.float_format = '{:.2f}'.format
url = '../input/personality-traits-of-twitter-users-celebrities/analisis.csv'

data = pd.read_csv(url, header='infer')
data.head()
#Checking for null/missing values

data.isna().sum()
#Renaming the columns

data = data.rename(columns={'usuario':'User','op':'Openness','co':'Conscientiousness','ex':'Extraversion',

                            'ag':'Agreeableness','ne':'Neuroticism','categoria':'Category' })
data['Category'] = data['Category'].astype('category')
data['wordcount'] = data['wordcount'].astype('int')



# Function to round up word count

def roundup_wordcount(count):

    count = round(count)

    return count



# Applying the function to Word Count Column

data['wordcount'] = data['wordcount'].apply(roundup_wordcount)
data.head()
data.describe().transpose()
# Function that shows the summary and density distribution of a numerical attribute:

def summary(x):

    x_min = data[x].min()

    x_max = data[x].max()

    Q2 = data[x].quantile(0.50)

    x_mean = data[x].mean()

    print(f'4 Point Summary of {x.capitalize()} Attribute:\n'

          f'{x.capitalize()}(min)   : {x_min}\n'

          f'Q2(Median)              : {Q2}\n'

          f'{x.capitalize()}(max)   : {x_max}\n'

          f'{x.capitalize()}(mean)  : {round(x_mean)}')



    fig = plt.figure(figsize=(15, 10))

    plt.subplots_adjust(hspace = 0.6)

    sns.set_palette('deep')

    

    plt.subplot(221)

    ax1 = sns.distplot(data[x], color = 'r')

    plt.title(f'{x.capitalize()} Density Distribution')

    

    plt.subplot(222)

    ax2 = sns.violinplot(x = data[x], palette = 'Accent', split = True)

    plt.title(f'{x.capitalize()} Violinplot')

    

    plt.subplot(223)

    ax2 = sns.boxplot(x=data[x], palette = 'cool', width=0.7, linewidth=0.6)

    plt.title(f'{x.capitalize()} Boxplot')

    

    plt.subplot(224)

    ax3 = sns.kdeplot(data[x], cumulative=True)

    plt.title(f'{x.capitalize()} Cumulative Density Distribution')

    

    plt.show()
summary('Openness')
summary('Conscientiousness')
summary('Extraversion')
summary('Agreeableness')
summary('Neuroticism')
summary('wordcount')
# Create a function that returns a Pie chart and a Bar Graph for the categorical variables:

def cat_view(x = 'Education'):

    """

    Function to create a Bar chart and a Pie chart for categorical variables.

    """

   

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))

    

     

    """

    Draw a Pie Chart on first subplot.

    """    

    s = data.groupby(x).size()



    mydata_values = s.values.tolist()

    mydata_index = s.index.tolist()

    

    explode = []

    

    for i in range(len(mydata_index)):

        explode.append(0.1)

    

    def func(pct, allvals):

        absolute = int(pct/100.*np.sum(allvals))

        return "{:.1f}%\n({:d})".format(pct, absolute)





    wedges, texts, autotexts = ax.pie(mydata_values, autopct=lambda pct: func(pct, mydata_values),explode=explode,

                                      textprops=dict(color="w", size=8))



    ax.legend(wedges, mydata_index,

              title=f'{x.capitalize()} Index',

              loc="center left",

              bbox_to_anchor=(1, 0, 0.5, 1))



    plt.setp(autotexts, size=12, weight="bold")



    ax.set_title(f'{x.capitalize()} Piechart', fontsize=16)

       



    fig.tight_layout()

    plt.show()
cat_view('Category')
# Creating a seperate dataset with 5 personality traits & word-count

data_sub = data[['Openness','Conscientiousness','Extraversion', 'Agreeableness', 'Neuroticism', 'wordcount']]



corr = data_sub.corr()

plt.figure(figsize=(8, 8))

g = sns.heatmap(corr, annot=True, cmap = 'PuBuGn_r', square=True, linewidth=1, cbar_kws={'fraction' : 0.02})

g.set_yticklabels(g.get_yticklabels(), rotation=0, horizontalalignment='right')

g.set_title("Correlation between each of Personality Traits & Word Count", fontsize=14)

plt.show()
# Creating a function to find the above average & below average 

def FindUsers(x):

    x_low = data[x].quantile(0.25)

    x_high = data[x].quantile(0.75)

    

    xx_df = data[['User', x]]

    

    xx_df_high = xx_df[xx_df[x] >= x_high]   # Finding High Value Users

    xx_df_low = xx_df[xx_df[x] <= x_low]     # Finding High Value Users

    

    print(f'Users with High {x.capitalize()}:\n')

    print(' , '.join(xx_df_high['User']))

    print()

    print(f'Users with Low {x.capitalize()}:\n')

    print(' , '.join(xx_df_low['User']))

    

    
FindUsers('Openness')
FindUsers('Conscientiousness')
FindUsers('Extraversion')
FindUsers('Agreeableness')
FindUsers('Neuroticism')
FindUsers('wordcount')