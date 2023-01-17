# importing necessary libraries

import numpy as np

import pandas as pd



from IPython.display import display



import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline

plt.style.use('ggplot')



import warnings

warnings.filterwarnings(action='once')
# reading supermarket_sales data

df = pd.read_csv('../input/supermarket-sales/supermarket_sales - Sheet1.csv')

print('(rows, columns):', df.shape)
# exploring data

df.head(10)
df.columns
# regenerating df with only necessary columns

df = df[['Gender', 'Product line', 'Quantity', 'Date', 'Time', 'Payment', 'gross income', 'Rating']]



# renaming columns for usability purposes

df.columns = ['gender', 'product_line', 'quantity', 'date', 'time', 'payment', 'gross_income', 'rating']



df.columns
def combine_date_time(data, date, time, drop=True, front=False):

    '''

    Takes two separate date and time columns

    and combine them into one datetime object column

    

    KEYWORD ARGUMENTS:

        # data -- DataFrame

            data source

            

        # date, time -- str

            names for date and time columns

            

        # drop -- bool, default: True

            if True, drops old date and time columns

            

        # front -- bool, default: False

            if True, brings new datetime columns to index 0

            

    OUTPUT:

        DataFrame with new datetime object column

        and old date and time columns dropped

    '''

    data['datetime'] = data[date] + ' ' + data[time]

    data.datetime = pd.to_datetime(data.datetime, infer_datetime_format=True)

    

    if drop == True:

        data.drop(columns=[date, time], inplace=True)

    

    if front == True:

        data = data.set_index('datetime').reset_index()
# converting date and time columns to one 'datetime' column

combine_date_time(df, 'date', 'time')
df.dtypes
# assessing missing values 

df.isnull().sum()
# defining a function to help in plotting graphs

def plot_grpah(figsize=(10,8), figtype=None, y=None, x=None, hue=None, labels=None, save=False):

    '''

    Plots y, x, and hue arrays using seaborn library

     

    KEYWORD ARGUMENTS:

        # figszie -- float, float, default: (10,8)

            (width, height) in inches



        # figtype -- str

            figure type, options: 'barplot', 'boxplot', and 'pointplot'

            

        # y, x, hue -- array

            inputs for plotting long-form data

            

        # labels -- list, default: None

            in order (figure_title, ylabel, xlabel, legend_title)

            

        # save -- bool, default: False

            save figure in 'png' format to main folder, uses figure's title as filename

    

    OUTPUT

        returns Axes object with the plot drawn onto it

    '''

    # plotting figure

    fig = plt.figure(figsize=figsize)

    

    if figtype == 'pointplot':

        ax = sns.pointplot(y=y, x=x, hue=hue)

    elif figtype == 'boxplot':

        ax = sns.boxplot(y=y, x=x, hue=hue)

    elif figtype == 'barplot':

        ax = sns.barplot(y=y, x=x, hue=hue, ci=False)

    else:

        print('figtype must be specified')

    

    # setting up labels

    fig = plt.title(labels[0])

    ax.set_ylabel(labels[1])

    ax.set_xlabel(labels[2])

    fig = plt.legend(title=labels[3], loc='upper left', bbox_to_anchor=(1.0, 0.5), ncol=1)

    

    # exporting and showing figure

    if save == True:

        plt.savefig(fname=labels[0].replace(" ", "_").lower(), dpi=72, bbox_inches='tight')       

    

    plt.show()
plot_grpah(

    figtype='pointplot', y=(df.gross_income/df.quantity), x=df.quantity, hue=df.gender, save=True,

    labels=['Gross Income by Gender in Terms of Quantity', 'Gross Income Per Item ($)', 'Quantity', 'Gender']

)
plot_grpah(

    figtype='boxplot', y=df.rating, x=df.payment, hue=df.gender, save=True,

    labels=['Customer Satisfaction by Gender in Terms of Payment Method', 'Rating', 'Payment Method', 'Gender']

)
plot_grpah(

    figsize=(17,8), figtype='barplot', y=(df.gross_income/df.quantity), x=df.datetime.dt.weekday_name, hue=df.product_line,

    labels=['Product Line Gross Income by Day of The Week', 'Gross Income Per Item ($)', 'Day of The Week', 'Product Line'],

    save=True

)