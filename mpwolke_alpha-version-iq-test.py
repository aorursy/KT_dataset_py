#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRmr9DLU2itRpL5VaXZFXiBSVgQDuXFDRqHDsyePcJNIiBea8jR&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

from plotly.offline import iplot

from sklearn.preprocessing import LabelEncoder

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/alphaversion-fullscale-iq-test-responses/data.csv', encoding='ISO-8859-2')

df.head()
def missing_values_table(df):

        # Total missing values

        mis_val = df.isnull().sum()

        

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Print some summary information

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        # Return the dataframe with missing information

        return mis_val_table_ren_columns
df_missing= missing_values_table(df)

df_missing
# Number of each type of column

df.dtypes.value_counts()
# Number of unique classes in each object column

df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
# Create a label encoder object

le = LabelEncoder()

le_count = 0



# Iterate through the columns

for col in df:

    if df[col].dtype == 'object':

        # If 2 or fewer unique categories

        if len(list(df[col].unique())) <= 2:

            # Train on the training data

            le.fit(df[col])

            # Transform both training and testing data

            df[col] = le.transform(df[col])

            #app_test[col] = le.transform(app_test[col])

            

            # Keep track of how many columns were label encoded

            le_count += 1

            

print('%d columns were label encoded.' % le_count)
# one-hot encoding of categorical variables

df = pd.get_dummies(df)

#app_test = pd.get_dummies(app_test)



print('Training Features shape: ', df.shape)

#print('Testing Features shape: ', app_test.shape)
ext_data = df[['VQ1s', 'testelapse', 'introelapse', 'endelapse', 'MQ6e']]

ext_data_corrs = ext_data.corr()

ext_data_corrs
# Copy the data for plotting

plot_data = ext_data.drop(columns = ['testelapse']).copy()



# Add in the age of the client in years

plot_data['introelapse'] = df['introelapse']



# Drop na values and limit to first 100000 rows

plot_data = plot_data.dropna().loc[:100000, :]



# Function to calculate correlation coefficient between two columns

def corr_func(x, y, **kwargs):

    r = np.corrcoef(x, y)[0][1]

    ax = plt.gca()

    ax.annotate("r = {:.2f}".format(r),

                xy=(.2, .8), xycoords=ax.transAxes,

                size = 20)



# Create the pairgrid object

grid = sns.PairGrid(data = plot_data, size = 3, diag_sharey=False,

                    hue = 'VQ1s', 

                    vars = [x for x in list(plot_data.columns) if x != 'VQ1s'])



# Upper is a scatter plot

grid.map_upper(plt.scatter, alpha = 0.2)



# Diagonal is a histogram

grid.map_diag(sns.kdeplot)



# Bottom is density plot

grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r);



plt.suptitle('VQ1s & introelapse Features Pairs Plot', size = 32, y = 1.05);
import missingno as msno

#msno.bar(df)
#msno.matrix(df)
#msno.heatmap(df)
#msno.dendrogram(df)
df.isnull().sum()
#df_1 = df.copy()

#df_1['VQ2a'].mean() #pandas skips the missing values and calculates mean of the remaining values.
# imputing with a constant



from sklearn.impute import SimpleImputer

df_constant = df.copy()

#setting strategy to 'constant' 

mean_imputer = SimpleImputer(strategy='constant') # imputing using constant value

df_constant.iloc[:,:] = mean_imputer.fit_transform(df_constant)

df_constant.isnull().sum()
from sklearn.impute import SimpleImputer

df_most_frequent = df.copy()

#setting strategy to 'mean' to impute by the mean

mean_imputer = SimpleImputer(strategy='most_frequent')# strategy can also be mean or median 

df_most_frequent.iloc[:,:] = mean_imputer.fit_transform(df_most_frequent)
df_most_frequent.isnull().sum()
df_knn = df.copy(deep=True)
from sklearn.impute import KNNImputer

df_knn = df.copy(deep=True)



knn_imputer = KNNImputer(n_neighbors=2, weights="uniform")

df_knn['RQ6a'] = knn_imputer.fit_transform(df_knn[['RQ6a']])
df_knn['RQ6a'].isnull().sum()
from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

df_mice = df.copy(deep=True)



mice_imputer = IterativeImputer()

df_mice['RQ6a'] = mice_imputer.fit_transform(df_mice[['RQ6a']])
df_mice['RQ6a'].isnull().sum()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSC45MgUFkcXqMBKdsOL0oSYBbbBHWfRub5K1R54SXHeOSumQ2O&usqp=CAU',width=400,height=400)