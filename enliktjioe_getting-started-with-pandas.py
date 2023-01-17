# Remove future warning message

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

data = pd.read_csv("../input/abalone.csv")

data
# TODO - here you have to write the code how did you find the answer.

data.memory_usage(deep=True).sum()
# TODO

data.columns
# TODO

data.index
# TODO

data.head()
# TODO

data['Weight'].tail(3)
# TODO

data.loc[576]['Diameter']
# TODO

data['Height'].describe()
# TODO

data[(data['Gender'] == "M") & (data['Weight'] < 0.75)].describe()
# TODO

data[data['Rings'] == 18].describe()
import numpy as np

import pandas as pd



# TODO

data = pd.read_table('../input/adult.csv', index_col='X')

data
# TODO

data.columns
# Matplotlib is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats 

# and interactive environments across platforms. Matplotlib can be used in Python scripts, the Python and IPython shells, 

# the Jupyter notebook, web application servers, and four graphical user interface toolkits.

import matplotlib.pyplot as plt



# allows to output plots in the notebook

%matplotlib inline 



# Set the default style

plt.style.use("ggplot") 
# Columns may contain non-numeric values, errors or missing values. Therefore, non-numeric values must be dealt with.

if data is not ...:

    pd.to_numeric(data['age'], errors='coerce').hist(bins=30) # ‘coerce’ -> invalid parsing will be set as NaN

else:

    print('Please define `data` in earlier subtasks')
# TODO 

import seaborn as sns



sns.distplot(data['salaries'], hist=True, kde=True, 

             bins=20, color = 'darkblue', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 2})
#TODO

total_rows = len(data.index)

total_missing = 0

if data is not data.empty:

    for column in data.columns:

        total_missing = len(data[data[column] == '?'])

        if total_missing > 0:

            print (f"Percentage of missing values in attribute {column} = {round((total_missing/total_rows)*100,2)}%")

else:

    print('Please define `data` in earlier subtasks')
if data is not data.empty:

    original_data = data.copy(deep=True) # Make a deep copy, including a copy of the data and the indices
# TODO

for column in data.columns:

    if data[column].dtype == 'O' : #dtype == 'O' stands for Python object, to check if data type is string or not

        data[column] = data[column].str.strip()

    else:

        print(f"column {column} is not string")
# Count differences

if data is not data.empty:

    data_all = pd.concat([original_data, data]).drop_duplicates()

    diff = data_all.shape[0] - data.shape[0]

    print ('Difference: ' + str(diff))
# TODO

data_clean = data.replace(['?','privat','UnitedStates', 'Unitedstates', 'Hong'], [np.nan,'Private','United-States', 'United-States', 'Hong-Kong'])

data_clean.head(10)
# TODO

if data_clean is not data_clean.empty:

    data_all = pd.concat([data, data_clean]).drop_duplicates()

    diff = data_all.shape[0] - data_clean.shape[0]

    print ('Difference: ' + str(diff))
# TODO

data['education'] = data['education'].astype('category')



data['education'].value_counts(normalize = True).reindex(['Doctorate', 'Masters', 

                                          'Bachelors', 'Some-college', 'Prof-school', 'Assoc-voc', 'Assoc-acdm',

                                          'HS-grad', '12th',

                                          '11th', '10th',

                                          '9th', '7th-8th',

                                          '5th-6th', '1st-4th',

                                          'Preschool'

                                         ]).plot(kind="bar")
# TODO

crosstab_df = pd.crosstab(data.occupation, data.education, values = data.salaries, aggfunc=np.mean).round(2)

crosstab_df
if crosstab_df is not crosstab_df.empty:

    for index, row in crosstab_df.iterrows():

        row.plot(kind = 'bar') # TODO 

        plt.show()