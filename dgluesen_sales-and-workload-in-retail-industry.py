# Array operations and useful analysis functionalities

import numpy as np

import pandas as pd



# Seaborn library for visualizations in the notebook

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("ticks")

%matplotlib inline





# Interactive widgets for user-selection

# This functionality can't be executed in every notebook viewer!

import ipywidgets as widgets

from ipywidgets import interact, interact_manual



# Scikit-learn for model building

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
df_goods = pd.read_excel('/kaggle/input/sales-and-workload-data-from-retail-industry/salesworkload.xlsx',            # file name

                         sheet_name = 0,                                                                             # index of sheet in file

                         header = 1,                                                                                 # row index of column names

                         na_values = ['#NV', 'Err:520'])                                                             # special treatment of missing-/error-values

df_hours = pd.read_excel('/kaggle/input/sales-and-workload-data-from-retail-industry/salesworkload.xlsx',

                         sheet_name = 1,

                         header = 5,

                         na_values = ['#NV', 'Err:520'])
print("Goods:")

print(df_goods.dtypes)

print(" ")

print("Hours:")

print(df_hours.dtypes)
goods_notnum = df_goods[['HoursOwn']].applymap(np.isreal).values

df_goods[~goods_notnum][['Dept_ID', 'HoursOwn']]
# Calculate the mean of HoursOwn for the specified department and without the cells containing a question mark

imp_dept6 = df_goods.loc[(df_goods['Dept_ID'] == 6.0) & (df_goods['HoursOwn'] != '?')]['HoursOwn'].mean()

imp_dept2 = df_goods.loc[(df_goods['Dept_ID'] == 2.0) & (df_goods['HoursOwn'] != '?')]['HoursOwn'].mean()



# Replace the question marks

df_goods.loc[(df_goods['Dept_ID'] == 6.0) & (df_goods['HoursOwn'] == '?'), ['HoursOwn']] = imp_dept6

df_goods.loc[(df_goods['Dept_ID'] == 2.0) & (df_goods['HoursOwn'] == '?'), ['HoursOwn']] = imp_dept2



# Give an information about the new values

print('Department 6: ' + str(imp_dept6))

print('Department 2: ' + str(imp_dept2))
df_goods['HoursOwn'] = pd.to_numeric(df_goods['HoursOwn'])   # convert column now to numeric

df_goods.dtypes                                              # check type of column
df_goods.describe(include = 'all')
df_goods[df_goods['Area (m2)'].isnull()]
df_goods = df_goods[df_goods['Area (m2)'].notnull()]   # drop all above shown rows with no/incomplete content

df_goods = df_goods.drop(['Customer'], axis = 1)       # drop "Customer" column, due to no values
# Subset the DataFrame as of May 2017

df_area_buffer = df_goods[df_goods['MonthYear'] == '05.2017']

# Extracting row indices of the subset

buffer_row_indices = df_goods[df_goods['MonthYear'] == '05.2017'].index



# Creating new DataFrame based on the indices

df_area = df_goods.loc[buffer_row_indices, :]



# Adding new column with identifier for store and department

df_area['ID'] = df_area['StoreID'].map(str) + df_area['Dept_ID'].map(str)

df_area = df_area[['ID', 'Area (m2)']]
df_goods['ID'] = df_goods['StoreID'].map(str) + df_goods['Dept_ID'].map(str)



df_cum = df_goods[['ID',

                   'HoursOwn',

                   'Sales units',

                   'Turnover']].groupby(['ID'], as_index = False).sum()



df_cum = pd.merge(df_cum, df_area,

                  on = 'ID',

                  how = 'left')



df_cum = pd.merge(df_cum, df_goods[['ID', 'StoreID', 'Dept_ID', 'Dept. Name']],

                  on = 'ID',

                  how = 'left').drop_duplicates()
df_hours = df_hours.drop(df_hours.columns[1:24], axis = 1)

df_hours = df_hours.drop(df_hours.columns[2:9], axis = 1)
df_cum = df_cum.rename(columns = {'Dept_ID': 'DeptID',

                                  'Dept. Name': 'DeptName',

                                  'Sales units': 'SalesUnits',

                                  'Area (m2)': 'Area'})
df_hours = df_hours.rename(columns = {'id': 'StoreID',

                                      '5.1': 'CumulativeHours'})
df = pd.merge(df_cum, df_hours, on = 'StoreID', how = 'left')    # Joining both tables

df['HoursRatio'] = df['HoursOwn'] / df['CumulativeHours']        # Create new ratio feature

df.head(3)                                                       # Check first data objects
df['ID'] = df['ID'].astype('category')

df['StoreID'] = df['StoreID'].astype('int').astype('category')

df['DeptID'] = df['DeptID'].astype('int').astype('category')
df.loc[:, ['DeptID', 'DeptName']].drop_duplicates().sort_values(by = ['DeptID'])
g_hoursratio = sns.FacetGrid(df,

                             col = 'DeptName',

                             col_wrap = 4,

                             sharex = False, sharey = False);

g_hoursratio.map(sns.distplot, 'HoursRatio', color = '#012363');
df['DeptID'] = df['DeptID'].cat.remove_categories([3])

df.dropna(inplace = True)
g_salesunits = sns.FacetGrid(df,

                             col = 'DeptName',

                             col_wrap = 4,

                             sharex = False, sharey = False);

g_salesunits.map(sns.distplot, 'SalesUnits', color = '#e3b900');
df['DeptID'] = df['DeptID'].cat.remove_categories([12, 15, 16, 17, 18])

df.dropna(inplace = True)
g_turnover = sns.FacetGrid(df,

                  col = 'DeptName',

                  col_wrap = 4,

                  sharex = False, sharey = False);

g_turnover.map(sns.distplot, 'Turnover', color = '#dd3e21');
%%capture

np.seterr(divide = 'ignore', invalid = 'ignore')

# The following FacetGrid throws an error (numpy: Invalid value

# encountered in true_divide), which has no effect. According to

# https://github.com/belltailjp/selective_search_py/issues/20

# the error is suppressed by this numpy command.
g_area = sns.FacetGrid(df,

                       col = 'DeptName',

                       col_wrap = 4,

                       sharex = False, sharey = False);

g_area.map(sns.distplot, 'Area', color = '#01673b');
df['DeptID'] = df['DeptID'].cat.remove_categories([11])

df.dropna(inplace = True)



# Get a sorted list of department IDs in scope for later use

final_dept_list = np.sort(df.loc[:, ['DeptName']]

                          .drop_duplicates()

                          .to_numpy()

                          .ravel())
cpal = ['#012363', '#e3b900', '#dd3e21', '#01673b', '#a6acaa', '#b01297', '#7EB00B', '#17E8DB', '#332506', '#701B0C']

pp = sns.pairplot(df,

                  hue = 'DeptName',

                  vars = ['HoursRatio',

                          'SalesUnits',

                          'Turnover',

                          'Area'],

                  palette = cpal)

pp.map_upper(sns.scatterplot)

pp.map_lower(sns.kdeplot)

pp.map_diag(sns.distplot)

pp._legend.remove()

handles = pp._legend_data.values()

labels = pp._legend_data.keys()

pp.fig.legend(handles = handles, labels = labels, loc = 'upper center', ncol = 5, frameon = False)

pp.fig.subplots_adjust(top = 0.9, bottom = 0.1)
df_hours = df.loc[:, ['StoreID', 'DeptID', 'HoursRatio']]

df_hours = df_hours.pivot(index = 'StoreID',

                          columns = 'DeptID',

                          values = 'HoursRatio')
df_sales = df.loc[:, ['StoreID', 'DeptID', 'SalesUnits']]

df_sales = df_sales.pivot(index = 'StoreID',

                          columns = 'DeptID',

                          values = 'SalesUnits')
df_area = df.loc[:, ['StoreID', 'DeptID', 'Area']]

df_area = df_area.pivot(index = 'StoreID',

                        columns = 'DeptID',

                        values = 'Area')
cor_ha = df_hours.corrwith(df_area, axis = 0)

cor_as = df_area.corrwith(df_sales, axis = 0)

cor_sh = df_sales.corrwith(df_hours, axis = 0)



cor = pd.concat([cor_ha, cor_as, cor_sh], axis = 1)

cor.columns = ['Hours-Area', 'Area-Sales', 'Sales-Hours']

cor = (cor.reset_index(drop=True)

       .unstack()

       .reset_index()

       .rename(columns={'level_0': 'Relation', 'level_1': 'DeptID', 0: 'PCC'}))
sns.set_context("notebook", font_scale = 1.5)

fig_boxplot = sns.boxplot(data = cor,

                          x = 'PCC', y = 'Relation',

                          width = 0.6,

                          palette = ['#99a7c0', '#f3e399', '#99c2b0'],

                          orient = 'h')

fig_boxplot.set_ylabel('')

sns.despine()

plt.xlabel(r"$\rho$")
sns.reset_orig()

g = sns.FacetGrid(df,

                  col = 'DeptName',

                  col_wrap = 3,

                  sharex = False, sharey = False);

g.map(sns.scatterplot, 'HoursRatio', 'SalesUnits', alpha = .8, color = '#012363')

g.add_legend()
def plot_lm(deptname):

    ax = sns.lmplot(data = df[df['DeptName'] == deptname],

                    x = 'HoursRatio',

                    y = 'SalesUnits')



@interact

def show_plot_lm(DeptName = final_dept_list):

    return plot_lm(DeptName)
@interact

def plot_model(deptname = final_dept_list):

    # Creating training and test set

    X_all = df[df['DeptName'] == deptname][['HoursRatio']].values

    y_all = df[df['DeptName'] == deptname][['SalesUnits']].values/1000000          # for smaller axis tick labels

    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all,

                                                                        test_size = 0.25,

                                                                        random_state = 42)

    

    # Initiating and fitting model

    lrm = LinearRegression()

    lrm.fit(X_train_all, y_train_all)

    

    # Parameters

    a = lrm.coef_[0]

    b = lrm.intercept_

    r2_training = lrm.score(X_train_all, y_train_all)

    r2_test = lrm.score(X_test_all, y_test_all)

    

    # Printing function

    print('Function: f(w) = %.3f * w + %.3f' % (a, b))

    print("Intercept: {}".format(a))

    print("Slope: {}".format(b))

    print('Training set R² score: {:.2f}'.format(r2_training))

    print('Test set R² score: {:.2f}'.format(r2_test))

    

    # Plotting

    df_concat_train = pd.DataFrame(np.hstack((X_train_all, y_train_all)))

    df_concat_train['Set'] = 'Train'

    df_concat_test = pd.DataFrame(np.hstack((X_test_all, y_test_all)))

    df_concat_test['Set'] = 'Test'

    df_concat_pred = pd.DataFrame(np.hstack((X_train_all, lrm.predict(X_train_all))))

    

    df_concat = pd.concat([df_concat_train, df_concat_test])

    

    df_concat.columns = ['X', 'y', 'Set']

    df_concat_pred.columns = ['X', 'y']

    

    ax = sns.scatterplot(data = df_concat,

                     x = 'X',

                     y = 'y',

                     hue = 'Set',

                     palette = ['#012363', '#e3b900'])

    

    sns.lineplot(data = df_concat_pred,

             x = 'X',

             y = 'y',

             color = '#dd3e21',

             ax = ax)

    

    sns.despine()

    plt.xlabel(r'Ratio of working hours $w$')

    plt.ylabel(r'Sales units $s$ [1e6]')

    

    return ax