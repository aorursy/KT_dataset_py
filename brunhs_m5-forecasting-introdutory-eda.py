#Data manipulation
import pandas as pd
import numpy as np

#Data visualization
import matplotlib.pylab as plt
import seaborn as sns
from itertools import cycle

pd.set_option('max_columns', 50)
# Read in the data
INPUT_DIR = '../input/m5-forecasting-accuracy'
calendar = pd.read_csv(f'{INPUT_DIR}/calendar.csv')
sales_train_validation = pd.read_csv(f'{INPUT_DIR}/sales_train_validation.csv')
sample_sub = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')
sell_prices = pd.read_csv(f'{INPUT_DIR}/sell_prices.csv')
# Printing shapes

print('The sell prices size is:',sell_prices.shape)
print('The calendar size is:',calendar.shape)
print('The sales_train_validation size is:',sales_train_validation.shape)
# Head of the data

sell_prices.head()
calendar.head()
sales_train_validation.head()
# fig, ax = plt.subplots()
# ax.pie(pd.DataFrame(sales_train_validation.groupby('cat_id').id.count()).reset_index(drop=True))



# First, let's gather all time data
time = [column for column in sales_train_validation.columns if 'd_' in column]

# Lets plot everything
sns.set(rc={'figure.figsize':(22.7,12.27)})
sns.set_style('whitegrid')
sns.set_context('talk')
plt.xticks(np.arange(min(calendar['date'].index), max(calendar['date'].index)+1, 150.0))
sns.lineplot(data = pd.concat([pd.DataFrame(sales_train_validation.loc[sales_train_validation['id'] == 'HOBBIES_1_004_CA_1_validation',
                                time].T.reset_index()).rename(columns={3: 'sales'}), calendar['date']], axis=1),
             x='date', y='sales').set(title='Time series from id HOBBIES_1_004_CA_1_validation')
plt.xticks(rotation=45)
categories = pd.DataFrame(sales_train_validation.groupby('cat_id')[time].sum().T.reset_index()).columns[1:]

for i in categories:
    # Lets plot everything
    sns.set(rc={'figure.figsize':(22.7,12.27)})
    sns.set_style('whitegrid')
    sns.set_context('talk')
    plt.xticks(np.arange(min(calendar['date'].index), max(calendar['date'].index)+1, 150.0))
    sns.lineplot(data = pd.concat([pd.DataFrame(sales_train_validation.groupby('cat_id')[time].sum().T.reset_index()), calendar['date']], axis=1),
                 x='date', y=i).set(title='Categories time series')
    plt.xticks(rotation=45)
    plt.legend(labels=categories.values)
#Let's take a look at the same graphic but separated by states!

categories = pd.DataFrame(sales_train_validation.groupby(['cat_id', 'state_id'])[time].sum().T.reset_index()).columns[1:]

for i in range(len(categories)):
    # Lets plot everything
    sns.set(rc={'figure.figsize':(22.7,16.27)})
    sns.set_style('whitegrid')
    sns.set_context('talk')
    plt.xticks(np.arange(min(calendar['date'].index), max(calendar['date'].index)+1, 150.0))
    sns.lineplot(data = pd.concat([pd.DataFrame(sales_train_validation.groupby(['cat_id', 'state_id'])[time].sum().T.reset_index()), calendar['date']], axis=1),
                 x='date', y=categories[i]).set(title='Categories time series')
    plt.xticks(rotation=45)
    plt.legend(labels=categories.values)

# #Can we see how stores are performing in each state?
# #Let's take a look at the same graphic but separated by states!

# categories = pd.DataFrame(sales_train_validation.groupby(['cat_id'])[time].sum().T.reset_index()).columns[1:]
# states = pd.DataFrame(sales_train_validation.groupby(['state_id'])[time].sum().T.reset_index()).columns[1:]
# stores = pd.DataFrame(sales_train_validation.groupby(['store_id'])[time].sum().T.reset_index()).columns[1:]


# for i in range(len(categories)):
#     # Lets plot everything
#     sns.set(rc={'figure.figsize':(22.7,16.27)})
#     sns.set_style('whitegrid')
#     sns.set_context('talk')
#     plt.subplot(3,1,)
#     plt.xticks(np.arange(min(calendar['date'].index), max(calendar['date'].index)+1, 150.0))
#     sns.lineplot(data = pd.concat([pd.DataFrame(sales_train_validation.groupby(['cat_id', 'state_id'])[time].sum().T.reset_index()), calendar['date']], axis=1),
#                  x='date', y=categories[i]).set(title='Categories time series')
#     plt.xticks(rotation=45)
#     plt.legend(labels=categories.values)
