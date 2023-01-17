import pandas as pd

import numpy as np

pd.options.display.latex.repr=True
!ls ../input/instacart-market-basket-analysis/
file_path = '../input/instacart-market-basket-analysis/'



load_data_dtype = {

    'order_id': np.uint32,

    'user_id': np.uint32,

    'eval_set': 'category',

    'order_number': np.uint8,

    'order_dow': np.uint8,

    'order_hour_of_day': np.uint8,

    # pandas 'gotcha'; leave as float:

    'days_since_prior_order': np.float16,

    'product_id': np.uint16,

    'add_to_cart_order': np.uint8,

    'reordered': np.bool

}



df_aisles = pd.read_csv(file_path + 'aisles.csv')

df_departments = pd.read_csv(file_path + 'departments.csv')

df_products = pd.read_csv(file_path + 'products.csv')



# Specify dtype to reduce memory utilization

df_order_products_prior = pd.read_csv(file_path + 'order_products__prior.csv',

                                      dtype=load_data_dtype)

df_order_products_train = pd.read_csv(file_path + 'order_products__train.csv',

                                      dtype=load_data_dtype)

df_orders = pd.read_csv(file_path + 'orders.csv', dtype=load_data_dtype)



# df_prior = full products from all prior orders

df_prior = pd.merge(df_orders[df_orders['eval_set'] == 'prior'],

                    df_order_products_prior,

                    on='order_id')



# # Useful DataFrame for aisle and department feature construction

# df_ad = pd.merge(df_prior, df_products, how='left',

#                  on='product_id').drop('product_name', axis=1)
# Mapping type between product_id and product_name

s_products = (df_products[['product_id', 'product_name']]

              .set_index('product_id')

              .iloc[:,0])
!ls ../input/instacart-top-n-random-forest-model/
file_path = '../input/instacart-top-n-random-forest-model/'
# Build dictionary of model predictions of form

# pred_dict[model_group][model_variant][ds]



import glob

from collections import defaultdict



mydict = lambda: defaultdict(mydict)

pred_dict = mydict()



# Rightmost split after '/' to remove path and

# remove last four characters '.csv'.

# Split each name on '-' into a list of three terms

# [model_group, model_variant, ds]

filenames = [path.rsplit('/')[-1][:-4].split('-')

             for path in glob.glob(file_path + '*.csv')]

# [Alternatively, transpose as array for clarity]

filenames_transposed = [list(phrase) for phrase in zip(*filenames)]



for term in zip(*filenames_transposed):

    pred_dict[term[0]][term[1]][term[2]] = pd.read_csv(file_path

                           + term[0] + '-'

                           + term[1] + '-' 

                           + term[2] + '.csv', 

                            index_col=['user_id', 'product_id'])
!ls ../input/instacart-feature-engineering/
# Open X for user lists and y for true orders



store = pd.HDFStore('../input/instacart-feature-engineering/io.h5', 'r')

store.open()



dsets = ['train', 'test', 'kaggle']



X = {ds : store['/X/' + str(ds)] for ds in dsets}

y = {ds : store['/y/' + str(ds)] for ds in dsets}



store.close()
# List of users in each dset

users = {ds : list(X[ds].index.get_level_values('user_id').drop_duplicates()) for ds in dsets}
# Split DataFrames

prior = dict.fromkeys(dsets)

orders = dict.fromkeys(dsets)

orders_full = dict.fromkeys(dsets)



# ad = dict.fromkeys(dsets)



for ds in dsets:

    prior[ds] = df_prior[df_prior['user_id'].isin(users[ds])]

    orders[ds] = df_orders[df_orders['user_id'].isin(users[ds])

                           & (df_orders.eval_set == 'prior')]

    orders_full[ds] = df_orders[df_orders['user_id'].isin(users[ds])]

#     ad[ds] = df_ad[df_ad['user_id'].isin(users[ds])]
del X
def prior_user_orders(user, ds='test'):

    """ Given prior DataFrame and user_id (in 'test'), return DataFrame

    with prior orders (descending order)."""



    prior_user_orders_df = (prior['test'][prior['test']['user_id'] == user]

                            [['order_number', 'add_to_cart_order', 'product_id']]

                            .pivot(index='add_to_cart_order',

                                   columns='order_number',

                                   values='product_id')

                            .fillna(0)

                            .astype('int64'))

    

    return prior_user_orders_df[prior_user_orders_df.columns[-1::-1]]
def true_user_order(user, ds='test'):

    """ Given user_id (in 'test'), return DataFrame of 

    'product_id's of user's true order."""

    

    true_user_order_df = (y[ds]

            .loc[user][y[ds].loc[user][0]]

            .reset_index()

            .drop(0, axis=1))

    

    true_user_order_df.index += 1

    

    return true_user_order_df
def pred_user_order(user, y_pred, ds='test'):

    """ Given user_id (in 'test'), return DataFrame of 

    'product_id's of user's predicted order."""

    

    pred_user_order_df = (y_pred

            .loc[user][y_pred.loc[user]]

            .dropna()              

            .reset_index()

            .drop('0', axis=1))

    

    pred_user_order_df.index += 1

    

    return pred_user_order_df
def combine_prior_ultimate(pred, true, prior):

    """Concatenate and transpose true, predict, and prior

    product order DataFrames."""

    

    combined_df = (pd.concat([pred, true, prior],

                             axis=1)

                   .fillna('')

                   .transpose())



    combined_df.index.name = prior.columns.name

    combined_df.columns.name = prior.index.name

    

    return combined_df
# Colors in order [TP, FP, FN, TN, empty]

default_colors = ['lawngreen',

                 'gold',

                 'deepskyblue',

                 'lightgrey',

                 'none']



def confusion_colors(combined_df,

                     css_attr='',

                     colors=default_colors):

    """Given a DataFrame as returned by combine_prior_ultimate,

    return a color mask for products in true, prediction, 

    overlap, neither, and empty with css style."""

    

    empty_mask = (combined_df.mask(combined_df == '', True) == True)

    

    true_products = list(filter(bool, list(combined_df.loc['true'])))

    true_mask = combined_df.isin(true_products)

    

    pred_products = list(filter(bool, list(combined_df.loc['prediction'])))    

    pred_mask = combined_df.isin(pred_products)    

    

    both_mask = true_mask & pred_mask

    neither_mask = ~ (true_mask | pred_mask) 

    

    color_mask = (both_mask * (css_attr + colors[0])

                  + (pred_mask & (~true_mask)) * (css_attr + colors[1])

                  + (true_mask & (~pred_mask)) * (css_attr + colors[2])

                  + (neither_mask & (~empty_mask)) * (css_attr + colors[3]) 

                  + (empty_mask) * (css_attr + colors[4]))

    

    return color_mask
def build_orders(user, y_pred, ds='test'):

    """For user and y_pred vector, convert true, predicted,

    and prior order product_ids to product_names and concatenate

    into a single DataFrame."""



    # Convert true product_ids to product_names and store as df

    true_user_order_df = pd.DataFrame(true_user_order(user, ds=ds)

                                      .iloc[:,0] #pd 0.24: .to_series

                                      .map(s_products)

                                      .rename('true'))



    # Convert pred product_ids to product_names and store as df

    pred_user_order_df = pd.DataFrame(pred_user_order(user, y_pred, ds=ds)

                                      .iloc[:,0] #pd 0.24: .to_series

                                      .map(s_products)

                                      .rename('prediction'))



    # Convert prior product_ids to product_names and store as df

    prior_user_orders_df = prior_user_orders(user, ds=ds)



    for col in prior_user_orders_df:

        prior_user_orders_df[col] = prior_user_orders_df[col].map(s_products)



    # Combine pred, true, prior

    combined_df = combine_prior_ultimate(

        pred_user_order_df,

        true_user_order_df,

        prior_user_orders_df)

    

    return combined_df
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

import matplotlib.ticker as ticker



# from matplotlib import rc

# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

# ## for Palatino and other serif fonts use:

# #rc('font',**{'family':'serif','serif':['Palatino']})

# rc('text', usetex=True)



def plot_orders(ax, user, model, ds='test',

                user_title=True, model_title=True,

                colors=default_colors):

    """Return Axes for plot of a user's order history with predictions

    and true ultimate orders as the first two rows.

    

    Parameters

    ----------

    ax : Axes

        The axes on which to draw



    user : int

        user_id

    

    model : list

        of the form [model_group, model_variant]



    ds : string

        the dset of user_id

    

    user_title : bool

        Display user_id in title

                

    model_title : bool

        Display model_variant in title

        

    colors : list of HTML color strings ordered as

        [TP, FP, FN, TN, empty]

                

    """



    y_pred = pred_dict[model[0]][model[1]][ds]

    

    # Prepare (n, m, 3) array of rgb values

    # from (n, m) array of color strings output from

    # confusion_colors(build_orders())

    color_str_to_rgb = np.vectorize(mcolors.to_rgba)

    rgb_array = np.array(

        color_str_to_rgb(

            confusion_colors(build_orders(user, y_pred))

            .values)

    ).transpose(1, 2, 0)

    

    # Plot rgb_array with various options    

    ax.imshow(rgb_array, aspect='equal')

    ax.autoscale_on=True

    

    # Make title

    title = ''

    if user_title==True:

        title += 'user_id = ' + str(user) + '\n'

    if model_title==True:

        title += model[0] + '-' + model[1]

    ax.set_title(title)



    # Make bounding boxes for true and predicted ultimate order

    ax.axhline(y=-0.5, color='k', linewidth=4)

    ax.axhspan(ymin=-0.5, ymax=0.5, fill=False, linewidth=2)

    ax.axhspan(ymin=0.5, ymax=1.5, fill=False, linewidth=2)



    # Remove ticks and spines

    ax.set_xticks([])

    ax.set_yticks([])



    for key in ax.spines.keys():

        ax.spines[key].set_visible(False)

    

    return ax
print(list(pred_dict.keys()))
print(list(pred_dict['N_threshold'].keys()))
# Select a few models as a list of tuples of the

# forms (model_group, model_variant):



models_list = [

    ('N_threshold', 'N_basket'),

    ('N_u', 'N_u'),

    ('top_N', '8'),

    ('top_N', '20')

]
# Choose a list of user_ids (in 'test')



ds = 'test'



user_list = [users[ds][17], users[ds][18], users[ds][19]]
# Choose the 18th user (in 'test')

# (Why the 18th? Because the 17th is too damn predictable!)

# and the first model in models_list



user = user_list[1]

model_group, model_variant = models_list[0][0], models_list[0][1]
# Display with product_names

(build_orders(user, pred_dict[model_group][model_variant][ds])

 .style

 .apply(confusion_colors, axis=None, css_attr='background-color: '))
fig, ax = plt.subplots(figsize=(10,10))



ax = plot_orders(ax, user, (model_group, model_variant))
# Iterate over models for given user



fig, axs = plt.subplots(1, len(models_list),

                        sharex=True,

                        figsize=(16,4))



for model in range(len(models_list)):

    axs[model] = plot_orders(axs[model],

                             user,

                             models_list[model],

                             user_title=False)
# Iterate over models_list and user_list



import itertools



fig, axs = plt.subplots(len(models_list), len(user_list),

                        sharex=True, sharey=True,

                        figsize=(16,24))



for (row, col) in itertools.product(range(len(models_list)),

                                    range(len(user_list))):



    axs[row, col] = plot_orders(axs[row, col],

                                user_list[col],

                                models_list[row])