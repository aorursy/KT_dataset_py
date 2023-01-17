%matplotlib inline



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
df = pd.read_csv("../input/tensorflow-speech-recognition-vae-latent-variables/autoencoder-results.csv")



# add some columns



# train or test

df['sample'] = df['fname'].str.split('/').str[0]



# labels for train data samples

df['label'] = None

df.loc[df['sample'] == 'train', 'label'] = df['fname'].str.split('/').str[2]



# whether this was part of the classes this was trained

# on (which is equivalent whether this was one of the

# classes to be predicted other than 'unknown' and 'silence')

labels_to_predict = 'yes no up down left right on off stop go silence unknown'.split()



df['label_to_predict'] = df['label'].isin(labels_to_predict)
df.head()
df.columns.tolist()
mu_names    = sorted([ col for col in df.columns if col.startswith('mu')])

sigma_names = sorted([ col for col in df.columns if col.startswith('sigma')])
def plot_latent_variables(varnames, groups, normed = False, log = True):

    

    plt.figure(figsize = (15,15))



    # find range for common binning of histograms

    bins = np.linspace(df[varnames].min().min(), 

                     df[varnames].max().max(),

                 101)



    for index, colname in enumerate(varnames):

        plt.subplot(4,3, index + 1)



        for group in groups:

            

            rows = group['selector'](df)

        

            plt.hist(df[rows][colname], 

                     bins = bins, 

                     alpha = 0.3, 

                     log = log, 

                     color = group.get('color', None),

                     histtype = 'stepfilled',

                     label    = group.get('label', None),

                     normed   = normed

                    )





        plt.title(colname)

        plt.legend()

        plt.grid()
# filters for plotting train vs. test sample distributions

train_test_groups = [

    dict(selector = lambda df: df['sample'] == 'test', label = 'test', color = 'red'),

    dict(selector = lambda df: df['sample'] == 'train', label = 'train', color = 'blue'),

                      ]
# filters for plotting train label to predict ('core') vs. train other label

core_other_label_groups = [

    dict(selector = lambda df: (df['sample'] == 'train') & (df['label_to_predict']), label = 'train core', color = 'red'),

    dict(selector = lambda df: (df['sample'] == 'train') & (~ df['label_to_predict']), label = 'train other', color = 'blue'),

                      ]
plot_latent_variables(mu_names, 

                      core_other_label_groups,

                      True)
plot_latent_variables(sigma_names, 

                      core_other_label_groups,

                      True)
plot_latent_variables(mu_names, 

                      train_test_groups,

                      True)
plot_latent_variables(sigma_names, 

                      train_test_groups,

                      True)