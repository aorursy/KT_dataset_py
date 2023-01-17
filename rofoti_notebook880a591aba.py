#******************************************************************************

# Importing packages

#******************************************************************************

#-----------------------------

# Standard libraries

#-----------------------------

import os

import sys

import random

import numpy as np

import pandas as pd

import scipy

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import skew
class DataFrameDescriber():

    '''

    '''

    def __init__(self, df=None):

        if not isinstance(df, pd.DataFrame):

            pass

        else:

            self.perc_miss = self.get_percent_missing(df);

            self.miss_loc = self.get_miss_loc(df)

            self.skewed_feats_sr = self.get_skew(df)

        #end

        return

    #end



    def log_or_print(self, message, logger=None):

        if not logger:

            print(message)

            sys.stdout.flush()

        else:

            logger.info(message)

        #end

        return

    #end



    def get_skew(self, df):

        numeric_feat_ls = list(df.dtypes[df.dtypes != "object"].index)

        skewed_feats_sr = df[numeric_feat_ls].apply(lambda x: skew(x.dropna()))

        return skewed_feats_sr

    #end



    def get_miss_loc(self, df):

        return pd.isnull(df).any(1).nonzero()[0] #create an array with the index of the rows where missing data are present

    #end



    def get_percent_missing(self, df):

        inds = self.get_miss_loc(df) #retrieve the index of rows containing missing values

        return 1.0*len(inds)/len(df) #return percent of rows containing missing values

    #end



    def describe(self, df, logger=None):

        self.perc_miss = self.get_percent_missing(df);

        self.miss_loc = self.get_miss_loc(df)

        self.skewed_feats_sr = self.get_skew(df)

        self.log_or_print('Numeric column description:')

        self.log_or_print(df.describe(), logger) #describe the dataset

        self.log_or_print('Percent of rows containing missing data: %.1f' %self.perc_miss, logger)

        self.log_or_print('Description of the DataFrame by filtering out rows that DO NOT contain missing data:', logger)

        self.log_or_print(df.iloc[self.get_miss_loc(df)].describe())

        self.log_or_print('Skewness of numerci features:')

        self.log_or_print(self.get_skew(df), logger)

        self.log_or_print('To retrieve percent missing: DataFrameDescriber(df).perc_miss', logger)

        self.log_or_print('To retrieve index of rows with missing values: DataFrameDescriber(df).miss_loc', logger)

        self.log_or_print('To retrieve skewness of numeric feats: DataFrameDescriber(df).get_skew', logger)



        return

    #end



#end



class EDAPlotter():

    '''

    '''

    def __init__(self):

        pass

    #end



    def heatmap_corr(self, df, sample=10000, savefig=False):

        '''

        Creates a heatmap of correlation from a dataframe

        '''

        sample = min(len(df), sample)

        plt.figure() #initialize the figure

        corrmat = df.sample(sample).corr() # build the matrix of correlation from the dataframe after sampling

        f, ax = plt.subplots(figsize=(12, 9))   # set up the matplotlib figure

        sns.heatmap(corrmat, vmax=1.0, vmin=-1.0, square=True); # draw the heatmap using seaborn

        if savefig:

            fig.savefig('./heatmap_corr.png')

        #end

        return

    #end



    def scatter_corr(self, df, response_label, sample=10000, 

                     display=4, savefig=False):

        sample = min(len(df), sample)

        num_df = df[list(df.dtypes[df.dtypes != "object"].index)].sample(sample) #extract numerical features and sample down

        n_feats = num_df.shape[1]

        n_plots = np.floor(n_feats / display) + 1

        for iplot in range(int(n_plots)):

            fig = plt.figure(figsize=(16,16))

            locs = np.hstack(([0], range(iplot*display+1, (iplot+1)*display)))

            sns.pairplot(num_df.iloc[:,locs], diag_kind='kde', 

                         hue=response_label, palette='Set1');

            if savefig:

                fig.savefig('./scatter_corr_' + str(iplot) + '.png')

            #end

        #end

        return

    #end



    def label_regression_stack(self, df, response_label, sample=10000, 

                               display=4, savefig=False):

        '''

        '''

        sample = min(len(df), sample)

        plt.figure()

        num_feat_ls = list(df.dtypes[df.dtypes != "object"].index)

        num_df = df[num_feat_ls].sample(sample)

        n_feats = num_df.shape[1]

        n_plots = np.floor(n_feats / display) + 1

        for iplot in range(int(n_plots)):

            fig = plt.figure()

            subplot_ls = range(iplot*display, min(n_feats, (iplot+1)*display))

            sns.pairplot(data=num_df, 

                         x_vars=[num_feat_ls[el] for el in subplot_ls], 

                         y_vars=response_label);

            if savefig:

                fig.savefig('./scatter_reg_' + str(iplot) + '.png')

            #end

        #end

        return

    #end



#end
md_df = pd.read_csv('../input/menu.csv')

md_df.head(3)
DataFrameDescriber().describe(md_df)
miss_percent = DataFrameDescriber(md_df).perc_miss

ind_miss = DataFrameDescriber(md_df).miss_loc

skew = DataFrameDescriber(md_df).get_skew
EDAPlotter().heatmap_corr(md_df)
EDAPlotter().label_regression_stack(md_df, 'Calories')