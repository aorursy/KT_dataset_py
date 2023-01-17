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
train_df = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')

test_df = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')
train_df.describe()
import plotnine as p9 # create plots



train_melt_df = pd.melt(train_df,

                       id_vars=["time"],

                       value_vars=["signal", "open_channels"])



# Create plot

time_train_plot = p9.ggplot(data=train_melt_df,

                         mapping=p9.aes(x='time', 

                                        y='value',

                                       color='variable'))



# Draw the plots using a line graph

time_train_plot = time_train_plot + p9.geom_line()



# Create subplots

time_train_plot = time_train_plot + p9.facet_wrap(['variable'], 2)



# get figure size

time_train_plot + p9.theme(

    figure_size=(20, 18),

    aspect_ratio=1/10)
train_df.loc[:,['signal','open_channels']].corr(method="spearman")
import plotnine as p9 # create plots



# Create plot

time_test_plot = p9.ggplot(data=test_df,

                         mapping=p9.aes(x='time', 

                                        y='signal'))



# Draw the plots using a line graph

time_test_plot = time_test_plot + p9.geom_line()



# get figure size

time_test_plot + p9.theme(

    figure_size=(20, 18),

    aspect_ratio=1/10)