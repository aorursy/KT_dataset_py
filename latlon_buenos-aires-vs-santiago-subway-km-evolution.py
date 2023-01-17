# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output



lines_df = pd.read_csv("../input/lines.csv")



# Lines of the Buenos Aires "Subte" system

ba_lines = lines_df[lines_df["system_id"] == 254].id



# Lines of the Santiago "Metro" system

santiago_lines = lines_df[lines_df["system_id"] == 267].id



df = pd.read_csv("../input/tracks.csv")



ba_tracks = df[df['line_id'].isin(ba_lines)]

santiago_tracks = df[df['line_id'].isin(santiago_lines)]



data = []



for y in range(1910, 2018):

    d = {'year': y,

         'Subte Buenos Aires':ba_tracks[ba_tracks["opening"] <= y].length.sum()/1000,

         'Metro Santiago':santiago_tracks[santiago_tracks["opening"] <= y].length.sum()/1000}

    data.append(d)



dataset = pd.DataFrame(data)

dataset.set_index('year')

plot = dataset.plot(x='year')

plot.set_ylabel("km")