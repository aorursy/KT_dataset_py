# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML
import math
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
from plotly import figure_factory as FF
init_notebook_mode()
VideoReview = pd.read_csv('../input/video_review.csv')
VideoFootageInjury = pd.read_csv('../input/video_footage-injury.csv')
PlayInfo = pd.read_csv('../input/play_information.csv')
VidoeFootageControl = pd.read_csv('../input/video_footage-control.csv')
RateOfHeadInjury = (len(VideoReview)/len(PlayInfo))*100
print ('Rate of Punt Play Head Injury: %.2f%%' % RateOfHeadInjury)
print (len(VideoReview))
table_data = [['Play ID', 'Unique Punt Injury', 'Injury Due to Illegal Play', 'Penalty Under New Helmet Rule'],
              ['3129', 'No', 'Yes', 'No'],
              ['1212', 'Yes', 'No', 'No'],
              ['905', 'Yes', 'No', 'No'], 
              ['2342', 'No', 'No', 'Yes'],
              ['3509', 'No', 'No', 'Yes'],
              ['3278', 'No', 'No', 'Yes'],
              ['2902', 'No', 'Yes', 'Yes'],
              ['2918', 'Yes', 'No', 'No'],
              ['3746', 'Yes', 'No', 'No'],
              ['3609', 'Yes', 'No', 'No'],
              ['2667', 'Yes', 'No', 'Yes'],
              ['3312', 'Yes', 'No', 'No'],
              ['1988', 'No', 'No', 'Yes'],
              ['1407', 'Yes', 'No', 'No'],
              ['733', 'No', 'Yes', 'Yes'],
              ['2208', 'No', 'No', 'Yes'],
              ['2792', 'No', 'No', 'Yes']]
# Initialize a figure with FF.create_table(table_data)
#figure = py.figure_factory.create_table(table_data, height_constant=60)
figure = FF.create_table(table_data)
iplot(figure)
table_data = [['Play ID', 'Block on Defenseless Player', 'Blocked into Another Player', 'Injury from Play Spread'],
              ['2587', 'Yes', 'No', 'No'],
              ['1045', 'Yes', 'Yes', 'No'],
              ['3663', 'No', 'Yes', 'No'],
              ['3468', 'No', 'Yes', 'No'], 
              ['1976', 'Yes', 'Yes', 'No'],
              ['2341', 'Yes', 'Yes', 'No'],
              ['2764', 'Yes', 'No', 'No'],
              ['1088', 'Yes', 'No', 'No'],
              ['2792', 'Yes', 'No', 'No'],
              ['1683', 'Yes', 'No', 'No'],
              ['538', 'No', 'No', 'No'],
              ['3630', 'No', 'No', 'Yes'],
              ['2489', 'No', 'No', 'Yes'],
              ['183', 'No', 'No', 'Yes'],
              ['1526', 'No', 'No', 'Yes'],
              ['2072', 'No', 'No', 'Yes'],
              ['602', 'No', 'No', 'Yes']]
# Initialize a figure with FF.create_table(table_data)
#figure = py.figure_factory.create_table(table_data, height_constant=60)
figure = FF.create_table(table_data)
iplot(figure)