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

# Info about computing valeus based on indicators: https://www.europeansocialsurvey.org/docs/methodology/ESS_computing_human_values_scale.pdf
ESS_ind = pd.read_csv("/kaggle/input/ess9-preprocessed-data/Data_imputed.csv")
# List of 21 indicators in the ESS dataset measuring 10 Schwarz values in the correct order: indexed from 0 to 20
indicators = ['ipcrtiv','imprich','ipeqopt','ipshabt','impsafe','impdiff','ipfrule','ipudrst', 'ipmodst', 'ipgdtim','impfree',
          'iphlppl', 'ipsuces', 'ipstrgv', 'ipadvnt', 'ipbhprp', 'iprspot', 'iplylfr', 'impenv', 'imptrad', 'impfun']

# List of 10 Schwarz values to be calculated based on 21 indicators
values = ['conformity', 'tradition', 'benevolence', 'universalism', 'self_direction', 'stimulation', 'hedonism', 'achievement', 'power', 'security']
ESS_ind["mrat"] = ESS_ind[indicators].apply(np.mean, axis = 1)
# conformity
ESS_ind[values[0]] = ESS_ind[[indicators[6], indicators[15]]].apply(np.mean, axis = 1) - ESS_ind["mrat"]

# tradition
ESS_ind[values[1]] = ESS_ind[[indicators[8], indicators[19]]].apply(np.mean, axis = 1) - ESS_ind["mrat"]

# benevolence
ESS_ind[values[2]] = ESS_ind[[indicators[11], indicators[17]]].apply(np.mean, axis = 1) - ESS_ind["mrat"]

# universalism
ESS_ind[values[3]] = ESS_ind[[indicators[2], indicators[7], indicators[18]]].apply(np.mean, axis = 1) - ESS_ind["mrat"]

# self_direction
ESS_ind[values[4]] = ESS_ind[[indicators[0], indicators[10]]].apply(np.mean, axis = 1) - ESS_ind["mrat"]

# stimulation
ESS_ind[values[5]] = ESS_ind[[indicators[5], indicators[14]]].apply(np.mean, axis = 1) - ESS_ind["mrat"]

# hedonism
ESS_ind[values[6]] = ESS_ind[[indicators[9], indicators[20]]].apply(np.mean, axis = 1) - ESS_ind["mrat"]

# achievement
ESS_ind[values[7]] = ESS_ind[[indicators[3], indicators[12]]].apply(np.mean, axis = 1) - ESS_ind["mrat"]

# power
ESS_ind[values[8]] = ESS_ind[[indicators[1], indicators[16]]].apply(np.mean, axis = 1) - ESS_ind["mrat"]

# security
ESS_ind[values[9]] = ESS_ind[[indicators[4], indicators[13]]].apply(np.mean, axis = 1) - ESS_ind["mrat"]

# Dropping indicators, because they aren't needed anymore
ESS_values = ESS_ind.drop(indicators, axis = 1)
ESS_values[values]
ESS_values.to_csv("ESS_values.csv", index = False)