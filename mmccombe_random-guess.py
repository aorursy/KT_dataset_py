#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Mon May 25 16:55:40 2020



@author: madelinemccombe

"""

import pandas as pd

import numpy as np



samp = pd.read_csv('../input/predicting-bank-telemarketing/samp_submission.csv')



samp.columns



samp.Predicted = np.random.choice(range(2), size=samp.shape[0], p=[0.5, 0.5])



samp.to_csv('first_test.csv', index=False)