
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
# === Import Basic Modules ===
import pandas as pd
import numpy as np
import os
import sys
import csv
import yaml
import json
import logging

# === Value Map ===
value_map = {'not available in demo dataset' : 'na', 'Not Socially Engaged' : 'ns', '(not set)' : 'nt', '(not provided)' : 'np'}


# Read Data
outList = []
completeStr = []
with open('../input/train.csv', 'r') as thisCsv:
    csvReader = csv.reader(thisCsv, quoting=csv.QUOTE_ALL, skipinitialspace=True, delimiter=',')
    header = next(csvReader)
    for rowIndex, thisRow in enumerate(csvReader):
        outDict = {}
        thisDict = zip(header, thisRow)
        for key_value in thisDict:
            ky = key_value[0]
            if ky in ['device', 'geoNetwork', 'trafficSource', 'totals']:
                vl = json.loads(key_value[1])
                for k,v in vl.items():
                    if ky == 'trafficSource' and k == 'adwordsClickInfo':
                        for ik, iv in v.items():
                            if ik == 'targetingCriteria':
                                continue
                            if iv in value_map:
                                iv = value_map[iv]
                                outDict[ky + '_' + k + '_' + ik] = iv
                            else:
                                outDict[ky + '_' + k + '_' + ik] = iv
                    if ky == 'trafficSource' and k != 'adwordsClickInfo':
                        if v in value_map:
                            v = value_map[v]
                        outDict[ky + '_' + k] = v
                    if ky == 'device':
                        if v in value_map:
                            v = value_map[v]
                        outDict[ky + '_' + k] = v
                    if ky == 'geoNetwork':
                        if v in value_map:
                            v = value_map[v]
                        outDict[ky + '_' + k] = v
                    if ky == 'totals':
                        if v in value_map:
                            v = value_map[v]
                        outDict[ky + '_' + k] = v
            else:
                outDict[ky] =  key_value[1]
        outList.append(outDict)


dfTrain = pd.DataFrame(outList)
dfTrain.shape
