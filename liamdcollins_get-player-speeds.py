import os
import pandas as pd
from datetime import datetime
import ciso8601
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import math

seasons = ["reg-wk1-6", "reg-wk7-12", "reg-wk13-17", "post"]
years = ["2016", "2017"]
years = ["2017"]


for year in years:
    for season in seasons:
        currentFile = '../input/NGS-' + year + '-' + season + '.csv'
        print("Loading:", currentFile)
        ngsDataRaw = pd.read_csv(currentFile)
        print("Processing...")
        # drop columns with no GSISID
        ngsData = ngsDataRaw.loc[ngsDataRaw["GSISID"] == ngsDataRaw["GSISID"]].copy()
        ngsData.sort_values(by=["Time"], inplace=True)
        # add extra column
        ngsData['speed'] = ''
        ngsData['Elapsed Time'] = np.NaN
        # get unique GIDs
        uniqueGIDs = ngsData["GSISID"].unique()

        j = 0
        for GID in uniqueGIDs:
            j += 1
            print("GID", GID, "(", j, "out of", len(uniqueGIDs), ")")
            GIDSubset = ngsData.loc[ngsData["GSISID"] == GID]
            uniquePlayIDs = GIDSubset["PlayID"].unique()
            
            for play in uniquePlayIDs:
                currentSet = GIDSubset[GIDSubset["PlayID"] == play]
                d0 = ciso8601.parse_datetime(currentSet.iloc[0]['Time'])
                d2 = d0
                for i in range(1,len(currentSet.index)):
                    d1 = d2
                    d2 = ciso8601.parse_datetime(currentSet.iloc[i]['Time'])
                    timeDelta = d2-d1
                    timeDeltaSeconds = (timeDelta.seconds) + (timeDelta.microseconds / 1000000)
                    distance = currentSet.iloc[i]["dis"]
                    ngsData.at[currentSet.iloc[i].name, 'speed'] = distance/timeDeltaSeconds
                    
            sys.stdout.write(" Complete.\n")

        ngsData.to_csv(year + '-' + season + '-output.csv', index = False)
