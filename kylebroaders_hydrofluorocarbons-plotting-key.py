# Import statements to get started
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os

allfiles = []
# Get the full filepath for each file available to this notebook
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        fullpath = os.path.join(dirname, filename)
        allfiles.append(fullpath)
        print(fullpath)
alldata = []
for file in allfiles:
    if file[-3:] == "txt":
        newdata = pd.read_fwf(file)
        alldata.append(newdata)
    elif file[-3:] == "tsv":
        newdata = pd.read_csv(file, sep='\t')
        alldata.append(newdata)
for i in range(len(alldata)):
    if 'decdate' in alldata[i]:
        alldata[i] = alldata[i].sort_values('decdate')
    else:
        alldata[i] = alldata[i].sort_values('dec_date')
for data in alldata:
    plt.figure(figsize=(10,5))
    sns.lineplot(x=data.iloc[:,1], y=data.iloc[:,6])