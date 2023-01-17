import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

train = pd.read_csv("../input/train.csv", index_col=None)
train.head()
test = pd.read_csv("../input/test.csv", index_col=None)
test.head()
print ("train.shape:", train.shape)
print ("test.shape:", test.shape)
target = train.TARGET
print (target.describe())
plt.hist(target)
plt.ylabel("freq")
plt.xlabel("target value")
plt.show()
uncomList = list(set(train.columns) ^ set(test.columns))
print (uncomList)
train.drop("TARGET", axis=1, inplace=True)
combi = pd.concat([train, test], axis=0)
print ("train.shape[0] + test.shape[0]:", train.shape[0]+test.shape[0])
print ("combi.shape:", combi.shape)
floatList = []
intList = []
objectList = []

for t in combi.columns:
    if combi[t].dtypes==np.float64 or combi[t].dtypes==np.float32:
        floatList.append(t)
    elif combi[t].dtypes==np.int64 or combi[t].dtypes==np.int32:
        intList.append(t)
    else:
        objectList.append(t)
        
print ("The number of float columns:", len(floatList))
print ("The number of int columns:", len(intList))
print ("The number of non-numeric columns:", len(objectList))
combiNan = np.sum(combi.isnull())

combiNanCounter = 0
combiNanCol = []

for n in range(len(combiNan)):
    if combiNan[n] > 0:
        print (combiNan.index[n])
        combiNanCol.append(n)
    combiNanCounter += 1
    
print ("Checked columns:", combiNanCounter)
print ("Columns with Nan:", len(combiNanCol))
uniq10 = []
uniq100 = []
uniqMany = []

for u in combi.columns:
    if combi[u].nunique() <= 10:
        uniq10.append(u)
    elif combi[u].nunique() > 10 & combi[u].nunique() <= 100:
        uniq100.append(u)
    else:
        uniqMany.append(u)
        
print ("The number of columns with <= 10 unique values:", len(uniq10))
print ("The number of columns with 10<x<=100 unique values", len(uniq100))
print ("The number of columns with >100 unique values:", len(uniqMany))
# Check only for the colums with float values to avoid categoricals to be incorporated
combiFloat = combi[floatList]

# <Removed from ver3>
# May not need this feature scaling (0-1) since probably matplotlib could deal with it
# But basically pearson correlation coefficient is sensitive to scale so this is just to make sure
# from sklearn.preprocessing import MinMaxScaler
# combiFloat = combiFloat.apply(lambda x: MinMaxScaler().fit_transform(x))

# get correlation coefficient as a matrix
corrFloat = combiFloat.corr()
print ("The shape of correlation coefficient matrix:", corrFloat.shape)
# Below code URL: https://stanford.edu/~mwaskom/software/seaborn/examples/many_pairwise_correlations.html

# Generate a mask for the upper triangle
mask = np.zeros_like(corrFloat, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 17))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corrFloat, mask=mask, cmap=cmap, vmin=-1, vmax=1,
            square=True, xticklabels=5, yticklabels=5,
            linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
# threshold is arbitrary, but in this example threshold = +/-0.8 (Pearson's correlation coefficient)

# check the number of unique combinations of 2 variables
import itertools
pairs = list(itertools.combinations(corrFloat.columns, 2))
print ("Variables pairs:", len(pairs))

hiCor = []
hiCorCounter = 0
for i in range(corrFloat.shape[1]):
    for j in range(0, i):
        if corrFloat[corrFloat.columns[i]][j] > 0.8 or corrFloat[corrFloat.columns[i]][j] < -0.8:
            hiCor.append(corrFloat.index[j])
        hiCorCounter += 1

# get unique values from the list
hiCor = list(set(hiCor))

print ("Checked pairs:", hiCorCounter)
print ("Columns to be removed due to high correlation:", len(hiCor))

combi.drop(hiCor, axis=1, inplace=True)

print ("New combi shape:", combi.shape)
train = combi[:train.shape[0]]
train["TARGET"] = target
test = combi[train.shape[0]:]

print ("new train shape:", train.shape)
print ("new test shape:", test.shape)