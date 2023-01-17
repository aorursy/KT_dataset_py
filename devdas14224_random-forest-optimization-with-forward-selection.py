import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from pandas.tools.plotting import scatter_matrix
df = pd.read_csv("../input/train.csv")
df.head(5)
data = df.drop(["Id","SalePrice"],axis=1)
labels = df["SalePrice"]
columnNames = data.columns
print(columnNames)
newData = pd.DataFrame()

for i in range(len(columnNames)):
    
    currData = data[columnNames[i]]
    curDesc = currData.describe()
    
    
    
    if len(curDesc) == 4:  
        
        currData = currData.replace(np.nan,"yes",regex=True) 
        
        labeler = LabelEncoder()
        labeler = labeler.fit(currData)
        labelOut = labeler.transform(currData)
        
        newData[columnNames[i]]=labelOut

    else:
        currData = currData.replace(np.nan,0,regex=True) 
        newData[columnNames[i]]=currData

dataScale = StandardScaler()
dataScale = dataScale.fit(newData)
scaleData = dataScale.transform(newData)

scaleData = pd.DataFrame(scaleData,columns=data.columns)
#Split the training Set to create a train and test set
X_train, X_test, y_train, y_test = train_test_split(scaleData, labels, test_size=0.33, random_state=42)
labels.hist(figsize=(20,20))
plt.show()
newData.hist(figsize=(30,30))
plt.show()
housingCorrMatrix = df.corr()

sb.set(style="white")

mask = np.zeros_like(housingCorrMatrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(20, 20))

cmap = sb.diverging_palette(220, 10, as_cmap=True)

sb.heatmap(housingCorrMatrix, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
housingCorrMatrix["SalePrice"].sort_values(ascending=False)
subset = pd.DataFrame()
testSubset = pd.DataFrame()

initDiff = 48000

keepCols = []
tossCols = []

colNames = X_train.columns

for i in range(len(colNames)):
    
    subset[colNames[i]]=X_train[colNames[i]]
    testSubset[colNames[i]]=X_test[colNames[i]]

    rfClf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    rfClf = rfClf.fit(subset,y_train)
    rfOut = rfClf.predict(testSubset)
    
    rfVal = [rfOut,list(y_test)]
    rfCoeff= np.corrcoef(rfVal[0],rfVal[1])
    forestMeanDiff = mean_absolute_error(rfVal[0],rfVal[1])
    
    
    if forestMeanDiff<initDiff:
        initDiff = forestMeanDiff
        keepCols.append(colNames[i])
        print("Kept: "+str(colNames[i])+"- Correlation: "+str(rfCoeff[0,1])+"| Mean Diff: "+str(forestMeanDiff)+"| R2 Score: "+str(r2_score(rfVal[1],rfVal[0])))
        
    else:
        subset.drop(colNames[i],axis=1)
        testSubset.drop(colNames[i],axis=1)
        tossCols.append(colNames[i])
        print("Dropped: "+str(colNames[i])+"- Correlation: "+str(rfCoeff[0,1])+"| Mean Diff: "+str(forestMeanDiff)+"| R2 Score: "+str(r2_score(rfVal[1],rfVal[0])))
plt.plot(rfVal[0], marker='', color='blue', linewidth=2, label="Ouput")
plt.plot(rfVal[1], marker='', color='orange', linewidth=2,linestyle='dashed',label="Ground Truth")
plt.legend()
print(keepCols)