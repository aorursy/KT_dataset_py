import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import display, Markdown # For displaying texts more elegantly
pd.options.display.float_format = '{:,.1f}'.format #For showing a single decimal after point

sampleDf = pd.read_csv('../input/data.csv')
sampleDf.head(10)
sampleDf.columns.to_series().groupby(sampleDf.dtypes).groups
sampleDf.describe()
sampleDf = sampleDf.drop(columns = ["Unnamed: 32"])
def check_duplicates(df, colName):
    temp = df.duplicated(colName).sum()
    if temp == 0:
        return "No duplicate values found in the {} column of the dataframe. ".format(colName)
    else:
        return "There are {} duplicates in the {} column of the dataframe. ".format(temp, colName)
display(Markdown(check_duplicates(sampleDf, 'id')))
sampleDf = sampleDf.set_index('id')
temp = sampleDf.groupby("diagnosis").count()
temp = temp.rename(columns={'radius_mean': "diagnosis"})
temp.plot(kind='pie', y="diagnosis", autopct='%1.0f%%', 
          startangle=90, shadow=True, #explode=[0.01, 0.01, 0.01],
          fontsize=11, legend=False, title="Breast Cancer Type Distributions")
copyDf = sampleDf.copy()
for col in copyDf.columns:
    if copyDf[col].dtype != 'O':
        copyDf[col] = (copyDf[col]-copyDf[col].min())/(copyDf[col].max()-copyDf[col].min()) 
meltedDf = pd.melt(copyDf, id_vars='diagnosis', value_vars=copyDf.drop('diagnosis', axis=1).columns, value_name='value')
g = sns.FacetGrid(meltedDf, col="variable", hue="diagnosis", col_wrap=5)
g.map(sns.distplot, 'value', kde=False).add_legend()
qDf = sampleDf.drop('diagnosis', axis=1)
plt.figure(figsize=(15,15))
sns.heatmap(qDf.corr(), annot=True, linewidths=.5, fmt= '.1f')
tempDf = qDf.corr()
buffer = tempDf.columns.tolist()
remainingFeatures = []
while buffer:
    col = buffer[0]
    strongCorrList = tempDf[col][tempDf[col]>=0.85].index.tolist()
    remainingFeatures.append(col)
    for col in strongCorrList:
        if col in buffer:
            buffer.remove(col)

featuresToDrop = []            
buffer = tempDf.columns.tolist()
for feature in buffer:
    if not feature in remainingFeatures:
        featuresToDrop.append(feature)

qDf = qDf.drop(featuresToDrop, axis = 1 )
plt.figure(figsize=(12,12))
sns.heatmap(qDf.corr(), annot=True, linewidths=.5, fmt= '.1f')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score

y = sampleDf.diagnosis
# split data train 80 % and test 20 %
x_train, x_test, y_train, y_test = train_test_split(qDf, y, test_size=0.2, random_state=41)

#random forest classifier with n_estimators=10 (default)
clf_rf = RandomForestClassifier(random_state=42)      
clr_rf = clf_rf.fit(x_train,y_train)

ac = accuracy_score(y_test,clf_rf.predict(x_test))
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,clf_rf.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")