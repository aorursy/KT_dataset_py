# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
color = sns.color_palette()
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as lr

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

inputFolder = "../input"

import os
excels = os.listdir(inputFolder)
print(excels)

# excel di interesse: bSoftPlanning e la tab omonima -> per provare a predirre le ore future
# bSoftMonitoring nel tab TaskDumpMonitoring contiene tutti i task con le varie statistiche
# SoftwareRequestView nel tab omonimo contiene alcune informazioni sulle feature (ma non il team), da incrociare con TaskDumpMonitoring
xlsbSoftPlanning = pd.ExcelFile(inputFolder + "/" + excels[4])
xlsbSoftMonitoring = pd.ExcelFile(inputFolder + "/" + excels[1])
xlsbSoftwareRequestView = pd.ExcelFile(inputFolder + "/" + excels[2])

taskDumpDf = pd.read_excel(xlsbSoftMonitoring, 'TaskDumpMonitoring')
srDumpDf = pd.read_excel(xlsbSoftwareRequestView, 'SoftwareRequestView')
bSoftPlanningDf = pd.read_excel(xlsbSoftPlanning, 'BSoftPlanning')

#Mostro le colonne disponibili in ogni DataFrame

def dfPreview(df, name):
    print(name + ' data: \nRows: {}\nCols: {}'.format(df.shape[0],df.shape[1]))
    print(df.columns)

dfPreview(taskDumpDf, 'Task dump')
dfPreview(srDumpDf, 'Software request')
dfPreview(bSoftPlanningDf, 'Planning')

def printNullValues(df, name):
    print(name + ' columns with null values:\n', df.isnull().sum())
    print("-"*10)

#Vediamo se mancano dati
#Il dump dei task non contiene dati nulli -> Ottimo!
printNullValues(taskDumpDf, 'Train Task')

#Il dump delle SR ha vuote solo le delivery deadline -> normale
printNullValues(srDumpDf, 'Train SR')

#Notiamo subito che mancano molte Delivery deadline per le attività in pianificazione -> male
printNullValues(bSoftPlanningDf, 'Train Planning')


#due modalità di sostituzione dei dati nulli, ma in questo caso non ci serve!
#dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
#dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

#Rimuoviamo colonne inutili
def dropFromDf(colums, df):
    df.drop(colums, axis=1, inplace = True)
    

dropFromDf(['Work Item Type','Work Item ID', 'AreaPath', 'State', 'Planned Release', 'Parent Title'], taskDumpDf)
#per ora limito l'analisi al dump dei task
#dropFromDf(['PassengerId','Cabin', 'Ticket'], srDumpDf)   
#dropFromDf(['PassengerId','Cabin', 'Ticket'], bSoftPlanningDf) 

#Mostra il numero di task assegnati per ogni team
taskDumpDf['Team'].value_counts().plot.bar()

#per visualizzare un grafico a torta
plt.figure(figsize=(10,10))
temp_series = taskDumpDf['Activity'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))
plt.pie(sizes, labels=labels, 
        autopct='%1.1f%%', startangle=200)
plt.title("Activity distribution", fontsize=15)
plt.show()

grouped_df = taskDumpDf.groupby(["Team"])["Completed Work"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(grouped_df['Team'].values, grouped_df['Completed Work'].values, alpha=0.8, color=color[2])
plt.ylabel('Completed Work', fontsize=12)
plt.xlabel('Team', fontsize=12)
plt.title("Team vs. mean completed work", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()

#Trasformo le string in numeri (labels)
labelEncoder = LabelEncoder()
taskDumpDf['Team'] = labelEncoder.fit_transform(taskDumpDf['Team'])
labelEncoder = LabelEncoder()
taskDumpDf['Activity'] = labelEncoder.fit_transform(taskDumpDf['Activity'])
labelEncoder = LabelEncoder()
taskDumpDf['Parent Work Item Type'] = labelEncoder.fit_transform(taskDumpDf['Parent Work Item Type'])
labelEncoder = LabelEncoder()
taskDumpDf['CATEGORIA'] = labelEncoder.fit_transform(taskDumpDf['CATEGORIA'])

#Mappa di correlazione
corrmat = taskDumpDf.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
    
    
sns.set(style="white")
df = taskDumpDf.loc[:,['Team','Completed Work','Original Estimated', 'Parent Work Item Type', 'Activity']]
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=3)
    
X = taskDumpDf[['Team','Remaining Work','Original Estimated', 'Parent Work Item Type', 'Activity', 'CATEGORIA']]
y = taskDumpDf['Completed Work']

X_train, X_test, Y_train, Y_test = tts(X, y, test_size = 0.2, random_state = 42)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

#Utilizziamo un semplice linear regressor
clf=lr()
clf.fit(X_train,Y_train)
accuracy=clf.score(X_test,Y_test)

#Bene ma non benissimo xD
"Accuracy: {}%".format(int(round(accuracy * 100)))

# Any results you write to the current directory are saved as output.
