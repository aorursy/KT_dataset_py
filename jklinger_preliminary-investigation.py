import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn.tree import DecisionTreeClassifier, export_graphviz
#---- Prepare some data types
dtype = {"Age" : np.float64,
         "Pclass" : pd.Categorical,
         "Sex" : pd.Categorical,         
         "Embarked" : pd.Categorical}

drops = ["Cabin","Ticket","Age","PassengerId"]

predictorVar = "Survived"
basicallyMrs = ["Mme.","Ms.","Mlle."]
notFancyTitles  = ["Mr.","Miss.","Mrs.","Master."]
uniqueTitles = {}
#___________________
#---- Function to extract the title of the passenger
def HasAFancyTitle(df):
    fullName = df["Name"]
    #---- Assume full name in the form: Bloggs, Mr. Joe Alex
    allTitles = [name for name in fullName.split() if "." in name]
    title = allTitles[0]
    #---- Convert titles that are basically Mrs to Mrs
    if title in basicallyMrs:
        title = "Mrs."
    #---- Add up unique titles for our records    
    if not title in uniqueTitles:
        uniqueTitles[title] = 0
    uniqueTitles[title] += 1
    
    #---- Now confirm if the title is fancy or not
    return int(not (title in notFancyTitles))

#___________________
#---- Convert categorical values in independent binary features
def CategConvert(df):
    #---- Loop over columns
    for col in df:
        #---- Extract categorical data
        if df[col].dtype != pd.Categorical:
            continue
        #---- Convert to binary dummies
        df_bin = pd.get_dummies(df[col])
        #---- Give the columns sensible names
        newNameMap = {}
        for col_bin in df_bin:
            newNameMap[col_bin] = col+"_"+col_bin
        df_bin.rename(columns=newNameMap,inplace=True)
        #---- Concat the dummies and drop the old column
        df = pd.concat([df,df_bin],axis=1)
        df = df.drop([col], axis = 1 )
    return df

#___________________
#---- Helper function to read the csv file
def ReadCSV(fileName):
    df = pd.read_csv(fileName, dtype=dtype, )
    df = df.drop(drops, axis = 1)
    df["Name"] = df.apply(HasAFancyTitle,axis=1)
    df.rename(columns={"Name":"HasAFancyTitle"},inplace=True)
    df = CategConvert(df)    
    return df

#___________________
#---- Stolen from http://stackoverflow.com/questions/29432629/correlation-matrix-using-pandas
def PlotCorr(df,size=15):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.
    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr,cmap=cm.get_cmap("YlOrRd"))
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar(cax)
#___________________
#---- "main"

#---- Read the csv files
train = ReadCSV("../input/train.csv")
test = ReadCSV("../input/test.csv")
nullRows = train[train.isnull().any(axis=1)]
if len(nullRows) > 0:
    print("Null values at positions:")
    print(nullRows)
#---- Show unique titles
print("\nUnique titles are:")
for k,v in uniqueTitles.items():
    print(k,v)
#---- Print some statistics
print("Top of the training data:\n")
print(train.head())
print("\nSummary statistics of training data:\n")
print(train.describe())
#---- Show correlations
PlotCorr(train,10)
df = train[train.columns[1:-1]].apply(lambda x: x.corr(train[predictorVar])).sort_values()
fig = plt.figure(figsize=(15,5))
ax = df.plot(marker="o")
ax.set_xticks(np.arange(len(df)))
ax.set_xticklabels(list(df.index))
ax.yaxis.grid(True)

print(df)
#---- Condense binary variables
#---- Split training into train and validation
#---- Next test different methods: dt, random forests, nn --> compare results 
#---- Then apply best to test sample

#---- Fit a decision tree
y = train["Survived"]
X = train.drop(['Survived'], axis=1)
dt = DecisionTreeClassifier(min_samples_split=200, random_state=99)
dt.fit(X, y)

#get_code(dt, list(train.drop(['Survived'],axis=1)), [1,0])