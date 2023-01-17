import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/titanic/train.csv")

df1 = pd.read_csv("/kaggle/input/titanic/test.csv")
df2 = df.drop(["Name","Ticket","Fare","Cabin"], axis = "columns")

df2.dropna()

df2.Age.fillna(int(df.Age.mean()))

df2["Embarked"].dropna()

gender = pd.get_dummies(df2.Sex)
df2 = pd.concat([df2,gender],axis="columns")

df3 = df2.drop(["Sex","Embarked","Age"],axis="columns")

df3["age"] = df2.Age

df3["age"].fillna(df2["Age"].mean(),inplace = True)
X = df3.drop("Survived",axis= "columns")

Y = df3["Survived"]
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()

def plotCorrelationMatrix(df, graphWidth):

    filename  = None

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix', fontsize=15)

    plt.show()

def plotConfussionMatrix(y_pred,y):

    from sklearn.metrices import confussion_matrix as confm

    cm = confm(y,y_pred)

    plt.figure(figsize = (10,7))

    import seaborn as sn

    sn.heatmap(cm,annot=True)

    plt.xlabel("prediction")

    plt.ylabel("Truth")

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.1000000f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()

plotPerColumnDistribution(df3,8,10)

plotCorrelationMatrix(df3,10)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X,Y)
import pickle as p

with open("Titanic","wb") as f:

    p.dump(model,f)
df2 = df1.drop(["Name","Ticket","Fare","Cabin","Embarked"],axis = "columns")

df3 = df2

df3.Age.fillna(df3.Age.mean())
gender1 = pd.get_dummies(df3.Sex)

df4 = pd.concat([df3,gender1],axis="columns")

df5 = df4.drop(["Sex","Age"],axis="columns")

df5 = pd.concat([df4,df3.Age],axis="columns")

df6 = df5.drop(["Sex","Age"],axis="columns")

df6 = pd.concat([df6,df4.Age],axis = "columns")

df7 = df6

df7["Age"] = df6["Age"].fillna(df4.Age.mean())
Y_pred = model.predict(df7)

Survived = Y_pred

DF = df7
PassengerId = df7["PassengerId"]
DF = DF.drop(["Pclass","SibSp","Parch","female","male","Age","PassengerId"],axis="columns")
DF["PassengerId"] = PassengerId

DF["Survived"] = Survived

DF.to_csv("sub.csv",index=False)