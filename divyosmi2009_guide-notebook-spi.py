import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression,LogisticRegression

import matplotlib.pyplot as plt 

import random as rn
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    col = ["red","orange","cyan","yellow"]

    c = rn.choices(col)

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

            valueCounts.plot.bar(color=c)

        else:

            columnDf.hist(color = c)

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()
df = pd.read_csv("/kaggle/input/sodium-potassium-imbalance/Sodium_Potassium_Imbalance.csv")

x = df.drop("Has_Sodium_Potassium_Imbalance",axis="columns")

y = df["Has_Sodium_Potassium_Imbalance"]
plotPerColumnDistribution(x,15,15)
log_reg_model = LogisticRegression()

log_reg_model.fit(x,y)
cout = 0

coutw = 0
y_p = log_reg_model.predict(x)

for i in range(len(y)):

    if y[i] == y_p[i]:

        cout += 1

    else:

        coutw += 1

    
plt.bar(cout,cout)

plt.bar(coutw,coutw)
cout,coutw=0,0 
DF = pd.read_csv("/kaggle/input/sodium-potassium-imbalance/Patent_Per_Year_in_India.csv")

X = DF.drop("No_Of_Patients",axis = "columns")

Y = DF["No_Of_Patients"] 
lin_reg_model = LinearRegression()

lin_reg_model.fit(X,Y)
Y = list(Y)
Y_p = lin_reg_model.predict(X)

for i in range(len(Y)):

    if Y[i] == Y_p[i]:

        cout += 1

    else:

        coutw += 1
plt.bar(cout,cout)

plt.bar(coutw,coutw)
plt.subplot(1,2,1)

plt.scatter(X,Y)

plt.subplot(1,2,2)

plt.scatter(X,Y)

plt.plot(X,lin_reg_model.predict(X))
with open("ppyii","wb") as f:

    import pickle as pi

    pi.dump(lin_reg_model,f)
with open("spicm","wb") as f:

    import pickle as pi

    pi.dump(log_reg_model,f)