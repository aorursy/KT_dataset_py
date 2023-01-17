import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline 

import seaborn as sns

plt.rcParams["figure.figsize"]=12,6
df=pd.read_csv("/kaggle/input/diamonds/diamonds.csv",index_col=0)

df
#CUT

def cut_codes(cut):

    if cut=="Ideal":

        return 5

    elif cut=="Premium":

        return 4

    elif cut=="Very Good":

        return 3

    elif cut=="Good":

        return 2

    else:

        return 1

df["cut"]=df["cut"].apply(lambda x:cut_codes(x))



#COLOUR

def color_codes(color):

    if color=="D":

        return 7

    elif color=="E":

        return 6

    elif color=="F":

        return 5

    elif color=="G":

        return 4

    elif color=="H":

        return 3

    elif color=="I":

        return 2

    else:

        return 1

df["color"]=df["color"].apply(lambda x:color_codes(x))



#CLARITY

def clarity_codes(clarity):

    if clarity=="I1":

        return 8

    elif clarity=="SI2":

        return 7

    elif clarity=="SI1":

        return 6

    elif clarity=="VS2":

        return 5

    elif clarity=="VS1":

        return 4

    elif clarity=="VVS2":

        return 3

    elif clarity=="VVS1":

        return 2

    else:

        return 1

df["clarity"]=df["clarity"].apply(lambda x:clarity_codes(x))



df.head()
df.isnull().sum()
df.describe()
print("The number of rows with a value of 0 for x are ",(df["x"]==0).sum(),".")

print("The number of rows with a value of 0 for y are ",(df["y"]==0).sum(),".")

print("The number of rows with a value of 0 for z are ",(df["z"]==0).sum(),".")

print("The total number of rows with a value of 0 are ",((df["x"]==0)|(df["y"]==0)|(df["z"]==0)).sum(),".")
df.drop(df[(df["x"]==0)|(df["y"]==0)|(df["z"]==0)].index,inplace=True)
fig=plt.figure(figsize=(16,16))

ax1=plt.subplot2grid((4,3),(0,0),colspan=2,rowspan=2)

ax2=plt.subplot2grid((4,3),(0,2))

ax3=plt.subplot2grid((4,3),(1,2))

ax4=plt.subplot2grid((4,3),(2,0))

ax5=plt.subplot2grid((4,3),(2,1))

ax6=plt.subplot2grid((4,3),(2,2))

ax7=plt.subplot2grid((4,3),(3,0))

ax8=plt.subplot2grid((4,3),(3,1))

ax9=plt.subplot2grid((4,3),(3,2))



sns.scatterplot(x=df["carat"],y=df["price"],color="lavender",ax=ax1)

sns.scatterplot(x=df["x"],y=df["price"],color="powderblue",ax=ax2)

sns.scatterplot(x=df["y"],y=df["price"],color="lightblue",ax=ax3)

sns.scatterplot(x=df["depth"],y=df["price"],color="palegreen",ax=ax4)

sns.scatterplot(x=df["table"],y=df["price"],color="lightgreen",ax=ax5)

sns.scatterplot(x=df["z"],y=df["price"],color="skyblue",ax=ax6)

sns.scatterplot(x=df["cut"],y=df["price"],color="lightsalmon",ax=ax7)

sns.scatterplot(x=df["color"],y=df["price"],color="palevioletred",ax=ax8)

sns.scatterplot(x=df["clarity"],y=df["price"],color="gold",ax=ax9)



plt.tight_layout(pad=1,h_pad=1,w_pad=1)
from sklearn.model_selection import train_test_split



from sklearn.linear_model import LinearRegression



from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

outliers=["Including the Outliers","Removing the Visual Outliers","Removing the Calculated Outliers"]

rmse=[]

r2score=[]
df1=df.copy()



print("The number of rows include all the outliers are",df1.shape[0],".")
x1=df1.drop(["price"],axis=1)

y1=df1["price"]

x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.3,random_state=5)



lr1=LinearRegression()

lr1.fit(x1_train,y1_train)

y1_predict=lr1.predict(x1_test)



print("MAE: %.2f"%mean_absolute_error(y1_test,y1_predict))

print("MSE: %.2f"%mean_squared_error(y1_test,y1_predict))

print("RMSE: %.2f"%np.sqrt(mean_absolute_error(y1_test,y1_predict)))

print("R2: %.2f"%r2_score(y1_test,y1_predict))



rmse.append(np.sqrt(mean_absolute_error(y1_test,y1_predict)))

r2score.append(r2_score(y1_test,y1_predict))
sns.distplot(y1_test,hist=True,color="lightskyblue",label="Actual Values")

sns.distplot(y1_predict,hist=True,color="plum",label="Predicted Values")

plt.legend()

plt.xlabel("Price")
f,axes=plt.subplots(2,1,sharex=True)

sns.boxplot(y1_test,color="lightskyblue",whis=4,ax=axes[0])

sns.boxplot(y1_predict,color="plum",whis=7,ax=axes[1])



axes[0].set_xlabel("")

plt.xlabel("Price")



axes[0].set_ylabel("Actual Price")

axes[1].set_ylabel("Predicted Price")
df2=df.copy()



visual=df2[(df2["y"]>30)|(df2["z"]>10)|(df2["depth"]<50)|(df2["depth"]>75)|(df2["table"]<45)|(df2["table"]>75)]

print("There are {num} rows of visual outliers where y >30, z >10, depth <50 and >75, and table <45 and >75.".format(num=visual.shape[0]))
df2.drop(visual.index,inplace=True)



print("After dropping the Visual Outliers, the number of rows are now",df2.shape[0],".")
x2=df2.drop(["price"],axis=1)

y2=df2["price"]

x2_train,x2_test,y2_train,y2_test=train_test_split(x2,y2,test_size=0.3,random_state=5)



lr2=LinearRegression()

lr2.fit(x2_train,y2_train)

y2_predict=lr2.predict(x2_test)



print("MAE: %.2f"%mean_absolute_error(y2_test,y2_predict))

print("MSE: %.2f"%mean_squared_error(y2_test,y2_predict))

print("RMSE: %.2f"%np.sqrt(mean_absolute_error(y2_test,y2_predict)))

print("R2: %.2f"%r2_score(y2_test,y2_predict))



rmse.append(np.sqrt(mean_absolute_error(y2_test,y2_predict)))

r2score.append(r2_score(y2_test,y2_predict))
sns.distplot(y2_test,hist=True,color="lightskyblue",label="Actual Values")

sns.distplot(y2_predict,hist=True,color="plum",label="Predicted Values")

plt.legend()

plt.xlabel("Price")
f,axes=plt.subplots(2,1,sharex=True)

sns.boxplot(y2_test,color="lightskyblue",whis=4,ax=axes[0])

sns.boxplot(y2_predict,color="plum",whis=7,ax=axes[1])



axes[0].set_xlabel("")

plt.xlabel("Price")



axes[0].set_ylabel("Actual Price")

axes[1].set_ylabel("Predicted Price")
df3=df.copy()



Q1=df3.quantile(0.25)

Q3=df3.quantile(0.75)

IQR=Q3-Q1



col=list(df3.columns)



print("Number of Calculated Outliers")

print(df3[(df3[col]<(Q1[col]-1.5*IQR[col]))|(df3[col]>(Q3[col]+1.5*IQR[col]))].count())
def c_outliers(col):

    return df3[(df3[col]<(Q1[col]-1.5*IQR[col]))|(df3[col]>(Q3[col]+1.5*IQR[col]))]



for col in df3:

    df3.drop(c_outliers(col).index,inplace=True)

    

print("After dropping the Calculated Outliers, the number of rows are now",df3.shape[0],".")
x3=df3.drop(["price"],axis=1)

y3=df3["price"]

x3_train,x3_test,y3_train,y3_test=train_test_split(x3,y3,test_size=0.3,random_state=5)



lr3=LinearRegression()

lr3.fit(x3_train,y3_train)

y3_predict=lr3.predict(x3_test)



print("MAE: %.2f"%mean_absolute_error(y3_test,y3_predict))

print("MSE: %.2f"%mean_squared_error(y3_test,y3_predict))

print("RMSE: %.2f"%np.sqrt(mean_absolute_error(y3_test,y3_predict)))

print("R2: %.2f"%r2_score(y3_test,y3_predict))



rmse.append(np.sqrt(mean_absolute_error(y3_test,y3_predict)))

r2score.append(r2_score(y3_test,y3_predict))
sns.distplot(y3_test,hist=True,color="lightskyblue",label="Actual Values")

sns.distplot(y3_predict,hist=True,color="plum",label="Predicted Values")

plt.legend()

plt.xlabel("Price")
f,axes=plt.subplots(2,1,sharex=True)

sns.boxplot(y3_test,color="lightskyblue",whis=4,ax=axes[0])

sns.boxplot(y3_predict,color="plum",whis=7,ax=axes[1])



axes[0].set_xlabel("")

plt.xlabel("Price")



axes[0].set_ylabel("Actual Price")

axes[1].set_ylabel("Predicted Price")
scores=pd.DataFrame({"Data":outliers,"RMSE":rmse,"R2-Scores":r2score})

scores
fig,ax1=plt.subplots()



ax1.plot(scores["Data"],scores["RMSE"],color="sandybrown",marker="o")

ax1.set_ylabel("Root Mean Square Error",fontsize=12,color="sandybrown")

for label in ax1.get_yticklabels():

    label.set_color("sandybrown")

    

ax2=ax1.twinx()

ax2.plot(scores["Data"],scores["R2-Scores"],color="yellowgreen",marker="^")

ax2.set_ylabel("R Squared Score",fontsize=12,color="yellowgreen")

for label in ax2.get_yticklabels():

    label.set_color("yellowgreen")