import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string as st # to get set of characters
col = [a for a in st.ascii_uppercase[:10]]
tmp = np.random.randint(1,30,1000).reshape(100,10)
df = pd.DataFrame(tmp,columns=col)
df.head(2)
df.describe()
df["categ"] = np.random.choice(col[:3],100)
date = pd.date_range("1/1/2018",periods=100)
df["date"] = date
df.set_index(date,inplace=True)
df.drop("date",axis=1,inplace=True)
df.head(2)
df.head(1)
sns.jointplot(df.A,df.D) # Default joint plot with scatter plot
sns.jointplot(df.A,df.D,kind="kde") # Default joint plot with scatter plot
sns.jointplot(df.A,df.D,kind="reg") # Default joint plot with scatter plot
sns.jointplot(df.A,df.D,kind="resid") # Default joint plot with scatter plot
sns.jointplot(df.A,df.D,kind="hex") # Default joint plot with scatter plot
sns.jointplot(df.A,df.D,kind="reg",color="g") # Default joint plot with scatter plot
sns.regplot(df.A,df.B)
sns.regplot(df.A,df.B,x_jitter=2.9)
sns.regplot(df.A,df.B,x_jitter=2.9,marker="^",color="r")
sns.regplot(df.A,df.B,x_jitter=2.3,marker=">",color="r",line_kws={"color":"m","linewidth":5.1},ci=100,x_estimator=np.median)
sns.regplot(df.A,df.B,x_jitter=2.3,marker=">",color="r",line_kws={"color":"m","linewidth":5.1},ci=100,x_estimator=np.std)
sns.regplot(df.A,df.B,x_jitter=2.3,marker=">",color="r",line_kws={"color":"m","linewidth":5.1},ci=100,x_estimator=np.mean)
df.head(1)
sns.set_style("darkgrid")
sns.pairplot(df[["A","B","C","categ"]],hue="categ",kind="reg")
sns.set_style("darkgrid")
sns.pairplot(df,hue="categ")
sns.set_style("darkgrid")
sns.pairplot(df[["A","B","C","categ"]],hue="categ",diag_kind="hist")
sns.set_style("darkgrid")
sns.pairplot(df[["A","B","C","categ"]],hue="categ",diag_kind="hist",palette="husl")
sns.set_style("darkgrid")
sns.pairplot(df[["A","B","C","categ"]],hue="categ",diag_kind="hist",palette="husl",markers=["D","<",">"])
sns.set_style("darkgrid")
sns.pairplot(df,vars=["A","B","C"],hue="categ",diag_kind="hist",palette="husl",markers=["D","<",">"])
sns.set_style("darkgrid")
sns.pairplot(df,vars=["A","B","C"],hue="categ",diag_kind="hist",palette="husl",markers=["D","<",">"])
sns.set_style("darkgrid")
sns.pairplot(df,x_vars=["A","B"],y_vars=["C","D"],hue="categ",palette="husl",markers=["^","<",">"])