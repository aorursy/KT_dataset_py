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
plt.figure(figsize=(16,4))
g = sns.barplot(df.index,df.A)
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(16,4))
g = sns.barplot(df.index,df.A)
for i,v in enumerate(df.A):
    g.text(i,v,v)
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(16,4))
g = sns.barplot("B","A",data=df)
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(16,4))
sns.barplot("categ","A",data=df,estimator=np.median) # Estimator = np.mean,np.std etc can be used
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(16,4))
sns.barplot("B","A",data=df[:50],hue="categ") # Hue can be used as any categorical value
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(16,4))
sns.barplot("B","A",data=df[:20],orient="h") # Hue can be used as any categorical value
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(16,4))
sns.barplot("categ","A",data=df,palette="winter") # palette = RdBu, coolwarm,winter etc.
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(16,4))
sns.barplot("categ","A",data=df,color="r") # color = to make all bar of same color
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(16,4))
sns.barplot("categ","A",data=df,color="r",capsize=.1) # capsize = to add bar on error lines 
                                                        # (blackone on bar)
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(16,4))
sns.barplot("categ","A",data=df,color="r",capsize=.1,errcolor=".9") # errcolor = color for error bar 
                                                                    #(gray range any value in "")
plt.xticks(rotation=90)
plt.show()
df.head(1)
plt.figure(figsize=(16,4))
sns.lineplot("categ","B",data=df) # Default line plot
plt.figure(figsize=(16,4))
sns.lineplot("categ","B",data=df,err_style="bars",color="r",estimator=np.std) 
                # err_style bar or band(default)
df.head(1)
plt.figure(figsize=(16,4))
sns.distplot(df.A) # default plot
plt.figure(figsize=(16,4))
sns.distplot(df.A,kde=False) # Making KDE off
plt.figure(figsize=(16,4))
sns.distplot(df.A,bins=20) # bin size can be any number
plt.figure(figsize=(16,4))
sns.distplot(df.A,hist=False) # making hist OFF (basically converting hist to equvalent KDE plot)
plt.figure(figsize=(16,4))
sns.distplot(df.A,bins=25,rug=True,color="r") # rug, keepign data points in plot
plt.figure(figsize=(16,4))
sns.distplot(df.A,bins=25,rug=True,color="orange",vertical=True) # making horozontal plot
df.head(1)
plt.figure(figsize=(16,4))
sns.scatterplot(df.categ,df.B) # default plot
plt.figure(figsize=(16,4))
sns.scatterplot(df.A,df.B,hue=df.categ,palette="RdBu",size=df.categ,x_jitter=30,alpha=.8) # default plot
plt.figure(figsize=(16,4))
sns.scatterplot(df.A,df.B,hue=df.categ,size=df.categ,estimator=np.std) # default plot
df.head(1)
plt.figure(figsize=(20,8))
sns.boxplot(df.categ,df.E)
plt.show()
plt.figure(figsize=(20,8))
sns.boxplot(df.E,df.categ)
plt.show()
plt.figure(figsize=(20,8))
sns.boxplot(df.categ,df.E,palette="coolwarm")
plt.show()
plt.figure(figsize=(20,8))
sns.boxplot(df.categ,df.E,palette="coolwarm",saturation=.2,width=.4,dodge=False,fliersize=20)
# fliersize is outlier size
plt.show()
plt.figure(figsize=(20,8))
sns.boxplot(df.categ,df.E,palette="coolwarm",saturation=.2,width=.4,dodge=False,fliersize=20,linewidth=5)
# linewidth, any number
plt.show()
plt.figure(figsize=(20,8))
sns.boxplot(df.categ,df.E,palette="RdBu",saturation=.2,width=.4,dodge=False,fliersize=20,whis=.9)
# whis is whisker size, to control outliers
plt.show()
plt.figure(figsize=(20,8))
sns.boxplot(df.categ,df.E,color="r",saturation=.8,width=.4,dodge=False,notch=True)
# notch is to display confidence level of median
plt.show()
plt.figure(figsize=(20,8))
sns.boxplot(df.categ,df.E,palette="coolwarm")
sns.swarmplot(df.categ,df.F,color="red",size=10,linewidth=1.2,edgecolor="green")
plt.show()
plt.figure(figsize=(20,8))
sns.boxplot(df.categ,df.E,palette="coolwarm")
sns.swarmplot(df.categ,df.F,hue=df.C,color="red",size=10,linewidth=1.2,edgecolor="green")
plt.show()
plt.figure(figsize=(20,8))
sns.boxenplot(df.categ,df.E,palette="RdBu")
plt.show()
plt.figure(figsize=(20,8))
sns.boxenplot(df.categ,df.E,palette="RdBu")
sns.swarmplot(df.categ,df.F,hue=df.C,color="red",size=10,linewidth=1.2,edgecolor="green")
plt.show()
plt.figure(figsize=(20,8))
sns.violinplot(df.categ,df.E,palette="coolwarm") # Default plot
plt.show()
plt.figure(figsize=(20,8))
sns.violinplot(df.categ,df.E,palette="coolwarm",bw=.3) # bw is for kernel bandwidth
plt.show()
plt.figure(figsize=(20,8))
sns.violinplot(df.categ,df.E,palette="coolwarm",cut=.2) # cut is again for kernel bandwidth, see the doc
plt.show()
plt.figure(figsize=(20,8))
sns.violinplot(df.categ,df.E,palette="coolwarm",gridsize =2) # gridsize is for computing kernel density
plt.show()
plt.figure(figsize=(20,8))
sns.violinplot(df.categ,df.E,palette="coolwarm",width=2) # width is for width of violin
plt.show()
plt.figure(figsize=(20,8))
sns.violinplot(df.categ,df.E,palette="coolwarm",inner="box") # inner is for displaying datapoints
plt.show()
plt.figure(figsize=(20,8))
sns.stripplot(df.categ,df.C,palette="BuGn",jitter=0,size=12,edgecolor="red",linewidth=.4) 
                                            # jitter is to see the distribution clearly
plt.show()
plt.figure(figsize=(20,8))
sns.stripplot(df.categ,df.C,palette="BuGn",jitter=.1,size=12,edgecolor="red",linewidth=.4) 
                                            # jitter is to see the distribution clearly
plt.show()
pg = sns.PairGrid(df,vars=["A","B","C"])
pg = pg.map(sns.scatterplot)
pg = sns.PairGrid(df,vars=["A","B","C"])
pg = pg.map_diag(plt.hist)
pg = pg.map_offdiag(plt.scatter)
pg = sns.PairGrid(df,vars=["A","B","C"],palette="RdBu",hue="categ")
pg = pg.map_diag(plt.hist)
pg = pg.map_lower(sns.kdeplot)
pg = pg.map_upper(plt.scatter)
pg.add_legend()
pg = sns.PairGrid(df,vars=["A","B","C"],palette="RdBu",hue="categ")
pg = pg.map_diag(sns.distplot)
pg = pg.map_lower(sns.kdeplot)
pg = pg.map_upper(plt.scatter)
pg.add_legend()
pg = sns.PairGrid(df,vars=["D","E","F"],palette="RdBu",hue="categ",hue_kws={"marker":["D","^","+"]})
pg = pg.map_diag(plt.hist,)
pg = pg.map_lower(sns.kdeplot)
pg = pg.map_upper(plt.scatter,s=50)
pg.add_legend()
ran_num = np.random.randint(1,50,100).reshape(10,10)
plt.figure(figsize=(10,4))
sns.heatmap(ran_num,linewidths=1,annot=True)
plt.figure(figsize=(10,4))
sns.heatmap(ran_num,linewidths=1,annot=True,vmax=40,vmin=15) # vmin/vmax to color only within the range, 
                                                             # below range will be black