import pandas as pd
df1=pd.read_csv('../input/world-university-rankings/cwurData.csv')
df1.head(10)
df1.loc[0:50,"institution":"national_rank"] 
df1.loc[50:0:-1,"institution":"national_rank"] 
df1.loc[0:50,"publications":] 
df1.groupby('institution').patents.sum().sort_values(ascending=True)
df2=df1[df1.country=="Australia"]
df2
import matplotlib.pyplot as plt
uni1=df1.head(15)
plt.scatter(uni1.country,uni1.institution,) 
plt.xlabel('countries')
plt.ylabel('universities')
plt.title('rank')
plt.show()
Aussi=df2[(df2.world_rank)&(df2.year==2015)]
plt.hist(Aussi.world_rank,bins=20) 
plt.title('Universities of Australia')
plt.xlabel("Rank of institutions")
plt.show()
df2.institution[df2.world_rank < 100]
USA=df1[df1.country=="USA"]
u1=USA.institution
u2=df2.institution
count_of_USA=u1.count() 
count_of_df2=u2.count() 
sizes=[573,20]
labels=['USA','Australia']
colors=['yellow','green']
plt.pie(sizes,labels=labels,colors=colors,autopct='%1.1f%%')
plt.show()
list1=df1.institution.head(10)
list2=df1.patents.head(10)
z=zip(list1,list2)
z_list=list(z)
print(z_list)
df1.describe()
uniplot = df1[["world_rank","quality_of_education","score"]]
uniplot.plot()
uniplot.plot(subplots=True)
uniplot.plot(kind = "scatter",x="world_rank",y = "score")
uniplot.plot(kind = "hist",y ="score",bins = 50)

