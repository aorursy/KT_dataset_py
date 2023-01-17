# Importing required packages
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os
print(os.listdir("../input"))

#loading the data 

bcancer_df =pd.read_csv('../input/haberman.csv')
print(bcancer_df)

bcancer_df =pd.read_csv("../input/haberman.csv",header=None, 
                        names=['Patients-age','op_year','positive_aux_nodes','surv_status_more_than_5_years'])
bcancer_df.info()
bcancer_df.head()
print(list(bcancer_df["surv_status_more_than_5_years"].unique()))
bcancer_df["surv_status_more_than_5_years"]=bcancer_df["surv_status_more_than_5_years"].map({1:"yes",2:"no"})

bcancer_df.info()
bcancer_df.head()
bcancer_df.describe()
print(bcancer_df["surv_status_more_than_5_years"].value_counts())

#   PDF:Probability desnity function 
for i,columns in enumerate(list(bcancer_df.columns)[:-1]):
    sns.set_style("whitegrid")
    a=sns.FacetGrid(bcancer_df,hue="surv_status_more_than_5_years",size=7)
    a.map(sns.distplot,columns).add_legend()
    plt.title("PDF plot for {}".format(columns))
    plt.ylabel("frequency range")
    plt.show()
#CDF :Cummulative distribution function

import numpy as np

sns.set_style("whitegrid")

counts,bin_edges=np.histogram(bcancer_df["Patients-age"],bins=10,density=True)
print(counts)
print(bin_edges)

pdf = counts/(sum(counts))

cdf =np.cumsum(pdf)

plt.plot(bin_edges[:-1],pdf)

plt.plot(bin_edges[:-1],cdf)

print(plt.title("CDF Age plot"))
plt.xlabel("Patients AGE")
plt.legend(['Yes','No'])

plt.ylabel("Percentage of CDF")
plt.show()
plt.close()
counts,bin_edges=np.histogram(bcancer_df["op_year"],bins=10,density=True)

print(counts)
print(bin_edges)

pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[:-1],pdf)
plt.plot(bin_edges[:-1],cdf)
print(plt.title("CDF Operation year plot"))
plt.legend(['Yes','No'])
plt.xlabel("Operation Year")
plt.ylabel("Percentage of CDF")



plt.show()

plt.close()

counts,bin_edges=np.histogram(bcancer_df["positive_aux_nodes"],bins=10,density=True)
print(counts)
print(bin_edges)
pdf=counts/(sum(counts))
plt.plot(bin_edges[:-1],pdf)
plt.plot(bin_edges[:-1],cdf)
print(plt.title("CDF positive axilliary nodes plot"))
plt.legend(['Yes','No'])
plt.xlabel("positive axillary nodes")
plt.ylabel("Percentage of CDF")
plt.show()


a=sns.boxplot(x="surv_status_more_than_5_years",y="Patients-age",data=bcancer_df)
print(a)
plt.title('fig1_boxplot (surv_status_more_than_5_years vs Patients-age)')
b=sns.boxplot(x="surv_status_more_than_5_years",y="op_year",data=bcancer_df)
plt.title('fig1.2_boxplot (surv_status_more_than_5_years , op_year)')
sns.boxplot(x="surv_status_more_than_5_years",y="positive_aux_nodes",data=bcancer_df)
plt.title('fig1.3_boxplot (surv_status_more_than_5_years , positive_aux_nodes)')
for i,columns in enumerate(list(bcancer_df.columns)[:-1]):

    #print(columns)
    # print(i)
    sns.violinplot(x="surv_status_more_than_5_years",y=columns,data=bcancer_df,size=10)
    plt.title("Fig {0} (surv_status_more_than_5_years , {1})".format(i,columns))
    plt.show()
bcancer_df.plot(kind='scatter',x='positive_aux_nodes',y="op_year")
plt.title("Fig 6.1 (Positive axilliary nodes , opeartion year)")
plt.close()
bva=sns.FacetGrid(bcancer_df,hue="surv_status_more_than_5_years",size=5)
bva.map(plt.scatter,"Patients-age","op_year")
bva.add_legend()
plt.title("Fig 6.2 (Patients-age , Opearation year)")
plt.show()

bva=sns.FacetGrid(bcancer_df,hue="surv_status_more_than_5_years",size=5)
bva.map(plt.scatter,"Patients-age","positive_aux_nodes")
bva.add_legend()
plt.title("Fig 6.3 (Patients-age , Positive axilliary nodes)" )
plt.show()

bva=sns.FacetGrid(bcancer_df,hue="surv_status_more_than_5_years",size=5)
bva.map(plt.scatter,"positive_aux_nodes","op_year")
bva.add_legend()
plt.title("Fig 6.4(Positive axilliary nodes , opeartion year)")
plt.show()


sns.pairplot(bcancer_df,hue="surv_status_more_than_5_years",size=4)
plt.show()
