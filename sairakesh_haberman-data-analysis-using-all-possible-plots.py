#Import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
#load the data set
haberman=pd.read_csv('../input/haberman.csv')
haberman
haberman.head()
#assign coloumn names manually
haberman.columns = ["Age", "Year of operation", "axillary nodes", "Survival status after 5 years"]
haberman.columns
#Top 5 data elements with header
haberman.head()
# Get how many data elements and columns are there
haberman.shape
# Get  data type of variables
haberman.dtypes
# modify the target column values to be meaningful as well as categorical
haberman['Survival status after 5 years'] = haberman['Survival status after 5 years'].map({1:"survived", 2:"not survived"})
haberman['Survival status after 5 years'] = haberman['Survival status after 5 years'].astype('category')
haberman.head()
print(haberman.iloc[:,-1].value_counts())
haberman.dtypes
haberman["Age"].value_counts()[:10]
#PDF AND HISTOGRAM PLOTS
for idx, feature in enumerate(list(haberman.columns)[:-1]):
    sns.set_style(style="whitegrid")
    sns.FacetGrid(haberman,hue=('Survival status after 5 years'),size=5)\
             .map(sns.distplot,feature)\
             .add_legend()
    plt.xlabel(feature)         
    plt.ylabel('PDF')
    plt.legend()
    plt.title("survival status with respective {}".format(feature))
plt.show()
#cdf 
#plot cdf for age
for idx, feature in enumerate(list(haberman.columns)[:-1]):
    count,bin_edges=np.histogram(haberman[feature],bins=10,density=True)
    print("Bin Edges: {}".format(bin_edges))
    pdf = count/sum(count)
    print("PDF: {}".format(pdf))
    cdf=np.cumsum(pdf)
    plt.plot(bin_edges[1:],pdf)
    plt.plot(bin_edges[1:],cdf)
    plt.xlabel(feature)
    plt.ylabel('PDF')
    plt.legend(['Pdf for the patients who dead Within 5 years',
            'Cdf for the patients who dead within 5 years'])
    plt.show()
#Statistical Description data of people who survived more than 5 years
haberman_Survived = haberman[haberman["Survival status after 5 years"] == 'survived']
print ("Summary of patients who are survived more than 5 years")
haberman_Survived.describe()
#Statistical Description data of people who survived less than 5 years
haberman_notSurvived = haberman[haberman["Survival status after 5 years"] == 'not survived']
print ("Summary of patients who are survived not more than 5 years")
haberman_notSurvived.describe()
haberman['Age'].mean()
haberman.Age[haberman.Age == haberman.Age.max()]
haberman.Age[haberman.Age == haberman.Age.min()]
#box plots
for idx, feature in enumerate(list(haberman.columns)[:-1]):
    sns.boxplot(x="Survival status after 5 years",y=feature,data=haberman)
    plt.show()
#violin plot
for idx, feature in enumerate(list(haberman.columns)[:-1]):
    sns.violinplot(x="Survival status after 5 years",y=feature,data=haberman,size=8)
    plt.show()
    
#PAIR PLOTS
sns.pairplot(haberman, hue='Survival status after 5 years', size=4)

plt.legend(['not survived','survived'])
plt.show()
#contour plot
sns.jointplot(x="Age",y="Year of operation",data=haberman,kind='kde')
plt.show()