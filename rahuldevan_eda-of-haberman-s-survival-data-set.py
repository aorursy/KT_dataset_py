import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

df = pd.read_csv('../input/haberman.csv',header=None)
#df.head()
#Since file do not have column headers these are inserted
#Col 0 -"Age" Age of Patients
#col 1 -"YearOfOperation" - year in which patient was operated 19XX
#col 2 - "PoistiveCancerNodes" - Number of positive axillary lymph node is a lymph node in 
#                                the area of the armpit (axilla) to which cancer has spread
#col 3 - "Survival status "

#https://stackoverflow.com/questions/34091877/how-to-add-header-row-to-a-pandas-dataframe

df.columns = ["Age","YearOfOperation","PoistiveCancerNodes","SurvivalStatus"]
#df.head()
DataPoints, Features = df.shape
l= df["SurvivalStatus"].unique()
x,y= df["SurvivalStatus"].value_counts()
print ("Total number of Datapoints or patients records -",DataPoints,'\n\n'
      'Total Number Features or Independent Variable -',Features-1,'\n')

#        'The patient survived 5 years or longer has class label' ,y)
print('The patient survived 5 years or longer has class label \"{}\" and \"{}\" datapoint \n'.format(l[0],x))
print('The patient died within 5 year has class label \"{}\" and \"{}\" datapoints'.format(l[1],y))
      
#Converting Our data in two blocks
#1. for morethan 5 years
#2 . for less than 5 years
MoreThan5 = df.loc[df["SurvivalStatus"]==1]
MoreThan5 = MoreThan5.iloc[:,:-1]
LessThan5 =df.loc[df["SurvivalStatus"]==2]
LessThan5 = LessThan5.iloc[:,:-1]

#Median 
from statsmodels import robust
print("Statistics for the patient survived 5 years or longer")
print("-----------------------------------------------------")

print("Median Age -\t\t\t",int(np.median(MoreThan5.Age)),'Years')
print("MAD of Age -\t\t\t",robust.mad(MoreThan5.Age))
print("90 percentile age -\t\t",int(np.percentile(MoreThan5.Age,90)))
print("IQR [25,75,100]percentile age -\t",np.percentile(MoreThan5.Age,np.arange(25, 125, 25)))
print('\n')
print("Median YearOfOperation -\t",int(np.median(MoreThan5.YearOfOperation)))
print('\n')
print("Median Poistive Cancer Nodes -\t\t\t\t",int(np.median(MoreThan5.PoistiveCancerNodes)))
print("MAD Poistive Cancer Nodes -\t\t\t\t",robust.mad(MoreThan5.PoistiveCancerNodes))
print("90 percentile Poistive Cancer Nodes -\t\t\t",int(np.percentile(MoreThan5.PoistiveCancerNodes,90)))
print("95 percentile Poistive Cancer Nodes -\t\t\t",int(np.percentile(MoreThan5.PoistiveCancerNodes,95)))
print("IQR [25,75,100]percentile Poistive Cancer Nodes -\t",np.percentile(MoreThan5.PoistiveCancerNodes,np.arange(25, 125, 25)))

print('\n\n')
from statsmodels import robust
print("Statistics for the patient survived less than5 years")
print("-----------------------------------------------------")
print("Median Age -\t\t\t",int(np.median(LessThan5.Age)),'Years')
print("MAD of Age -\t\t\t",robust.mad(LessThan5.Age))
print("90 percentile age -\t\t",int(np.percentile(LessThan5.Age,90)))
print("IQR [25,75,100]percentile age -\t",np.percentile(LessThan5.Age,np.arange(25, 125, 25)))
print('\n')
print("Median Year Of Operation -\t",'19'+str(int(np.median(LessThan5.YearOfOperation))))
print('\n')
print("Median Poistive Cancer Nodes -\t\t\t\t",int(np.median(LessThan5.PoistiveCancerNodes)))
print("MAD Poistive Cancer Nodes -\t\t\t\t",robust.mad(LessThan5.PoistiveCancerNodes))
print("90 percentile Poistive Cancer Nodes -\t\t\t",int(np.percentile(LessThan5.PoistiveCancerNodes,90)))
print("95 percentile Poistive Cancer Nodes -\t\t\t",int(np.percentile(LessThan5.PoistiveCancerNodes,95)))
print("IQR [25,75,100]percentile Poistive Cancer Nodes -\t", 
      np.percentile(LessThan5.PoistiveCancerNodes,np.arange(25, 125, 25)));

# Analysis Using Age
c=131
bins=6

for i in MoreThan5.columns:
    counts, bin_edges = np.histogram(MoreThan5[i], bins=bins, density = True)
    pdf = counts/(sum(counts))
#     print(pdf);
#     print(bin_edges);
    
    cdf = np.cumsum(pdf)
    plt.figure(1)
    plt.subplot(c)
    plt.title("MoreThan5year-"+i)
    plt.plot(bin_edges[1:],pdf,label="pdf")
#     plt.legend("PDF",loc=1)
    plt.plot(bin_edges[1:], cdf,label="cdf")
    plt.legend(loc=1)
    
    plt.subplots_adjust(top=0.92, bottom=0.5, left=4, right=6, hspace=0.25,wspace=0.35)
    c+=1
    
      
    

c=131
for i in LessThan5.columns:
    counts, bin_edges = np.histogram(LessThan5[i], bins=bins, density = True)
    pdf = counts/(sum(counts))
#     print(pdf);
#     print(bin_edges);

    cdf = np.cumsum(pdf)
    
    plt.figure(2)
    plt.subplot(c)
    plt.title("LessThan5year-"+i)
    plt.plot(bin_edges[1:],pdf,label="pdf")
#     plt.legend("PDF",loc=1)
    plt.plot(bin_edges[1:], cdf,label="cdf")
    plt.legend(loc=1)
    
    plt.subplots_adjust(top=0.92, bottom=0.5, left=4, right=6, hspace=0.25,wspace=0.35)
    c+=1

#Box plots
# refernce --https://stackoverflow.com/questions/41384040/subplot-for-seaborn-boxplot
# reference -- https://jovianlin.io/data-visualization-seaborn-part-2/
# g = sns.FacetGrid(df, col="day", size=4, aspect=.5)

f, axes = plt.subplots(1, 3, figsize=(15, 7))
plt.subplots_adjust(top=0.85, wspace=0.3)
sns.boxplot(x='SurvivalStatus',y="Age", data=df,orient='v',ax=axes[0])
sns.boxplot(x='SurvivalStatus',y="YearOfOperation", data=df,orient='v',ax=axes[1])
sns.boxplot(x='SurvivalStatus',y="PoistiveCancerNodes", data=df,orient='v',ax=axes[2])

#Box plots
# refernce --https://stackoverflow.com/questions/41384040/subplot-for-seaborn-boxplot
# reference -- https://jovianlin.io/data-visualization-seaborn-part-2/
# g = sns.FacetGrid(df, col="day", size=4, aspect=.5)

f, axes = plt.subplots(1, 3, figsize=(15, 7))
plt.subplots_adjust(top=0.85, wspace=0.3)
sns.violinplot(x='SurvivalStatus',y="Age", data=df,orient='v',ax=axes[0])
sns.violinplot(x='SurvivalStatus',y="YearOfOperation", data=df,orient='v',ax=axes[1])
sns.violinplot(x='SurvivalStatus',y="PoistiveCancerNodes", data=df,orient='v',ax=axes[2])
#pair wise plot
# cols =["Age","YearOfOperation","PoistiveCancerNodes","SurvivalStatus"]
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df,hue="SurvivalStatus", size=3);
plt.show()





#  )
# fig = pp.fig 
# fig.subplots_adjust(top=0.93, wspace=0.3)
# fig.suptitle('Wine Attributes Pairwise Plots', fontsize=14, fontweight='bold')
# cols =["Age","YearOfOperation","PoistiveCancerNodes","SurvivalStatus"]
sns.FacetGrid(df, hue="SurvivalStatus", size=5) \
   .map(sns.distplot, "Age") \
   .add_legend();
plt.show();

sns.FacetGrid(df, hue="SurvivalStatus", size=5) \
   .map(sns.distplot, "PoistiveCancerNodes") \
   .add_legend();
plt.show();

sns.FacetGrid(df, hue="SurvivalStatus", size=5) \
   .map(sns.distplot, "YearOfOperation") \
   .add_legend();
plt.show();
print("Plot for More than 5 year - relationship between PoistiveCancerNodes V/s Age ")
sns.jointplot(x="PoistiveCancerNodes", y="Age", data=MoreThan5, kind="kde");
# sns.jointplot(x=LessThan5['Age'], y=MoreThan5['Age'], kind="kde");
plt.show();

sns.jointplot(x="YearOfOperation", y="Age", data=MoreThan5, kind="kde");
plt.show();

# sns.jointplot(x="YearOfOperation", y="PoistiveCancerNodes", data=MoreThan5, kind="kde");
# plt.show();

print("Plot for Less than 5 year - relationship between PoistiveCancerNodes V/s Age ")
sns.jointplot(x="PoistiveCancerNodes", y="Age", data=LessThan5, kind="kde");
# sns.jointplot(x=LessThan5['Age'], y=MoreThan5['Age'], kind="kde");
plt.show();

sns.jointplot(x="YearOfOperation", y="Age", data=LessThan5, kind="kde");
plt.show();

# sns.jointplot(x="YearOfOperation", y="PoistiveCancerNodes", data=LessThan5, kind="kde");
# plt.show();

