#Assignment On Haberman's Survival Datasets

"""ABOUT THE HABERMAN DATASET

The dataset contains cases from a study that was conducted between 1958 and 1970 at the
University of Chicago's Billings Hospital on the survival of patients who had undergone
surgery for breast cancer.

***Objective***
1.Perform univariate analysis:
  A.Probability Density Function (PDF)
  B.Cumulative Density Function (CDF)
  C.Box-plots
  D.violin Plots

2.Perform Bi-variate analysis:
  A.Scatter plots
  B.Pair-plots

***Attribute Information***

  1.Age_Of_Patient at time of operation (numerical)
  2.Patient's Operation_Year (year - 1900, numerical)
  3.Number of positive Axil_nodes detected (numerical)
  4.Survival_Status (1 = Patient survived 5 years or longer 
                     2 = Patient died within 5 years)"""


#important Packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

report = pd.read_csv('../input/haberman.csv')#dataframe=report
report.head()#for displaying top 5 rows




print(report.shape)#gives the information about no. of data points(rows) and features(coloumns)
print(report.columns)#display our dataset's columns name

report["Survival_Status"].value_counts()#1 = the patient survived 5 years or longer 
                                        #2 = the patient died within 5 years
report.info()# Infomation of data
report.describe()# Description of data
"""=> Univariate Analysis:

  1.Univariate analysis is the simplest form of statistical analysis. 

  2.The examination of the distribution of cases on only ONE VARIABLE 
    at a time. 
    
  3.e.g weight of college students.

=> Portability Disribution Function:

  1.The PDF is used to specify the probability of the random variable 
    falling within a particular range of values.
    
  2.Distribution plots are used to visually assess how the data points
   are distributed w.r.t its frequency.
    
  3.PDFs are used to denote the probability density function.

  4.Here the height of the bar denotes the percentage of data points
   under the corresponding group."""
#showing datasets using HISTOGRAM
import warnings# used to ignore warnings
warnings.filterwarnings("ignore")

sns.FacetGrid(report, hue="Survival_Status", size=5) \
.map(sns.distplot, "Age_Of_Patient") \
.add_legend();
plt.show();
print('Means:')
print("Mean age of patients survived:", 
      round(np.mean(report[report['Survival_Status'] == 1]['Age_Of_Patient'])))
print("Mean age of patients not survived:", 
      round(np.mean(report[report['Survival_Status'] == 2]['Age_Of_Patient'])))


print('\nMedian:')
print("Median age of patients survived:", 
      round(np.median(report[report['Survival_Status'] == 1]['Age_Of_Patient'])))
print("Median age of patients not survived:", 
      round(np.median(report[report['Survival_Status'] == 2]['Age_Of_Patient'])))

#mean value of death patients(52) is more than survived(54) 
#median value of death patients(52) is more than survived(53)
import warnings# used to ignore warnings
warnings.filterwarnings("ignore")

#diaplaying graph w.r.t Patients years
sns.FacetGrid(report, hue='Survival_Status', size=5)\
   .map(sns.distplot,"Operation_Year")\
   .add_legend();

#diaplaying graph w.r.t Axil_nodes
sns.FacetGrid(report, hue='Survival_Status', size=5)\
   .map(sns.distplot,"Axil_nodes")\
   .add_legend();


plt.show();


"""***Observation***
    
    From above three graph, it is very clear 'Axil_nodes' gives far better infotmation than 'Age_of_Patient',
    (DUE TO SEPERATION OF THIER HISTOGRAMS FIGURES).
    
    therefore axil_nodes>Age_of_Patient>Operation_Year."""
      
"""***Cumulative Distribution Function***
    
  1.The cumulative density function of a real-valued random variable X,
     or just distribution function of X,evaluated at x, is the probability
     that X will take a value less than or equal to x. 
        
  2.In the case of a continuous distribution, it gives the area
    under the probability density function.
    
  3.Its give whole area information to the corresponding dataset, more efficient from PDFs.

  4.CDFs are also used to specify the distribution of multivariate random variables."""


#displaying PDF and CDF
#bin:If bins is a string, it defines the method used to calculate the optimal bin width,as defined by histogram_bin_edges.
#density:The result is the value of the probability density function not as y-axis counts
#bin_edges:Return the bin edges (length(hist)+1).
plt.figure(figsize=(20,5))#providing seperate figures

for idx, feature in enumerate(list(report.columns)[:-1]):#for loop
    
    plt.subplot(1, 3, idx+1)
    print("********* "+feature+" *********")
    
    counts, bin_edges = np.histogram(report[feature], bins=10, density=True)
    print("Bin Edges: {}".format(bin_edges))
    
    pdf = counts/sum(counts)
    print("PDF: {}".format(pdf))
    
    cdf = np.cumsum(pdf)
    print("CDF: {}".format(cdf))
    
    plt.plot(bin_edges[1:], pdf, bin_edges[1:], cdf)#displaying histograms
    plt.xlabel(feature)
#The Big advantage of using PDF and CDF together i.e we get together exact information about data.
"""***Observation***
    
  1.From graphs we are getting the information i.e CDF is denoted by Orange color and PDF is denoted by blue color.
    
  2.In these graphs the y-axis points in all the three graphs aretends to 100 value at the end.#(Density=True)
    
  3.We can also observe that:
      a.There are 40% of cancer patients having age <=50
      b.There are 42% of cancer patients having year_of_treatment <=63
      c.There are 40% of cancer patients having positive_lymph_nodes <=12"""
"""***Box Plots And Whiskers***

   1.A Box Plot and whisker plot is a method for graphically depictinggroups of numerical data through their quartiles.
    
   2.Box plots may also have lines extending vertically from the boxes (whiskers) indicating variability outside the 
     upper(MAX VALUE) and lower quartiles(MIN VALUE).
    
   3.BoxPlot is useful when we need infomation in Percentile(Below-25%-50%-75%-Above).

   4.Box and whisker plots are also very useful when large numbers of observations are involved and when two or more
     data sets are being compared.
        

***The outliers are displayed as points outside the box***

   1. Q1 - 1.5*IQR
   2. Q1 (25th percentile)
   3. Q2 (50th percentile or median)
   4. Q3 (75th percentile)
   5. Q3 + 1.5*IQR"""
#Box Plots w.r.t Age_Of_Patient
plt.figure(figsize=(8,5))
sns.boxplot(x='Survival_Status',y="Age_Of_Patient",data=report)


#Box Plots w.r.t Operation_Year
plt.figure(figsize=(8,5))
sns.boxplot(x='Survival_Status',y="Operation_Year",data=report)

#Box Plots w.r.t Axil_nodes
plt.figure(figsize=(8,5))
sns.boxplot(x='Survival_Status',y="Axil_nodes",data=report)
plt.legend()
plt.show()
    
print("Here we can see that Patient with less Axil_nodes have more rate of survival")
"""***Violin plots***
    
    1.A violin plot is a method of plotting numeric data.
      It is similar to box plot with a rotated kernel density plot on each side.
    
    2.A violin plot has four layers
        a)95% confidence interval
        b)Density Plot(Width=Frequency)
        c)Median
        d)Interquartile range
        
    3.It is combination of Boxplot and Histpgram."""
    
#Violin plots w.r.t Age_Of_Patient
plt.figure(figsize=(8,5))
sns.violinplot(x='Survival_Status',y="Age_Of_Patient",data=report)

#Violin plots w.r.t Operation_Year
plt.figure(figsize=(8,5))
sns.violinplot(x='Survival_Status',y="Operation_Year",data=report)

#Violin plots w.r.t Axil_nodes
plt.figure(figsize=(8,5))
sns.violinplot(x='Survival_Status',y="Axil_nodes",data=report)
plt.show()

print('\n***OBSERVATION***')
print("Median Age of patients survived:", 
      round(np.median(report[report['Survival_Status'] == 1]['Age_Of_Patient'])))
print("Median Year of patients not survived:", 
      round(np.median(report[report['Survival_Status'] == 1]['Operation_Year'])))
print("Median Axil_nodes of patients not survived:", 
      round(np.median(report[report['Survival_Status'] == 1]['Axil_nodes'])))
"""***Bi-variate Analysis***
    
    1.Bivariate analysis is the simultaneous analysis of two variables (attributes).
    
    2.It explores the concept of relationship between two variables, whether there exists an association 
      and the strength of this association.
        
    3.The best example of Bi-variate is- Pair plots and Scatter plots."""
#2-D Scatter Plots
print("***Observation***\n1=Age_Of_Patient\n2=Operation_Year\nThis data is not sufficient to analyse any result.")

plt.figure(figsize=(8,5))
sns.FacetGrid(report,hue='Survival_Status',size=4)\
   .map(plt.scatter,"Age_Of_Patient","Operation_Year")\
   .add_legend();

plt.show


print("***Observation***\n1=Age_Of_Patient\n2=Axil_nodes\n(In y-axis)Above 5 and below 30 there excessive amount of Axil_nodes.")
   
plt.figure(figsize=(8,5))
sns.FacetGrid(report,hue='Survival_Status',size=4)\
   .map(plt.scatter,"Age_Of_Patient","Axil_nodes")\
   .add_legend();

plt.show
#3D scattered plot.(resource kaggle)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig=plt.figure()
ax=fig.add_subplot(111, projection='3d')

x=report["Age_Of_Patient"]
y=report["Operation_Year"]
z=report["Axil_nodes"]

ax.scatter(x,y,z,marker='o', c='r');

ax.set_xlabel('Age_Of_Patient')
ax.set_ylabel('Operation_Year')
ax.set_zlabel('Axil_nodes')

plt.show()
"""***Pair plots***
    
    1.A “Pair plots” is also known as a scatterplot, in which one variable in the same data row is matched
      with another variable’s value.
        
    2.Pair plots are just elaborations on showing all variables paired with all the other variables.
        
    3.It  is used to visualize the relationship between two variables.
    
    4.It results best optimisation of data sets by analysing given different pairs.
    
    5.Effective for less than 5 or 6 fetaures(columns)."""
#Pair Plot 
plt.close();#releasing the memory
sns.set_style('whitegrid');
sns.pairplot(report,hue='Survival_Status',size=3,diag_kind='kde');
plt.show()
# Multivariate probability density
sns.jointplot(x= 'Age_Of_Patient',kind = 'kde', y='Operation_Year', data =report)
plt.show()

print("***Observations***\nBy scattering the data points between Age_Of_Patient \nand Axil_nodes, we can see the better seperation\nbetween the two clases than other scatter plots.")

"""***Final Observations***

1. Patient with 0 Axil_nodes have more rate of survival.

2. Patient who gone through surgery between age 45-55 in 1959-1964 can be classified as survived.

3  Patient age between 30-40 can be classified as survived."""
