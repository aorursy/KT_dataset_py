"""
Last amended: 16th May, 2020

Objectives:

          1. Understanding basics of matplotlib
          2. Learn to use jupyter markdown script 
          3. Uses data of einstein city - uncover/einstein/diagnosis-of-covid-19-and-its-clinical-spectrum.csv


"""
# 1.0 Call libraries
%reset -f
# 1.1 For data manipulations
import numpy as np
import pandas as pd
# 1.2 For plotting
import matplotlib.pyplot as plt
#import matplotlib
#import matplotlib as mpl     # For creating colormaps
import seaborn as sns
# 1.3 For data processing
from sklearn.preprocessing import StandardScaler
# 1.4 OS related
import os


# 1.5 Display multiple outputs from a jupyter cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# 1.6 Read Dataset & display dtypes
ad = pd.read_csv("../input/uncover/einstein/diagnosis-of-covid-19-and-its-clinical-spectrum.csv")
ad.dtypes
# 1.7 display dataset related information
ad.head(3)
ad.info()               # Also informs how much memory dataset takes and status of nulls
ad.shape                # (5644, 111)
ad.columns.values       # Dislay column names
len(ad)                 # Also give number of rows 5644
# 1.8 Patient Age Quantile data value counts
len(ad.patient_age_quantile.unique())                   # 20 - number of unique values out of 5644 rows
ad.patient_age_quantile.value_counts()                  # Quantile wise counts
# 1.9 sars_cov_2_exam_result data value counts
len(ad.sars_cov_2_exam_result.unique())                # 2 - number of Unique values in the column
ad.sars_cov_2_exam_result.value_counts()               # Resilt wise counts - 5086 -Ve; 558 +Ve
#1.10 Some more counts
ad.patient_addmited_to_regular_ward_1_yes_0_no.value_counts()        # 79 - No. of patients admitted to Regular ward
ad.patient_addmited_to_semi_intensive_unit_1_yes_0_no.value_counts() # 50 - No. of patients admitted to Semi ICU 
ad.patient_addmited_to_intensive_care_unit_1_yes_0_no.value_counts() # 41 - No. of patients admitted to ICU 
ad.hemoglobin.value_counts()                                         # 84 - Count of haemoglobin samples - values wise
len(ad.hemoglobin.unique())                                          # 85 - Unique haemoglobin data in the column

#############################
# 2.0 Engineering  features
#############################
# 2.1 Descretise continuos columns
# Haemoglobin is cut to three buckets l - Low, n - Normal, g - Good
ad["hemoglobin_range"] = pd.cut(
                       ad['hemoglobin'],
                       bins = 3,           
                       labels= ["l", "n", "g"]
                      )
#2.2 Display new column "hemoglobin_range"
ad.head(10)                          #2.2 Display new column "hemoglobin_range"
ad.hemoglobin_range.value_counts()   # Count of haemoglobin samples - range wise - 
len(ad.hemoglobin_range.unique())    # 4 - Unique haemoglobin_range data in the column
##################
# 3 Plotting
##################

# Question 3.1: How is Age distributed? 
# Age quantile is well spread out
sns.distplot(ad.patient_age_quantile)
sns.despine() # Plot  without spine

# 3.2 Distribution of Age - More plot configurations are demonstrated by setting the attributes of the graphs
ax= sns.distplot(ad.patient_age_quantile)
ax.set( xlim =(-100,30),                    # The axis is shifted to -10 to bring the plot to center
        xlabel= "age of persons",           # X axis is given name instead of column name
        ylabel = "Density",                 # Y axis is given name
        title= "Density of Age",            # Title is given to the plot
        xticks = list(range(0, 20, 2))      # The X range is reduced from default 5 to 2
        )
# 4.0 Using Series.map() method along with a function for data transformation 
# 4.1 'Patient_addmited_to_regular_ward_1_yes_0_no' is transformed from String to Numeric into a new column "isregularward"
# The value 'f' is assigned '0' and the value 't' is assigned '0'
def isregularward(x):
    if x == "f":
        return 0            # No
    if x == "t":
        return 1            #Yes
    
ad['isregularward'] = ad['patient_addmited_to_regular_ward_1_yes_0_no'].map(lambda x : isregularward(x))   # Regular Ward
#4.1 continued - display the new column data type, unique value wise counts, unique values and count of unique values
ad.dtypes["patient_addmited_to_regular_ward_1_yes_0_no"]
ad.dtypes["isregularward"]
ad.isregularward.value_counts() 
ad.isregularward.unique()
len(ad.isregularward.unique()) 
# 4.2 'patient_addmited_to_semi_intensive_unit_1_yes_0_no' from String to Numeric into a new column "issemiicu"
# The value 'f' is assigned '0' and the value 't' is assigned '0'

def issemiicu(x):
    if x == "f":
        return 0            # No
    if x == "t":
        return 1            #Yes
    
ad['issemiicu'] = ad['patient_addmited_to_semi_intensive_unit_1_yes_0_no'].map(lambda x : issemiicu(x))   # Semi ICU Ward
#4.2 continued - display the new column data type, unique value wise counts, unique values and count of unique values
ad.dtypes["patient_addmited_to_semi_intensive_unit_1_yes_0_no"]
ad.dtypes["issemiicu"]
ad.issemiicu.value_counts() 
ad.issemiicu.unique()
len(ad.issemiicu.unique()) 
# 4.3 'patient_addmited_to_intensive_care_unit_1_yes_0_no' from String to Numeric into a new column "isicu"
# The value 'f' is assigned '0' and the value 't' is assigned '0'

def isicu(x):
    if x == "f":
        return 0            # No
    if x == "t":
        return 1            #Yes
    
ad['isicu'] = ad['patient_addmited_to_intensive_care_unit_1_yes_0_no'].map(lambda x : isicu(x))   # ICU Ward
#4.3 continued - display the new column data type, unique value wise counts, unique values and count of unique values
ad.dtypes["patient_addmited_to_intensive_care_unit_1_yes_0_no"]
ad.dtypes["isicu"]
ad.isicu.value_counts() 
ad.isicu.unique()
len(ad.isicu.unique()) 
# 4.4 'sars_cov_2_exam_result' from String to Numeric into a new column "iscovidpositive"
# The value 'f' is assigned '0' and the value 't' is assigned '0'

def iscovidpositive(x):
    if x == "negative":
        return 0            # No
    if x == "positive":
        return 1            #Yes
    
ad['iscovidpositive'] = ad['sars_cov_2_exam_result'].map(lambda x : iscovidpositive(x))   # Is the patient COVID +Ve
#4.4 continued - display the new column data type, unique value wise counts, unique values and count of unique values
ad.dtypes["sars_cov_2_exam_result"]
ad.dtypes["iscovidpositive"]
ad.iscovidpositive.value_counts() 
ad.iscovidpositive.unique()
len(ad.iscovidpositive.unique()) 
# 5.0 Draw multiple Distribution Plots using for loops
# 5.1 To draw distribution of 'patient_age_quantile' and 'hemoglobin' using for loop

# Using for loop to plot all at once
columns = ['patient_age_quantile', 'hemoglobin'] # Create a list of columns 
fig = plt.figure(figsize = (10,10))              # define the figure size
for i in range(len(columns)):                    # number of times the for loop runs i.e. the length of the list
    plt.subplot(2,2,i+1)
    sns.distplot(ad[columns[i]])
    
# The hemoglobin distribution seems to be slightly right skewed
# 6.0 Relationship of numeric variable with a categorical variable
# 6.1 Box Plot of relationship of 'patient_age_quantile' with 'iscovidpositive'
sns.boxplot(x = 'iscovidpositive',       # Discrete
            y = 'patient_age_quantile',  # Discrete
            data = ad
            )

# Median Age Quantile : For iscovidpositive = 0 --> around 9.5 : iscovidpositive = 1 --> around 11
# IQR   : For iscovidpositive = 0 --> around 10 quantiles : iscovidpositive = 1 --> around 7 quantiles 
# With reference to IQR - confirmed +Ve cases are between  quantile 7 to 15
# 6.2 More such relationships through Box plots using for-loop
columns = ['patient_age_quantile', 'isregularward', 'issemiicu', 'isicu'] # list for Y axis
catVar = ['sars_cov_2_exam_result', 'hemoglobin_range' ]                  # list for X axis


# 6.3 Now for loop. First create pairs of cont and cat variables
mylist = [(cont,cat)  for cont in columns  for cat in catVar]
mylist

# 6.4 Now run-through for-loop
fig = plt.figure(figsize = (10,10))
for i, k in enumerate(mylist):
    #print(i, k[0], k[1])
    plt.subplot(4,2,i+1)
    sns.boxplot(x = k[1], y = k[0], data = ad)

# Interpretations from the graphs
# (1,2) - The hemoglobin is low in the age quantiles of 10 and above; however it is normal and good in case of quantiles less than 15
# (3,2) & (4,2) - Hospitalisation in Semi-ICU and ICU is seen in cases of low Hemoglobin

# 6.5 More such relationships through Box plots with "Notch" using for-loop
columns = ['patient_age_quantile', 'isregularward', 'issemiicu', 'isicu']
catVar = ['sars_cov_2_exam_result', 'hemoglobin_range' ]


# 6.6 Now for loop. First create pairs of cont and cat variables
mylist = [(cont,cat)  for cont in columns  for cat in catVar]
mylist

# 6.7 Now run-through for-loop with NOTCH
fig = plt.figure(figsize = (10,10))
for i, k in enumerate(mylist):
    #print(i, k[0], k[1])
    plt.subplot(4,2,i+1)
    sns.boxplot(x = k[1], y = k[0], data = ad, notch = True)
    
# Interpretations from the graphs
# (1,2) - The hemoglobin is low in the age quantiles of 10 and above; 
#     however it is normal and good in case of quantiles less than 15
# (3,2) & (4,2) - Hospitalisation in Semi-ICU and ICU is seen in cases of low Hemoglobin; 
#     The median was not visible in plain box plots and is visible in box plots with notch

# 7.0 Joint Plots
# 7.1 An example of joint plot between Patient Age Quantile & Admitted to Regular ward
sns.jointplot(ad.patient_age_quantile, ad.isregularward)
# Nothing may be interpreted with this plot
# 7.2 Joint plot
# patient_age_quantile Vs hemoglobin

# In this plot no interpretation could be drawn
sns.jointplot(ad.patient_age_quantile, ad.hemoglobin) 

# Maximum patients appear to be having normal hemoglobin and in the age quantile around 15
sns.jointplot(ad.patient_age_quantile, ad.hemoglobin, kind = "kde") 

# In this plot no interpretation could be drawn
sns.jointplot(ad.patient_age_quantile, ad.hemoglobin, kind = "hex")

# 7.3 Joint plot
# iscovidpositive Vs hemoglobin

# In this plot no interpretation could be drawn
sns.jointplot(ad.iscovidpositive, ad.hemoglobin)

# In this plot no interpretation could be drawn
sns.jointplot(ad.iscovidpositive, ad.hemoglobin, kind = "hex")
# 7.4 Joint plot
# hemoglobin Vs iscovidpositive
sns.jointplot(ad.hemoglobin, ad.iscovidpositive)

sns.jointplot(ad.hemoglobin, ad.iscovidpositive, kind = "hex")

# 8.0 Relationship of a categorical to another categorical variable
#catVar = ['sars_cov_2_exam_result', 'hemoglobin_range' ]
sns.barplot(x = 'sars_cov_2_exam_result',
            y = 'hemoglobin',
            estimator = np.mean,      # As there are multiple occurrences of sars_cov_2_exam_result, sum up 'hemoglobin_range'
            ci = 95,                 # Estimate default confidence interval using bootstrapping
            data = ad
            )
#9.0 Relationship between two categorical and one numeric variable
grouped = ad.groupby(['sars_cov_2_exam_result', 'hemoglobin_range'])
df_wh = grouped['patient_age_quantile'].sum().unstack()
df_wh

sns.heatmap(df_wh, cmap = plt.cm.Blues)

# Maximum patients lie in the block Covid -Ve and normal hemoglobin