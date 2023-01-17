# Include pandas to manipulate dataframes and series

import pandas as pd

# Include matplotlib to plot diagrams

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import numpy as np



# Read human resources data from CSV data file

raw_data = pd.read_csv("../input/HR_comma_sep.csv")
# Print the shape of the dataset (number of samples and features)

print(raw_data.shape)

# Print the features of the dataset

raw_features_name = raw_data.columns.tolist()

print(raw_features_name)
raw_data.head()


salary_map = {"low":0, "medium":.5, "high":1}



#given a dataset row, maps the salary from {"low", "medium", "high"} to a range from 0 to 1 

def map_func ( row ):

    return salary_map [ row["salary"] ]



raw_data["numeric_salary"] = raw_data.apply (map_func, axis=1)





# Reorganise the dataset:

#     put the 'left' feature at the end

#     remove "salary" column (keep only "numeric_salary")

#     rename "sales" column to "department"

raw_data = raw_data.rename(columns={'sales': 'department'})

features_name = raw_data.columns.tolist()

features_name.remove('left')

organized_features_name = features_name + ['left']

organized_features_name.remove('salary')



hr_data = raw_data[organized_features_name]



# Print the reorganised dataset first values

hr_data.head()
corr = hr_data.corr()

sns.heatmap(corr, cmap="bwr", vmin=-1, vmax=1, annot=True, fmt=".2f")
time_data = hr_data["time_spend_company"]

sns.boxplot(y=time_data)

time_data.describe()
#'satisfaction_level', 'last_evaluation', 'number_project',

#       'average_montly_hours', 'time_spend_company', 'Work_accident',

#       'promotion_last_5years', 'department', 'numeric_salary', 'left']

sns.distplot(hr_data['satisfaction_level'], kde=False)
N = 10

left_data = []

noleft_data = []

for i in range (N):

    min = i/N

    max = (i+1)/N

    if (i==N-1):

        max = max+0.01

    #print (min, max)

    left_count = hr_data[ (hr_data['left']==1) & (min <= hr_data["satisfaction_level"]) &  (hr_data["satisfaction_level"] < max) ].shape[0]

    noleft_count = hr_data[ (hr_data['left']==0) & (min <= hr_data["satisfaction_level"]) &  (hr_data["satisfaction_level"] < max) ].shape[0]

    left_data.append (left_count)

    noleft_data.append (noleft_count)

    

#print (left_data)

#print (noleft_data)

#print (sum(left_data) + sum(noleft_data))

#print (hr_data.shape[0])



ind = np.arange(N)    # the x locations for the groups

width = 0.35       # the width of the bars: can also be len(x) sequence



p1 = plt.bar(ind, left_data, width, color='#d62728')

p2 = plt.bar(ind, noleft_data, width, bottom=left_data)



plt.ylabel('Left')

plt.title('Left by satisfaction level')

plt.xticks(ind, [str(i/N)+"-"+str((i+1)/N) for i in ind])

plt.legend((p1[0], p2[0]), ('left', 'no left'))



plt.show()

hr_data["satisfaction_level"].head()
