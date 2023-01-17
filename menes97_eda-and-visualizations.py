
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")  # load data
data.info()
#rename columns
data.rename(columns=({'gender':'Gender','race/ethnicity':'Race/Ethnicity'
                     ,'parental level of education':'Parental_Level_of_Education'
                     ,'lunch':'Lunch','test preparation course':'Test_Preparation_Course'
                      ,'math score':'Math_Score','reading score':'Reading_Score'
                     ,'writing score':'Writing_Score'}),inplace=True)
data.columns
data.head()
data.describe().T
data.columns
def bar_plot(variable):
  

    # get feature
    var = data[variable]
    # count number of categorical variable(value/sample)
    varValue = var.value_counts()
    
    # visualize
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))
category1= ['Gender', 'Race/Ethnicity', 'Parental_Level_of_Education', 'Lunch',
       'Test_Preparation_Course']

for i in category1 :
    bar_plot(i)
def plot_hist(variable) :
    plt.figure(figsize=(9,3))
    plt.hist(data[variable], bins = 20)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("distribution with hist".format(variable))
    plt.show()

numericvar = ['Math_Score', 'Reading_Score',
       'Writing_Score']
for i in numericvar:
    plot_hist(i)
        
from pandas.api.types import CategoricalDtype
race_cat = ["group A","group B","group C","group D","group E"]
data["Race/Ethnicity"] =data["Race/Ethnicity"].astype(CategoricalDtype(categories = race_cat, ordered = True))
data["Race/Ethnicity"].head(1)
data[['Race/Ethnicity',
      'Math_Score',
      'Reading_Score',
      'Writing_Score']].groupby(['Race/Ethnicity']).agg('median')
data[['Parental_Level_of_Education',
      'Math_Score',
      'Reading_Score',
      'Writing_Score']].groupby(['Parental_Level_of_Education']).agg('median').sort_values(by="Math_Score",ascending = False)
data[['Gender',
      'Math_Score',
      'Reading_Score',
      'Writing_Score']].groupby(['Gender']).agg('median').sort_values(by="Math_Score",ascending = False)
data[['Lunch',
      'Math_Score',
      'Reading_Score',
      'Writing_Score']].groupby(['Lunch']).agg('median').sort_values(by="Math_Score",ascending = False)
data[['Test_Preparation_Course',
      'Math_Score',
      'Reading_Score',
      'Writing_Score']].groupby(['Test_Preparation_Course']).agg('median').sort_values(by="Math_Score",ascending = False)
Score = ['Math_Score', 'Reading_Score',
       'Writing_Score']
Category = ['Gender', 'Race/Ethnicity', 'Parental_Level_of_Education', 'Lunch',
       'Test_Preparation_Course']
for i in Category:
    for x in Score:
        print(data[[i,x]].groupby([i], as_index = False).mean().sort_values(by=x,ascending = False),"\n")
    
data.columns
plt.subplot(1, 3, 1)
sns.distplot(data['Math_Score'])

plt.subplot(1, 3, 2)
sns.distplot(data['Reading_Score'])

plt.subplot(1, 3, 3)
sns.distplot(data['Writing_Score'])

plt.show()

ax1 =(sns
 .FacetGrid(data,
              hue= "Race/Ethnicity",
              height = 5,
              xlim = (0,100))
 .map(sns.kdeplot,"Math_Score", shade= True)
 .add_legend()
);


ax2 = (sns
 .FacetGrid(data,
              hue= "Race/Ethnicity",
              height = 5,
              xlim = (0,100))
 .map(sns.kdeplot,"Reading_Score", shade= True)
 .add_legend()
);


ax3 = (sns
 .FacetGrid(data,
              hue= "Race/Ethnicity",
              height = 5,
              xlim = (0,100))
 .map(sns.kdeplot,"Writing_Score", shade= True)
 .add_legend()
);
plt.show()
data.columns
(sns
 .FacetGrid(data,
              hue= "Gender",
              height = 5,
              xlim = (0,100))
 .map(sns.kdeplot,"Math_Score", shade= True)
 .add_legend()
);

(sns
 .FacetGrid(data,
              hue= "Gender",
              height = 5,
              xlim = (0,100))
 .map(sns.kdeplot,"Writing_Score", shade= True)
 .add_legend()
);

(sns
 .FacetGrid(data,
              hue= "Gender",
              height = 5,
              xlim = (0,100))
 .map(sns.kdeplot,"Reading_Score", shade= True)
 .add_legend()
);

sns.catplot(x = "Race/Ethnicity", y = "Math_Score",  hue = "Gender",kind = "point", data = data );
sns.catplot(x = "Race/Ethnicity", y = "Reading_Score",  hue = "Gender",kind = "point", data = data );
sns.catplot(x = "Race/Ethnicity", y = "Writing_Score",  hue = "Gender",kind = "point", data = data );
sns.lmplot( x = "Math_Score" ,y = "Reading_Score" , hue = "Gender", col = "Race/Ethnicity", data =data);
data.columns
with sns.axes_style('white'):
    sns.jointplot("Writing_Score", "Reading_Score", data, kind='kde');

