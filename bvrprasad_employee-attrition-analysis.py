import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import matplotlib.gridspec as gridspec

from matplotlib import colors



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Load the data and have a quick peek at it using info, describe and head in order to 

# understand what we are dealing with

df = pd.read_csv("../input/HR_comma_sep.csv")



df.info()

# info() reveals that most of the data is of numberical type, except for sales and salary
# And a quick peek using head

df.head()
# And describe - to get first statistics

# Work_accident and promotion_last_5years have only one value, which is also the max value.

# So, these two parameters might not have significant influence on attrition

df.describe().T
# Taking a quick look at the unique values in sales and salary reveals the following

## The sales column seems to store the department information

print(df.sales.unique().tolist())



## There are only three range of salaries provided

print(df.salary.unique().tolist())
# Taking a quick look at the unique values in salary reveals the following

## And there are only three range of salaries provided

df.salary.unique().tolist()
# It's time for some plot to explore the data at hand



# This function plots a violin plot (as it also gives the density) of the feature (what) against 'left'.

def plotv(df, what):

    f, ax = plt.subplots(nrows=1, ncols=2)

    stayed = df[df.left==0][what]

    left = df[df.left == 1][what]

    ax[0].violinplot(stayed.values.tolist(), showmeans=False, showmedians=True)

    ax[0].set_title('Stayed:' + what)

    ax[1].violinplot(left.values.tolist(), showmeans=False, showmedians=True)

    ax[1].set_title('Left:' + what)

    plt.show()



def plotVals():

    # Plot all numerical columns against the target column, which is 'left' in this case

    for colname in df.columns:

        if colname not in ['left', 'sales', 'salary']:

            plotv(df, colname)



plotVals()

# Here are some of the key observations



## 1. Satisfaction level:: is less densed at the lower values and seems to be 

## somewhat evenly spread starting around 0.4 and higher.

### Lower satisfaction (around 0.4) level seems to influence leaving

### Ironically higher satisfaction levels (0.7 and above) also has minor effect on leaving. 

#### Might be that these people are looking for more challenges.



## 2. Last evaluation:: the plot seems to be similar to the satisfaction level, except that the

## lower ranges are not applicable



## 3. Number of projects:: leaving is more densed on the lesser values of 2 and correspondingly, 

## staying at these levels is significantly very minimum



## 4. Average monthly hours:: here values between 170 and 210 tend not to leave that much. And values

## outside this range tend to leave significantly



## 5. Time spend company:: this seems to be the years spent in the company. Except for small 

## differences both the plots look similar. Only signifacnt observations are::

### People tend to stay at least for 2.5 years and if they stayed for more than 6 years, 

### then they dont leave



## 6. Work accident:: is populated only around zero & one. This column might be a binary value!

### Both plots are equally densed around zero. Though the dense for one is more for people who 

### stayed back. So, very few people had an accident and out of this very very few people left.



## 7. Promotion last 5 years:: is populated only around zero & one. This column might be a binary value!

### The values are equally densed around zero. So, no promotion seems to have no significant influence.

### Though there is tiny dense around one for staying!
# Now let us also explore the non-numerical parameters:: salary and sales (departments)

# I will output the values, but for the plots I will use the percentages, as they give a

# better observation.



def plot_stacked(colname):

    dfg = df.groupby([colname, 'left']).agg({'left':'count'})

    print(dfg)

    vals = [i[0] for i in dfg.values]

    stayed = vals[::2]

    left = vals[1::2]

    

    # Calculate the percentage values for better observation

    stayedp = []

    leftp = []

    for i in range(len(stayed)):

        total = stayed[i]+left[i]

        stayedp.append(float(stayed[i]/total))

        leftp.append(float(left[i]/total))

        

    levels = dfg.index.levels[0].tolist()

    N = len(levels)

    ind = np.arange(N)

    width = 0.25

    p1 = plt.bar(ind, stayedp, width, color = 'g')

    p2 = plt.bar(ind, leftp, width, color = 'r', bottom=stayedp)

    plt.ylabel('Stayed/Left')

    plt.ylim([0,1.2])

    plt.title('Stayed vs Left based on: ' + colname)

    plt.xticks(ind + width/2., levels, rotation=45)

    plt.legend((p1[0], p2[0]), ('Stayed', 'Left'))

    fig = plt.gcf()

    fig.set_size_inches(9, 5, forward=True)

    plt.show()



plot_stacked('salary')

plot_stacked('sales')



# Here are some of the observations



## 1. Salary:: is only provided in levels low, medium and high

### very few people earning higher salary have left

### more people earning lower salary have left than the one earning medium salary



## 2. Sales (department):: there are totally 10 departments, with varying number of 

## employees - sales being the highest.

### Management and RandD seems to leave less. RandD - is an interesting name for a department. Do we need to treat this special? 

### HR and accounting tend to leave more.

### Others departments seems to have similar pattern
# Let's split the data and train for some prediction

columns = df.columns.tolist()

target = 'left'

# I will initially drop the target plus ...

# least significant columns promotion_last_5years, work_accident

# and also salary and sales, as I need to figure out how to replace these with levels

columns = [col for col in columns if col not in [target, 'sales', 'salary', 'promotion_last_5years', 'work_accident']]

print(columns)
from sklearn.model_selection import train_test_split



# Take 70% of the data for train set

train = df.sample(frac=0.7, random_state=1)

train.describe().T
# The remaining 30% for the test set

test = df[~df.index.isin(train.index)]

test.describe().T
# Try with random forest

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error



# Initialize the model with some parameters.

model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)



# Fit the model to the data.

model.fit(train[columns], train[target])



# Make predictions.

predictions = model.predict(test[columns])



# Compute the error.

mean_squared_error(predictions, test[target])



# Observation:: the prediction error seems to be significantly less (< 0.02).

# This is without using the salary, department, promotion_last_5years and work_accident.

# As already observed in the exploratory analysis, salary is a significant feature for high income

# employees. I will make one more try including that column after transforming it's value to levels.