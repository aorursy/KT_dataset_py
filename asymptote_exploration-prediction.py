import pandas as pd

import numpy as np

import matplotlib.mlab as mlab

import matplotlib.pyplot as plt

%matplotlib inline



plt.style.use('ggplot')



blue_color =  '#7F7FFF'

red_color = '#BF3F7F'

green_color = '#9ACD32'



data = pd.read_csv('../input/HR_comma_sep.csv')



print("Number of rows: {}".format(data.shape[0]))

print("Number of columns: {}\n".format(data.shape[1]))



print("Column Names:")

print("----------------")

for col in data.columns:

    print(col+" ("+str(data[col].dtype)+")")

print("----------------\n")



print("Any NaN values in data: " + str(data.isnull().values.any()))



data.head()
data=data.rename(columns = {'sales':'department'})

data.columns = [x.lower() for x in data.columns]



data['left'] = data['left'].astype('str')

data['work_accident'] = data['work_accident'].astype('str')

data['promotion_last_5years'] = data['promotion_last_5years'].astype('str')



print("Column Names:")

print("----------------")

for col in data.columns:

    print(col+" ("+str(data[col].dtype)+")")

print("----------------")



data.head(8)
import copy



##########################################################

# Create fiure with 4 subplots

f, ax = plt.subplots(2,2,figsize=(14,14))



(ax1, ax2, ax3, ax4) = ax.flatten()



##########################################################

# Bar Chart of Left Column

left_count = data['left'].value_counts()

left_indices = left_count.index.tolist()

left_values = left_count.values.tolist()



if (left_indices[0] == '1'):

    left_indices[0] = 'Left'

    left_indices[1] = 'Stayed'

else:

    left_indices[0] = 'Stayed'

    left_indices[1] = 'Left'

    

y_pos = np.arange(len(left_values))    

bars=ax1.bar(y_pos, left_values, align='center')



bars[0].set_color(green_color)

bars[1].set_color(red_color)



# Add counts on Bars

def autolabel(rects):

    for rect in rects:

        ax1.text(rect.get_x() + rect.get_width()/2.,

                rect.get_y() + rect.get_height()-1000,

                '%d' % int(rect.get_height()),

                ha='center', va='bottom')

autolabel(bars)



# Add Text showing percentage of employees who left

emp_left = left_values[1]

perc_left = emp_left/sum(left_values) * 100.

ax1.text(0.55, 8000, "Turnover Percentage:\n {:.2f}%".format(perc_left), fontsize=11)



ax1.set_xticks(y_pos)

ax1.set_xticklabels(left_indices)

ax1.set_ylabel('Frequency')

ax1.set_title('Employees Status: Stayed vs Left')





##########################################################

# Histogram of Satisfaction Level: I want 20 bins in range (0-1)

ax2.hist(data['satisfaction_level'], bins=20, range=(0,1), alpha=0.5)

ax2.set_title('Histogram: Satisfaction Level')

ax2.set_ylabel('Frequency')

ax2.set_xlabel('Satisfaction-Level')





##########################################################





n, bins, patches = ax3.hist(data['satisfaction_level'], bins=20, range=(0,1), alpha=0.5)

## 

left_in_bins = []

for i in range(len(bins)-1):

    start = bins[i]

    end = bins[i+1]

    

    left_emp = len(data.loc[(data['satisfaction_level']>=start) & (data['satisfaction_level']<end) & (data['left'] == '1')])

    left_in_bins.append(left_emp)





index = 0

for_legend = None

for p in patches:

    patch = copy.copy(p)

    patch.set_height(left_in_bins[index])

    #patch.set_color(red_color)

    patch.set_hatch('//')

    patch.set_alpha(1.0)

    ax3.add_patch(patch)

    if index==1:

        for_legend = patch

    index = index + 1

ax3.set_title('Histogram: Satisfaction-Level with Left-Status')

ax3.set_ylabel('Frequency')

ax3.set_xlabel('Satisfaction-Level')

ax3.legend([for_legend], ['Employees Left'])



##########################################################



n, bins, patches = ax4.hist(data['satisfaction_level'], bins=20, range=(0,1), alpha=0.5)

## 

left_in_bins = []

for i in range(len(bins)-1):

    start = bins[i]

    end = bins[i+1]

    

    left_emp = len(data.loc[(data['satisfaction_level']>=start) & (data['satisfaction_level']<end) & (data['left'] == '1')])

    left_in_bins.append(left_emp)





index = 0

for p in patches:

    patch = copy.copy(p)

    patch.set_height(left_in_bins[index])

    if index in range(3):

        patch.set_color('r')

    elif index in range(7,10):

        patch.set_color('y')

    elif index in range(14,20):

        patch.set_color('b')

        

    else:

        patch.set_color(red_color)

        patch.set_alpha(1.0)

    ax4.add_patch(patch)

    index = index + 1

ax4.set_title('Histogram:Satisfaction-Level with Left-Status Segments')

ax4.set_xlabel('Satisfaction-Level')

ax4.set_ylabel('Frequency')



plt.show()



seg_one_left = sum(left_in_bins[0:3])

print((seg_one_left/emp_left) * 100.)



seg_two_left = sum(left_in_bins[7:10])

print((seg_two_left/emp_left) * 100.)



seg_three_left = sum(left_in_bins[14:20])

print((seg_three_left/emp_left) * 100.)
import copy 







##########################################################

# Create fiure with 3 subplots

f, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(10,25))



##########################################################

# Departments Bar Chart

dept_count = data['department'].value_counts()

dept_indices = dept_count.index.tolist()

dept_values = dept_count.values.tolist()



# Employees left in certain department

emp_left = []

for dept in dept_indices:

    left_emp = len(data.loc[(data['department']==dept) & (data['left'] == '1')])

    emp_left.append(left_emp)



# Percentage of employees who left in certain department

emp_left_perc = [(ai/bi)*100. for ai,bi in zip(emp_left,dept_values)]





y_pos = np.arange(len(dept_values))

bars=ax1.bar(y_pos, dept_values, align='center', color=blue_color,edgecolor='black')

emp_left_bars=ax1.bar(y_pos, emp_left, align='center',color=blue_color,hatch='//',edgecolor='black')



# Add counts on Bars

def autolabel(rects, ax):

    for rect in rects:

        ax.text(rect.get_x() + rect.get_width()/2.,

                rect.get_y() + rect.get_height(),

                "{:d}".format(int(rect.get_height())),

                ha='center', va='bottom')

autolabel(bars, ax1)



# Add percentage on Bars

def autolabel_emp(rects, ax):

    index = 0

    for rect in rects:

        ax.text(rect.get_x() + rect.get_width()/2.,

                rect.get_y() + rect.get_height(),

                "{:.0f}%".format(int(emp_left_perc[index])),

                ha='center', va='bottom')

        index = index + 1

autolabel_emp(emp_left_bars, ax1)

    



ax1.set_xticks(y_pos)

ax1.set_xticklabels(dept_indices)

for tick in ax1.get_xticklabels():

    tick.set_rotation(45)



ax1.set_ylabel('Frequency')

ax1.set_title('Employees who left w.r.t Departments')

ax1.legend((bars[0], emp_left_bars[0]), ('Employees in Dept.', 'Employees Left'))



##########################################################

# Salary Bar Chart



sal_count = data['salary'].value_counts()

sal_indices = sal_count.index.tolist()

sal_values = sal_count.values.tolist()



# Employees left w.r.t salary

emp_left = []

for sal in sal_indices:

    left_emp = len(data.loc[(data['salary']==sal) & (data['left'] == '1')])

    emp_left.append(left_emp)



# Percentage of employees who left in certain salary range

emp_left_perc = [(ai/bi)*100. for ai,bi in zip(emp_left,sal_values)]





y_pos = np.arange(len(sal_values))

bars=ax2.bar(y_pos, sal_values, align='center', color=blue_color,edgecolor='black')

emp_left_bars=ax2.bar(y_pos, emp_left, align='center',color=blue_color,hatch='//',edgecolor='black')



autolabel(bars,ax2)

autolabel_emp(emp_left_bars,ax2)



ax2.set_xticks(y_pos)

ax2.set_xticklabels(sal_indices)

for tick in ax1.get_xticklabels():

    tick.set_rotation(45)



ax2.set_ylabel('Frequency')

ax2.set_title('Employees who left w.r.t Salary')

ax2.legend((bars[0], emp_left_bars[0]), ('Employees in Salary-Range.', 'Employees Left'))



##########################################################

dept_count = data['department'].value_counts()

dept_indices = dept_count.index.tolist()

dept_values = dept_count.values.tolist()



low_sal = []

med_sal = []

high_sal = []

for dept in dept_indices:

    low_sal.append(len(data.loc[(data['department']==dept) & (data['salary'] == 'low')]))

    med_sal.append(len(data.loc[(data['department']==dept) & (data['salary'] == 'medium')]))

    high_sal.append(len(data.loc[(data['department']==dept) & (data['salary'] == 'high')]))



y_pos = np.arange(len(dept_values))

low_bars=ax3.bar(y_pos, low_sal, align='center', color=red_color)

med_bars=ax3.bar(y_pos, med_sal, align='center', color=green_color,bottom=low_sal)

high_bars=ax3.bar(y_pos, high_sal, align='center', color=blue_color,bottom=np.add(low_sal, med_sal))



ax3.set_xticks(y_pos)

ax3.set_xticklabels(dept_indices)

for tick in ax3.get_xticklabels():

    tick.set_rotation(45)





ax3.set_ylabel('Frequency')

ax3.set_title('Departments w.r.t Salary')

ax3.legend((high_bars[0],med_bars[0],low_bars[0]), ('High Salary','Medium Salary','Low Salary'))



plt.show()



##########################################################

# Create fiure with 4 subplots

f, ax = plt.subplots(2,2,figsize=(12,12))



(ax1, ax2, ax3, ax4) = ax.flatten()



##########################################################

# Work-Accident Bar Chart

acc_count = data['work_accident'].value_counts()

acc_indices = acc_count.index.tolist()

acc_values = acc_count.values.tolist()





# Employees left w.r.t Accidents

emp_left = []

for acc in acc_indices:

    left_emp = len(data.loc[(data['work_accident']==acc) & (data['left'] == '1')])

    emp_left.append(left_emp)



# Percentage of employees w.r.t Accidents

emp_left_perc = [(ai/bi)*100. for ai,bi in zip(emp_left,acc_values)]





y_pos = np.arange(len(acc_values))

bars=ax1.bar(y_pos, acc_values, align='center', color=blue_color, edgecolor='black')

emp_left_bars=ax1.bar(y_pos, emp_left, align='center', color=blue_color, hatch='//',edgecolor='black')



#ax1.



# Add counts on Bars

def autolabel(rects, ax):

    for rect in rects:

        ax.text(rect.get_x() + rect.get_width()/2.,

                rect.get_y() + rect.get_height(),

                "{:d}".format(int(rect.get_height())),

                ha='center', va='bottom')

autolabel(bars, ax1)



# Add percentage on Bars

def autolabel_emp(rects, ax):

    index = 0

    for rect in rects:

        ax.text(rect.get_x() + rect.get_width()/2.,

                rect.get_y() + rect.get_height(),

                "{:.0f}%".format(int(emp_left_perc[index])),

                ha='center', va='bottom')

        index = index + 1

autolabel_emp(emp_left_bars, ax1)

    



ax1.set_xticks(y_pos)

#ax1.set_xticklabels(acc_indices)

ax1.set_xticklabels(["No Work-Accident","Work-Accident"])

ax1.set_ylabel('Frequency')

ax1.set_title('Employees who left w.r.t Accidents')

ax1.legend([emp_left_bars[0]], ['Employees Left'])



##########################################################

# Promotion Bar Chart

promo_count = data['promotion_last_5years'].value_counts()

promo_indices = promo_count.index.tolist()

promo_values = promo_count.values.tolist()





# Employees left w.r.t promotion

emp_left = []

for p in promo_indices:

    left_emp = len(data.loc[(data['promotion_last_5years']==p) & (data['left'] == '1')])

    emp_left.append(left_emp)



# Percentage of employees w.r.t promotion

emp_left_perc = [(ai/bi)*100. for ai,bi in zip(emp_left,promo_values)]





y_pos = np.arange(len(promo_values))

bars=ax2.bar(y_pos, promo_values, align='center', color=blue_color, edgecolor='black')

emp_left_bars=ax2.bar(y_pos, emp_left, align='center', color=blue_color, hatch='//',edgecolor='black')



# Add counts on Bars

def autolabel(rects, ax):

    for rect in rects:

        ax.text(rect.get_x() + rect.get_width()/2.,

                rect.get_y() + rect.get_height(),

                "{:d}".format(int(rect.get_height())),

                ha='center', va='bottom')

autolabel(bars, ax2)



# Add percentage on Bars

def autolabel_emp(rects, ax):

    index = 0

    for rect in rects:

        ax.text(rect.get_x() + rect.get_width()/2.,

                rect.get_y() + rect.get_height(),

                "{:.0f}%".format(int(emp_left_perc[index])),

                ha='center', va='bottom')

        index = index + 1

autolabel_emp(emp_left_bars, ax2)





ax2.set_xticks(y_pos)

ax2.set_xticklabels(["Not-Promotion","Promoted"])

ax2.set_ylabel('Frequency')

ax2.set_title('Employees who left w.r.t Promotion')

ax2.legend([emp_left_bars[0]], ['Employees Left'])



##########################################################

# No. of Projects Bar Chart

proj_count = data['number_project'].value_counts()

proj_indices = proj_count.index.tolist()

proj_values = proj_count.values.tolist()





# Employees left w.r.t No. of Projects

emp_left = []

for p in proj_indices:

    left_emp = len(data.loc[(data['number_project']==p) & (data['left'] == '1')])

    emp_left.append(left_emp)



# Percentage of employees w.r.t No. of Projects

emp_left_perc = [(ai/bi)*100. for ai,bi in zip(emp_left,proj_values)]





y_pos = np.arange(len(proj_values))

bars=ax3.bar(y_pos, proj_values, align='center', color=blue_color, edgecolor='black')

emp_left_bars=ax3.bar(y_pos, emp_left, align='center', color=blue_color, hatch='//',edgecolor='black')



# Add counts on Bars

def autolabel(rects, ax):

    for rect in rects:

        ax.text(rect.get_x() + rect.get_width()/2.,

                rect.get_y() + rect.get_height(),

                "{:d}".format(int(rect.get_height())),

                ha='center', va='bottom')

autolabel(bars, ax3)



# Add percentage on Bars

def autolabel_emp(rects, ax):

    index = 0

    for rect in rects:

        ax.text(rect.get_x() + rect.get_width()/2.,

                rect.get_y() + rect.get_height(),

                "{:.0f}%".format(int(emp_left_perc[index])),

                ha='center', va='bottom')

        index = index + 1

autolabel_emp(emp_left_bars, ax3)

    



ax3.set_xticks(y_pos)

ax3.set_xticklabels(proj_indices)

ax3.set_ylabel('Frequency')

ax3.set_xlabel('No. of Projects')

ax3.set_title('Employees who left w.r.t No. of Projects')

ax3.legend([emp_left_bars[0]], ['Employees Left'])



##########################################################

# Time Spend Bar Chart

spend_count = data['time_spend_company'].value_counts()

spend_indices = spend_count.index.tolist()

spend_values = spend_count.values.tolist()





# Employees left w.r.t Time Spend

emp_left = []

for ts in spend_indices:

    left_emp = len(data.loc[(data['time_spend_company']==ts) & (data['left'] == '1')])

    emp_left.append(left_emp)



# Percentage of employees w.r.t No. of Projects

emp_left_perc = [(ai/bi)*100. for ai,bi in zip(emp_left,spend_values)]





y_pos = np.arange(len(spend_values))

bars=ax4.bar(y_pos, spend_values, align='center', color=blue_color, edgecolor='black')

emp_left_bars=ax4.bar(y_pos, emp_left, align='center', color=blue_color, hatch='//',edgecolor='black')



# Add counts on Bars

def autolabel(rects, ax):

    for rect in rects:

        ax.text(rect.get_x() + rect.get_width()/2.,

                rect.get_y() + rect.get_height(),

                "{:d}".format(int(rect.get_height())),

                ha='center', va='bottom')

autolabel(bars, ax4)



# Add percentage on Bars

def autolabel_emp(rects, ax):

    index = 0

    for rect in rects:

        ax.text(rect.get_x() + rect.get_width()/2.,

                rect.get_y() + rect.get_height(),

                "{:.0f}%".format(int(emp_left_perc[index])),

                ha='center', va='bottom')

        index = index + 1

autolabel_emp(emp_left_bars, ax4)

    



ax4.set_xticks(y_pos)

ax4.set_xticklabels(spend_indices)

#ax3.set_xticklabels(["No Work-Accident","Work-Accident"])

ax4.set_ylabel('Frequency')

ax4.set_xlabel('Time Spend in Company')

ax4.set_title('Employees who left w.r.t Time Spend in Comp.')

ax4.legend([emp_left_bars[0]], ['Employees Left'])





plt.show()
##########################################################

# Create fiure with 4 subplots

f, ax = plt.subplots(2,2,figsize=(13,13))



(ax1, ax2 , ax3, ax4) = ax.flatten()



##########################################################

# Histogram of Last Eval: I want 20 bins in range (0-1)

ax1.hist(data['last_evaluation'], bins=20, range=(0,1), alpha=0.5, color='b')

ax1.set_title('Histogram: Last-Evaluation')

ax1.set_ylabel('Frequency')

ax1.set_xlabel('Last-Evaluation Scores')







##########################################################

# Histogram of Avg. Monthly Hours: I want 20 bins 

ax2.hist(data['average_montly_hours'], bins=20, alpha=0.5, color='b')

ax2.set_title('Histogram: Avg. Monthly Hours')

ax2.set_ylabel('Frequency')

ax2.set_xlabel('Avg. Monthly Hours')



##########################################################

# Histogram of Last Eval: I want 20 bins in range (0-1)

n, bins, patches = ax3.hist(data['last_evaluation'], bins=20, range=(0,1), alpha=0.5, color='b')

left_in_bins = []

for i in range(len(bins)-1):

    start = bins[i]

    end = bins[i+1]

    

    left_emp = len(data.loc[(data['last_evaluation']>=start) & (data['last_evaluation']<end) & (data['left'] == '1')])

    left_in_bins.append(left_emp)



index = 0

for_legend = None

for p in patches:

    patch = copy.copy(p)

    patch.set_height(left_in_bins[index])

    #patch.set_color(red_color)

    patch.set_hatch('//')

    patch.set_alpha(1.0)

    ax3.add_patch(patch)

    if index==1:

        for_legend = patch

    index = index + 1



ax3.set_title('Histogram: Last-Evaluation with Left Status')

ax3.set_ylabel('Frequency')

ax3.set_xlabel('Last-Evaluation Scores')

ax3.legend([for_legend], ['Employees Left'])

 

##########################################################

# Histogram of Avg. Monthly Hours: I want 20 bins 

n, bins, patches = ax4.hist(data['average_montly_hours'], bins=20, alpha=0.5, color='b')

left_in_bins = []

for i in range(len(bins)-1):

    start = bins[i]

    end = bins[i+1]

    

    left_emp = len(data.loc[(data['average_montly_hours']>=start) & (data['average_montly_hours']<end) & (data['left'] == '1')])

    left_in_bins.append(left_emp)



index = 0

for_legend = None

for p in patches:

    patch = copy.copy(p)

    patch.set_height(left_in_bins[index])

    #patch.set_color(red_color)

    patch.set_hatch('//')

    patch.set_alpha(1.0)

    ax4.add_patch(patch)

    if index==1:

        for_legend = patch

    index = index + 1

ax4.set_title('Histogram: Avg. Monthly Hours with Left Status')

ax4.set_ylabel('Frequency')

ax4.set_xlabel('Avg. Monthly Hours')

ax4.legend([for_legend], ['Employees Left'])



plt.show()
high_sat_churners = data[(data['satisfaction_level']>=0.7) & (data['satisfaction_level']<0.95) & (data['left'] == '1')]

print("No. of Employees Left (with High Satisfaction) {:d}".format(len(high_sat_churners)))



f, ax = plt.subplots(3,2,figsize=(12,24))

(ax1, ax2, ax3, ax4, ax5, ax6) = ax.flatten()



##########################################################

# Explore High-Satisfaction Churners



high_sat_churners['salary'].value_counts().plot(kind='bar', ax=ax1, title='Salary')

high_sat_churners['department'].value_counts().plot(kind='bar', ax=ax2, title='Department')

high_sat_churners['promotion_last_5years'].value_counts().plot(kind='bar', ax=ax3, title='Promotion')

high_sat_churners['time_spend_company'].value_counts().plot(kind='bar', ax=ax4, title='Time Spend in Comp.')

high_sat_churners['number_project'].value_counts().plot(kind='bar', ax=ax5, title='No. of Projects')

high_sat_churners['average_montly_hours'].plot(kind='hist', ax=ax6, title='Avg. Monthly hours')



print(high_sat_churners['salary'].value_counts())

print(high_sat_churners['promotion_last_5years'].value_counts())

print(high_sat_churners['time_spend_company'].value_counts())

print(high_sat_churners['number_project'].value_counts())



#print(high_sat_churners['average_montly_hours'].value_counts())



plt.show()



##########################################################

# Explore High-Satisfaction Churners with High-Salary



f, ax = plt.subplots(2,1,figsize=(5,8))

(ax1, ax2) = ax.flatten()

high_sal_high_sat_churners = data[(data['satisfaction_level']>=0.7) & (data['satisfaction_level']<0.95) & (data['left'] == '1') & (data['salary'] == 'high')]

print(high_sal_high_sat_churners['promotion_last_5years'].value_counts())

high_sal_high_sat_churners['promotion_last_5years'].value_counts().plot(kind='bar', title='Promotion of 15 high-salary high-satisfaction churners', ax=ax1)

high_sal_high_sat_churners['average_montly_hours'].plot(kind='hist', title='Working Hours of 15 high-salary high-satisfaction churners', ax=ax2)



plt.show()



##########################################################

# Explore High-Satisfaction Churners who got promoted



plt.figure()

promoted_high_sat_churners = data[(data['satisfaction_level']>=0.7) & (data['satisfaction_level']<0.95) & (data['left'] == '1') & (data['promotion_last_5years'] == '1')]

print(len(promoted_high_sat_churners))

promoted_high_sat_churners['salary'].value_counts().plot(kind='bar', title='Salary of 4 promoted high-satisfaction churners')

plt.show()

from sklearn.model_selection import train_test_split



data = pd.read_csv('../input/HR_comma_sep.csv')



# Convert all nominal to numeric.

data['sales'].replace(['sales', 'accounting', 'hr', 'technical', 'support', 'management',

        'IT', 'product_mng', 'marketing', 'RandD'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], inplace = True)

data['salary'].replace(['low', 'medium', 'high'], [0, 1, 2], inplace = True)

###########################################

# Train & Test Data



data_X = data.copy()

data_y = data_X['left']

del data_X['left']



train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size = 0.2, random_state = 1234)



print("Train Dataset rows: {} ".format(train_X.shape[0]))

print("Test Dataset rows: {} ".format(test_X.shape[0]))







###########################################



## Uncomment the code below to run parameter search



"""from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV



rf = RandomForestClassifier(n_estimators=10)





param_grid = {

    'n_estimators': [10, 15, 20, 25, 30],

    'max_features': ['auto', 2,3,4,5],

    'min_samples_leaf': [1, 3, 5],

    'criterion': ["gini", "entropy"]

}

CV_rfc = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5)

CV_rfc.fit(train_X, train_y)

print(CV_rfc.best_params_)"""

###########################################







#############################################

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix



rf = RandomForestClassifier(n_estimators =25, max_features=4, min_samples_leaf=1, criterion='entropy')



rf.fit(train_X, train_y)

train_accuracy = rf.score(train_X, train_y)

print("Accuracy (Train) {:.04f}".format(train_accuracy))



test_accuracy = rf.score(test_X, test_y)

print("Accuracy (Test) {:.04f}".format(test_accuracy))

pred_y = rf.predict(test_X)



confusion_matrix(test_y, pred_y)
feature_importance = rf.feature_importances_

features_list = data.columns



# make importances relative to max importance

feature_importance = 100.0 * (feature_importance / feature_importance.max())



# A threshold below which to drop features from the final data set. Specifically, this number represents

# the percentage of the most important feature's importance value

fi_threshold = 15



# Get the indexes of all features over the importance threshold

important_idx = np.where(feature_importance > fi_threshold)[0]



# Create a list of all the feature names above the importance threshold

important_features = features_list[important_idx]

#print "n", important_features.shape[0], "Important features(>", fi_threshold, "% of max importance):n", 

#        important_features



# Get the sorted indexes of important features

sorted_idx = np.argsort(feature_importance[important_idx])[::-1]

print("nFeatures sorted by importance (DESC):n" + str(important_features[sorted_idx]))



# Adapted from http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html

pos = np.arange(sorted_idx.shape[0]) + .5

plt.subplot(1, 2, 2)

plt.barh(pos, feature_importance[important_idx][sorted_idx[::-1]], align='center')

plt.yticks(pos, important_features[sorted_idx[::-1]])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.draw()

plt.show()