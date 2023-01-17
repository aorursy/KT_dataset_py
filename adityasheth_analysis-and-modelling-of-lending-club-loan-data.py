# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Importing the libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# Reading the dataset
data = pd.read_csv("../input/loan.csv")
data_1 = pd.DataFrame(data) # Creating a copy

# Checking the dataset
data.head()
data.tail()
data.describe()
data = data.iloc[:,2:-30].values
# Setting the target vector
status = data[:,14]
unique_labels = np.unique(status, return_counts = True)
# print(unique_labels)

plt.figure()
plt.bar(unique_labels[0],unique_labels[1])
plt.xlabel('Type of label')
plt.ylabel('Frequency')
plt.title('Status categories')
plt.show()

category = unique_labels[0]
frequency = unique_labels[1]
category_count = np.vstack((category,frequency))
category_list = np.array(category_count.T).tolist()
category_list_1 = pd.DataFrame(category_list)
print(category_list_1)

category_one_data = data_1[data_1.loan_status == "Fully Paid"]
category_two_data = data_1[data_1.loan_status == "Charged Off"]
new_data = np.vstack((category_one_data,category_two_data))
# new_data_copy = pd.DataFrame(new_data)
new_data = new_data[:,2:-30]
new_data_df = pd.DataFrame(new_data)
# Creating bins for various installment amounts
installment_amt = new_data[:,5]
bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
installment_amt = installment_amt.astype(float).reshape(installment_amt.size,1)
binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))                  
installment_groups = (np.array(np.unique(binned_installment_amt, return_counts = True))).T

# A bar plot to figure out the distribution of installment amount
plt.figure()
plt.bar(installment_groups[:,0],installment_groups[:,1])
plt.xlabel('Installment_amt_grp')
plt.ylabel('Frequency')
plt.title('Distribution of Installment amount categories')
plt.show()

# Appending the installment_groups to status
status_new = new_data_df[14]
factored_status = np.array(pd.factorize(status_new)) # 0's = Fully Paid, 1's = Charged Off
status_labels = pd.DataFrame(factored_status[0])
status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt,status_labels)))
status_installment_groups.columns = ['Installment_amt_grp','status_labels'] 

# Looking for a trend in the defaulted observations
Charged_off = status_installment_groups[status_installment_groups.status_labels == 1]
temp_1 = Charged_off.iloc[:,0].values
plot_var_1 = np.array(np.unique(temp_1, return_counts = True))
plot_var_1 = plot_var_1[:,:-1]
plot_var_11 = plot_var_1.T # Eliminating the 10th, since as only one reading

# Looking for a trend in the successful observations
Fully_paid = status_installment_groups[status_installment_groups.status_labels == 0]
temp_2 = Fully_paid.iloc[:,0].values
plot_var_2 = np.array(np.unique(temp_2, return_counts = True))
plot_var_22 = plot_var_2.T

# Concatenating the two variables
plot_var_stack = np.hstack((plot_var_11,plot_var_22))
plot_var_stack = pd.DataFrame(plot_var_stack)
plot_var_stack = plot_var_stack.drop(plot_var_stack.columns[2], axis=1)
plot_var_stack.columns = ['Installment_amt_grp','Charged Off','Fully Paid']

# Percent stacked
# From raw value to percentage
totals = [i+j for i,j in zip(plot_var_stack['Charged Off'], plot_var_stack['Fully Paid'])]
C_Off = [i / j * 100 for i,j in zip(plot_var_stack['Charged Off'], totals)]
mean_C_Off = np.mean(C_Off)
F_Paid = [i / j * 100 for i,j in zip(plot_var_stack['Fully Paid'], totals)]
plot_var_stack = np.array(plot_var_stack)
group_number = plot_var_stack[:,0]
p1 = plt.bar(group_number, C_Off, color='#7f6d5f', edgecolor='white', width=0.5)
p2 = plt.bar(group_number, F_Paid, bottom=C_Off, color='#557f2d', edgecolor='white', width=0.5)
#Axes.axhline(y=mean_C_Off)
plt.xlabel('Installment_amt_grp')
plt.ylabel('Percent loan status')
plt.title('Installment amount categories')
plt.legend((p1, p2), ('Charged Off', 'Fully Paid'), loc = 'upper right')
plt.show()

installment_grade = new_data[:,6]
# print(np.unique(installment_grade, return_counts = True))
installment_grade_list = np.array(np.unique(installment_grade, return_counts = True))
installment_grade_df = pd.DataFrame(installment_grade_list.T)
print(installment_grade_df)

# Distribution of Installment grade
plt.figure()
plt.bar(installment_grade_df[0],installment_grade_df[1])
plt.xlabel('Installment_grade')
plt.ylabel('Frequency')
plt.title('Distribution of Installment grade categories')
plt.show()

installment_grade = pd.DataFrame(installment_grade)
status_installment_grade = pd.DataFrame(np.hstack((installment_grade,status_labels)))
status_installment_grade.columns = ['Installment_grade','status_labels']

# Looking for a trend in the defaulted observations
Charged_off_grade = status_installment_grade[status_installment_grade.status_labels == 1]
temp_11 = Charged_off_grade.iloc[:,0].values
plot_var_grade = np.array(np.unique(temp_11, return_counts = True))
plot_var_grade_11 = plot_var_grade.T 

# Looking for a trend in the successful observations
Fully_Paid_grade = status_installment_grade[status_installment_grade.status_labels == 0]
temp_22 = Fully_Paid_grade.iloc[:,0].values
plot_var_grade_2 = np.array(np.unique(temp_22, return_counts = True))
plot_var_grade_22 = plot_var_grade_2.T # Eliminating the 10th, since as only one reading

# Concatenating the two variables
plot_var_stack_1 = np.hstack((plot_var_grade_11,plot_var_grade_22))
plot_var_stack_1 = pd.DataFrame(plot_var_stack_1)
plot_var_stack_1 = plot_var_stack_1.drop(plot_var_stack_1.columns[2], axis=1)
plot_var_stack_1.columns = ['Installment_grade_grp','Charged Off','Fully Paid']

# Percent stacked
# From raw value to percentage
totals = [i+j for i,j in zip(plot_var_stack_1['Charged Off'], plot_var_stack_1['Fully Paid'])]
C_Off = [i / j * 100 for i,j in zip(plot_var_stack_1['Charged Off'], totals)]
mean_C_Off = np.mean(C_Off)
F_Paid = [i / j * 100 for i,j in zip(plot_var_stack_1['Fully Paid'], totals)]
# plot_var_stack_1 = np.array(plot_var_stack_1)
group_number = plot_var_stack_1['Installment_grade_grp']
p1 = plt.bar(group_number, C_Off, color='#7f6d5f', edgecolor='white', width=0.5)
p2 = plt.bar(group_number, F_Paid, bottom=C_Off, color='#557f2d', edgecolor='white', width=0.5)
#Axes.axhline(y=mean_C_Off)
plt.xlabel('Installment_grade')
plt.ylabel('Percent loan status')
plt.title('Installment grade categories')
plt.legend((p1, p2), ('Charged Off', 'Fully Paid'), loc = 'upper right')
plt.show()

home_status = new_data_df[10]
# print(np.unique(home_status, return_counts = True))
home_status_list = np.array(np.unique(home_status, return_counts = True))
home_status_df = pd.DataFrame(home_status_list.T)
print(home_status_df)

# Distribution of Emp_length
plt.figure()
plt.bar(home_status_df[0],home_status_df[1])
plt.xlabel('Home Status')
plt.ylabel('Frequency')
plt.title('Home Status categories')
plt.show()

home_status = pd.DataFrame(home_status)
status_home_status = pd.DataFrame(np.hstack((home_status,status_labels)))
status_home_status.columns = ['Home Status','status_labels']

# Looking for a trend in the defaulted observations
Charged_off_home_status = status_home_status[status_home_status.status_labels == 1]
temp_41 = Charged_off_home_status.iloc[:,0].values
plot_var_home_status = np.array(np.unique(temp_41, return_counts = True))
plot_var_home_status_44 = pd.DataFrame(plot_var_home_status.T) 

# Looking for a trend in the successful observations
Fully_Paid_home_status = status_home_status[status_home_status.status_labels == 0]
temp_42 = Fully_Paid_home_status.iloc[:,0].values
plot_var_home_status_2 = np.array(np.unique(temp_42, return_counts = True))
plot_var_home_status_55 = pd.DataFrame(plot_var_home_status_2.T) # Eliminating the 10th, since as only one reading
plot_var_home_status_55 = plot_var_home_status_55.drop(0) # Eliminating the home status = "any", since as only one reading

# Concatenating the two variables
plot_var_stack_3 = np.hstack((plot_var_home_status_44,plot_var_home_status_55))
plot_var_stack_3 = pd.DataFrame(plot_var_stack_3)
plot_var_stack_3 = plot_var_stack_3.drop(plot_var_stack_3.columns[2], axis=1)
plot_var_stack_3.columns = ['Home Status','Charged Off','Fully Paid']

# Percent stacked
# From raw value to percentage
totals = [i+j for i,j in zip(plot_var_stack_3['Charged Off'], plot_var_stack_3['Fully Paid'])]
C_Off = [i / j * 100 for i,j in zip(plot_var_stack_3['Charged Off'], totals)]
mean_C_Off = np.mean(C_Off)
F_Paid = [i / j * 100 for i,j in zip(plot_var_stack_3['Fully Paid'], totals)]
#plot_var_stack_3 = np.array(plot_var_stack_3)
group_number = plot_var_stack_3['Home Status']
p1 = plt.bar(group_number, C_Off, color='#7f6d5f', edgecolor='white', width=0.5)
p2 = plt.bar(group_number, F_Paid, bottom=C_Off, color='#557f2d', edgecolor='white', width=0.5)
#Axes.axhline(y=mean_C_Off)
plt.xlabel('Home Status')
plt.ylabel('Percent loan status')
plt.title('Home Status categories')
plt.legend((p1, p2), ('Charged Off', 'Fully Paid'), loc = 'upper right')
plt.show()

## Now checking the effect of annual income on loan status
# Creating bins for various income amounts
annual_income = new_data[:,11]
#bins_2 = np.linspace(annual_income.min(), annual_income.max(), 3)
bins_2 = np.array([40000,70000,100000,150000])
annual_income = annual_income.astype(float).reshape(annual_income.size,1)
binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))                  
annual_groups = (np.array(np.unique(binned_annual_income, return_counts = True))).T

# A bar plot to figure out the distribution of income amount
plt.figure()
plt.bar(annual_groups[:,0],annual_groups[:,1])
plt.xlabel('Annual income amount group')
plt.ylabel('Frequency')
plt.title('Annual income amount categories')
plt.legend(loc="upper right")
plt.show()

# Appending the income_groups to status
status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income,status_labels)))
status_annual_groups.columns = ['Annual_income_grp','status_labels'] 

# Looking for a trend in the defaulted observations
Charged_off_annual_income = status_annual_groups[status_annual_groups.status_labels == 1]
temp_51 = Charged_off_annual_income.iloc[:,0].values
plot_var_annual_income = np.array(np.unique(temp_51, return_counts = True))
plot_var_annual_income_66 = pd.DataFrame(plot_var_annual_income.T) 

# Looking for a trend in the successful observations
Fully_Paid_annual_income = status_annual_groups[status_annual_groups.status_labels == 0]
temp_52 = Fully_Paid_annual_income.iloc[:,0].values
plot_var_annual_income_2 = np.array(np.unique(temp_52, return_counts = True))
plot_var_annual_income_77 = pd.DataFrame(plot_var_annual_income_2.T) # Eliminating the 10th, since as only one reading
#plot_var_annual_income_55 = plot_var_home_status_55.drop(0) # Eliminating the home status = "any", since as only one reading

# Concatenating the two variables
plot_var_stack_4 = np.hstack((plot_var_annual_income_66,plot_var_annual_income_77))
plot_var_stack_4 = pd.DataFrame(plot_var_stack_4)
plot_var_stack_4 = plot_var_stack_4.drop(plot_var_stack_4.columns[2], axis=1)
plot_var_stack_4.columns = ['Annual Income Group','Charged Off','Fully Paid']

# Percent stacked
# From raw value to percentage
totals = [i+j for i,j in zip(plot_var_stack_4['Charged Off'], plot_var_stack_4['Fully Paid'])]
C_Off = [i / j * 100 for i,j in zip(plot_var_stack_4['Charged Off'], totals)]
mean_C_Off = np.mean(C_Off)
F_Paid = [i / j * 100 for i,j in zip(plot_var_stack_4['Fully Paid'], totals)]
#plot_var_stack_4 = np.array(plot_var_stack_4)
group_number = plot_var_stack_4['Annual Income Group']
p1 = plt.bar(group_number, C_Off, color='#7f6d5f', edgecolor='white', width=0.5)
p2 = plt.bar(group_number, F_Paid, bottom=C_Off, color='#557f2d', edgecolor='white', width=0.5)
#Axes.axhline(y=mean_C_Off)
plt.xlabel('Annual income amount group')
plt.ylabel('Percent loan status')
plt.title('Annual income amount categories')
plt.legend((p1, p2), ('Charged Off', 'Fully Paid'), loc = 'upper right')
plt.show()

# Separating the variable under investigation
state = new_data_df[21]
#print(np.unique(state, return_counts = True))
state_list = np.array(np.unique(state, return_counts = True))
state_df = pd.DataFrame(state_list.T)
print(state_df)

# Distribution of Emp_length
plt.figure()
plt.bar(state_df[0],state_df[1])
plt.xlabel('State')
plt.ylabel('Frequency')
plt.title('State')
plt.show()

state = pd.DataFrame(state)
status_state = pd.DataFrame(np.hstack((state,status_labels)))
status_state.columns = ['State','status_labels']

# Looking for a trend in the defaulted observations
Charged_off_state = status_state[status_state.status_labels == 1]
temp_61 = Charged_off_state.iloc[:,0].values
plot_var_state = np.array(np.unique(temp_61, return_counts = True))
plot_var_state_88 = pd.DataFrame(plot_var_state.T) 

# Looking for a trend in the successful observations
Fully_Paid_state = status_state[status_state.status_labels == 0]
temp_62 = Fully_Paid_state.iloc[:,0].values
plot_var_state_2 = np.array(np.unique(temp_62, return_counts = True))
plot_var_state_99 = pd.DataFrame(plot_var_state_2.T) 

plot_var_state_88 = plot_var_state_88.drop(7)
plot_var_state_99 = plot_var_state_99.drop([7,21,28]) # Eliminating the home status = "any", since as only one reading

# Concatenating the two variables
plot_var_stack_5 = np.hstack((plot_var_state_88,plot_var_state_99))
plot_var_stack_5 = pd.DataFrame(plot_var_stack_5)
plot_var_stack_5 = plot_var_stack_5.drop(plot_var_stack_5.columns[2], axis=1)
plot_var_stack_5.columns = ['state','Charged Off','Fully Paid']

# Percent stacked
# From raw value to percentage
totals = [i+j for i,j in zip(plot_var_stack_5['Charged Off'], plot_var_stack_5['Fully Paid'])]
C_Off = [i / j * 100 for i,j in zip(plot_var_stack_5['Charged Off'], totals)]
mean_C_Off = np.mean(C_Off)
F_Paid = [i / j * 100 for i,j in zip(plot_var_stack_5['Fully Paid'], totals)]
#plot_var_stack_5 = np.array(plot_var_stack_5)
group_number = plot_var_stack_5['state']
p1 = plt.bar(group_number, C_Off, color='#7f6d5f', edgecolor='white', width=0.5)
p2 = plt.bar(group_number, F_Paid, bottom=C_Off, color='#557f2d', edgecolor='white', width=0.5)
plt.xlabel('State')
plt.ylabel('Percent loan status')
plt.title('State')
plt.legend((p1, p2), ('Charged Off', 'Fully Paid'), loc = 'upper right')
plt.show()

###### Sort in order and print top 5 states with max default % ########
# Concatenating C_Off and state
C_Off = pd.DataFrame(C_Off)
temp_plot = np.hstack((plot_var_stack_5, C_Off))
temp_plot = pd.DataFrame(temp_plot)
temp_plot.columns = ['state','Charged Off','Fully Paid','% Charged Off']
temp_plot = np.array(temp_plot.sort_values(by = '% Charged Off',ascending = False))
print(temp_plot[0:5,(0,3)])

temp_plot = pd.DataFrame(temp_plot)
temp_plot.columns = ['state','Charged Off','Fully Paid','% Charged Off']
temp_plot = temp_plot.drop(['Charged Off', 'Fully Paid'], axis = 1)

# Chloropleth map for better visualization
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
 
for col in temp_plot.columns:
    temp_plot[col] = temp_plot[col].astype(str)

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

#temp_plot['text'] = temp_plot['state'] + '<br>' +\
#    'Default rate '+ temp_plot['% Charged Off']

data_chloropleth = [ dict(
        type ='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = temp_plot['state'],
        z = temp_plot['% Charged Off'].astype(float),
        locationmode = 'USA-states',
        #text = temp_plot['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Default rate")
        ) ]

layout = dict(
        title = 'State-wise % Charged Off<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict(type='albers usa'),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
#fig = dict(data=data_chloropleth, layout=layout)
#py.iplot(fig, image = 'png', filename = 'test-image-2', show_link = False)
# Change to py.iplot
chloromap = go.Figure(data = data_chloropleth, layout = layout)
iplot(chloromap, validate=False)

# Separating the variable under investigation
ver_stat = new_data_df[12]
#print(np.unique(ver_stat, return_counts = True))
ver_stat_list = np.array(np.unique(ver_stat, return_counts = True))
ver_stat_df = pd.DataFrame(ver_stat_list.T)
print(ver_stat_df)

# Distribution of Emp_length
plt.figure()
plt.bar(ver_stat_df[0],ver_stat_df[1])
plt.xlabel('Verification Status')
plt.ylabel('Frequency')
plt.title('Verification Status')
plt.show()

ver_stat = pd.DataFrame(ver_stat)
status_ver_stat = pd.DataFrame(np.hstack((ver_stat,status_labels)))
status_ver_stat.columns = ['Verification Status','status_labels']

# Looking for a trend in the defaulted observations
Charged_off_ver_stat = status_ver_stat[status_ver_stat.status_labels == 1]
temp_71 = Charged_off_ver_stat.iloc[:,0].values
plot_var_ver_stat = np.array(np.unique(temp_71, return_counts = True))
plot_var_ver_stat_101 = pd.DataFrame(plot_var_ver_stat.T) 

# Looking for a trend in the successful observations
Fully_Paid_ver_stat = status_ver_stat[status_ver_stat.status_labels == 0]
temp_72 = Fully_Paid_ver_stat.iloc[:,0].values
plot_var_ver_stat_2 = np.array(np.unique(temp_72, return_counts = True))
plot_var_ver_stat_111 = pd.DataFrame(plot_var_ver_stat_2.T) 

# Concatenating the two variables
plot_var_stack_6 = np.hstack((plot_var_ver_stat_101,plot_var_ver_stat_111))
plot_var_stack_6 = pd.DataFrame(plot_var_stack_6)
plot_var_stack_6 = plot_var_stack_6.drop(plot_var_stack_6.columns[2], axis=1)
plot_var_stack_6.columns = ['Verification Status','Charged Off','Fully Paid']

# Percent stacked
# From raw value to percentage
totals = [i+j for i,j in zip(plot_var_stack_6['Charged Off'], plot_var_stack_6['Fully Paid'])]
C_Off = [i / j * 100 for i,j in zip(plot_var_stack_6['Charged Off'], totals)]
mean_C_Off = np.mean(C_Off)
F_Paid = [i / j * 100 for i,j in zip(plot_var_stack_6['Fully Paid'], totals)]
#plot_var_stack_5 = np.array(plot_var_stack_5)
group_number = plot_var_stack_6['Verification Status']
p1 = plt.bar(group_number, C_Off, color='#7f6d5f', edgecolor='white', width=0.5)
p2 = plt.bar(group_number, F_Paid, bottom=C_Off, color='#557f2d', edgecolor='white', width=0.5)
plt.xlabel('Verification Status')
plt.ylabel('Percent loan status')
plt.title('Verification Status')
plt.legend((p1, p2), ('Charged Off', 'Fully Paid'), loc = 'upper right')
plt.show()
from wordcloud import WordCloud

# Employee Title
emp_title = new_data_df[8]
emp_title = pd.DataFrame(emp_title)
emp_title.columns = ['Employee Title']
emp_title = emp_title.dropna(axis=0, how='all')
wordcloud = WordCloud().generate(' '.join(emp_title['Employee Title']))

# Generate plot
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
# Title 
title = new_data_df[19]
title = pd.DataFrame(title)
title.columns = ['Title']
title = title.dropna(axis=0, how='all')
wordcloud3 = WordCloud().generate(' '.join(title['Title']))

# Generate plot
plt.imshow(wordcloud3)
plt.axis("off")
plt.show()
# Description
Description = new_data_df[17]
Description = Description.dropna(axis=0, how='all')
Description = list(Description)
Description_1 = []
i = 0 
for i in range(0,len(Description)):
    s = Description[i]
    s = s.replace("Borrower added on ", "")
    s = s.replace("<br>", "")
    Description_1.append(s)
    i = i+1
Description_1 = pd.DataFrame(Description_1)
Description_1.columns = ['Description']
wordcloud4 = WordCloud().generate(' '.join(Description_1['Description']))

# Generate plot
plt.imshow(wordcloud4)
plt.axis("off")
plt.show()

# Data pre-processing

# Dealing with na values
new_data_copy = np.vstack((category_one_data,category_two_data))
new_data_copy = pd.DataFrame(new_data_copy)
#print(np.shape(new_data_copy)) # Dimensions of the dataset
#print(new_data_copy.isnull().sum()) # Printing number of na values in each column 
#data_2 = new_data_copy.dropna(axis = 1, how = 'all') # Dropping columns where all values are na
data_2 = new_data_copy
#print(np.shape(data_2)) # Dimensions of new dataset
# We can observe that one of the column was removed since it was completely empty
data_dim = np.shape(data_2)

# We can see that a lot of columns contain 70% na values, which is no good for us
# Columns having more than 20-30% na values would not be of much help, thus eliminating them
col_nos = []
i = 0
for i in range (0,data_dim[1]):
    num_na_val = data_2[i].isnull().sum()
    if (num_na_val/len(data_2)) > 0.2:
        col_nos.append(i)
    i = i+1

data_2 = data_2.drop(data_2.columns[col_nos], axis = 1)
#print(data_2.isnull().sum())
np.shape(data_2)

# Now lets drop the columns like id, employee title, description,etc. which cannot be taken into consideration while modelling 
rename_var_1 = range(0,49)
data_2.columns = rename_var_1
cols_remove = [0,10,11,17,18,19,20,21]
data_2 = data_2.drop(data_2.columns[cols_remove], axis = 1)
np.shape(data_2)

rename_var_2 = range(0,41)
data_2.columns = rename_var_2
time_series_var = [12,17,34,36]
cat_var_cols = [4,7,8,9,11,14,16,18,19,20,24,25,32,33,37,38,39]
cat_plus_time_cols = [4,7,8,9,11,12,14,16,17,18,19,20,24,25,32,33,34,36,37,38,39]
cat_var_df = data_2.iloc[:,cat_var_cols].values
cat_var_df = pd.DataFrame(cat_var_df)
#cat_var_df.describe(include=['category'])
i = 0
unique_categories = []
for i in cat_var_df:
    un_cat = np.unique(cat_var_df[i])
    unique_categories.append(un_cat)
    i = i+1
    
# Removing more columns based on the above result
#print(unique_categories)
c = [11,12,13,15]
cat_var_df = cat_var_df.drop(cat_var_df.columns[c], axis = 1)
np.shape(cat_var_df)
r_var = range(0,13)
# We can observe that column 16 has 56 null values, let us replace them with the most frequent value of the column
cat_var_df.columns = r_var
#print(cat_var_df.isnull().sum())
# Taking care of missing values for categorical features
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(cat_var_df[[11]])
cat_var_df[11] = imputer.transform(cat_var_df[[11]])
#print(cat_var_df[11].isnull().sum())
#print(cat_var_df.isnull().sum())
renaming_df = range(0,13)
cat_var_df.columns = renaming_df
# We can clearly see that now there are no more na values
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
i = 0
for i in range(0,13):
    cat_var_df[i] = labelencoder_X.fit_transform(cat_var_df[i])
    #onehotencoder = OneHotEncoder(categorical_features = [i])
    i = i+1

# Taking care of missing values for remaining features
#print(data_2.isnull().sum())
data_2_copy = data_2
non_cat_var = data_2_copy.drop(data_2_copy.columns[cat_plus_time_cols], axis = 1)
rename_var = range(0,20)
non_cat_var.columns = rename_var


# Also dropping the target variable
Y = non_cat_var[[7]]
non_cat_var = non_cat_var.drop(non_cat_var.columns[7], axis = 1) 
#non_cat_var = non_cat_var.drop(non_cat_var.columns[0], axis = 1) # Dropping the variable id
renaming_df = range(0,19)
non_cat_var.columns = renaming_df
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
i = 0
for i in non_cat_var:
    imputer = imputer.fit(non_cat_var[[i]])
    non_cat_var[i] = imputer.transform(non_cat_var[[i]])
    i = i+1
    
#print(non_cat_var.isnull().sum())   

# We have no nan values in our non_cat_var now
# Let us now concatenate the categrical variables and non_categorical variables and form ourfeature matrix.
# Checking the dimensions
print(np.shape(non_cat_var))
print(np.shape(cat_var_df))
print(np.shape(Y))

X = np.hstack((non_cat_var,cat_var_df)) # Concatenating
# Label encoding the target variable 
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#Splitting the dataset in training and testing set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# Fitting XGBClassifier to the training data: Model_1
from xgboost import XGBClassifier
classifier_1 = XGBClassifier()
classifier_1.fit(X_train,Y_train)

# Fitting SVM to the training data: Model 2
from sklearn.svm import SVC
classifier_2 = SVC(kernel = 'linear', C = 1, probability = True, random_state = 0) # poly, sigmoid
classifier_2.fit(X_train,Y_train)

# Creating and Fitting Random Forest Classifier to the training data: Model 3
from sklearn.ensemble import RandomForestClassifier
classifier_3 = RandomForestClassifier(n_estimators = 5, criterion = 'entropy')
classifier_3.fit(X_train,Y_train)

# Fitting classifier to the training data: Model 4 
from sklearn.linear_model import LogisticRegression
classifier_4 = LogisticRegression(penalty = 'l1', random_state = 0)
classifier_4.fit(X_train,Y_train)

# Fitting Balanced Bagging Classifier to the training data: Model 5
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import RandomForestClassifier
classifier_5 = BalancedBaggingClassifier(base_estimator = RandomForestClassifier(criterion='entropy'),
                                       n_estimators = 5, bootstrap = True)
classifier_5.fit(X_train,Y_train)

# Fitting Decision Tree to the training data: Model 6
from sklearn.tree import DecisionTreeClassifier
classifier_6 = DecisionTreeClassifier()
classifier_6.fit(X_train,Y_train)

# Predicting the results
y_pred_1 = classifier_1.predict(X_test)
y_pred_2 = classifier_2.predict(X_test)
y_pred_3 = classifier_3.predict(X_test)
y_pred_4 = classifier_4.predict(X_test)
y_pred_5 = classifier_5.predict(X_test)
y_pred_6 = classifier_6.predict(X_test)

# Creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm_1 = confusion_matrix(Y_test,y_pred_1)
accuracy_1 = (cm_1[0,0]+cm_1[1,1])/len(Y_test)

cm_2 = confusion_matrix(Y_test,y_pred_2)
accuracy_2 = (cm_2[0,0]+cm_2[1,1])/len(Y_test)

cm_3 = confusion_matrix(Y_test,y_pred_3)
accuracy_3 = (cm_3[0,0]+cm_3[1,1])/len(Y_test)

cm_4 = confusion_matrix(Y_test,y_pred_4)
accuracy_4 = (cm_4[0,0]+cm_4[1,1])/len(Y_test)

cm_5 = confusion_matrix(Y_test,y_pred_5)
accuracy_5 = (cm_5[0,0]+cm_5[1,1])/len(Y_test)

cm_6 = confusion_matrix(Y_test,y_pred_6)
accuracy_6 = (cm_6[0,0]+cm_6[1,1])/len(Y_test)


print("Accuracy_XGBoost:",accuracy_1*100,'%',"\nAccuracy_SVC:",accuracy_2*100,'%',"\nAccuracy_RF:",accuracy_3*100,'%',"\nAccuracy_Logistic:",accuracy_4*100,'%',
      "\nAccuracy_BalancedBagging:",accuracy_5*100,'%',"\nAccuracy_DecisionTree:",accuracy_6*100,'%')
