#Import the District 5 SHSAT dataset

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #plotting
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier #random forest classifier
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score #classifier performance
from sklearn.metrics import recall_score #classifier performance

#Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

#Explore the dataset for the test registration and testers
print("Looking at the first few rows of the dataset: ")
test_df = pd.read_csv("../input/data-science-for-good/D5 SHSAT Registrations and Testers.csv")
test_df.head()
#Let's take a look at some basic high level points about the testing data we have

num_schools = test_df['DBN'].unique().size
num_test_groups = test_df['DBN'].count()
years_taken = np.sort(test_df['Year of SHST'].unique())
grade_levels = np.sort(test_df['Grade level'].unique())

print("Number of unique test groups (i.e. school/class/year combos): ", num_test_groups)
print("Number of schools for which there is data: ", num_schools)
print("Years for the test data: ", years_taken)
print("Grade levels for the test data: ", grade_levels)
#Looking at the overall trend in time among schools
percent_taken_dict = {}
percent_taken_stats_dict = {}
df_by_year_dict = {}

#Add two columns: percentage of students who took the test and percentage of students who signed up for the test 
percent_taken = pd.DataFrame({'Percent Class Test Taken': test_df['Number of students who took the SHSAT']/test_df['Enrollment on 10/31']*100})
percent_signed = pd.DataFrame({'Percent Class Registered for Test': test_df['Number of students who registered for the SHSAT']/test_df['Enrollment on 10/31']*100})
school_test_df = test_df.merge(percent_taken, left_index=True, right_index=True).merge(percent_signed, left_index=True, right_index=True)

for year in years_taken:
    year_df = school_test_df.loc[lambda school_test_df: school_test_df['Year of SHST'] == year, :]
    df_by_year_dict[year] = year_df
    percent_taken_dict[year] = year_df['Number of students who took the SHSAT'].sum()/year_df['Enrollment on 10/31'].sum()
    
    percent_taken_stats_dict[year] = year_df['Percent Class Test Taken'].describe()
    
print(pd.DataFrame(percent_taken_stats_dict))

#In general, there are apparent trends in the percentage of students taking the test by year. The number of classes observed increases from 33 to 35 from 2013 to 2014,
#stays constant from 2014 to 2015 and increases from 35 to 37 from 2015 to 2016. The mean of percent students taking the test ranges from 11.70%
#to 11.92%. The standard deviation ranges from 10.79% to 12.54%. Overall, there is no strong trend in either direction with time.

#Showing a basic boxplot of the percent taken by year
school_test_df.boxplot(column='Percent Class Test Taken', by='Year of SHST', figsize=[14,8])
plt.show()


#Looking at the trends between grade levels (i.e. is there a big difference in enrollment from grade 8 to grade 9?)
percent_taken_stats_by_grade = {}

for grade in grade_levels:
    grade_df = school_test_df.loc[lambda school_test_df: school_test_df['Grade level'] == grade, :]
    
    percent_taken_stats_by_grade[grade] = grade_df['Percent Class Test Taken'].describe()
    
print(pd.DataFrame(percent_taken_stats_by_grade))

#There does seem to be a significant difference in percentage of students taking the test between grade levels. The mean/std dev. of percent of students taking the test in the eight grade
#is 18.39%/10.36% and 1.64%/2.84% in the ninth grade. The maximum percentage of studenst taking the test in the eight grade was 45.92% and only 13.15% in the ninth grade.

#Showing a basic boxplot of the percent taken by grade
school_test_df.boxplot(column='Percent Class Test Taken', by='Grade level', figsize=[14,8])
plt.show()
#Loading teh included NYC school data and transforming it to better fit our analysis
total_school_df = pd.read_csv("../input/data-science-for-good/2016 School Explorer.csv")

#Prepare a much smaller table that only included the school for which we have testing data
#The "DBN" field in the test file is the same key as the "Location Code" in the School Explorer file

#Get a list of all school info location codes
all_location_codes = total_school_df['Location Code'].values
#keep track of the indices where the schools in the total_school_df are
merger_index = []

#for each unique dbn code in the test_df
for dbn in school_test_df['DBN'].unique():
    #if it matches the location code in the total_school_df
    for i,loc_code in enumerate(all_location_codes):
        if loc_code == dbn:
            #record index
            merger_index.append(i)

#Create a smaller df with only schools of interest
school_df_unformatted = total_school_df.iloc[merger_index, :]

#Remove all columns not relevant to analysis
removed_columns = ['Adjusted Grade', 'New?', 'Other Location Code in LCGMS', 'School Name', 'SED Code']
for col in list(school_df_unformatted):
    if 'Grade 3' in col or 'Grade 4' in col or 'Grade 5' in col:
        removed_columns.append(col)
        
school_df_unformatted = school_df_unformatted.drop(columns = removed_columns)

#Normalize ELA/Math test scores by class size (find % of students who scored 4s)
grades = ['Grade 6 ', 'Grade 7 ', 'Grade 8 ']
to_norm_cols_suffix = ['4s - All Students', '4s - American Indian or Alaska Native', '4s - Black or African American', '4s - Hispanic or Latino', '4s - Asian or Pacific Islander', '4s - White', '4s - Multiracial', '4s - Limited English Proficient', '4s - Economically Disadvantaged']
for grade in grades:
    for subject in ['ELA ', 'Math ']:
        total_col = grade + subject + '- All Students Tested'
        denom = school_df_unformatted[total_col]
        for suff in to_norm_cols_suffix:
            norm_col = grade + subject + suff;
            new_col = norm_col + ' - Normalized'
            school_df_unformatted[new_col]=school_df_unformatted[norm_col].divide(denom)
            school_df_unformatted[new_col].fillna(value=0, inplace=True)
        
#Merge the school df and the test results df, clean up the dataset and look at correlation coefficients between
#all fields and the percent class test taken
test_and_school_df_unformatted = school_test_df.merge(school_df_unformatted, left_on='DBN', right_on='Location Code')

#Need to clean the '%' out of certain columns or they are read as strings
percent_columns = ['Percent ELL', 'Percent Asian', 'Percent Black', 'Percent Hispanic', 'Percent Black / Hispanic', 'Percent White', 'Student Attendance Rate', 'Percent of Students Chronically Absent', 'Rigorous Instruction %', 'Collaborative Teachers %', 'Supportive Environment %', 'Effective School Leadership %', 'Strong Family-Community Ties %', 'Trust %']

for col in percent_columns:
    for i,row in test_and_school_df_unformatted.iterrows():
        val_string = row.loc[col]
        val_num = float(val_string.replace("%", ""))
        test_and_school_df_unformatted.at[i, col] = val_num
    

#test_and_school_df1 = test_and_school_df_unformatted.convert_objects(convert_numeric=True)
test_and_school_df1 = test_and_school_df_unformatted.infer_objects()


#Look for correlations between percent test taken and hand-selected features
corr_coeffs = test_and_school_df1.corr()['Percent Class Test Taken']
corr_rows_removed = ['Enrollment on 10/31', 'Number of students who took the SHSAT', 'District', 'Zip', 'Grade 8 ELA - All Students Tested', 'Grade 8 Math - All Students Tested', 'Percent Class Test Taken', 'Number of students who registered for the SHSAT', 'Longitude', 'Latitude']

for row_label, val in corr_coeffs.iteritems():
    if 'Grade 6' in row_label or 'Grade 7' in row_label or '- Normalized' in row_label:
        corr_rows_removed.append(row_label)
        

#Drop values that are non-sensical in a correlatin analysis and all N/A values as well
corr_coeffs = pd.DataFrame(corr_coeffs.drop(corr_rows_removed).dropna(how='all').sort_values())

corr_coeffs.style.bar(align='zero', color=['#5fba7d'])


#Grade level has the second largest correlation coefficient. However, it's unclear if including data
#about grade 9 classes is useful. There are far fewer slots for grade 10 entry into the specialized high schools,
#thus I think students udnerstand the odds and are less willing to take the test. Further, many who would
#have been inclined to take the test have probably done so. I'm shaping the data to look at 8th grade test takers
#only to see if the trends become more obvious.

#create a new df for just 8th grade test results
test_and_school_df_8thgrade = test_and_school_df1.copy()
removed_rows = []
for i,row in test_and_school_df1.iterrows():
    if row['Grade level'] == 9:
        removed_rows.append(i)
test_and_school_df_8thgrade = test_and_school_df_8thgrade.drop(index=removed_rows)

#find correlation coefficients
corr_coeffs_8thgrade = test_and_school_df_8thgrade.corr()['Percent Class Test Taken']
corr_rows_removed = ['Enrollment on 10/31', 'Number of students who took the SHSAT', 'District', 'Zip', 'Grade 6 ELA - All Students Tested', 'Grade 6 Math - All Students Tested', 'Grade 7 ELA - All Students Tested', 'Grade 7 Math - All Students Tested', 'Grade 8 ELA - All Students Tested', 'Grade 8 Math - All Students Tested', 'Percent Class Test Taken', 'Number of students who registered for the SHSAT']

for row_label, val in corr_coeffs_8thgrade.iteritems():
    if 'Grade 6' in row_label or 'Grade 7' in row_label or '- Normalized' in row_label:
        corr_rows_removed.append(row_label)


#Drop values that are non-sensical in a correlatin analysis and all N/A values as well
corr_coeffs_8thgrade = pd.DataFrame(corr_coeffs_8thgrade.drop(corr_rows_removed).dropna(how='all').sort_values())

corr_coeffs_8thgrade.style.bar(align='zero', color=['#5fba7d'])

#Let's break the data up into a couple categories and look at the covariance matrices, can we reduce the features there?
#Start with test scores of just 8th graders

test_and_school_df_8thgrade_test_scores = pd.DataFrame()

#Normalize ELA/Math test scores by class size (find % of students who scored 4s)
grades = ['Grade 6 ', 'Grade 7 ', 'Grade 8 ']
#grades = ['Grade 7 ']
test_cols_suffix = ['4s - All Students', '4s - American Indian or Alaska Native', '4s - Black or African American', '4s - Hispanic or Latino', '4s - Asian or Pacific Islander', '4s - White', '4s - Multiracial', '4s - Limited English Proficient', '4s - Economically Disadvantaged']
for grade in grades:
    for subject in ['ELA ']:
        for suff in to_norm_cols_suffix:
            norm_col = grade + subject + suff;
            test_and_school_df_8thgrade_test_scores[norm_col]=test_and_school_df_8thgrade[norm_col]
            
#Add average scores
test_and_school_df_8thgrade_test_scores['Average ELA Proficiency'] = test_and_school_df_8thgrade['Average ELA Proficiency']

state_test_scores_coeffs = test_and_school_df_8thgrade_test_scores.corr()

#Expand the dataset to include NY School Demographics and Accountability Snapshot, redcued and free lunch stats

#Read in the file
school_demo_all_df = pd.read_csv("../input/ny-school-demographics-and-accountability-snapshot/2006-2012-school-demographics-and-accountability-snapshot.csv")
#Reduce the df to only the relevant schools for the most recent year available (2012)
school_demo_reduced_df = school_demo_all_df.loc[lambda school_demo_all_df: school_demo_all_df['schoolyear'] == 20112012]

#Merge the free and reduced lunch columns form the school demographics snapshot file
test_and_school_df_8thgrade_plusdemo = test_and_school_df_8thgrade.merge(school_demo_reduced_df.loc[:, ['DBN','fl_percent','frl_percent']], left_on='DBN', right_on='DBN', how='left')
#Merge the fl_percent and frl_percent columns, one of the other is filled and we will treat them as both here

finding_nulls_series = np.bitwise_and(test_and_school_df_8thgrade_plusdemo.loc[:,['frl_percent']].isnull(), test_and_school_df_8thgrade_plusdemo.loc[:,['fl_percent']].isnull())

num_empty_entries = finding_nulls_series.sum()

print('Of 78 test entries, ', num_empty_entries[0], 'did not have data on free and reduced lunch data reported')


#Plotting to show the students who do well on standardized tests but don't take the SHSAT

plt.figure(1, figsize=(15, 7))
plt.subplot(121)
plt.scatter(test_and_school_df_8thgrade.loc[test_and_school_df_8thgrade['Year of SHST'] == 2016,'Grade 8 Math 4s - All Students'], test_and_school_df_8thgrade.loc[test_and_school_df_8thgrade['Year of SHST'] == 2016, 'Number of students who took the SHSAT'])
plt.plot([0, 10, 20, 30, 40, 50], [0, 10, 20, 30, 40, 50], 'r')
plt.xlabel('Number of students who scored a 4 on the Math Standardized Test')
plt.ylabel('Number of students who took the SHSAT')

plt.subplot(122)
plt.scatter(test_and_school_df_8thgrade.loc[test_and_school_df_8thgrade['Year of SHST'] == 2016,'Grade 8 ELA 4s - All Students'], test_and_school_df_8thgrade.loc[test_and_school_df_8thgrade['Year of SHST'] == 2016, 'Number of students who took the SHSAT'])
plt.plot([0, 10, 20, 30, 40, 50], [0, 10, 20, 30, 40, 50], 'r')
plt.xlabel('Number of students who scored a 4 on the ELA Standardized Test')
plt.ylabel('Number of students who took the SHSAT')
plt.suptitle('Finding High Potential Students Not Taking the SHSAT')
plt.show()


#Plotting a Histogram of high performing students missed


plt.hist(test_and_school_df_8thgrade.loc[test_and_school_df_8thgrade['Year of SHST'] == 2016,'Grade 8 Math 4s - All Students'].subtract(test_and_school_df_8thgrade.loc[test_and_school_df_8thgrade['Year of SHST'] == 2016, 'Number of students who took the SHSAT']))
plt.ylabel('Number of schools')
plt.xlabel('Number of students who scored a 4 on the Math Standardized Test who didn\'t take the SHSAT')
plt.show()
#Building a classifier that finds high potential, but overlooked students

#Make a new column in the dataframe of high potential but overlooked students
test_and_school_df_8thgrade['High Potential Overlooked Classification'] = pd.Series(False, index = test_and_school_df_8thgrade.index)
test_and_school_df_8thgrade['High Potential Overlooked Classification'].mask((test_and_school_df_8thgrade['Number of students who took the SHSAT']<test_and_school_df_8thgrade['Grade 8 Math 4s - All Students']) | (test_and_school_df_8thgrade['Number of students who took the SHSAT']<test_and_school_df_8thgrade['Grade 8 ELA 4s - All Students']),other=True, inplace=True)

#Make a list of columns to include in the features
#train_cols = ['Economic Need Index', 'Grade 8 ELA 4s - All Students', 'Grade 8 Math 4s - All Students']
train_cols = ['Grade 8 ELA 4s - All Students', 'Grade 8 Math 4s - All Students']
#train_cols = ['Strong Family-Community Ties %', 'Percent of Students Chronically Absent', 'Percent Asian']
#train_cols = ['Percent Black / Hispanic', 'Supportive Environment %', 'Collaborative Teachers %', 'Student Attendance Rate', 'Economic Need Index']
#train_cols = ['Percent Black / Hispanic', 'Supportive Environment %', 'Collaborative Teachers %', 'Strong Family-Community Ties %']

#Use only the data form 2016
x_all = test_and_school_df_8thgrade.loc[test_and_school_df_8thgrade['Year of SHST'] == 2016,train_cols]
y_all = test_and_school_df_8thgrade.loc[test_and_school_df_8thgrade['Year of SHST'] == 2016, 'High Potential Overlooked Classification']

#Iterate through the random forest kernels and different test sets
num_trials = 100
score = np.zeros(num_trials)

for i in range(num_trials):
    #Split the data into testing and training data
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.25, random_state=i)
    #Build a random forest classifier to find overlooked students
    model2 = RandomForestClassifier(n_estimators=10, random_state=i)
    #Fit the model to the training data
    model2.fit(x_train, y_train)
    #Use developed model to predict the classifications of the test data
    y_test_model = model2.predict(x_test)
    #Find the score
    score[i] = model2.score(x_test, y_test)

#Print results
mean_score = np.mean(score)
print("The average score of the classifier over", num_trials, "trials is: {:.2}".format(mean_score))



#Running a lienar regression on students lost (rather than a classifer)
test_and_school_df_8thgrade_2016only = test_and_school_df_8thgrade.loc[test_and_school_df_8thgrade['Year of SHST'] == 2016, :].copy()

test_and_school_df_8thgrade_2016only['Num Students Overlooked'] = test_and_school_df_8thgrade_2016only.loc[:,'Grade 8 Math 4s - All Students'].sub(test_and_school_df_8thgrade_2016only.loc[:,'Number of students who took the SHSAT'])

# #find correlation coefficients
# corr_coeffs_8thgrade_2016 = test_and_school_df_8thgrade_2016only.corr()['Num Students Overlooked']
# corr_rows_removed = ['District', 'Zip', 'Num Students Overlooked', 'Latitude', 'Longitude']

# for row_label, val in corr_coeffs_8thgrade_2016.iteritems():
#     if 'Grade 6' in row_label or 'Grade 7' in row_label or 'Normalized' in row_label:
#         corr_rows_removed.append(row_label)

# #Drop values that are non-sensical in a correlatin analysis and all N/A values as well
# corr_coeffs_8thgrade_2016 = pd.DataFrame(corr_coeffs_8thgrade_2016.drop(corr_rows_removed).dropna(how='all').sort_values())

# corr_coeffs_8thgrade_2016.style.bar(align='zero', color=['#5fba7d'])


#Remove rows for which there was no state testing data
test_and_school_df_8thgrade_2016only = test_and_school_df_8thgrade_2016only.loc[lambda df: df['Grade 8 Math - All Students Tested'] != 0]

#Define the training columns
train_cols = ['Grade 8 Math 4s - All Students']
#train_cols = ['Grade 8 ELA 4s - All Students', 'Grade 8 Math 4s - All Students']
#train_cols = ['Percent Black / Hispanic', 'Supportive Environment %', 'Collaborative Teachers %', 'Student Attendance Rate', 'Economic Need Index']
x_train = test_and_school_df_8thgrade_2016only.loc[:,train_cols]
y_train = test_and_school_df_8thgrade_2016only.loc[:,'Num Students Overlooked']



#Build the linear regression model
model = linear_model.LinearRegression()
#Fit the model
model.fit(x_train, y_train)
#Print the score
model_score = model.score(x_train, y_train)
#Declare the model's values as a seperate variable
y_model = model.predict(x_train)


#Show the relationship between number of students recieving a 4 on teh math section of the standardized test
#and the number of overlooked students
#Plot the data we are trying to fit vs. the model's output
plt.scatter(x_train, y_train, c='b', label='2016 Harlem School Data')
x_model_plot = np.array([0, 5, 10, 15, 20, 25, 30, 45, 50])
y_model_plot = np.add(model.intercept_, np.multiply(model.coef_[0], x_model_plot))
plt.plot(x_model_plot, y_model_plot, 'g', label='Linear Regression Model')
plt.ylabel('Number of High Potential Overlooked Students')
plt.xlabel('Number of students who scored a 4 on the Math Standardized Test')
plt.legend()
plt.text(35, 12, "y = {:.2f} + {:.2}x".format(model.intercept_, model.coef_[0]))
plt.show()


#Show some statistics about the final model (i.e., parameters, performance)
error = y_model - y_train
print("The regression model's score is: {:.2}".format(model_score))
print("The model's mean error is: {:.2}".format(np.mean(error)))
print("The model's std. dev. of error is: {:.2}".format(np.std(error)))

print("The model's y-intercept is {:.2f} and slope is {:.2}".format(model.intercept_, model.coef_[0]))


plt.hist(error)
plt.title('Distribution of error of the students overlooked linear regression model')
plt.xlabel('Number of Students Overlooked Error')
plt.show()


