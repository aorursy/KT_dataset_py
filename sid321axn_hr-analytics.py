import numpy as np # linear algebra

# import stat

from scipy import stats

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting intuitive plots and graphs

%matplotlib inline 

pd.set_option('display.max_columns', 500) # To display all the columns of dataframe

pd.set_option('max_colwidth', -1) # To set the width of the column to maximum

import warnings # To ignore warnings

warnings.filterwarnings("ignore")

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import seaborn as sns
## Functions for summary and plotting

# Get the summary info of data set

def tableSummary(df):

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing %'] = 100* np.round(df.isnull().sum().values/len(df),2)  

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values



    for name in summary['Name'].value_counts().index:

        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 



    return summary



# defining function for plotting

def univariate_percent_plot(cat):

    fig = plt.figure(figsize=(18,12))

    cmap=plt.cm.Blues

    cmap1=plt.cm.coolwarm_r

    ax1 = fig.add_subplot(221)

    ax2 = fig.add_subplot(222)

    

    result = df.groupby(cat).apply (lambda group: (group.Attrition == 0).sum() / float(group.Attrition.count())

         ).to_frame('Loyal')

    result['Left'] = 1 -result.Loyal

    result.plot(kind='bar', stacked = True,colormap=cmap1, ax=ax1)

    ax1.set_title('stacked Bar Plot of '+ cat +' (in %)', fontsize=14)

    ax1.set_ylabel('% Attrition Rate')

    ax1.legend(loc="lower right")

    loans_by_annual = df.groupby([cat, 'Attrition']).size()

    loans_by_annual.unstack().plot(kind='bar', stacked=True,ax=ax2)

    ax2.set_title('stacked Bar Plot of '+ cat +' (in %)', fontsize=14)

    ax2.set_ylabel('Number of Employees')

    

    

    plt.show()
#importing dataset

df = pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
#Lets see the sample of the dataset.

df.head()
# Get the data set summary 

df_info = tableSummary(df)

df_info
# As per the data description we will convert categorical numerically encoded columns into their categories



# binning worklifebalance

def bin_work_life(n):

    if n ==1:

        return 'Bad'

    elif n ==2:

        return 'Good'

    elif n ==3:

        return 'Better'

    elif n ==4:

        return 'Best'



df['WorkLifeBalance'] = df['WorkLifeBalance'].apply(lambda x: bin_work_life(x))



# binningPerformanceRating

def bin_PerformanceRating(n):

    if n ==1:

        return 'Low'

    elif n ==2:

        return 'Good'

    elif n ==3:

        return 'Excellent'

    elif n ==4:

        return 'Outstanding'



df['PerformanceRating'] = df['PerformanceRating'].apply(lambda x: bin_PerformanceRating(x))



# binning RelationshipSatisfaction

def bin_RelationshipSatisfaction(n):

    if n ==1:

        return 'Low'

    elif n ==2:

        return 'Medium'

    elif n ==3:

        return 'High'

    elif n ==4:

        return 'Very High'



df['RelationshipSatisfaction'] = df['RelationshipSatisfaction'].apply(lambda x: bin_RelationshipSatisfaction(x))



# binning JobSatisfaction

def bin_JobSatisfaction(n):

    if n ==1:

        return 'Low'

    elif n ==2:

        return 'Medium'

    elif n ==3:

        return 'High'

    elif n ==4:

        return 'Very High'



df['JobSatisfaction'] = df['JobSatisfaction'].apply(lambda x: bin_JobSatisfaction(x))



# binning JobInvolvement

def bin_JobInvolvement(n):

    if n ==1:

        return 'Low'

    elif n ==2:

        return 'Medium'

    elif n ==3:

        return 'High'

    elif n ==4:

        return 'Very High'



df['JobInvolvement'] = df['JobInvolvement'].apply(lambda x: bin_JobInvolvement(x))



def bin_EnvironmentSatisfaction(n):

    if n ==1:

        return 'Low'

    elif n ==2:

        return 'Medium'

    elif n ==3:

        return 'High'

    elif n ==4:

        return 'Very High'



df['EnvironmentSatisfaction'] = df['EnvironmentSatisfaction'].apply(lambda x: bin_EnvironmentSatisfaction(x))





def bin_Education(n):

    if n ==1:

        return 'Below College'

    elif n ==2:

        return 'College'

    elif n ==3:

        return 'Bachelor'

    elif n ==4:

        return 'Master'

    elif n==5:

        return 'Doctor'



df['Education'] = df['Education'].apply(lambda x: bin_Education(x))



# encoding attrition to binary value 0 and 1 0 for no attrition and 1 for attrition

df['Attrition'] = df['Attrition'].apply(lambda x: 0 if x=='No' else 1)



# converting Attrition datatype to integer type

df['Attrition'] = df['Attrition'].apply(lambda x: pd.to_numeric(x))



# segregating dataframes based on class label for plotting in future



attr_1=df[df['Attrition']==1]

attr_0=df[df['Attrition']==0]

# Get the data set summary again

df_info = tableSummary(df)

df_info
# dropping columns having 1 unique value and employee number



cols_to_drop = ['EmployeeCount','Over18','StandardHours','EmployeeNumber']

# dropping columns

df = df.drop(cols_to_drop, axis=1)

# Get the data set summary again

df_info = tableSummary(df)

df_info
# Plotting attrition of employees

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=(14,6))



ax1 = df['Attrition'].value_counts().plot.pie( x="Attrition" ,y ='no.of employees', 

                   autopct = "%1.0f%%",labels=["Working Employees","Attrition Employee"], startangle = 60,ax=ax1);

ax1.set(title = 'Percentage of Attrition of Employee in Dataset')



ax2 = df["Attrition"].value_counts().plot(kind="barh" ,ax =ax2)

for i,j in enumerate(df["Attrition"].value_counts().values):

    ax2.text(.5,i,j,fontsize=12)

ax2.set(title = 'No. of Attrition in Dataset')

plt.show()
fig = plt.figure(figsize=(15,5))

ax1 = plt.subplot2grid((1,2),(0,0))

sns.distplot(attr_0['Age'])

plt.title('AGE DISTRIBUTION OF LOYAL EMPLOYEES', fontsize=15, weight='bold')



ax1 = plt.subplot2grid((1,2),(0,1))

sns.countplot(attr_0['Gender'], palette='viridis')

plt.title('GENDER DISTRIBUTION OF LOYAL EMPLOYEES', fontsize=15, weight='bold' )

plt.show()



fig = plt.figure(figsize=(15,5))

ax1 = plt.subplot2grid((1,2),(0,0))

sns.distplot(attr_1['Age'])

plt.title('AGE DISTRIBUTION OF ATTRITION EMPLOYEES', fontsize=15, weight='bold')



ax1 = plt.subplot2grid((1,2),(0,1))

sns.countplot(attr_1['Gender'], palette='viridis')

plt.title('GENDER DISTRIBUTION OF ATTRITION EMPLOYEES', fontsize=15, weight='bold' )

plt.show()
fig, ax = plt.subplots(figsize=(10,4))



# Horizontal Bar Plot

title_cnt=df.Education.value_counts().sort_values(ascending=False).reset_index()

mn= ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1], color='silver')

mn[0].set_color('lightskyblue')

mn[4].set_color('crimson')





# Remove axes splines

for s in ['top','bottom','left','right']:

    ax.spines[s].set_visible(False)



# Remove x,y Ticks

ax.xaxis.set_ticks_position('none')

ax.yaxis.set_ticks_position('none')



# Add padding between axes and labels

ax.xaxis.set_tick_params(pad=5)

ax.yaxis.set_tick_params(pad=10)



# Add x,y gridlines

ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.4)



# Show top values 

ax.invert_yaxis()



# Add Plot Title

ax.set_title('Education Distribution of Employees in IBM',

             loc='center', pad=10, fontsize=16)

plt.yticks(weight='bold')





# Add annotation to bars

for i in ax.patches:

    ax.text(i.get_width()+10, i.get_y()+0.5, str(round((i.get_width()), 2)),

            fontsize=10, fontweight='bold', color='grey')

plt.yticks(weight='bold')

plt.xticks(weight='bold')

# Show Plot

plt.show()



attr_1=df[df['Attrition']==1]

fig, ax = plt.subplots(figsize=(10,4))



# Horizontal Bar Plot

title_cnt=attr_1.Education.value_counts().sort_values(ascending=False).reset_index()

mn= ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1], color='silver')

mn[0].set_color('red')

mn[4].set_color('blue')





# Remove axes splines

for s in ['top','bottom','left','right']:

    ax.spines[s].set_visible(False)



# Remove x,y Ticks

ax.xaxis.set_ticks_position('none')

ax.yaxis.set_ticks_position('none')



# Add padding between axes and labels

ax.xaxis.set_tick_params(pad=5)

ax.yaxis.set_tick_params(pad=10)



# Add x,y gridlines

ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.4)



# Show top values 

ax.invert_yaxis()



# Add Plot Title

ax.set_title('Education wise Distribution of Attrition Rate in IBM',

             loc='center', pad=10, fontsize=16)

plt.yticks(weight='bold')





# Add annotation to bars

for i in ax.patches:

    ax.text(i.get_width()+10, i.get_y()+0.5, str(round((i.get_width()), 2)),

            fontsize=10, fontweight='bold', color='grey')

plt.yticks(weight='bold')

plt.xticks(weight='bold')

# Show Plot

plt.show()
univariate_percent_plot('WorkLifeBalance')

#Exploring the attrition rate based on WorkLifeBalance

plot_criteria= ['WorkLifeBalance', 'Attrition']

cm = sns.light_palette("red", as_cmap=True)

(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)
univariate_percent_plot('JobSatisfaction')

#Exploring the attrition rate based on JobSatisfaction

plot_criteria= ['JobSatisfaction', 'Attrition']

cm = sns.light_palette("red", as_cmap=True)

(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)
univariate_percent_plot('JobInvolvement')

#Exploring the attrition rate based on JobInvolvement

plot_criteria= ['JobInvolvement', 'Attrition']

cm = sns.light_palette("red", as_cmap=True)

(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)

univariate_percent_plot('EnvironmentSatisfaction')

#Exploring the attrition rate based on EnvironmentSatisfaction

plot_criteria= ['EnvironmentSatisfaction', 'Attrition']

cm = sns.light_palette("red", as_cmap=True)

(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)

univariate_percent_plot('RelationshipSatisfaction')

#Exploring the attrition rate based on RelationshipSatisfaction

plot_criteria= ['RelationshipSatisfaction', 'Attrition']

cm = sns.light_palette("red", as_cmap=True)

(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)

univariate_percent_plot('PerformanceRating')

#Exploring the attrition rate based on PerformanceRating

plot_criteria= ['PerformanceRating', 'Attrition']

cm = sns.light_palette("red", as_cmap=True)

(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)

fig = plt.figure(figsize=(12,8))

sns.distplot(attr_0[['DistanceFromHome']],label="Loyal", hist=False)

sns.distplot(attr_1[['DistanceFromHome']],label="Left", hist=False)

plt.title('Distribution of Distance from Home')

plt.show()
# binning distance_from_home

def dist_home(n):

    if n <= 10:

        return 'Near'

    elif n > 10 and n <=25:

        return 'Far'

    elif n > 25:

        return 'Very far'

    

    

df['dist_home_bins'] = df['DistanceFromHome'].apply(lambda x: dist_home(x))
univariate_percent_plot('dist_home_bins')

#Exploring the attrition rate based on dist_home_bins

plot_criteria= ['dist_home_bins', 'Attrition']

cm = sns.light_palette("red", as_cmap=True)

(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)
univariate_percent_plot('Department')

#Exploring the attrition rate based on Department

plot_criteria= ['Department', 'Attrition']

cm = sns.light_palette("red", as_cmap=True)

(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)
univariate_percent_plot('BusinessTravel')

#Exploring the attrition rate based on BusinessTravel

plot_criteria= ['BusinessTravel', 'Attrition']

cm = sns.light_palette("red", as_cmap=True)

(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)
univariate_percent_plot('JobRole')

#Exploring the attrition rate based on JobRole

plot_criteria= ['JobRole', 'Attrition']

cm = sns.light_palette("red", as_cmap=True)

(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)
univariate_percent_plot('MaritalStatus')

#Exploring the attrition rate based on MaritalStatus

plot_criteria= ['MaritalStatus', 'Attrition']

cm = sns.light_palette("red", as_cmap=True)

(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)
fig = plt.figure(figsize=(12,8))

sns.distplot(attr_0[['MonthlyIncome']],label="Loyal", hist=False)

sns.distplot(attr_1[['MonthlyIncome']],label="Left", hist=False)

plt.title('Distribution of Monthly Income of Loyal vs Left Employees')

plt.show()
# binning total monthly_sal

bins = [5000, 10000, 15000, 20000]

df['dist_month_sal'] = pd.cut(df['MonthlyIncome'], bins)



univariate_percent_plot('dist_month_sal')

#Exploring the attrition rate based on dist_work_years

plot_criteria= ['dist_month_sal', 'Attrition']

cm = sns.light_palette("red", as_cmap=True)

(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)

fig = plt.figure(figsize=(12,8))

sns.distplot(attr_0[['NumCompaniesWorked']],label="Loyal", hist=False)

sns.distplot(attr_1[['NumCompaniesWorked']],label="Left", hist=False)

plt.title('Distribution of NumCompaniesWorked')

plt.show()
univariate_percent_plot('OverTime')

#Exploring the attrition rate based on OverTime

plot_criteria= ['OverTime', 'Attrition']

cm = sns.light_palette("red", as_cmap=True)

(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


univariate_percent_plot('EducationField')

#Exploring the attrition rate based on EducationField

plot_criteria= ['EducationField', 'Attrition']

cm = sns.light_palette("red", as_cmap=True)

(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)
fig = plt.figure(figsize=(12,8))

sns.distplot(attr_0[['PercentSalaryHike']],label="Loyal", hist=False)

sns.distplot(attr_1[['PercentSalaryHike']],label="Left", hist=False)

plt.title('Distribution of PercentSalaryHike')

plt.show()
# binning %salary hike

bins = [10, 15, 20, 25]

df['dist_sal_hike'] = pd.cut(df['PercentSalaryHike'], bins)



univariate_percent_plot('dist_sal_hike')

#Exploring the attrition rate based on dist_work_years

plot_criteria= ['dist_sal_hike', 'Attrition']

cm = sns.light_palette("red", as_cmap=True)

(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)

fig = plt.figure(figsize=(12,8))

sns.distplot(attr_0[['TotalWorkingYears']],label="Loyal", hist=False)

sns.distplot(attr_1[['TotalWorkingYears']],label="Left", hist=False)

plt.title('Distribution of TotalWorkingYears')

plt.show()



# binning total working years

bins = [0, 5, 10, 15, 20, 25, 30, 35, 40]

df['dist_work_years'] = pd.cut(df['TotalWorkingYears'], bins)



univariate_percent_plot('dist_work_years')

#Exploring the attrition rate based on dist_work_years

plot_criteria= ['dist_work_years', 'Attrition']

cm = sns.light_palette("red", as_cmap=True)

(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)

fig = plt.figure(figsize=(12,8))

sns.distplot(attr_0[['YearsAtCompany']],label="Loyal", hist=False)

sns.distplot(attr_1[['YearsAtCompany']],label="Left", hist=False)

plt.title('Distribution of YearsAtCompany')

plt.show()
# binning total working years

bins = [5, 10, 15, 20, 25, 30]

df['dist_years_comp'] = pd.cut(df['YearsAtCompany'], bins)



univariate_percent_plot('dist_years_comp')

#Exploring the attrition rate based on dist_work_years

plot_criteria= ['dist_years_comp', 'Attrition']

cm = sns.light_palette("red", as_cmap=True)

(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)

fig = plt.figure(figsize=(12,8))

sns.distplot(attr_0[['YearsInCurrentRole']],label="Loyal", hist=False)

sns.distplot(attr_1[['YearsInCurrentRole']],label="Left", hist=False)

plt.title('Distribution of YearsInCurrentRole')

plt.show()
fig = plt.figure(figsize=(12,8))

sns.distplot(attr_0[['YearsSinceLastPromotion']],label="Loyal", hist=False)

sns.distplot(attr_1[['YearsSinceLastPromotion']],label="Left", hist=False)

plt.title('Distribution of YearsSinceLastPromotion')

plt.show()
fig = plt.figure(figsize=(12,8))

sns.distplot(attr_0[['YearsWithCurrManager']],label="Loyal", hist=False)

sns.distplot(attr_1[['YearsWithCurrManager']],label="Left", hist=False)

plt.title('Distribution of YearsWithCurrManager')

plt.show()
univariate_percent_plot('TrainingTimesLastYear')

#Exploring the attrition rate based on TrainingTimesLastYear

plot_criteria= ['TrainingTimesLastYear', 'Attrition']

cm = sns.light_palette("red", as_cmap=True)

(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)
df.groupby('Gender')['Attrition'].mean().sort_values(ascending=False)
# defining a function which takes a categorical variable and plots the Attrition rate

# segmented by Gender 

def plot_segmented(cat_var):

    plt.figure(figsize=(10, 6))

    sns.barplot(x=cat_var, y="Attrition", hue="Gender", data=df)

    plt.show()
# Education: segmented by Gender

plot_segmented('Education')
# EducationField: segmented by Gender

plot_segmented('EducationField')
# WorkLifeBalance: segmented by Gender

plot_segmented('WorkLifeBalance')
# EnvironmentSatisfaction: segmented by Gender

plot_segmented('EnvironmentSatisfaction')
# JobSatisfaction: segmented by Gender

plot_segmented('JobSatisfaction')
# JobInvolvement: segmented by Gender

plot_segmented('JobInvolvement')
# OverTime: segmented by Gender

plot_segmented('OverTime')
# BusinessTravel: segmented by Gender

plot_segmented('NumCompaniesWorked')
# Distance from Home: segmented by Gender

plot_segmented('dist_home_bins')
# Department: segmented by Gender

plot_segmented('Department')
# MaritalStatus: segmented by Gender

plot_segmented('MaritalStatus')
# RelationshipSatisfaction: segmented by Gender

plot_segmented('RelationshipSatisfaction')
# BusinessTravel: segmented by Gender

plot_segmented('BusinessTravel')
# NumCompaniesWorked: segmented by Gender

plot_segmented('NumCompaniesWorked')
# dist_work_years: segmented by Gender

plot_segmented('dist_work_years')
# dist_sal_hike: segmented by Gender

plot_segmented('dist_sal_hike')
# dist_month_sal: segmented by Gender

plot_segmented('dist_month_sal')
# dist_years_comp: segmented by Gender

plot_segmented('dist_years_comp')
# Violin Plots

f, (ax) = plt.subplots(1, 1, figsize=(12, 6))

f.suptitle('Experience vs Monthly Income', fontsize=14)



sns.swarmplot(x="dist_work_years", y="MonthlyIncome", data=df,  ax=ax)

ax.set_xlabel("Total work experience in years",size = 12,alpha=0.8)

ax.set_ylabel("Monthly Income in US$",size = 12,alpha=0.8)

# Violin Plots

f, (ax) = plt.subplots(1, 1, figsize=(12, 6))

f.suptitle('Job Satisfaction vs Monthly Income', fontsize=14)



sns.boxplot(x="JobSatisfaction", y="MonthlyIncome",hue='Attrition', data=df,  ax=ax,palette="Blues")

ax.set_xlabel("Job Satisfaction",size = 12,alpha=0.8)

ax.set_ylabel("Monthly Income in US$",size = 12,alpha=0.8)

# Violin Plots

f, (ax) = plt.subplots(1, 1, figsize=(12, 6))

f.suptitle('Education vs Monthly Income', fontsize=14)



sns.boxplot(x="Education", y="MonthlyIncome",hue='Attrition', data=df,  ax=ax,palette="Blues")

ax.set_xlabel("Education",size = 12,alpha=0.8)

ax.set_ylabel("Monthly Income in US$",size = 12,alpha=0.8)

# Violin Plots

f, (ax) = plt.subplots(1, 1, figsize=(12, 6))

f.suptitle('Education vs Monthly Income', fontsize=14)



sns.boxplot(x="Department", y="MonthlyIncome",hue='Attrition', data=df,  ax=ax,palette="Paired")

ax.set_xlabel("Education",size = 12,alpha=0.8)

ax.set_ylabel("Monthly Income in US$",size = 12,alpha=0.8)

# Violin Plots

f, (ax) = plt.subplots(1, 1, figsize=(12, 6))

f.suptitle('DistanceFromHome vs OverTime', fontsize=14)



sns.boxplot(x="OverTime", y="DistanceFromHome",hue='Attrition', data=df,  ax=ax,palette="Paired")

ax.set_xlabel("OverTime",size = 12,alpha=0.8)

ax.set_ylabel("DistanceFromHome",size = 12,alpha=0.8)

# Violin Plots

f, (ax) = plt.subplots(1, 1, figsize=(12, 6))

f.suptitle('JobInvolvement vs PercentSalaryHike', fontsize=14)



sns.boxplot(x="JobInvolvement", y="PercentSalaryHike",hue='Attrition', data=df,  ax=ax,palette="Paired")

ax.set_xlabel("JobInvolvement",size = 12,alpha=0.8)

ax.set_ylabel("PercentSalaryHike",size = 12,alpha=0.8)

# Violin Plots

f, (ax) = plt.subplots(1, 1, figsize=(12, 6))

f.suptitle('JobSatisfaction vs PercentSalaryHike', fontsize=14)



sns.boxplot(x="JobSatisfaction", y="PercentSalaryHike",hue='Attrition', data=df,  ax=ax,palette="Paired")

ax.set_xlabel("JobSatisfaction",size = 12,alpha=0.8)

ax.set_ylabel("PercentSalaryHike",size = 12,alpha=0.8)
