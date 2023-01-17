# import required packages

import pandas as pd

import numpy as np

import seaborn as sns

from scipy.stats import norm

from scipy import stats

import matplotlib.pyplot as plt

%matplotlib inline

%config InlineBackend.figure_format='retina'
# loading the 'Medical Appointment No Slows' dataset

df = pd.read_csv('../input/noshowappointments/KaggleV2-May-2016.csv')



# data overview

print('The dataset has :',df.shape[0],' rows, and :',df.shape[1],' columns.')

df.head()
# get information about data

df.info()
# describing data

df.describe()
# check for negative values in the 'Age' column

df[df.Age < 0]
# checking for missing values

df.isnull().any()
# checking for duplicate data

df.duplicated().sum()
# Print Unique Values

print('there are ',df.Gender.nunique()," Unique Values in `Gender` =>" , df.Gender.unique())

print('there are ',df.Scholarship.nunique()," Unique Values in `Scholarship` =>" , df.Scholarship.unique())

print('there are ',df.Hipertension.nunique()," Unique Values in `Hipertension` =>" , df.Hipertension.unique())

print('there are ',df.Diabetes.nunique()," Unique Values in `Diabetes` =>" , df.Diabetes.unique())

print('there are ',df.Alcoholism.nunique()," Unique Values in `Alcoholism` =>" , df.Alcoholism.unique())

print('there are ',df.Handcap.nunique()," Unique Values in `Handcap` =>" , df.Handcap.unique())

print('there are ',df.SMS_received.nunique()," Unique Values in `SMS_received` =>" , df.SMS_received.unique())

print('there are ',df['No-show'].nunique()," Unique Values in `No-show` =>" , df['No-show'].unique())

print('there are ',df.Neighbourhood.nunique(),"Unique Values in `Neighbourhood` =>" , df.Neighbourhood.unique())



# current columns' names

df.columns
# Rename incorrect columns' names and makes it all a lower case



new_labls = ['patient_Id', 'appointment_id', 'gender', 'scheduled_day',

       'appointment_day', 'age', 'neighbourhood', 'scholarship', 'hypertension',

       'diabetes', 'alcoholism', 'handicap', 'sms_received', 'appointment']

df.columns = new_labls
# Convert PatientId from Float to Integer

df['patient_Id'] = df['patient_Id'].astype('int64')
# convert all the categorical variables from 'int' to 'category' type



df['gender'] = df['gender'].astype('category')

df['scholarship'] = df['scholarship'].astype('category')

df['hypertension'] = df['hypertension'].astype('category')

df['diabetes'] = df['diabetes'].astype('category')

df['alcoholism'] = df['alcoholism'].astype('category')

df['handicap'] = df['handicap'].astype('category')

df['sms_received'] = df['sms_received'].astype('category')

df['appointment'] = df.appointment.astype('category')
# update variables' categories



df.appointment.cat.rename_categories(['Show','No Show'], inplace = True)



df.gender.cat.rename_categories(['Female','Male'], inplace = True)



df.scholarship.cat.rename_categories(['No Scholarship','Scholarship'], inplace = True)



df.hypertension.cat.rename_categories(['No Hypertension','Hypertension'], inplace = True)



df.diabetes.cat.rename_categories(['No Diabetes','Diabetes'], inplace = True);

# Convert 'scheduled_day' and 'appointment_day' from 'object' type to 'datetime64[ns]'

df['scheduled_day'] = pd.to_datetime(df['scheduled_day']).dt.date.astype('datetime64[ns]')

df['appointment_day'] = pd.to_datetime(df['appointment_day']).dt.date.astype('datetime64[ns]')



df.head()

df.info()
# Row with incorrect age value (-1)

incorrect_row = df[df['age'] == -1].index

print(df.shape)

df[df['age'] == -1]
# Removing the row with incorrect age value (-1)

df.drop(incorrect_row ,axis=0,inplace=True)

print(df.shape)

df[df.age == -1]
# create 'age_category' variable for make the analysis easier

df['age_category'] = pd.cut(df.age,bins=[-1,2,17,39,60,116],labels=['0 - 2','3 - 17','18 - 39','40 - 59', '60 - 115'],)

# Trimming data regarding the variables needed in my research questions

df = df.iloc[: , np.r_[2:3,5:6,7:10,13:15]]



df.head()

# Create a function to generate a countplot for one categorical variable

# it should provide a dataset as 'df', a name of categorical variable as 'vb' and number of color as 'i'.



def count_plot_one_vb(df, vb,i):

    base_color = sns.color_palette()[i]

    sns.countplot(data = df, x = vb, color = base_color)

    plt.xlabel(vb.upper())

    plt.ylabel('Number of patients \n ')

    plt.title('Number of patients by '+ vb +' \n',fontsize=16)

    # add annotations

    n_points = df.shape[0]

    gen_counts = df[vb].value_counts()

    locs, labels = plt.xticks() # get the current tick locations and labels



    # loop through each pair of locations and labels

    for loc, label in zip(locs, labels):



        # get the text property for the label to get the correct count

        count = gen_counts[label.get_text()]

        pct_string = '{:0.1f}%'.format(100*count/n_points)



        # print the annotation just below the top of the bar

        plt.text(loc, count-8, pct_string, ha = 'center', fontsize=11, color = 'black')

    return 
# Create a funtion with 3 arguments >> (dataframe as 'df', variable 1 as 'vb1' and variable 2 as 'vb2') to generate:

# 1)- a pivot table between the two variables,

# 2)- a rate bar chart with:

# - the first variable as 'vb1' in horizontal axis(x),

# - the second variable as 'vb2' proportion in (y) axis, and show in legend),

# -- this function concerns only categorical variables --



def rate_bar_chart_2vb(df, vb1,vb2):

    

    # pivot-table 

    df_by_vb_count = df.pivot_table(index = vb1, columns = vb2, values = 'age', aggfunc = 'count',margins = True)

    

    #rate bar chart

    df_by_vb = pd.crosstab(df[vb1], df[vb2], normalize = 'index')

    df_by_vb = np.round((df_by_vb * 100), decimals=2)

    ax = df_by_vb.plot.bar(figsize=(10,5));

    vals = ax.get_yticks()

    ax.set_yticklabels(['{:3.0f}%'.format(x) for x in vals]);

    ax.set_xticklabels(df_by_vb.index,rotation = 0, fontsize = 15);

    ax.set_title('\n '+ vb2.upper() + ' (%) by ' + df_by_vb.index.name + '\n', fontsize = 15)

    ax.set_xlabel(df_by_vb.index.name.upper(), fontsize = 12)

    ax.set_ylabel('(Percentage %)', fontsize = 12)

    ax.legend(loc = 'upper left',bbox_to_anchor=(1.0,1.0), fontsize= 12)

    rects = ax.patches

    

    # Add Data Labels

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2, 

                height + 2, 

                str(height)+'%', 

                ha='center', 

                va='bottom',

                fontsize = 12)

        

    return  df_by_vb_count 
# Create a funtion with 2 arguments >> (datafram as 'df' and binary variable as 'vb') to generate:

# 1)- a pivot table between the appointment variable and the binary variable in arguments,

# 2)- stacked bar chart with:

# - the variable 'vb' in horizontal axis(x),

# - the 'appointment' variable in (y) axis.



def stacked_bar_appointment_with_vb(df, vb):

    

    # pivot table 'gender' by 'appointment' 

    group_vb_all= df.pivot_table(index = vb, columns = 'appointment', values = 'age', aggfunc = 'count',margins = True)

    

    # plot

    group_vb = pd.pivot_table(df,index = vb, columns = 'appointment', values = 'age',aggfunc = 'count')

    ind = range(len(df[vb].value_counts()))

    width = 0.8

    p1=plt.bar(ind, group_vb.Show, width)

    p2 = plt.bar(ind, group_vb['No Show'], width)

    plt.legend(['Show','No Show'])

    plt.xticks(ind, group_vb.index)

    plt.ylabel('Number of patients')

    plt.title('Number of patients (Show/No Show) appointment by ' + vb + '\n ', fontsize=16)

    

    return group_vb_all
# number of patients grouping by 'appointment' variable

app = df.groupby('appointment')['appointment'].count()

sumapp = app.sum()

print(app)

print('\n From ',sumapp,' patents, there are ', app[1], ' patients showed up at their appointments, which represents', 

      np.round(app[1]/sumapp,4)*100, '% of the total,','\n and ',app[0], ' patients no show, which represents', 

      np.round(app[0]/sumapp,4)*100, '% of the total.')
count_plot_one_vb(df, 'appointment',7)

plt.title('Number of patients by (showing/no showing) at the appointment',fontsize=16);
# number of patients grouping by 'gender' variable

gd = df.groupby('gender')['gender'].count()

sumgd = gd.sum()

print(gd)

print('\n From ',sumgd,' patents, there are ', gd[0], ' female, which represents', 

      round(gd[0]/sumgd,2)*100, '% of the total,','\n and ',gd[1], ' male, which represents', 

      round(gd[1]/sumgd,2)*100, '% of the total.')
# gender count plot

count_plot_one_vb(df, 'gender',0)
# stacked bar chart 'appointment' by Gender

stacked_bar_appointment_with_vb(df, 'gender')
# bar chart comparison femal and mal by 'appointment' variable.

rate_bar_chart_2vb(df, 'gender', 'appointment')
# unique values

print(df.age.nunique())

np.sort(df.age.unique())
# frequency 

df.age.value_counts()
# 

print(df.age.describe())
# simple histogram of 'age' variable

plt.hist(df.age,bins=10);

# violin plot of the 'age' series

plt.violinplot(df.age, showmedians=True)

plt.xlim([0.5,1.5])

plt.title('Age',fontsize=(14))

plt.ylabel('Years',fontsize=(12))

plt.grid();
#histogram 'age' series and normal probability plot

sns.distplot(df.age, bins=10, fit=norm)

fig = plt.figure()

res = stats.probplot(df.age, plot=plt)

plt.grid();
# create two dataframes frames one for 'female' and another for 'male'

female = df[df.gender == 'Female']

male = df[df.gender == 'Male']

# histogram for each Gender

female.age.hist(label='Female', alpha=0.5, bins=10)

male.age.hist(label='Male', alpha=0.5, bins=10)

plt.ylabel('Number of patients')

plt.xlabel('Age')

plt.title('Distribution of Age by Gender')

plt.legend();
# number of patients grouping by 'age_category' variable

ac = df.groupby('age_category')['age_category'].count()

sumac = ac.sum()

print(ac)

print('\n From ',sumac,' patents, there are: \n -', ac[0], ' in [0 - 2] category, which represents', 

      round(ac[0]/sumac,3)*100, '% of the total,\n -', ac[1], ' in [3 - 17] category, which represents', 

      round(ac[1]/sumac,3)*100, '% of the total,\n -', ac[2], ' in [18 - 39] category, which represents', 

      round(ac[2]/sumac,3)*100, '% of the total,\n -', ac[3], ' in [40 - 59] category, which represents', 

      round(ac[3]/sumac,4)*100, '% of the total,\n -',' and ',ac[4], ' in [60 - 115] category, which represents', 

      round(ac[4]/sumac,3)*100, '% of the total.')
# bar plot with proportion age category

count_plot_one_vb(df,'age_category',4)
# stacked bar chart 'appointment' by age_category 

stacked_bar_appointment_with_vb(df, 'age_category')



# bar chart comparison proportion of (show/no show) by 'age_category' variable.

rate_bar_chart_2vb(df, 'age_category', 'appointment')
# bar chart comparison proportion of (Female/Male) by 'age_category' variable.

rate_bar_chart_2vb(df, 'age_category', 'gender')
# number of patients grouping by 'scholarship' variable

sh = df.groupby('scholarship')['scholarship'].count()

sumsh = sh.sum()

print(sh)

print('\n From ',sumsh,' patents, there are ', sh[1], ' who have a scholarship, which represents', 

      round(sh[1]/sumsh,3)*100, '% of the total,','\n and ',sh[0], ' who haven not a scholarship, which represents', 

      round(sh[0]/sumsh,3)*100, '% of the total.')
# scholarship count plot

count_plot_one_vb(df, 'scholarship',2)
# stacked bar chart 'appointment' by Scholarship 

stacked_bar_appointment_with_vb(df, 'scholarship')



# bar chart comparison the proportion of (show/no show) by 'scholarship' variable.

rate_bar_chart_2vb(df, 'scholarship', 'appointment')
# number of patients grouping by 'hypertension' variable

hy = df.groupby('hypertension')['hypertension'].count()

sumhy = hy.sum()

print(hy)

print('\n From ',sumhy,' patents, there are ', hy[1], ' who have an hypertension, which represents', 

      round(hy[1]/sumhy,3)*100, '% of the total,','\n and ',hy[0], ' who haven not an hypertension, which represents', 

      round(hy[0]/sumhy,4)*100, '% of the total.')
# hypertension count plot

count_plot_one_vb(df, 'hypertension',3)
# stacked bar chart 'appointment' by hypertension 

stacked_bar_appointment_with_vb(df, 'hypertension')



# bar chart comparison the proportion of (show/no show) by 'hypertension' variable.

rate_bar_chart_2vb(df, 'hypertension','appointment')
# bar chart comparison the proportion of (hypertension/no hypertension) by 'age_category' variable.

rate_bar_chart_2vb(df, 'age_category', 'hypertension')
# number of patients grouping by 'diabetes' variable

db = df.groupby('diabetes')['diabetes'].count()

sumdb = db.sum()

print(db)

print('\n From ',sumdb,' patents, there are ', db[1], ' who have an diabetes, which represents', 

      round(db[1]/sumdb,4)*100, '% of the total,','\n and ',db[0], ' who haven not an diabetes, which represents', 

      round(db[0]/sumdb,4)*100, '% of the total.')
# diabetes count plot

count_plot_one_vb(df, 'diabetes',8)
# stacked bar chart 'appointment' by diabetes 

stacked_bar_appointment_with_vb(df, 'diabetes')



# bar chart comparison the proportion of (show/no show) by 'diabetes' variable.

rate_bar_chart_2vb(df, 'diabetes','appointment')
# bar chart comparison the proportion of (diabetic/no diabetic) by 'age_category' variable.

rate_bar_chart_2vb(df, 'age_category', 'diabetes')
# bar chart comparison the proportion of (hypertension/no hypertension) by 'diabetes' variable.

rate_bar_chart_2vb(df, 'diabetes', 'hypertension')