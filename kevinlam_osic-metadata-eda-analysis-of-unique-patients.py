# OS (working folder) packages

import os



# DataFrame packages

import pandas as pd



# Visualisation packages

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap



from IPython.display import display, display_html



# Calculation packages 

import numpy as np



sns.set_style("whitegrid")
# Kevin - Data functions - filter dataframe according to specific value(s) in column "str_col"

def filter_data(data, str_col, list_filter):

    '''Example: filter_data(train_df, 'Patient', ['ID00007637202177411956430'])'''

    df = data.copy()

    df = df[df[str_col].isin(list_filter)]

    return(df)
# Plotting functions are written below, click "Code" to reveal.
# Kevin - plotting functions

def plt_fvc_vs_sex1(df, col_str, xlabel, subtitle, suptitle):

    '''few-time use only, see in sections'''

    

    bins = 30

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16,6))



    # Plot 1 (Left)

    sns.distplot(df[col_str], label = 'Male', bins = bins, ax = ax1)

    ax1.set(xlabel = xlabel, ylabel = "Density", title = subtitle + " (overall)")



    # Plot 2 (Right)

    # or use plt.hist() instead of sns.distplot()

    sns.distplot(filter_data(df, 'Sex', ['Male'])[col_str], label = 'Male', bins = bins, ax = ax2)

    sns.distplot(filter_data(df, 'Sex', ['Female'])[col_str], label = 'Female', bins = bins, ax = ax2)

    ax2.legend(loc = 'upper right')

    ax2.set(xlabel = xlabel, ylabel = "Density", title = subtitle + " (by gender)")



    # Overall info

    plt.suptitle(suptitle)

    plt.show()

    

def plt_fvc_vs_sex2(df1, df2, col_str, xlabel, subtitle1, subtitle2, suptitle):

    '''few-time use only, see in sections'''

    

    bins = 30

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16,6))



    # plot 1

    sns.distplot(filter_data(df1, 'Sex', ['Male'])[col_str], label = 'Male', bins = bins, ax = ax1)

    sns.distplot(filter_data(df1, 'Sex', ['Female'])[col_str], label = 'Female', bins = bins, ax = ax1)



    # plot 1 - legends

    ax1.legend(loc = 'upper right')

    ax1.set(xlabel = xlabel, ylabel = "Density", title = subtitle1 + " (by gender)")



    # plot 2

    sns.distplot(filter_data(df2, 'Sex', ['Male'])[col_str], label = 'Male', bins = bins, ax = ax2)

    sns.distplot(filter_data(df2, 'Sex', ['Female'])[col_str], label = 'Female', bins = bins, ax = ax2)



    # plot 2 - legends

    ax2.legend(loc = 'upper right')

    ax2.set(xlabel = xlabel, ylabel = "Density", title = subtitle2 + " (by gender)")



    # Overall plot info

    plt.suptitle(suptitle)

    plt.show()

    



def plt_fvc_vs(df, x, y, z, color_palette):

    '''Example: plt_fvc_vs(train_df, x = 'Percent', y = 'FVC', z = 'Age', color_palette = 'YlGn')'''

    

    my_cmap = ListedColormap(sns.color_palette(color_palette))



    plt.figure(figsize = (15,9))

    plt.scatter(df[x], df[y], c = df[z], cmap=my_cmap)

    plt.xlabel(x); plt.ylabel(y); plt.title(y + " vs. " + x + " (" + z + " in colours)")

    plt.colorbar()

    plt.show()



def plt_fvc_vs_sns(df, x, y, z, color_palette):

    '''Example: plt_fvc_vs_sns(df = train_df, x = 'Percent', y = 'FVC', z = 'Sex', color_palette = 'CMRmap_r')'''

    df1 = df.copy()

    df_unique = len(df[z].unique())

    

    if (z == "Sex_Smoke"):

        df1 = df1.sort_values(z, ascending = False)

    

    sns.lmplot(x = x, y = y, data = df1, hue = z,

               fit_reg = False, legend = True, palette = sns.color_palette(color_palette, df_unique), height = 6, aspect = 2)

    ax = plt.gca().set_title(y + " vs. " + x + " (" + z + " in colours)")

    

###### Functions for extra plots

def plt_fvc_vs_subplot(df, x, y, z, color_palette, ax):

    '''Example: plt_fvc_vs_subplot(train_df, x = 'Percent', y = 'FVC', z = 'Age', color_palette = 'YlGn', ax = ax1)'''

    

    my_cmap = ListedColormap(sns.color_palette(color_palette))



    im = ax.scatter(df[x], df[y], c = df[z], cmap = my_cmap)

    ax.set(xlabel = x, ylabel = y, title = y + " vs. " + x + " (" + z + " in colours)")

    plt.colorbar(im, ax = ax)

    

def plt_fvc_vs_compare(df1, df2, x, y, z1, z2, color_palette):

    '''Example: plt_fvc_compare(first_visit, last_visit, x = 'Percent', y = 'FVC', z = 'Age', color_palette = 'YlGn')'''

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20,8))

    plt_fvc_vs_subplot(df1, x = x, y = y, z = z1, color_palette = color_palette, ax = ax1)

    plt_fvc_vs_subplot(df2, x = x, y = y, z = z2, color_palette = color_palette, ax = ax2)

    if(z1 == z2):

        plt.suptitle(y + " vs. " + x + " (" + z1 + " in colours) - (first vs. last visit are left/right)")

    else:

        plt.suptitle(y + " vs. " + x + " (" + z1 + " and " + z2 + " in colours (left/right))")
# List files and folders available

list(os.listdir("../input/osic-pulmonary-fibrosis-progression"))
train_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")

test_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")

sample_submission = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
display(train_df.head(6))

display(test_df.head())
# Natalia - Patients



# Number of unique patients

print('Number of unique patients in training set: {}'.format(train_df['Patient'].nunique()))

print('Number of unique patients in test set: {}'.format(test_df['Patient'].nunique()), '\n')



# Patient ID

patient_id_train = set(train_df['Patient'].unique())

patient_id_test = set(test_df['Patient'].unique())

print('Patients in both training and test set:')

print(list(patient_id_train.intersection(patient_id_test)))
# Below: We write a data frame patient_info_df, which filters train_df for unique patients.

# We also add some summary statistics. Click "Code" to reveal.
# Natalia - Obtain patient information



# Obtain FVC, Percent, Weeks statistics

patient_info_df = train_df.groupby(['Patient','Sex','Age','SmokingStatus']).agg({'FVC': ['count','mean','std','min','max'],'Percent': ['mean','std','min','max'],'Weeks': ['min','max']}) 



# Rename columns

patient_info_df.columns = ["_".join(x) for x in patient_info_df.columns.ravel()]

patient_info_df = patient_info_df.rename(columns = {'FVC_count':'Count'})



# Obtain range for weeks

patient_info_df['Weeks_range'] = patient_info_df['Weeks_max'] - patient_info_df['Weeks_min']



# Reset index

patient_info_df = patient_info_df.drop(['Weeks_min','Weeks_max'],1).reset_index()



# Combine Sex and SmokingStatus

patient_info_df = patient_info_df.assign(Sex_Smoke = patient_info_df.Sex.astype(str) + '_' + patient_info_df.SmokingStatus.astype(str))
# Display dataframe

display(patient_info_df)
# Below: We filter the dataframe train_df according to the first and last visits.

# We keep corresponding values. Click "Code" to reveal.
# Nikolas - Dataframes with first and last visits only

first_visit = pd.DataFrame(data = None, columns = train_df.columns)

last_visit = pd.DataFrame(data = None, columns = train_df.columns)



for pat in patient_info_df['Patient']: 

     new = filter_data(train_df,'Patient',[pat]).iloc[0]

     first_visit = first_visit.append(new,ignore_index = True)

     new = filter_data(train_df,'Patient',[pat]).iloc[-1]

     last_visit = last_visit.append(new,ignore_index = True)
display(first_visit.head())

display(last_visit.head())
# Nikolas

patient_info_df_copy = patient_info_df.copy()

patient_info_df_copy['Sex'].replace('Female',0,inplace=True)

patient_info_df_copy['Sex'].replace('Male',1,inplace=True)



patient_info_df_copy['SmokingStatus'].replace('Never smoked', 0, inplace = True)

patient_info_df_copy['SmokingStatus'].replace('Ex-smoker', 1, inplace = True)

patient_info_df_copy['SmokingStatus'].replace('Currently smokes', 0, inplace = True)
# Plot of correlation heatmap of feature summaries and responses

plt.gcf().subplots_adjust(bottom = 0.15)



corr = patient_info_df_copy.corr()

top_corr_features = corr.index



plt.figure(figsize = (12,10))

ax = sns.heatmap(patient_info_df_copy[top_corr_features].corr(),annot=True,cmap="RdYlGn")



bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)

ax.set_title("Correlation heatmap of feature sumamries and response (filtered for unique patients)")

plt.show()
# Nikolas 



plt.figure(figsize = (14,6))

ax = sns.countplot(x = "Weeks", data = train_df, color = 'lightskyblue')

ax.set_title("Histogram of weeks (how many patients visited at particular 'week')")



for ind, label in enumerate(ax.get_xticklabels()):

    if ind % 10 == 0:  # every 10th label is kept

        label.set_visible(True)

    else:

        label.set_visible(False)  
# Plot number of visits

plt.figure(figsize = (14,6))

ax = sns.countplot(x = "Count", data = patient_info_df, color = 'olivedrab')

ax.set(xlabel = "Number of visits", title = "Number of patient visits")

plt.show()
# Natalia - Weeks in which a patient visits (baseline is first FVC measurement)

train_df['Weeks'].describe()



# Plot gaps between visits

# Notice some patients stop visits at week 27, most go up to week 50+

patient_weeks = {}

plt.figure(figsize = (12,10))



for i,j in enumerate(patient_id_train):

    

    weeks = np.array(filter_data(train_df, 'Patient', [j])['Weeks'])

    weeks = weeks - weeks[0]

    patient_weeks[j] = weeks

    plt.plot(weeks)

    plt.xticks(np.arange(0,10,1))

    plt.xlabel("Visits")

    plt.ylabel("Weeks")

    

    if i <= 20:

        print(j, weeks)    

    else:

        continue



plt.title("Weeks in which a patient visits (baseline is first FVC measurement)")

plt.xlabel("i-th visit"), plt.ylabel("'Week' corresponding to i-th visit")

plt.show()
# Natalia

# Sex, Smoking Status 

def display_side_by_side(*args):

    html_str=''

    for df in args:

        html_str+=df.to_html()

    display_html(html_str.replace('table','table style="display:inline"'),raw=True)

    

display_side_by_side(patient_info_df.groupby(['Sex']).count()['Patient'].to_frame(),

                 patient_info_df.groupby(['SmokingStatus']).count()['Patient'].to_frame(),

                 patient_info_df.groupby(['Sex','SmokingStatus']).count()['Patient'].to_frame())
# Below: Code to create barplot, with analysis of categorical data. Click "Code" to reveal.
# Extract the data to add to the barplots

male_smoking = filter_data(patient_info_df, 'Sex', ['Male']).groupby(['SmokingStatus']).count()['Patient']

female_smoking = filter_data(patient_info_df, 'Sex', ['Female']).groupby(['SmokingStatus']).count()['Patient']

smoking_sex1 = filter_data(patient_info_df, 'SmokingStatus', ['Currently smokes']).groupby(['Sex']).count()['Patient']

smoking_sex2 = filter_data(patient_info_df, 'SmokingStatus', ['Ex-smoker']).groupby(['Sex']).count()['Patient']

smoking_sex3 = filter_data(patient_info_df, 'SmokingStatus', ['Never smoked']).groupby(['Sex']).count()['Patient']



# labels for x and y axis

smokingstatus_lab = ['Currently smokes', 'Ex-smoker', 'Never smoked']

sex_lab = ['Female', 'Male']



fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,6))



# Plot 1

ax1.bar(x = smokingstatus_lab, height = male_smoking, bottom = female_smoking, label = 'Male')

ax1.bar(x = smokingstatus_lab, height = female_smoking, label = 'Female')



# Plot 1 - details

ax1.set(xlabel = "SmokingStatus", ylabel = "Counts", title = "Number of smokers, conditioned on Sex")

ax1.legend()



# Plot 2

ax2.bar(x = sex_lab, height = smoking_sex1, label = 'Currently Smokes')

ax2.bar(x = sex_lab, height = smoking_sex2, bottom = smoking_sex3 + smoking_sex1, label = 'Ex-smoker')

ax2.bar(x = sex_lab, height = smoking_sex3, bottom = smoking_sex1, label = 'Never smoked')



# Plot 2 - details

ax2.set(xlabel = "Sex", ylabel = "Counts", title = "Number of sex, conditioned on SmokingStatus")

plt.legend()



# Overall info

fig.suptitle('Comparison of Categorical Data (Filtered for unique patients)')

plt.show()
# Density plot of percent (not filtered for unique)

plt_fvc_vs_sex1(train_df, 'Percent', "Percent of Patient FVC compared to usual FVC", 'Histogram of Percent',

               "Histogram of the distribution of Percent (Not filtered for unique patients)")



# Density plot of percent (first and last visit)

plt_fvc_vs_sex2(first_visit, last_visit, 'Percent', "Percent of Patient FVC compared to usual FVC",

                "Histogram of Percent, first visit", "Histogram of Percent, last visit",

                "Comparison of percentage values, for first and last visit")
# Density plot of age (filtered for unique)

plt_fvc_vs_sex1(patient_info_df, 'Age', "Age (in years)", 'Histogram of Age',

               "Histogram of the distribution of Age (filtered for unique patients)")
# Nikolas - Time Series of FVC 



# most patients have their FVCs not change massively between successive visits

plt.figure(figsize = (12,8))

for pat in patient_id_train:

       filt = filter_data(train_df, 'Patient',[pat])['FVC'].tolist()

       plt.plot(range(1,len(filt)+1),filt)



plt.xlabel("i-th visit"); plt.ylabel("FVC value"); plt.title("FVC levels by visit")

plt.show()  
# Histogram/density for FVC (not filtered)

plt_fvc_vs_sex1(train_df, 'FVC', 'Lung Capacity (ml)', 'Histogram of FVC', "Overall FVC comparison (not filtered for unique patients)")



# Histogram/density for mean FVC (filtered)

plt_fvc_vs_sex1(patient_info_df, 'FVC_mean', 'Lung Capacity (ml)', 'Histogram of FVC_mean', "Overall FVC comparison (filtered for unique patients)")



# For Male patients, there seems to be a shift towards the right, while this 

# is not the case for Female Patients
# define color palettes

# col_list = ['CMRmap_r','gist_ncar_r', 'hsv_r', 'jet_r', 'nipy_spectral_r', 'rainbow_r']



color_sex = 'nipy_spectral_r'

color_age = 'nipy_spectral_r'

color_percent = 'gist_ncar_r'
# Natalia - 6 Categories



plt_fvc_vs_sns(df = patient_info_df, x = 'Age', y = 'FVC_mean', z = 'Sex_Smoke', color_palette = color_sex)

plt_fvc_vs_sns(df = patient_info_df, x = 'Percent_mean', y = 'FVC_mean', z = 'Sex_Smoke', color_palette = color_sex)

# plt_fvc_vs_sns(df = patient_info_df, x = 'Weeks_range', y = 'FVC_mean', z = 'Sex_Smoke', color_palette = 'CMRmap_r')
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (20,8))



plt_fvc_vs_subplot(patient_info_df, x = 'Percent_mean', y = 'FVC_mean', z = 'Age', color_palette = color_age, ax = ax1)

plt_fvc_vs_subplot(patient_info_df, x = 'Age', y = 'FVC_mean', z = 'Percent_mean', color_palette = color_percent, ax = ax2)



plt.suptitle("FVC_mean vs. continuous (summary) statistics (filtered for unique patients)")

plt.show()
# Kevin - Nikolas Modification

# Continuous conditions (see "z = [...]")



# FVC vs. Age (by Percent)

plt_fvc_vs_compare(first_visit, last_visit, x = 'Age', y = 'FVC', z1 = 'Percent', z2 = 'Percent', color_palette = color_percent)



# FVC vs. Age (by Weeks)

plt_fvc_vs_compare(first_visit, last_visit, x = 'Age', y = 'FVC', z1 = 'Weeks', z2 = 'Weeks', color_palette = color_age)
# Add Sex_smoke status

train_df_cat = train_df.copy()

train_df_cat = train_df_cat.assign(Sex_Smoke = train_df_cat.Sex.astype(str) + '_' + train_df_cat.SmokingStatus.astype(str))

display(train_df_cat.head())
plt_fvc_vs_compare(train_df, train_df, x = 'Percent', y = 'FVC', z1 = 'Age', z2 = 'Weeks', color_palette = color_age)

plt_fvc_vs_compare(train_df, train_df, x = 'Age', y = 'FVC', z1 = 'Percent', z2 = 'Weeks', color_palette = color_percent)
plt_fvc_vs_sns(df = train_df_cat, x = 'Percent', y = 'FVC', z = 'Sex_Smoke', color_palette = color_sex)

plt_fvc_vs_sns(df = train_df_cat, x = 'Age', y = 'FVC', z = 'Sex_Smoke', color_palette = color_sex)

plt_fvc_vs_sns(df = train_df_cat, x = 'Weeks', y = 'FVC', z = 'Sex_Smoke', color_palette = color_sex)
# Below are previously used plots, which we felt are not needed to present (click "Code" and "Output" to show).
# Previously used plots:

import plotly.express as px



# Nikolas - interactive plots



# Kevin version of Nikolas (static instead of interactive)

plt_fvc_vs_sns(patient_info_df, x = 'Weeks_range', y = 'Age', z = 'Sex_Smoke', color_palette = 'nipy_spectral_r')



# Age / Weeks scatterplot colored by Sex

fig = px.scatter(train_df, x="Weeks", y="Age", color='Sex')

fig.show()



# Age / Weeks scatterplot colored by SmokingStatus

fig = px.scatter(train_df, x="Weeks", y="Age", color='SmokingStatus')

fig.show()





# Histogram/density for intial and final FVC (filtered) - NOT RELEVANT

plt_fvc_vs_sex2(first_visit, last_visit, 'FVC', 'Lung Capacity (ml)', 'Histogram with initial FVC ', 'Histogram with final FVC ',

                'Histograms with intial and final FVC (filtered for unique patients)')



plt_fvc_vs_sns(train_df, x = 'Weeks', y = 'Age', z = 'Sex', color_palette = 'nipy_spectral_r')

plt_fvc_vs_sns(train_df, x = 'Weeks', y = 'Age', z = 'SmokingStatus', color_palette = 'nipy_spectral_r')



plt_fvc_vs_sns(df = first_visit, x = 'Age', y = 'FVC', z = 'Sex', color_palette = color_sex)

plt_fvc_vs_sns(df = first_visit, x = 'Age', y = 'FVC', z = 'SmokingStatus', color_palette = color_sex)



# FVC vs. Percent (by Age)

plt_fvc_vs_compare(first_visit, last_visit, x = 'Percent', y = 'FVC', z1 = 'Age', z2 = 'Age', color_palette = color_age)



# FVC vs. weeks

plt_fvc_vs_compare(train_df, train_df, x = 'Weeks', y = 'FVC', z1 = 'Percent', z2 = 'Age', color_palette = color_age)



# # FVC vs Percent_mean (by Sex and Smoking Status)

plt_fvc_vs_sns(df = patient_info_df, x = 'Percent_mean', y = 'FVC_mean', z = 'Sex', color_palette = color_sex)

plt_fvc_vs_sns(df = patient_info_df, x = 'Percent_mean', y = 'FVC_mean', z = 'SmokingStatus', color_palette = color_sex)



# # FVC vs Age (by Sex and Smoking Status)

plt_fvc_vs_sns(df = train_df, x = 'Age', y = 'FVC', z = 'Sex', color_palette = color_sex)

plt_fvc_vs_sns(df = train_df, x = 'Age', y = 'FVC', z = 'SmokingStatus', color_palette = color_sex)



# # FVC vs Weeks (by Sex and Smoking Status)

plt_fvc_vs_sns(df = train_df, x = 'Weeks', y = 'FVC', z = 'Sex', color_palette = color_sex)

plt_fvc_vs_sns(df = train_df, x = 'Weeks', y = 'FVC', z = 'SmokingStatus', color_palette = color_sex)