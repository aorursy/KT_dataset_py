# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib import colors

from matplotlib.ticker import PercentFormatter

import seaborn as sns



# scipy.special for sigmoid function

import scipy.special



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv', index_col='PassengerId')

df_test = pd.read_csv('../input/test.csv', index_col='PassengerId')

df_gender_sub = pd.read_csv("../input/gender_submission.csv", index_col='PassengerId')
# Storing the target separately

Survived = df_train.loc[:,'Survived']

df_train = df_train.drop(['Survived'], axis=1).copy()



# Saving index for train test split 

train_index = df_train.index

test_index = df_test.index



# Concate the two datasets

df_all = pd.concat([df_train, df_test])



# dont needed anymore

##del df_train

##del df_test
# Function for nullanalysis

def nullAnalysis(df):

    tab_info=pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})



    tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values (nb)'}))

    tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100)

                         .T.rename(index={0:'null values (%)'}))

    return tab_info
# Show the null values

nullAnalysis(df_all)
# First 10 datarows where age is null

df_all[df_all.loc[:,'Age'].isnull()].head(10)
# Average age overall

print("Average age of a passengers: ", round(df_all.loc[:,'Age'].agg('mean'),0))
# Average age per class

df_all.groupby('Pclass')['Age'].agg('mean')
# Setting the average age of each class for the missing values inside the corresponding class

df_all.loc[(df_all['Age'].isnull()) & (df_all['Pclass'] == 1), ['Age']] = round(df_all.groupby('Pclass')['Age'].agg('mean')[1],0)

df_all.loc[(df_all['Age'].isnull()) & (df_all['Pclass'] == 2), ['Age']] = round(df_all.groupby('Pclass')['Age'].agg('mean')[2],0)

df_all.loc[(df_all['Age'].isnull()) & (df_all['Pclass'] == 3), ['Age']] = round(df_all.groupby('Pclass')['Age'].agg('mean')[3],0)
df_all[df_all['Fare'].isnull()]
df_all.groupby('Pclass', as_index=False)['Fare'].agg('mean')
# Setting Fare to mean fare of pclass

df_all.loc[1044,['Fare']] = 13.30
# Count all Cabins with NaN data

print("Count of cabins with nan data: ")

df_all.loc[(df_all.loc[:,'Cabin'].isnull()) == True]['Name'].count()
# Group by Pclasses

df_all.groupby('Pclass').agg('count')[['Name','Cabin']]
(df_all.groupby('Pclass').agg('count')['Cabin'] / df_all.groupby('Pclass').agg('count')['Name'])*100
df_all[df_all['Cabin'].str.contains(' ', regex=False) == True].sort_values(by='Cabin')
# Creating a data frame for the Cabin values to split multiple values into seperate columns

df_cabin_expand = df_all.loc[:,'Cabin'].str.split(' ', expand=True)



# Group all doubled values by the first value

df_cabin_expand[df_cabin_expand.loc[:,1].isnull() == False].groupby([0]).count()
df_all[df_all['Embarked'].isnull()]
# Fill the two missing Embarked features by using the next valid value

df_all['Embarked'] = df_all['Embarked'].fillna(method='bfill')
# Split Name feature strings into several columns

df_name_salutation = df_all.loc[:,'Name'].str.split(' ', expand=True).copy()

df_name_salutation.groupby(1).count()
# Extract Salutation from every column based on the '.'

df_newsal_1 = df_name_salutation[df_name_salutation[1].str.contains('.', regex=False)][1]

df_newsal_2 = df_name_salutation[df_name_salutation[2].str.contains('.', regex=False)][2]

df_newsal_3 = df_name_salutation[(df_name_salutation[3].isnull() == False) & (df_name_salutation[3].str.contains('.', regex=False))][3]



# Rename column for append 

df_newsal_2 = df_newsal_2.rename(1)

df_newsal_3 =  df_newsal_3.rename(1)



# Append both salutations results to one column and rename them

df_newsal = df_newsal_1.append([df_newsal_2, df_newsal_3])

df_newsal = df_newsal.rename('Salutation')



# Concatenate them to the main dataframe

df_all = pd.concat([df_all,df_newsal],axis=1)
df_all.groupby('Salutation').count()
# Create new feature Family true/false

df_all.loc[:,'Family'] = ((df_all['SibSp'] > 0) | (df_all['Parch'] > 0)).replace(True, 1, inplace=False)

df_all.loc[:,'Family'] = df_all.loc[:,'Family'].astype(int)
# Splitting the name feature into seperate strings

df_familynames = df_all.loc[:,'Name'].str.split(' ', expand=True).copy()



# Families with single last name

l_singleLastname = df_familynames[(df_familynames[0].str.contains(',', regex=False)==True)].index



# Families with double last name

l_doubleLastname = df_familynames[(df_familynames[0].str.contains(',', regex=False)==False) 

                                  & (df_familynames[1].str.contains(',', regex=False)==True)].index



# Families with double last name and more

l_doubleLastnameSpec = df_familynames[(df_familynames[0].str.contains(',', regex=False)==False) 

                                      & (df_familynames[1].str.contains(',', regex=False)==False) 

                                      & (df_familynames[2].str.contains(',', regex=False)==True)].index



# Create all last names for the single named, double named and multiple named passengers

df_singleLastname = df_familynames.loc[l_singleLastname,0]

df_doubleLastname = (df_familynames.loc[l_doubleLastname,0] 

                     + ' ' + df_familynames.loc[l_doubleLastname,1])

df_doubleLastnameSpec = (df_familynames.loc[l_doubleLastnameSpec,0] 

                         + ' ' + df_familynames.loc[l_doubleLastnameSpec,1] 

                         + ' ' + df_familynames.loc[l_doubleLastnameSpec,2])





# Rename column for append 

df_singleLastname = df_singleLastname.rename('Lastname')

df_doubleLastname = df_doubleLastname.rename('Lastname')

df_doubleLastnameSpec =  df_doubleLastnameSpec.rename('Lastname')



# Creating a Lastname column in initial dataframe

df_familynames['Lastname'] = df_singleLastname.append([df_doubleLastname, df_doubleLastnameSpec])

df_lastname_count = df_familynames.groupby('Lastname', as_index=False).count()



# dropping all columns except for the first one

df_lastname_count = df_lastname_count.drop([1,2,3,4,5,6,7,8,9,10,11,12,13], axis=1)

df_familynames = df_familynames.drop([0,1,2,3,4,5,6,7,8,9,10,11,12,13], axis=1)



# Joining the grouped by lastname counts to the PassengersId

df_familynames = df_familynames.join(df_lastname_count.set_index('Lastname'), on='Lastname')



#renaming column

df_familynames = df_familynames.rename(columns = {0: "Number_of_Familymembers"})
# Mergin the Lastnames and their family member count to original df

df_all = pd.merge(df_all , df_familynames, right_index=True, left_index=True)

# Removing the trailing comma

df_all['Lastname'] = df_all['Lastname'].str.rstrip(',')
# Set Number_of_Familymembers = 0 when traveling alone

df_all.loc[df_all['Family'] == 0,'Number_of_Familymembers'] = 1 
# Gender distribution

df_all.groupby(['Parch']).agg('count')
# Visualizing pie chart



fig, ax = plt.subplots(figsize=(10,7))



# Size and explsion

size_out = 3

size_in = 1

explode_out = (0.2,0.2)

explode_in = (0.3,0.3,0.3,0.3,0.3,0.3)



cmap = plt.get_cmap('tab20c')



outer_colors = cmap(np.array([8,0]))

inner_colors = cmap(np.array([11,10,9,3,2,1]))



patches1, texts1, autotexts1 = ax.pie(df_all.groupby(['Sex']).count().Name, radius=3, colors=outer_colors,

       labels=df_all.groupby(['Sex']).count().Name.index,autopct='%1.1f%%',pctdistance=0.85,

       wedgeprops=dict(width=size_out, edgecolor='black'),

       explode = explode_out)



patches2, texts2, autotexts2 = ax.pie(df_all.groupby(['Sex','Pclass']).count().Name, radius=2, colors=inner_colors,

       labels=[1,2,3,1,2,3],autopct='%1.1f%%', labeldistance=0.88,pctdistance=0.55,

       wedgeprops=dict(width=size_in, edgecolor='black'),

      explode = explode_in)



# Centre Cirle

centre_circle = plt.Circle((0,0),1.5,color='black', fc='white',linewidth=0)

fig = plt.gcf()

fig.gca().add_artist(centre_circle)



#plt.rcParams['font.size'] = 10.0

#plt.rc_context





# Define the labels on the outer plot

for t in texts1:

    t.set_size('large')

for t in autotexts1:

    t.set_size('large')

#autotexts1[0].set_color('y')





# Define the labels on the inner plot

for t in texts2:

    t.set_size('large')

for t in autotexts2:

    t.set_size('large')

#autotexts2[0].set_color('d')





# Setting legend

ax.legend(loc='lower right', bbox_to_anchor=(0.7, 0., 0.5, 0.5), shadow=1,title='Legend',

          handletextpad=1, labelspacing=0.5 , fontsize='12', labels=['female','male','1. class','2. class', '3. class','1. class','2. class', '3. class'])





ax.set(aspect="equal", title='Gender Distribution')

plt.axis('equal')

plt.show()
# Sclicing the three classes

df_firstclass_ages = df_all[df_all.loc[:,'Pclass'] == 1]['Age'].copy()

df_secondclass_ages = df_all[df_all.loc[:,'Pclass'] == 2]['Age'].copy()

df_thirdclass_ages = df_all[df_all.loc[:,'Pclass'] == 3]['Age'].copy()



# Combining all classes in an array

df_all_class_ages =[df_firstclass_ages.values,

                    df_secondclass_ages.values,

                    df_thirdclass_ages.values]



# Font dictionary

font = {'color':  'black',

        'weight': 'normal',

        'size': 18,

}



# Building the figure and the axes for the plot

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 6) )



# plot violin plot

parts = axes.violinplot(df_all_class_ages

                   ,showmeans=False,

                    showmedians=True)

axes.set_title('Age Distribution per Class', fontdict=font, fontsize=25)



# Styling every violin in the graph

for pc in parts['bodies']:

    pc.set_facecolor('#FF8C00')

    pc.set_edgecolor('#000000')

    pc.set_linewidth(2)

    pc.set_alpha(0.7)





# adding horizontal grid lines

axes.yaxis.grid(True)

axes.set_xticks([y + 1 for y in range(len(df_all_class_ages))])

axes.set_xlabel('Class',fontdict=font, labelpad=20, size=20)

axes.set_ylabel('Age', fontdict=font,labelpad=20, size=20)





axes.vlines(1, df_firstclass_ages.describe()['25%'], df_firstclass_ages.describe()['75%'], color=['#000000'], linestyle='-', lw=5)

axes.vlines(2, df_secondclass_ages.describe()['25%'], df_secondclass_ages.describe()['75%'], color=['#000000'], linestyle='-', lw=5)

axes.vlines(3, df_thirdclass_ages.describe()['25%'], df_thirdclass_ages.describe()['75%'], color=['#000000'], linestyle='-', lw=5)

#axes.vlines(2, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)



# add x-tick labels

plt.setp(axes, xticks=[y + 1 for y in range(len(df_all_class_ages))],

         xticklabels=['First', 'Second','Third'])





plt.show()
# Grouping by and reset index

df_sal_distr = df_all.groupby('Salutation').count()

df_sal_distr.reset_index(level=0, inplace=True)

df_sal_distr = df_sal_distr[['Salutation','Pclass']]

#rename the column

df_sal_distr = df_sal_distr.rename(columns = {'Pclass':"Salutation_Count"})

                                              

df_sal_distr
# The dataset with survived information

df_survivalinfo = pd.concat([df_all.loc[train_index,:], Survived], axis=1)
# Survival distribution per Sex

gp_survived_gender = df_survivalinfo.groupby(['Survived','Sex'])['Name'].count()[1]



# Gender Survival

gp_gender_survived = df_survivalinfo.groupby(['Sex','Survived']).count()['Name']



# Survival distribution Y/N 

gp_survived_yn = df_survivalinfo.groupby(['Survived']).agg('count')['Name']



# Survival total female / male 

gp_survival_total = df_survivalinfo.groupby(['Sex','Survived']).count().xs('Name', axis=1)





# Labels and size based on survival group by (df_survivalinfo)

#labels_suvinf = ['not suvived','survived']

sizes_suvinf = [gp_survived_yn[y] for y in range(len(gp_survived_yn))]

explode_suvinf = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')



# Labels and size based on gender group by (gp_sex_survived)

labels_sexinf = [gp_survived_gender.index[y] for y in range(len(gp_survived_gender.index))]

sizes_sexinf = [gp_survived_gender[y] for y in range(len(gp_survived_gender))]

explode_sexinf = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')



# Labels and size MALE Survivor Distribution

labels_maleinf = [gp_gender_survived['male'].index[y] for y in range(len(gp_gender_survived['male'].index))]

sizes_maleinf = [gp_gender_survived['male'][y] for y in range(len(gp_gender_survived['male']))]

explode_maleinf = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')



# Labels and size FEMALE Survivor Distribution

labels_femaleinf = [gp_gender_survived['female'].index[y] for y in range(len(gp_gender_survived['female'].index))]

sizes_femaleinf = [gp_gender_survived['female'][y] for y in range(len(gp_gender_survived['female']))]

explode_femaleinf = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')



# Labels and size total survival data

labels_totalsuvinf = [gp_survival_total.index[y] for y in range(len(gp_gender_survived.index))]

sizes_totalsuvinf = [gp_survival_total[y] for y in range(len(gp_survival_total))]

explode_totalsuvinf = (0.3, 0.0, 0.1, 0.0)  # not survived (fm), survived(fm), not survived(m), survived(m)





# Font dictionary

font = {'color':  'black',

        'weight': 'normal',

        #'size': 15,

        'fontsize':15

}



# Color maps for the pies

cmap = plt.get_cmap('tab20c')

survivedcolor = cmap(np.array([5,1]))

survivorallcolor = cmap(np.array([9,10,0,1]))



# Figure and axes of the plot / 4 * 2 plots 

gridsize = (4,2)

fig1 = plt.figure(figsize=(14,10))

ax1 = plt.subplot2grid(gridsize, (0,0))

ax2 = plt.subplot2grid(gridsize, (0,1))

ax3 = plt.subplot2grid(gridsize, (1,0))

ax4 = plt.subplot2grid(gridsize, (1,1))

ax5 = plt.subplot2grid(gridsize, (2,0), colspan= 2, rowspan= 2)



## fig1 configs

fig1.suptitle('Distributions of 891 Passengers (Trainingset)', fontsize=25)



## ax1 

# Define first pie for survival true falls

ax1.pie(sizes_suvinf, 

        explode=explode_suvinf,

        #labels=labels_suvinf,

        autopct='%1.1f%%',

        shadow=True, startangle=90,

        colors=survivedcolor,

        labeldistance=1.15,

        pctdistance=0.55

       )

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

#ax1.fontdict=font

ax1.legend(loc='upper left',fontsize='12',labels=('not survived', 'survived'))

ax1.set_title('Survivor Distribution', fontdict=font, fontsize=20)



## ax2 

# Define second pie for sex to survival

ax2.pie(sizes_sexinf, 

        explode=explode_sexinf,

        #labels=labels_sexinf,

        autopct='%1.1f%%',

        shadow=True, startangle=90,

        colors=outer_colors,

        labeldistance=1.15,

        pctdistance=0.55)

ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

#ax2.fontdict=font

ax2.legend(loc='upper right', fontsize='12', labels=labels_sexinf )

ax2.set_title('Gender Survivors', fontdict=font, fontsize=20)



ax3.pie(sizes_maleinf, 

        explode=explode_maleinf,

        #labels=labels_sexinf,

        autopct='%1.1f%%',

        shadow=True, startangle=90,

        colors=survivedcolor,

        labeldistance=1.15,

        pctdistance=0.55)

#ax3.fontdict=font

ax3.legend(loc='lower left', fontsize='12', labels=['not survived','survived'] )

ax3.set_title('Male Survivors', fontdict=font, fontsize=20)

ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.





ax4.pie(sizes_femaleinf, 

        explode=explode_femaleinf,

        #labels=labels_sexinf,

        autopct='%1.1f%%',

        shadow=True, startangle=90,

        colors=survivedcolor,

        labeldistance=1.15,

        pctdistance=0.55)

#ax4.fontdict=font

ax4.legend(loc='lower right', fontsize='12', labels=['not survived','survived'] )

ax4.set_title('Female Survivors', fontdict=font, fontsize=20)

ax4.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.





ax5.pie(sizes_totalsuvinf, 

        explode=explode_totalsuvinf,

        #labels=labels_totalsuvinf,

        autopct='%1.1f%%',

        shadow=True, startangle=90,

        colors=survivorallcolor,

        labeldistance=1.15,

        pctdistance=0.55)

ax5.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

#ax4.fontdict=font

ax5.legend(loc='lower right', fontsize='12', labels=['female not survived','female survived','male not survived','male survived'] )

ax5.set_title('Overall Survivors', fontdict=font, fontsize=20)

ax5.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
# Grouped by Survived and Pclass

gp_survpclass = df_survivalinfo.groupby(['Survived','Pclass'])['Name'].count()



gridsize = (1,2)

fig1 = plt.figure(figsize=(12,8))

ax1 = plt.subplot2grid(gridsize, (0,0), colspan=2, rowspan=1)



# Bar chart design

bar_width = 0.35

cmap = plt.get_cmap('tab20b')



survbarcol = cmap(np.array([7]))

nsurvbarcol = cmap(np.array([12]))



barindex = gp_survpclass[0].index   # Group by index of bar plot data

xtickslables = [barindex[y] for y in range(len(gp_survpclass[0].index))]







ax1.bar(barindex

        ,gp_survpclass[0].values

        ,bar_width

        ,color=nsurvbarcol

        )



ax1.bar(barindex + bar_width

        ,gp_survpclass[1].values

        ,bar_width

        ,color=survbarcol)



ax1.set_xlabel('Ticket Class')

ax1.set_ylabel('Passenger Count')

ax1.set_title('Suvivors per Ticket Class')

ax1.set_xticks(barindex + bar_width / 2)

ax1.set_xticklabels(xtickslables)

ax1.legend(labels=('not survived','survived'))



fig.tight_layout()

plt.show()
ptbl = pd.DataFrame.pivot_table(df_survivalinfo, values=['Fare', 'Survived'], index=['Pclass'],

                     aggfunc={'Survived': ['sum'], 'Fare': [min,max,np.mean]})



ptbl

# seaborn's kdeplot, plots univariate or bivariate density estimates.

#Size can be changed by tweeking the value used

sns.FacetGrid(df_survivalinfo.loc[:,['Survived','Pclass']], hue="Survived", height=5).map(sns.kdeplot, "Pclass").add_legend()

plt.show()
# Group by salutation and survival

df_survival_sal = df_survivalinfo.groupby(['Salutation','Survived'], as_index=False)['Name'].count()



# Rename column

df_survival_sal = df_survival_sal.rename(columns={"Name": "Total"})



# Aggregate/Count all Salutations

df_survival_sal_total = df_survival_sal.groupby('Salutation', as_index=False)['Total'].agg(sum)



# Not survived Salutations incl. renaming of Total column for join

df_survival_sal_nsuv = pd.DataFrame(df_survival_sal[df_survival_sal['Survived'] == 0])

df_survival_sal_nsuv = df_survival_sal_nsuv.rename(columns={'Total':'Total_notSurvived'})

df_survival_sal_nsuv = df_survival_sal_nsuv[['Salutation','Total_notSurvived']]





# Survived salutations incl. renaming of Total column for join

df_survival_sal_suv = pd.DataFrame(df_survival_sal[df_survival_sal['Survived'] == 1])

df_survival_sal_suv = df_survival_sal_suv.rename(columns={'Total':'Total_Survived'})

df_survival_sal_suv = df_survival_sal_suv[['Salutation','Total_Survived']]



# Joining all salutation survival information together

df_survival_sal_total = df_survival_sal_total.join(df_survival_sal_suv.set_index('Salutation'), on='Salutation', how='outer')

df_survival_sal_total = df_survival_sal_total.join(df_survival_sal_nsuv.set_index('Salutation'), on='Salutation', how='outer')



# Fill the NaN with zeros

df_survival_sal_total = df_survival_sal_total.fillna(value=0)



df_survival_sal_total
sns.set(style="whitegrid")



# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(6, 15))



# Load the example car crash dataset

#crashes = sns.load_dataset("car_crashes").sort_values("total", ascending=False)

salutations = df_survival_sal_total



# Plot the total salutations

sns.set_color_codes("pastel")

sns.barplot(x="Total", y="Salutation", data=salutations,

            label="Total", color="b")



## Plot the survivals

sns.set_color_codes("muted")

sns.barplot(x="Total_Survived", y="Salutation", data=salutations,

            label="Survived", color="b")



# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 350), ylabel="",

       xlabel="Passengers Salutation (Total, Survived)")

sns.despine(left=True, bottom=True)
# Count of groupy

N = 9



# Group seperation by Survived Not-Survived

g_sur_Familymembers = df_survivalinfo[df_survivalinfo.loc[:,'Survived'] == 1].groupby('Number_of_Familymembers').count().Survived

g_nsur_Familymembers = df_survivalinfo[df_survivalinfo.loc[:,'Survived'] == 0].groupby('Number_of_Familymembers').count().Survived



sur_Std = g_sur_Familymembers.std(axis=0) 

sur_Mean = g_sur_Familymembers.mean(axis=0) 

nsur_Std = g_nsur_Familymembers.std(axis=0) 

nsur_Mean = g_nsur_Familymembers.mean(axis=0) 





# the x locations for the groups

ind  = np.arange(N)



width = 0.75       # the width of the bars: can also be len(x) sequence



p1 = plt.bar(ind, g_sur_Familymembers, width, yerr=sur_Mean)

p2 = plt.bar(ind, g_nsur_Familymembers, width,

             bottom=g_sur_Familymembers, yerr=nsur_Mean)





plt.ylabel('Passengers')

plt.xlabel('Number Family Members')

plt.title('Passenger Survival by Family Members')

plt.xticks(ind, ('1', '2', '3', '4', '5', '6', '7', '8', '11'))

plt.yticks(np.arange(-130, 670, 25))

plt.legend((p1[0], p2[0]), ('Survived', 'Not Survived'))



plt.show()
fammem_survived = df_survivalinfo['Number_of_Familymembers']

#[df_survivalinfo['Survived'] == 1]

#fare_survived = df_survivalinfo['Fare']



inputfeature = df_survivalinfo[df_survivalinfo['Survived'] == 1 ]['Number_of_Familymembers']



#df_survivalinfo['Number_of_Familymembers']



df_survivalinfo[df_survivalinfo['Survived'] == 1 ]



mu = inputfeature.describe()['mean']  # mean of distribution

sigma = inputfeature.describe()['std']  # standard deviation of distribution

x = mu + sigma * inputfeature.values



num_bins = 11



fig, ax = plt.subplots(figsize=(12,7))



# the histogram of the data

n, bins, patches = ax.hist(inputfeature,num_bins, density=1)



# add a 'best fit' line

y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *

     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))

ax.plot(bins, y, '--')

ax.set_xlabel('Number of Familymebers')

ax.set_ylabel('Probability density')

ax.set_title(r'Histogram of Family Member Density: $\mu='+ str(mu) +'$, $\sigma= $'+  str(sigma))



fig.tight_layout()

plt.show()
sns.set(style="whitegrid")



# Draw a nested barplot to show survival for class and sex

g = sns.catplot(x="Number_of_Familymembers", y="Survived", data=df_survivalinfo,

                #x="Pclass", y="Survived", hue='Number_of_Familymembers', data=df_survivalinfo,

                height=6, kind="bar", palette="muted")

g.despine(left=True)

g.set_ylabels("survival probability")
fare_survived = df_survivalinfo[df_survivalinfo['Survived'] == 1]['Fare']

#fare_survived = df_survivalinfo['Fare']



inputfeature = fare_survived



mu = inputfeature.describe()['mean']  # mean of distribution

sigma = inputfeature.describe()['std']  # standard deviation of distribution

x = mu + sigma * inputfeature.values



num_bins = 50



fig, ax = plt.subplots(figsize=(12,7))



# the histogram of the data

n, bins, patches = ax.hist(inputfeature,num_bins, density=1)



# add a 'best fit' line

y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *

     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))

ax.plot(bins, y, '--')

ax.set_xlabel('Fare Price')

ax.set_ylabel('Probability density')

ax.set_title(r'Histogram of Fare: $\mu=$'+ str(mu) +', $\sigma=$ ' + str(sigma))



fig.tight_layout()

plt.show()
# Pivot for age and fare

ptbl_survived = pd.DataFrame.pivot_table(df_survivalinfo, values=['Fare', 'Age', 'Survived'], index=['Sex', 'Pclass'],

                     aggfunc={'Fare': np.mean,'Age': [min, max, np.mean], 'Survived': ['sum']})

ptbl_survived
sns.pairplot(df_survivalinfo, hue='Survived')
# K-Nearest Neighbours imports

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



# Using here the full dataset because train and test set must have the same shape when using it with model



# One hot encoding for the categorical data

df_all_knn_hot = df_all.copy()

df_all_knn_hot = df_all_knn_hot.drop(['Name','Cabin','Fare','Ticket','Lastname'], axis=1)

df_all_knn_hot = pd.get_dummies(df_all_knn_hot, columns=['Sex','Salutation','Embarked'])







# Train test split only on test dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_all_knn_hot.loc[train_index,:], 

                                                    Survived, test_size = 0.30, random_state = 45)  ## 50  25 ## 0.25 25 , 25
from sklearn.model_selection import cross_val_score



# creating odd list of K for KNN

myList = list(range(1,50))



# subsetting just the odd ones

neighbors = list(myList)





# empty list that will hold cv scores

cv_scores = []



# perform 10-fold cross validation

for k in neighbors:

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')

    cv_scores.append(scores.mean())
# changing to misclassification error

MSE = [1 - x for x in cv_scores]



# determining best k

optimal_k = neighbors[MSE.index(min(MSE))]

print ("The optimal number of neighbors is %d" % optimal_k)



# plot misclassification error vs k

plt.plot(neighbors, MSE)

plt.xlabel('Number of Neighbors K')

plt.ylabel('Misclassification Error')

plt.show()
# K-Nearest Neighbours

from sklearn import metrics



from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix







X_train = X_train

y_train = y_train



# 3 Neighbors used from the misclassification error calculation

KNNC = KNeighborsClassifier(n_neighbors=3)

KNNC.fit(X_train, y_train)



y_pred = KNNC.predict(X_test)



# Summary of the predictions made by the classifier

print(classification_report(y_test, y_pred, target_names=['0','1']))





print ("Models accuracy score: ", accuracy_score(y_test, y_pred))
from yellowbrick.classifier import ClassificationReport



classes = ["will not suvive", "will survive"]



# Instantiate the classification model and visualizer

visualizer = ClassificationReport(KNNC, classes=classes, support=True)



visualizer.fit(X_train, y_train)  # Fit the visualizer and the model

visualizer.score(X_test, y_test)  # Evaluate the model on the test data

g = visualizer.poof()             # Draw/show/poof the data
# Plotting the Precision-Recall curve

y_proba_train = KNNC.predict_proba(X_train)[:, 1]

p, r, t = metrics.precision_recall_curve(y_train, y_proba_train)



plt.plot(r, p)

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.show()
titanic_submission = pd.DataFrame({'PassengerId':df_all_knn_hot.loc[test_index,:].index,

                                   'Survived':KNNC.predict(df_all_knn_hot.loc[test_index,:])})

titanic_submission.PassengerId = titanic_submission.PassengerId.astype(int)

titanic_submission.Survived = titanic_submission.Survived.astype(int)



# Overview how much suvived with k-nearest neighbor approach

titanic_submission.groupby('Survived').count()
titanic_submission.head(15)
# Submission to a csv file for competition upload.

titanic_submission.to_csv("titanic_submission_knn_4.csv", index=False)
# Random Forest imports

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier





# Using here the full dataset because train and test set must have the same shape when using it with model



# One hot encoding for the categorical data

df_all_rf_hot = df_all.copy()

df_all_rf_hot = df_all_rf_hot.drop(['Name','Cabin','Fare','Ticket','Lastname'], axis=1)

df_all_rf_hot = pd.get_dummies(df_all_rf_hot, columns=['Sex','Salutation','Embarked'])







# Train test split only on test dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_all_rf_hot.loc[train_index,:], 

                                                    Survived, test_size = 0.30, random_state = 45)  ## 50  25 ## 0.25 25 , 25
from sklearn.model_selection import cross_val_score



# creating odd list of K for KNN 

myList = list(range(1,30))



# subsetting just the odd ones

levels = list(myList)





# empty list that will hold cv scores

cv_scores = []



# perform 10-fold cross validation

for l in levels:

    rfc = RandomForestClassifier( n_estimators=100)

    scores = cross_val_score(rfc, X_train, y_train, cv=5, scoring='recall')

    cv_scores.append(scores.mean())
# changing to misclassification error 

MSE = [1 - x for x in cv_scores]



# determining best k

optimal_l = levels[MSE.index(min(MSE))]

print ("The optimal level depth is %d" % optimal_l)



# plot misclassification error vs k

plt.plot(levels, MSE)

plt.xlabel('Level l')

plt.ylabel('Misclassification Error')

plt.show()
# K-Nearest Neighbours

from sklearn import metrics



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



#from sklearn.ensemble import ExtraTreesClassifier

#from sklearn.tree import DecisionTreeClassifier









X_train = X_train

y_train = y_train



# Random forest classifier  

RFCC = RandomForestClassifier(max_depth=14,

                              n_estimators=5000)

#RFCC = RandomForestClassifier(n_estimators=1000)

RFCC.fit(X_train, y_train)



y_pred = RFCC.predict(X_test)



# Summary of the predictions made by the classifier

print(classification_report(y_test, y_pred, target_names=['0','1']))





print ("Models accuracy score: ", accuracy_score(y_test, y_pred))
from yellowbrick.classifier import ClassificationReport



classes = ["will not suvive", "will survive"]



# Instantiate the classification model and visualizer

visualizer = ClassificationReport(RFCC, classes=classes, support=True)



visualizer.fit(X_train, y_train)  # Fit the visualizer and the model

visualizer.score(X_test, y_test)  # Evaluate the model on the test data

g = visualizer.poof()             # Draw/show/poof the data
titanic_submission_rfc = pd.DataFrame({'PassengerId':df_all_rf_hot.loc[test_index,:].index,

                                   'Survived':RFCC.predict(df_all_rf_hot.loc[test_index,:])})

titanic_submission_rfc.PassengerId = titanic_submission_rfc.PassengerId.astype(int)

titanic_submission_rfc.Survived = titanic_submission_rfc.Survived.astype(int)



# Overview how much suvived with random forest approach

titanic_submission_rfc.groupby('Survived').count()
# Submission to a csv file for competition upload.

titanic_submission_rfc.to_csv("titanic_submission_rfc_5.csv", index=False)
# importing sigmoid function

import scipy.special

# Neural network class definition



class neuralNetwork:

    

    # Initialize

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):

        # Number of nodes in each layer

        self.inodes = inputnodes

        self.hnodes = hiddennodes

        self.onodes = outputnodes

        

        # Bulding the size of the weight matrices without normal distribution info

        ## self.wih = (np.random.rand(self.hnodes,self.inodes) - 0.5)

        ## self.who = (np.random.rand(self.onodes,self.hnodes) - 0.5)

        

        # Size of the weight matrice by random sample based on the normal distribution

        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))

        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        

        #self.whh = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))

        

        self.lrate = learningrate

        

        self.activationfunction = lambda x : scipy.special.expit(x);

        

        pass 

    

    def train(self, inputlist, targetlist):

        inputs = np.array(inputlist, ndmin=2).T

        targets = np.array(targetlist, ndmin=2).T

        

        hidden_inputs = np.dot(self.wih, inputs)

        hidden_outputs = self.activationfunction(hidden_inputs)

        

        # hidden layer 2

        #hidden2_inputs = np.dot(self.wih, inputs)

        #hidden2_outputs = self.activationfunction(hidden_inputs)

            

            

        final_inputs = np.dot(self.who, hidden_outputs)

        final_outputs = self.activationfunction(final_inputs)

        

        

        ## BACKPROPAGATION ##

        

        output_errors = targets - final_outputs

        

        hidden_errors = np.dot(self.who.T, output_errors)

        

        self.who += self.lrate * np.dot((output_errors * final_outputs * (1 - final_outputs)), np.transpose(hidden_outputs))        

        

        self.wih += self.lrate * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), np.transpose(inputs))

        

        pass

    

    def query(self, inputs_list):

        inputs = np.array(inputs_list, ndmin = 2).T

        

        hidden_inputs = np.dot(self.wih, inputs)

        hidden_outputs = self.activationfunction(hidden_inputs)

        

        final_inputs = np.dot(self.who, hidden_outputs)

        final_outputs = self.activationfunction(final_inputs)

        

        return final_outputs
# List to record the different configurations

# The idea is to store: nn_accuracy,learningrate,hidden_nodes,epochs

conf_performance_list = []
# Main configuration params for the nn

input_nodes = 29

hidden_nodes = 3

output_nodes = 2



learningrate = 0.4

nn_epochs = 1000     #453000   #49000
# Initialize neural network for titanic passengers

titanic_nn = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learningrate)
# train test split ( this function makes it very easy to split the data by percentage) 

X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(df_all_knn_hot.loc[train_index,:], Survived, test_size = 0.15, random_state = 45) 
# Prepare data for learning

# To make everything easier the X_train and the y_train (targets) will be 

# combined in one df again to loop over it easier.



Xy_train_nn = pd.concat([X_train_nn, y_train_nn], axis=1)
# train the nn multiple (epochs) times.

epochs = nn_epochs





for e in range(epochs):

    

    for row in Xy_train_nn.itertuples():

        

        inputs = (np.asfarray(row[1:30]) / 29 * 0.99) + 0.01

               

        targets = np.zeros(output_nodes) + 0.01

        targets[int(row[30])] = 0.99

             

        titanic_nn.train(inputs, targets)

        

        pass

    pass



    
# concatenate test set

Xy_test_nn = pd.concat([X_test_nn, y_test_nn], axis=1)
testrowindex = 2



#retrieving the results

nn_result = titanic_nn.query(Xy_test_nn[Xy_test_nn.columns[0:29]].values[testrowindex])
# List to store the true and false prediction results of the nn

scorecard = []

# List to store the matrix structure results to analyse it afterwards

matrixlist = []



for index,row in Xy_test_nn.iterrows():

    

    inputs = row[0:29].values

    

    correct_label = row[29]

    

    results = titanic_nn.query(inputs)

    

    label = np.argmax(results)

    

    print('PassengerID:', index, ' - Networks answer: ', label, ' --> Correct answer: ', correct_label)

    

    # Append all information to this list to get an overview about networks results and its matrix structure.

    matrixlist.append([results, label, correct_label])



    

    if(label == correct_label):

        scorecard.append(1)

    else:

        scorecard.append(0)

    pass     

        
scorecard_array = np.array(scorecard)

nn_accuracy = scorecard_array.sum() / scorecard_array.size



# Append current network settings to a list 

#   due to several runs of the network I will store here all configurations and 

#   its corresponding accuracy to adjust the model better.

conf_performance_list.append([nn_accuracy,learningrate,hidden_nodes,epochs])



print('Accuracy score by "75/15"-network: ', nn_accuracy)
# Configurations performance list

#   This list is used to show the different configuration in comparision to the accuracy to fine tune the nn

#   It will be commented out after I have found out the best fit

#   Columns: nn_accuracy,learningrate,hidden_nodes,epochs



# conf_performance_list
# train the nn multiple (epochs) times.

epochs = nn_epochs





for e in range(epochs):

    

    for row in Xy_test_nn.itertuples():

        

        inputs = (np.asfarray(row[1:30]) / 29 * 0.99) + 0.01        

        targets = np.zeros(output_nodes) + 0.01

        targets[int(row[30])] = 0.99

  

        titanic_nn.train(inputs, targets)

        

        pass

    pass
titanic_submission_nn = pd.DataFrame(columns=['PassengerId','Survived'])



for index,row in df_all_rf_hot.loc[test_index].iterrows():

    

    inputs = row[0:29].values

    

    results = titanic_nn.query(inputs)

    

    label = np.argmax(results)

        

    titanic_submission_nn = titanic_submission_nn.append({'PassengerId' : index , 'Survived': label} , ignore_index=True)

        

    pass
# Preparing the submission file

titanic_submission_nn.PassengerId = titanic_submission_nn.PassengerId.astype(int)

titanic_submission_nn.Survived = titanic_submission_nn.Survived.astype(int)



titanic_submission_nn.groupby('Survived').count()
titanic_submission_nn.to_csv("titanic_submission_nn_6.csv", index=False)
import tensorflow as tf
# train test split ( this function makes it very easy to split the data by percentage) 

X_train_tfnn, X_test_tfnn, y_train_tfnn, y_test_tfnn = train_test_split(df_all_knn_hot.loc[train_index,:], Survived, test_size = 0.15, random_state = 45) 



# scale the values between 0 and 1

X_train_tfnn = tf.keras.utils.normalize(np.asfarray(X_train_tfnn),axis= -1)

X_test_tfnn = tf.keras.utils.normalize(np.asfarray(X_test_tfnn),axis= -1)



y_train_tfnn = np.asfarray(y_train_tfnn) 

y_test_tfnn = np.asfarray(y_test_tfnn) 
# Model building

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))





model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy']

            )

model.fit(X_train_tfnn, y_train_tfnn, epochs = 200)
# Test the model and show the accuracy

val_loss, val_acc = model.evaluate(X_test_tfnn, y_test_tfnn)
# Save the model so far

model.save('titanic_survivor_predictor.model')
# Load the model again

new_model = tf.keras.models.load_model('titanic_survivor_predictor.model')