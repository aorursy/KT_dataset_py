import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

import warnings

warnings.filterwarnings('ignore')
print('This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them.') 

print('In particular, the Cleveland database is the only one that has been used by ML researchers to this date.')   

print('The "goal" field refers to the presence of heart disease in the patient.  It is integer valued from 0 (no presence) to 4.') 

print('Experiments with the Cleveland database have concentrated on simply attempting to distinguish presence (values 1,2,3,4)') 

print('from absence (value 0).\n') 

    

print('Source Information:') 

print('       -- 1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.') 

print('       -- 2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.') 

print('       -- 3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.') 

print('       -- 4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.') 
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")

df.head(5)
print('    #---------------------------------#')

print('    ####   Correlation Analysis   #####')

print('    #---------------------------------#\n')

print('Variables and their correlation to risk of Heart Disease')

df.corr() 

HD_corr = df.corr()["target"]

corr_HD = HD_corr.abs().sort_values(ascending=False)[1:]; corr_HD
print("I will select only the variables with greater than 20% correlation to Heart Disease to analyze")

df = df[['target', 'exang','cp', 'oldpeak','thalach','slope','thal','sex','age']];
#--------------------#

####  Pair Plot   ####

#--------------------#



col_list = ['steel grey']

col_list_palette = sns.xkcd_palette(col_list)

sns.set_palette(col_list_palette)

print('Create a pairplot to look identify data types and get a summary of the distributions')

sns.pairplot(data=df, markers="+", palette = sns.set_palette(col_list_palette)) #, corner=True)
#---------------------------------------#

#####   rename colmns for clarity   #####

#---------------------------------------#



df.rename(columns = {'age' : 'Age',

                     'sex' : 'Gender',

                     'cp' : 'Chest Pain',

                     'thalach':'Maximum Heart Rate',

                     'exang': 'Exercise Induced Angina',

                     'oldpeak':'Depression Induced by Exercise',

                     'thal':'Thalassemia',

                     'slope':'Exercise/Heart Rate Slope',

                     'target':'Heart Disease'}, inplace = True)

df.describe()
    #----------------------------------------------#

    #####    assign categorical data types    ######

    #####  change numbers to English labels   ######

    #----------------------------------------------#



#df.dtypes



#-----   Gender  -----#



# (1 = male; 0 = female)

g = {0:"Female", 1:"Male"}

df['Gender'] = [g[item] for item in df['Gender']]



#---- diagnosis of At Risk ----#



#df['Heart Disease'].value_counts()

#-- Value 0: < 50% diameter narrowing

#-- Value 1: > 50% diameter narrowing

hd = {0:"Healthy", 1:"At Risk"}

df['Heart Disease'] = [hd[item] for item in df['Heart Disease']]



#---   chest pain  ----#



#df['Chest Pain'].value_counts()

chestpain = {0: "Typical Angina",

            1: "Atypical Angina",

            2: "Non-Anginal Pain",

            3: "Asymptomatic"}

df['Chest Pain'] = [chestpain[item] for item in df['Chest Pain']]





#---   exercise induced angina (1 = yes; 0 = no)  ----#



#df['Exercise Induced Angina'].value_counts()

ang = {1:"No", 0:"Yes"}

df['Exercise Induced Angina'] = [ang[item] for item in df['Exercise Induced Angina']]



#--- Exercise ECG ST Segment/Heart Rate Slope  ----#



#df['Exercise ECG ST Segment/Heart Rate Slope'].value_counts()

# 1: upsloping, 2: flat, 3: downsloping

slp = {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}

df['Exercise/Heart Rate Slope'] = [slp[item] for item in df['Exercise/Heart Rate Slope']]



#------  Thalassemia   --------#



#thal = {3 : 'Normal', 6 : 'Fixed Defect',  7 : 'Reversable Defect'}

#df["Thalassemia"] = thal[item] for item in df["Thalassemia"]



#------   assign categorical data types   -------#



# df.Chest Pain.value_counts()

df[["Chest Pain", "Exercise Induced Angina",

    "Exercise/Heart Rate Slope", "Thalassemia",

    "Heart Disease"]] = df[["Chest Pain",

    "Exercise Induced Angina", "Exercise/Heart Rate Slope",

    "Thalassemia", "Heart Disease"]].astype('category')

# check to make sure it worked

df.dtypes
        #-----------------------------------#

        #####   Data Visualizations    ######

        #-----------------------------------#



#------------------------------------------------#

#----   set the parameters of all the plots  ----#

#------------------------------------------------#



col_list = ['blood','rosa']

col_list_palette = sns.xkcd_palette(col_list)

sns.set_palette(col_list_palette)



sns.set(rc={"figure.figsize": (10,6)},

            palette = sns.set_palette(col_list_palette),

            context="talk",

            style="ticks")



        #-------------------------------------------#

        ####  Count Plots for categorical data  #####

        #-------------------------------------------#



col_list = ['pale mauve','pale purple']

col_list_palette = sns.xkcd_palette(col_list)

sns.set_palette(col_list_palette)



#df.Sex.value_counts()

# how many people are in the study male and female?

sns.countplot(x = 'Gender', data=df, edgecolor = "black",

              palette = sns.set_palette(col_list_palette))

              #palette = sns.color_palette("PiYG", 3))

plt.suptitle('Study Participants by Gender', fontsize = 20)

plt.xlabel('', fontsize = 14)
#-----------------------#

#####  Age by Sex   #####

#-----------------------#



# change the bin height to reduce or enhance detail

g = sns.FacetGrid(data = df, col="Gender", hue = 'Gender',height = 6,

                  palette = sns.set_palette(col_list_palette))

                  #palette = sns.color_palette("PiYG", 3))

g.map(plt.hist, 'Age', edgecolor = "black") # bins = 20)

g.set_axis_labels('Age', 'Counts')

plt.suptitle('Age Distribution by Gender')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
col_list = ['mud brown','bubble gum pink']

col_list_palette = sns.xkcd_palette(col_list)

sns.set_palette(col_list_palette)



sns.countplot(x = 'Gender', data=df, hue = "Heart Disease",

              #palette = sns.color_palette(col_list_palette),

              #hue_order = ["At Risk", "Healthy"], 

              edgecolor = "black")

plt.suptitle('Study Participants by Heart Health and Gender', fontsize = 20)

plt.xlabel('', fontsize = 14)
g = sns.catplot(x="Heart Disease", hue="Gender", aspect = 2, # col="Sex"

                data=df, kind="count", height=5,

                hue_order = ["Female", "Male"], edgecolor = "black")

g.set_axis_labels("", "Counts")

plt.suptitle('At Risk & Healthy Study Participants', fontsize = 20)
#------------------------------------#

####   Exercise Induced Angina   #####

#------------------------------------#



col_list = ['mud brown','bubble gum pink']

col_list_palette = sns.xkcd_palette(col_list)

sns.set_palette(col_list_palette)





g = sns.catplot(x="Exercise Induced Angina", hue="Heart Disease", aspect = 1.8,

                data=df, kind="count", edgecolor = "black", height=5)

g.set_axis_labels("", "Counts")

plt.suptitle('Exercise Induced Angina & Heart Health', fontsize = 20)
#--------------------------------------#

#####  Chest Pain by Heart Health  #####

#--------------------------------------#



g = sns.catplot(x="Chest Pain", hue="Heart Disease", aspect = 2,

                data=df, kind="count", height=5, edgecolor = "black",

                hue_order = ["At Risk", "Healthy"])

g.set_axis_labels("", "Counts")

plt.suptitle('Chest Pain & Heart Health', fontsize = 20)
#-----------------------------#

######    Thalassemia   #######

#-----------------------------#



g = sns.catplot(x="Thalassemia", hue="Heart Disease", aspect = 1.8,

                data=df, kind="count", edgecolor = "black", height=5)

g.set_axis_labels("", "Counts")

plt.suptitle('Thalassemia & Heart Health', fontsize = 20)
#----------------------------------------------------------#

######    Exercise ECG ST Segment/Heart Rate Slope   #######

#----------------------------------------------------------#



g = sns.catplot(x="Exercise/Heart Rate Slope",

                hue="Heart Disease", aspect = 1.8,

                data=df, kind="count", edgecolor = "black", height=5)

g.set_axis_labels("", "Counts")

plt.suptitle('Exercise/Heart Rate Slope by Heart Health', fontsize = 20)
        #-----------------------------------------#

        #####   Distribution Visualizations   #####

        #-----------------------------------------#



#--------------------------------#

####   Age by At Risk   ####

#--------------------------------#



# change the bin height to reduce or enhance detail

g = sns.FacetGrid(data = df, col="Heart Disease",

                  hue = 'Heart Disease',height = 6)

g.map(plt.hist, 'Age', edgecolor = "black") # bins = 20)

g.set_axis_labels('Age', 'Counts')

plt.suptitle('Age Distribution & Heart Health')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#---------------------------------------------#

####   Heart Health and Heart Rate by Sex  ####

#---------------------------------------------#



g = sns.FacetGrid(data = df, col = "Heart Disease",

                  hue ='Heart Disease',height = 6)

g.map(plt.hist, 'Maximum Heart Rate', edgecolor = "black") # bins = 20)

g.set_axis_labels('Max Heart Rate', 'Counts')

plt.suptitle('Max Heart Rate & Heart Health')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#---------------------------------------------#

####   Heart Health and Heart Rate by Sex  ####

#---------------------------------------------#



g = sns.FacetGrid(data = df, row="Gender",col = "Heart Disease",

                  hue = 'Heart Disease',height = 6)

g.map(plt.hist, 'Maximum Heart Rate', edgecolor = "black") # bins = 20)

g.set_axis_labels('Max Heart Rate', 'Counts')

plt.suptitle('Heart Health by Gender & Max Heart Rate')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#------------------------------------------------------#

####   Depression Induced by Exercise by At Risk    ####

#------------------------------------------------------#



# change the bin height to reduce or enhance detail

g = sns.FacetGrid(data = df, col="Heart Disease", hue = 'Heart Disease',height = 6)

g.map(plt.hist, 'Depression Induced by Exercise', edgecolor = "black") # bins = 20)

g.set_axis_labels('Maximum Heart Rate Achieved', 'Counts')

plt.suptitle('ST Depression Induced by Exercise & Heart Health')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        #----------------------------------------#

        ####  Boxplots for categorical data  #####

        #----------------------------------------#



boxprops = {'edgecolor': 'k', 'linewidth': 2}

lineprops = {'color': 'k', 'linewidth': 2}

kwargs = {'hue_order': ["At Risk", "Healthy"]}

boxplot_kwargs = dict({'boxprops': boxprops, 'medianprops': lineprops,

                       'whiskerprops': lineprops, 'capprops': lineprops,

                       'width': 0.75}, **kwargs)
#-----------------------------#

#####   max heart rate   ######

#-----------------------------#



sns.boxplot(x = "Heart Disease", y = "Maximum Heart Rate", data=df, **boxplot_kwargs)

plt.suptitle('Maximum Heart Rate & Heart Disease', fontsize = 20)

plt.xlabel('', fontsize = 14)
#------------------------------------#

####   Exercise Induced Angina   #####

#------------------------------------#



# need to switch the order of the colors here

col_list = ['bubble gum pink', 'mud brown']

col_list_palette = sns.xkcd_palette(col_list)

sns.set_palette(col_list_palette)



sns.boxplot(x = "Exercise Induced Angina", y = "Maximum Heart Rate",

            data=df, **boxplot_kwargs) #hue_order = ["No", "Yes"])

plt.suptitle('Exercise Induced Angina & Heart Disease', fontsize = 20)

plt.xlabel('', fontsize = 14)
#####  Age  ######



# need to switch the order of the colors back

col_list = ['mud brown', 'bubble gum pink']

col_list_palette = sns.xkcd_palette(col_list)

sns.set_palette(col_list_palette)



sns.boxplot(x = "Heart Disease", y = "Age", data=df, **boxplot_kwargs)

plt.suptitle('Age by Heart Disease', fontsize = 20)

plt.xlabel('', fontsize = 14)

#-------------------------------------------------------#

#####  Bin the Age Data for presentation/analysis   #####

#-------------------------------------------------------#



mna = min(df.Age)-4 

mxa = max(df.Age)+3

range(mna, mxa)

b = np.linspace(mna, mxa, num = 12).astype(int)

df["Age_bins"] = pd.cut(df['Age'], bins = b); df["Age_bins"].head(10)



# then make the boxplot with binned age ranges

g = sns.boxplot(x = "Age_bins", y = "Maximum Heart Rate",data=df,

                **boxplot_kwargs, palette = 'Reds')

plt.suptitle('Age by Maximum Heart Rate', fontsize = 20)

plt.xlabel('Age', fontsize = 20); plt.xticks(rotation = 70)
#----------------------------------------------------#

#--- Age by Maximum Heart Rate and Heart Disease  ---#

#----------------------------------------------------#



g = sns.catplot(x="Age_bins", y="Maximum Heart Rate", hue = 'Heart Disease',

               data=df, kind="box", height = 10,**boxplot_kwargs) #hue_order=["Healthy", "At Risk"],

plt.suptitle('Age by Maximum Heart Rate & Heart Disease', fontsize = 20)

plt.xlabel('Age', fontsize = 20); plt.xticks(rotation = 70)
#--------------------------------------------------------------#

#--- Age by Maximum Heart Rate and Exercise Induced Angina  ---#

#--------------------------------------------------------------#



# need to reset the order

boxprops = {'edgecolor': 'k', 'linewidth': 2}

lineprops = {'color': 'k', 'linewidth': 2}

kwargs = {'hue_order': ["Yes", "No"]}

boxplot_kwargs = dict({'boxprops': boxprops, 'medianprops': lineprops,

                       'whiskerprops': lineprops, 'capprops': lineprops,

                       'width': 0.75}, **kwargs)



sns.catplot(x="Age_bins", y="Maximum Heart Rate", hue = 'Exercise Induced Angina',

               data=df, kind="box", height = 10,**boxplot_kwargs)

plt.suptitle('Age by Maximum Heart Rate and Exercise Induced Angina', fontsize = 20)

plt.xlabel('Age', fontsize = 20);plt.xticks(rotation = 70)
#--------------------------------------------------------------------#

#--- Chest Pain by Maximum Heart Rate and Exercise Induced Angina ---#

#--------------------------------------------------------------------#



sns.catplot(x="Chest Pain", y="Maximum Heart Rate", hue = 'Exercise Induced Angina',

               data=df, kind="box", height = 10, **boxplot_kwargs)

plt.suptitle('Chest Pain by Maximum Heart Rate and Exercise Induced Angina', fontsize = 20)

plt.xlabel('Age', fontsize = 20)
#----------------------------------------------------------#

#--- Chest Pain by Maximum Heart Rate and Heart Disease ---#

#----------------------------------------------------------#



sns.catplot(x="Chest Pain", y="Maximum Heart Rate",  hue = 'Heart Disease',

               data=df, kind="box", height = 10, **boxplot_kwargs) #hue_order=["Healthy", "At Risk"])

plt.suptitle('Chest Pain by Maximum Heart Rate and Heart Disease', fontsize = 20)

plt.xlabel('Age', fontsize = 20)
#---------------------------------------------------------------#

#--- Age by Depression Induced by Exercise and Heart Disease ---#

#---------------------------------------------------------------#



sns.catplot(x="Age_bins", y="Depression Induced by Exercise",

               hue = 'Heart Disease',data=df, kind="box", height = 10)

plt.suptitle('Age by Depression Induced by Exercise and Heart Disease', fontsize = 20)

plt.xlabel('Age', fontsize = 20); plt.xticks(rotation = 70)
#--------------------------------------------------------#

#--- Age by Depression Induced by Exercise and Angina ---#

#--------------------------------------------------------#



# need to switch the order of the colors back

col_list = ['bubble gum pink', 'mud brown']

col_list_palette = sns.xkcd_palette(col_list)

sns.set_palette(col_list_palette)



sns.catplot(x="Age_bins", y="Depression Induced by Exercise",

               hue = 'Exercise Induced Angina',data=df, kind="box", height = 10)

plt.suptitle('Age by Depression Induced by Exercise & Exercise Induced Angina', fontsize = 20)

plt.xlabel('Age', fontsize = 20); plt.xticks(rotation = 70)



# need to switch the order of the colors back

col_list = ['mud brown', 'bubble gum pink']

col_list_palette = sns.xkcd_palette(col_list)

sns.set_palette(col_list_palette)
        #------------------------------#

        #####     scatterplots     #####

        #------------------------------#



plt.subplots(figsize=(8,8))

sns.regplot(x="Age", y="Maximum Heart Rate", data=df, color='#fe86a4')

plt.suptitle('Age by Maxium Heart Rate')
#-----  Heart Disease   ------#



sns.lmplot(x="Age", y="Maximum Heart Rate", hue="Heart Disease",

           data=df, height = 9, hue_order=["At Risk", "Healthy"])

plt.suptitle('Age by Maxium Heart Rate & Heart Disease')
sns.lmplot(x="Age", y="Maximum Heart Rate", col="Heart Disease",

           hue="Heart Disease", data=df, height = 6, hue_order=["At Risk", "Healthy"])

plt.suptitle('Age by Maxium Heart Rate & Heart Disease')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#---------------------------------------#

#-----  Exercise Induced Angina   ------#

#---------------------------------------#



sns.lmplot(x="Age", y="Maximum Heart Rate", hue="Exercise Induced Angina",

           data=df, height = 9, hue_order = ["Yes", "No"])

plt.suptitle('Age by Maxium Heart Rate and Exercise Induced Angina')
sns.lmplot(x="Age", y="Maximum Heart Rate", col="Exercise Induced Angina",

           hue="Exercise Induced Angina", data=df, height = 6, hue_order = ["Yes", "No"])

plt.suptitle('Age by Maxium Heart Rate and Exercise Induced Angina')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#-----  Exercise ECG ST Segment/Heart Rate Slope   ------#



# rename the columns to fit the chart

df.rename(columns = {'Exercise/Heart Rate Slope':'ECG'}, inplace = True)



g = sns.lmplot(x="Age", y="Maximum Heart Rate", col="ECG",

           hue="ECG", data=df, height = 6, aspect = .6,

           palette = "Reds")

           #palette = sns.color_palette("PiYG", 10))

plt.suptitle('Age by Maxium Heart Rate & Exercise Segment/Heart Rate Slope')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])



# change the name back to the original

df.rename(columns = {'ECG':'Exercise/Heart Rate Slope'}, inplace = True)
#-----  Chest Pain   ------#



sns.lmplot(x="Age", y="Maximum Heart Rate", col="Chest Pain",

           hue="Chest Pain", data=df, height = 5,

           palette = "Reds")

           #palette = sns.color_palette("PiYG", 9))

plt.suptitle('Age by Maxium Heart Rate and Chest Pain')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        #-----------------------------------------------#

        #######     Statistical Data Analysis    ########

        #-----------------------------------------------#



# is there a significant difference between those at risk and healthy?



#------------------------------------------------------------#

#####    Statistical Comparison of at Risk and Healthy   #####

#####      compare the means of at Risk and Healthy      #####

#------------------------------------------------------------#



from scipy import stats

from statistics import variance



######XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX######

#-------------------------------------------------#

####   Healthy and At Risk of Heart Disease   #####

#-------------------------------------------------#

######XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX######



atRisk = df[df['Heart Disease'] == 'At Risk']

healthy = df[df['Heart Disease'] == 'Healthy']



#------------------------------#

####  Maximum Heart Rate   #####

#------------------------------#



atRisk_hrate = atRisk['Maximum Heart Rate']

healthy_hrate = healthy['Maximum Heart Rate']



# mean, variance, and standard deviation

(atRisk_hrate.mean(), healthy_hrate.mean())

(atRisk_hrate.var(), healthy_hrate.var())

(atRisk_hrate.std(), healthy_hrate.std())



#-------------------------------------#

####   Equal Variance and T Test  #####

#-------------------------------------#



####   are the variances equal?   #####

stats.levene(atRisk_hrate, healthy_hrate)

# tests the null hypothesis that all input samples are from

# populations with equal variances



####    are the means the same?   #####

stats.ttest_ind(atRisk_hrate, healthy_hrate, equal_var = False)

# two-sided test for the null hypothesis that 2 independent samples

# have identical average (expected) values.

#---------------------------------------------------------------#

####    Visualization of Health and At Risk Distributions    ####

#---------------------------------------------------------------#



plt.figure(figsize=(11,6))

sns.kdeplot(atRisk_hrate,lw=8, shade=True,label='At Risk', alpha = 0.7)

plt.axvline(np.mean(atRisk_hrate), linestyle='--', linewidth = 5) # color='blood'

sns.kdeplot(healthy_hrate ,lw=8, shade=True, label='Healthy', alpha = 0.7)

plt.axvline(np.mean(healthy_hrate), linestyle='--', linewidth = 5,color = '#fe86a4')

plt.suptitle('Maximum Heart Rate for Healthy and At Risk Patients', fontsize = 20)

plt.xlabel('Maximum Heart Rate');plt.ylabel('Frequency');plt.legend()
#----------------------------------------------------------#

####    Maximum Heart Rate & Exercise Induced Angina   #####

#----------------------------------------------------------#



angYes = df[df['Exercise Induced Angina'] == "Yes"]

angNo = df[df['Exercise Induced Angina'] == "No"]



angYes_hrate = atRisk['Maximum Heart Rate']

angNo_hrate = healthy['Maximum Heart Rate']



# mean, variance, and standard deviation

(angYes_hrate.mean(), angNo_hrate.mean())

(angYes_hrate.var(), angNo_hrate.var())

(angYes_hrate.std(), angNo_hrate.std())





#-------------------------------------#

####   Equal Variance and T Test  #####

#-------------------------------------#



####   are the variances equal?   #####

stats.levene(angYes_hrate, angNo_hrate)

# tests the null hypothesis that all input samples are from

# populations with equal variances



####    are the means the same?   #####

stats.ttest_ind(angYes_hrate, angNo_hrate, equal_var = False)

# two-sided test for the null hypothesis that 2 independent samples

# have identical average (expected) values.
#---------------------------------------------------#

####    Visualization of Angina Distributions    ####

#---------------------------------------------------#



plt.figure(figsize=(11,6))

sns.kdeplot(angYes_hrate,lw=8, shade=True, label='Yes', alpha = 0.7)

plt.axvline(np.mean(angYes_hrate), linestyle='--', linewidth = 5)

sns.kdeplot(angNo_hrate,lw=8, shade=True,label='No', alpha = 0.7)

plt.axvline(np.mean(angNo_hrate),color='#fe86a4', linestyle='--', linewidth = 5)

plt.suptitle('Maximum Heart Rate & Exercise Induced Angina', fontsize = 20)

plt.xlabel('Maximum Heart Rate');plt.ylabel('Frequency');plt.legend()


#------------------------------------------------------------#

#####    Statistical Comparison of at Risk and Healthy   #####

#####      means of Depression Induced by Exercise       #####

#------------------------------------------------------------#



# mean

mean_angYes = df[df['Heart Disease'] == 'Healthy']['Depression Induced by Exercise'].mean()

mean_angNo = df[df['Heart Disease'] == 'At Risk']['Depression Induced by Exercise'].mean()



# Standard Deviation

sd_angYes = df[df['Heart Disease'] == 'Yes']['Depression Induced by Exercise'].std()

sd_angNo = df[df['Heart Disease'] == 'No']['Depression Induced by Exercise'].std()



# subset the data

angYes = df[df['Heart Disease'] == 'Yes']['Depression Induced by Exercise']

angNo = df[df['Heart Disease'] == 'No']['Depression Induced by Exercise']

# test the means

stats.ttest_ind(angYes, angNo, equal_var = False)



#---------------------------------------------#

####     Depression Induced by Exercise    ####

####   for Healthy and At Risk Patients    ####

#---------------------------------------------#



healthy_x = df[df['Heart Disease']=='Healthy']['Depression Induced by Exercise']

atRisk_x = df[df['Heart Disease']=='At Risk']['Depression Induced by Exercise']



plt.figure(figsize=(11,6))

sns.kdeplot(atRisk_x,lw=8, shade=True,label='At Risk', alpha = 0.7)

plt.axvline(np.mean(atRisk_x), linestyle='--', linewidth = 5)

sns.kdeplot(healthy_x,lw=8, shade=True, label='Healthy', alpha = 0.7)

plt.axvline(np.mean(healthy_x), color='#fe86a4', linestyle='--', linewidth = 5)

plt.suptitle('Depression Induced by Exercise for Healthy and At Risk Patients', fontsize = 20)

plt.xlabel('Depression Induced by Exercise');plt.ylabel('Frequency');plt.legend()
#---------------------------------------------#

####     Depression Induced by Exercise    ####

####         Exercise Induced Angina       ####

#---------------------------------------------#



# mean

mean_angYes = df[df['Exercise Induced Angina'] == 'Yes']['Depression Induced by Exercise'].mean()

mean_angNo = df[df['Exercise Induced Angina'] == 'No']['Depression Induced by Exercise'].mean()



# Standard Deviation

sd_angYes = df[df['Exercise Induced Angina'] == 'Yes']['Depression Induced by Exercise'].std()

sd_angNo = df[df['Exercise Induced Angina'] == 'No']['Depression Induced by Exercise'].std()



# subset the data

angYes = df[df['Exercise Induced Angina'] == 'Yes']['Depression Induced by Exercise']

angNo = df[df['Exercise Induced Angina'] == 'No']['Depression Induced by Exercise']

# test the means

stats.ttest_ind(angYes, angNo, equal_var = False)
#####  Density Plot    #####



angNo_x = df[df['Exercise Induced Angina']=='No']['Depression Induced by Exercise']

angYes_x = df[df['Exercise Induced Angina']=='Yes']['Depression Induced by Exercise']



plt.figure(figsize=(11,6))

sns.kdeplot(angYes_x,lw=8, shade=True,label='Yes', alpha = 0.7)

plt.axvline(np.mean(angYes_x), linestyle='--', linewidth = 5)

sns.kdeplot(angNo_x,lw=8, shade=True, label='No', alpha = 0.7)

plt.axvline(np.mean(angNo_x), color='#fe86a4', linestyle='--', linewidth = 5)

plt.suptitle('Angina & Depression Induced by Exercise', fontsize = 20)

plt.xlabel('Depression Induced by Exercise');plt.ylabel('Frequency');plt.legend()