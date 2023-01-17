#importing packages

import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import os
#load data

data=pd.read_csv("../input/hepatitis-c-virus-hcv-for-egyptian-patients/HCV-Egy-Data.csv")
data.head()
#Explore Data

data.columns
#Rename Columns

data.columns=['Age ', 'Gender', 'BMI', 'Fever', 'Nausea/Vomting','Headache ',

       'Diarrhea','Fatigue','Jaundice',

       'Epigastric_pain ', 'WBC','RBC','HGB', 'Plat','AST_1','ALT_1',

       'ALT_4', 'ALT_12','ALT_24','ALT_36','ALT 48','ALT_after_24w',

       'RNA_Base','RNA 4','RNA_12', 'RNA_EOT','RNA_EF',

       'Baseline_histological_Grading','Baselinehistological_staging']
#Remove spaces at the ends of column names to avoid any further errors:

data.columns = data.columns.str.strip()
data.info()
data.describe()
#check for missing values

data.isnull().sum().sum()
#Change the classes columns to categorical for better visualization 



#categorize columns: Gender,Fever,Nausea/Vomting,Headache,Diarrhea,Fatigue,Jaundice

data_cat=data[['Gender','Fever','Nausea/Vomting','Headache','Fatigue','Jaundice','Diarrhea','Epigastric_pain',"Baselinehistological_staging"]]
data_cat.head()
data_cat=data_cat.astype('category')
data_cat.info()
#Replacing the values to names e.g. 1:Male,2:Female 1:Absent,2:present 

data_cat['Gender'].replace([1,2],['Male','Female'],inplace=True)

data_cat['Gender']
#check for the unique values for the other columns(Symptoms):

Symptoms_cols=data_cat[["Fever","Nausea/Vomting","Headache","Fatigue","Jaundice","Diarrhea",'Epigastric_pain']]



Symptoms_cols_values = data_cat[["Fever","Nausea/Vomting","Headache","Fatigue","Jaundice","Diarrhea",'Epigastric_pain']].values

unique_values =  np.unique(Symptoms_cols_values)



print(unique_values)
#Replacing values (1-2) to (Absent,Present) in Symptoms Features:



data_cat['Fever'].replace([1,2],['Absent','Present'],inplace=True)

data_cat['Nausea/Vomting'].replace([1,2],['Absent','Present'],inplace=True)

data_cat['Headache'].replace([1,2],['Absent','Present'],inplace=True)

data_cat['Fatigue'].replace([1,2],['Absent','Present'],inplace=True)

data_cat['Jaundice'].replace([1,2],['Absent','Present'],inplace=True)

data_cat['Diarrhea'].replace([1,2],['Absent','Present'],inplace=True)

data_cat['Epigastric_pain'].replace([1,2],['Absent','Present'],inplace=True)
#Doing the same for Histological Staging

data_cat['Baselinehistological_staging'].unique()
data_cat['Baselinehistological_staging'].replace([1,2,3,4],['Portal Fibrosis','Few Septa','Many Septa','Cirrhosis'],inplace=True)
data_cat.head()
#Gain insights of the categorical data:

sns.countplot(data_cat['Gender'],palette="Reds_r")

plt.show()
data['Gender'].value_counts()
#check Histological Grading and Staging across each Gender



sns.countplot(x=data['Baseline_histological_Grading'],hue=data_cat['Gender'],palette="Dark2")

plt.title("Gender Chart for Histological Grading")

plt.xlabel("Histological Grading")

plt.legend(bbox_to_anchor=(1,1))

plt.show()
#Histological Stages across each Gender

sns.countplot(x=data_cat['Baselinehistological_staging'],hue=data_cat['Gender'],palette="viridis")

plt.legend(bbox_to_anchor=(1,1))

plt.title("Gender Chart for Histological Staging")

plt.xlabel("Histological Staging")

plt.show()
data_cat['Histological Stages']=data_cat['Baselinehistological_staging']

data_cat['Histological Gradings']=data['Baseline_histological_Grading']
#parameters

order=['Present','Absent']

h_order=['Portal Fibrosis','Few Septa', 'Many Septa', 'Cirrhosis']

height=5

asp=0.7

palette=['winter','Spectral','coolwarm','gist_gray','plasma','Dark2','Pastel1']



#plots

FeverPltS = sns.catplot(x="Fever", hue="Histological Stages", col="Gender",

                data=data_cat, kind="count",

                 height=height, aspect=asp,order=order,hue_order=h_order,palette=palette[1])



NauseaPltS =sns.catplot(x="Nausea/Vomting", hue="Histological Stages", col="Gender",

                data=data_cat, kind="count",

                 height=height, aspect=asp,order=order,hue_order=h_order,palette=palette[0])





HeadachePltS =sns.catplot(x="Headache", hue="Histological Stages", col="Gender",

                data=data_cat, kind="count",

                 height=height, aspect=asp,order=order,hue_order=h_order,palette=palette[2])





FatiguePltS =sns.catplot(x="Fatigue", hue="Histological Stages", col="Gender",

                data=data_cat, kind="count",

                 height=height, aspect=asp,order=order,hue_order=h_order,palette=palette[3])





JaundicePltS =sns.catplot(x="Jaundice", hue="Histological Stages", col="Gender",

                data=data_cat, kind="count",

                 height=height, aspect=asp,order=order,hue_order=h_order,palette=palette[4])







DiarrheaPltS =sns.catplot(x="Diarrhea", hue="Histological Stages", col="Gender",

                data=data_cat, kind="count",

                 height=height, aspect=asp,order=order,hue_order=h_order,palette=palette[5])





Epigastric_painPltS =sns.catplot(x="Epigastric_pain", hue="Histological Stages", col="Gender",

                data=data_cat, kind="count",

                 height=height, aspect=asp,order=order,hue_order=h_order,palette=palette[6])



#parameters

order=['Present','Absent']

height=8

asp=1

palette=['winter','Spectral','coolwarm','gist_gray','plasma','Dark2','Pastel1']



#plots

FeverPltG = sns.catplot(x="Fever", hue="Histological Gradings", col="Gender",

                data=data_cat, kind="count",

                height=height, aspect=asp,order=order,palette="gnuplot")



NauseaPltG = sns.catplot(x="Nausea/Vomting", hue="Histological Gradings", col="Gender",

                data=data_cat, kind="count",

                height=height, aspect=asp,order=order,palette=palette[1])



HeadachePltG = sns.catplot(x="Headache", hue="Histological Gradings", col="Gender",

                data=data_cat, kind="count",

                height=height, aspect=asp,order=order,palette=palette[2])



FatiguePltG = sns.catplot(x="Fatigue", hue="Histological Gradings", col="Gender",

                data=data_cat, kind="count",

                height=height, aspect=asp,order=order,palette=palette[3])



JaundicePltG = sns.catplot(x="Jaundice", hue="Histological Gradings", col="Gender",

                data=data_cat, kind="count",

                height=height, aspect=asp,order=order,palette=palette[4],)



DiarrheaPltG = sns.catplot(x="Diarrhea", hue="Histological Gradings", col="Gender",

                data=data_cat, kind="count",

                height=height, aspect=asp,order=order,palette=palette[5],)



Epigastric_painPltG = sns.catplot(x="Epigastric_pain", hue="Histological Gradings", col="Gender",

                data=data_cat, kind="count",

                height=height, aspect=asp,order=order,palette=palette[6],)



#Age distrubtion across the dataset



sns.distplot(data.Age,bins=10,label="Age",color="green",rug=True)

plt.yticks([])

plt.title("Age Distribution")

plt.legend()

plt.show()
#Age with Gender

fig,axis=plt.subplots(2,1,figsize=(5,10))



sns.boxplot(x=data_cat["Gender"],y=data['Age'],ax=axis[0])

sns.violinplot(x=data_cat["Gender"],y=data['Age'],inner="quartile",bandwidth=0.2, ax=axis[1],palette='twilight')



plt.legend(bbox_to_anchor=(1,1))

plt.show()
#Aspartate Transminase distribution:



AST=data['AST_1']



sns.distplot(AST,bins=15,label="AST",color="green")

plt.xlabel("Aspartate Transaminase Ratio")

plt.yticks([])

plt.legend()

plt.show()

plt.savefig("ASTdist.png")



#AST vs Stages

plt.figure(figsize=(10,5))

sns.swarmplot(y=data['AST_1'],x=data_cat['Baselinehistological_staging'],hue=data_cat.Gender,palette="coolwarm")

plt.legend(bbox_to_anchor=(1,1))

plt.xlabel("Stages")

plt.ylabel("AST")

plt.title("The distribution of the AST Enzyme Across each Stage ")

plt.show()

plt.savefig("ASTstages.png")
#check the disribution of RBC,WBC and HGB

sns.set_style('whitegrid')



fig,axis=plt.subplots(3,1,figsize=(5,15))



RBCplt=sns.distplot(data['RBC'], kde = False, color ='red', bins = 20,ax=axis[0],label="RBC Distribution") 

WBCplt=sns.distplot(data['WBC'], kde = True, color ='blue', bins = 30,ax=axis[1],label="WBC Distribution")

HGBplt=plt.hist(data['HGB'] ,bins=30,label="HGB Distribution",color="green")





plt.legend()

plt.show()
#AST vs RBC & WBC 

sns.set_style("darkgrid")

fig,axis=plt.subplots(2,1,figsize=(8,15))



RBCASTplt=sns.kdeplot(data['RBC'],data['AST_1'],cmap="Reds",shade=True,shade_lowest=False,ax=axis[0])

RBCASTpltcontour=sns.kdeplot(data['RBC'],data['AST_1'],cmap="autumn",ax=axis[0])



WBCASTplt=sns.kdeplot(data['WBC'],data['AST_1'],cmap="Blues",shade=True,shade_lowest=False,ax=axis[1])

WBCASTpltcontour=sns.kdeplot(data['WBC'],data['AST_1'],cmap="winter",ax=axis[1])







plt.show()

#AST VS HGB(Blood Hemoglobin level) across Gender:
plt.figure(figsize=(10,5))



sns.swarmplot(y=data['AST_1'],x=data['HGB'],hue=data_cat.Gender,palette="ocean")



plt.legend(bbox_to_anchor=(1,1))

plt.xlabel("HGB Level")

plt.ylabel("AST")

plt.title("The distribution of the AST Enzyme Across each HGB Level ")



plt.show()

plt.savefig('asthgb.png')
#plat distribution 

sns.distplot(data.Plat,bins=8,axlabel="Platelets",rug=True,color='red')

plt.show()
#AST V PLAT

sns.set_style('whitegrid')

WBCASTplt=sns.kdeplot(data['Plat'],data['AST_1'],cmap="Reds",shade=True,shade_lowest=False)

WBCASTpltcontour=sns.kdeplot(data['Plat'],data['AST_1'],cmap="Reds")



plt.show()
ALTdata=data[['ALT_1', 'ALT_4', 'ALT_12', 'ALT_24', 'ALT_36','ALT 48', 'ALT_after_24w']]

ALTdata
ALTdata.columns=['ALT WEEK 1','ALT WEEK 4','ALT WEEK 12','ALT WEEK 24','ALT WEEK 36','ALT WEEK 48',"'ALT after 24 WEEKs '"]
#ALT Enzyme distribution across each week

sns.set_palette(palette="inferno")



for i, col in enumerate(ALTdata.columns):

    plt.figure(i)

    sns.distplot(ALTdata[col],rug=True)

#ALT WEEKS V stages

def ALTSwarmPlot(data,x,hue,figsize,palette):

    for i, col in enumerate(ALTdata.columns):

        plt.figure(i,figsize=figsize)

        sns.swarmplot(y=ALTdata[col],x=x,hue=hue,palette=palette)

        plt.yticks([])

        plt.legend(bbox_to_anchor=(1,1))

ALTSwarmPlot(data=ALTdata,x=data_cat['Histological Stages'], figsize=((8,5)),hue=data_cat['Gender'],palette="hls")
#ALT WEEKS V Grades

ALTSwarmPlot(data=ALTdata,x=data_cat['Histological Gradings'],figsize=(10,5),hue=data_cat['Gender'],palette="cubehelix")
#ALT WEEKS V HGB

ALTSwarmPlot(data=ALTdata,x=data['HGB'],figsize=(10,5),hue=data_cat['Gender'],palette="coolwarm")
print("All numerical columns: \n",data.columns,'\n'*2)#Gender is included for better visualization

print("ALT weeks columns: \n",ALTdata.columns)
#plot ALT WEEKS vs BMI:

def ALTHexPlot(data,x,figsize,color):

    for i, col in enumerate(ALTdata.columns):

        plt.figure(i,figsize=figsize)

        sns.jointplot(y=ALTdata[col],x=x,color=color,kind="hex")

        plt.yticks([])

        plt.legend(bbox_to_anchor=(1,1))

        plt.show()

ALTHexPlot(data=ALTdata,x=data['BMI'],figsize=((8,5)),color='magenta')

plt.savefig("ALT Vs BMI")
#ALT WEEKS VS WBCs

ALTHexPlot(data=ALTdata,x=data['WBC'],figsize=((8,5)),color='cyan')

#ALT WEEKS VS WBCs

ALTHexPlot(data=ALTdata,x=data['RBC'],figsize=((8,5)),color='red')

data.columns
#ALT Weeks VS Platelets

ALTHexPlot(data=ALTdata,x=data['Plat'],figsize=((8,5)),color='gray')
RNAdata=data[['RNA_Base','RNA 4','RNA_12','RNA_EOT','RNA_EF']]

RNAdata.head()
#Distribution of each RNA 

sns.set_palette(palette="Dark2")



for i, col in enumerate(RNAdata.columns):

    plt.figure(i)

    sns.distplot(RNAdata[col],rug=True)
#RNA at each Stage/Gradings.

def RNASwarmPlot(data,x,hue,figsize,palette):

    for i, col in enumerate(RNAdata.columns):

        plt.figure(i,figsize=figsize)

        sns.swarmplot(y=RNAdata[col],x=x,hue=hue,palette=palette)

        plt.legend(bbox_to_anchor=(1,1))

def RNABoxPlot(data,x,hue,figsize,palette):

    for i, col in enumerate(RNAdata.columns):

        plt.figure(i,figsize=figsize)

        sns.boxplot(y=RNAdata[col],x=x,hue=hue,palette=palette)

        plt.legend(bbox_to_anchor=(1,1))

#RNA VS STAGES

RNASwarmPlot(data=data,x=data_cat['Histological Stages'],figsize=(8,5),palette="cool",hue=data_cat['Gender'])
#RNA VS GRADINGS 

RNABoxPlot(data=data,x=data_cat['Histological Gradings'],figsize=(18,8),palette="coolwarm",hue=data_cat['Gender'])
data.head()


# Heatmap

sns.set(style="white")

# Create a correlation matrix

corr = data.corr()

# Creating a mask the size of our covariance matrix

mask = np.zeros_like(corr, dtype=bool)

mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11,9))



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr,mask=mask,cmap="twilight",square=True, 

            linewidth=.5)

ax.set_title('Multi-Collinearity of Features')

plt.savefig('correlation.png')

#stages

plt.figure(figsize=(8,12))



HGBStagingPlt=sns.countplot(y=data['HGB'],hue=data_cat['Baselinehistological_staging'],palette="twilight")



plt.legend(bbox_to_anchor=(1,1))

plt.show()
#Gradings

sns.set_style("whitegrid")

plt.figure(figsize=(8,12))



HGBGradingPlt=sns.countplot(y=data['HGB'],hue=data['Baseline_histological_Grading'],palette="twilight",saturation=6,)



plt.legend(bbox_to_anchor=(1,1))

plt.show()
#Dashboard

sns.set_style("darkgrid")

fig,axis=plt.subplots(4,2,figsize=(10,20))

k1=sns.countplot(x=data['Baseline_histological_Grading'],hue=data_cat['Gender'],palette="Dark2",ax=axis[0,0])

k2=sns.countplot(x=data_cat['Histological Stages'],hue=data_cat['Gender'],palette="viridis",ax=axis[0,1])

k3=sns.distplot(data['Age'],bins=10,label="Age",color="green",rug=True,ax=axis[1,0])

k4=sns.swarmplot(y=data['AST_1'],x=data_cat['Histological Stages'],hue=data_cat.Gender,palette="coolwarm",ax=axis[1,1])

k5=RBCASTplt=sns.kdeplot(data['RBC'],data['AST_1'],cmap="Reds",shade=True,shade_lowest=False,ax=axis[2,0])

k6=sns.swarmplot(y=data['ALT_after_24w'],x=data['HGB'],hue=data_cat.Gender,palette="ocean",ax=axis[2,1])

k7=HGBStagingPlt=sns.countplot(data['HGB'],hue=data_cat['Baselinehistological_staging'],palette="twilight",ax=axis[3,0])

k8=sns.distplot(data['Plat'],bins=8,axlabel="Platelets",rug=True,color='red',ax=axis[3,1])

plt.show()

plt.savefig("Dashboard.png")