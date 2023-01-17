#Import all required libraries for reading data, analysing and visualizing data

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.preprocessing import LabelEncoder
#Read the training & test data

liver_df = pd.read_csv('../input/indian-liver-patient-records/indian_liver_patient.csv')
#Inspecting the no of columns and rows in the data

liver_df.info()
liver_df.head(5)
#Inspecting the Gender column

liver_df.Gender.value_counts()

#Inspecting the dataset column which is the target variable

liver_df.Dataset.value_counts()

#inspecting all continuos columns 

liver_df.describe()
#The below figure shows the distribution of the data 

import seaborn as sns

fig,ax=plt.subplots(figsize=(5,5))

sns.set(style="darkgrid")

sns.countplot(data=liver_df,x='Dataset',ax=ax)

#The 

fig,ax=plt.subplots(figsize=(5,5))

sns.barplot(x=liver_df.Gender,y=liver_df.Age,ax=ax)

g=sns.FacetGrid(liver_df,col="Dataset",row="Gender")

g.map(plt.hist,"Age")

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Disease by Gender and Age');
g=sns.FacetGrid(liver_df,col="Gender",hue="Dataset",height=4)

g.map(plt.scatter,"Direct_Bilirubin","Total_Bilirubin")
sns.jointplot("Total_Bilirubin", "Direct_Bilirubin", data=liver_df, kind="reg")
g=sns.FacetGrid(liver_df,col="Gender",hue="Dataset",height=4)

g.map(plt.scatter,"Total_Protiens","Albumin")
sns.jointplot("Total_Protiens", "Albumin", data=liver_df, kind="reg")

g=sns.FacetGrid(liver_df,col="Gender",hue="Dataset",height=4)

g.map(plt.scatter,"Total_Protiens","Albumin")
g=sns.FacetGrid(liver_df,col="Gender",hue="Dataset",height=4)

g.map(plt.scatter,"Alamine_Aminotransferase","Aspartate_Aminotransferase")
sns.jointplot("Aspartate_Aminotransferase", "Alamine_Aminotransferase", data=liver_df, kind="reg")
#We can see from the above that Albumin_and_Globulin_Ratio the value ranges from 0.300000 to 2.800000 

#Filling the null values of Albumin_and_Globulin_Ratio column using mean value

liver_df.Albumin_and_Globulin_Ratio.fillna(liver_df.sns.mean(),inplace=True)
liver_df.info()
#Convert categorical variable "Gender" to indicator variables



liver_df=pd.get_dummies(liver_df, prefix = 'Gender').head()
X = liver_df.drop(['Dataset','Direct_Bilirubin','Albumin','Aspartate_Aminotransferase','Alkaline_Phosphotase'], axis=1)

X.head(3)
y=liver_df['Dataset']
# Correlation

liver_corr = X.corr()

liver_corr
plt.figure(figsize=(30, 30))

sns.heatmap(liver_corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},

           cmap= 'coolwarm')

plt.title('Correlation between features');
#Visualisation of distribution of continuos Column

fig=plt.figure(figsize=(10,5))

ax=fig.add_subplot(2,2,1)

ax.hist(liver_df.Age,color='orange')

ax.set_xlabel('Age')

ax2=fig.add_subplot(2,2,2)

ax2.hist(liver_df.Total_Bilirubin,color='green')

ax2=fig.add_subplot(2,2,3)

ax2.hist(liver_df.Direct_Bilirubin,color='blue')

ax2=fig.add_subplot(2,2,4)

ax2.hist(liver_df.Alkaline_Phosphotase,color='red')

plt.show()




