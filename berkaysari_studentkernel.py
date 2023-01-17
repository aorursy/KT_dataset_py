# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings("ignore")

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
data.head()

data.tail()
data.describe()
data.corr()
fg,ax = plt.subplots(figsize = (10,10))
sns.heatmap(data.corr(), annot = True, linewidth = 5, fmt ='.2f', ax=ax)
plt.show()
data.isnull().sum() # all data for null values
for i,col in enumerate(data.columns):
    print(i+1, ' .column is  ', col)
#rename columns

data.rename(columns = ({ 'gender': 'Gender', 'race/ethnicity' : 'Race/Ethnicity',
                        'parental level of education':'Parental_Level_of_Education',
                        'lunch':'Lunch', 'test preparation course':'Test_Preparation_Course',
                        'math score':'Math_Score', 'reading score':'Reading_Score',
                        'writing score':'Writing_Score' }), inplace = True)
#inplace = true metodu tabloyu yeniden gostermesini engelliyor
for i,col in enumerate(data.columns):
    print(i+1, ' .column is  ', col)
data['Gender'].value_counts()  #kacar adet gender oldugunu gosteriyor
plt.bar(data['Gender'].value_counts().index , data['Gender'].value_counts().values, width = 0.5, alpha = 0.5,color = ('pink','blue'))
plt.legend(loc=0)
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.title('Show of Gender Bar Plot')
plt.show()
plt.bar(data['Race/Ethnicity'].value_counts().index , data['Race/Ethnicity'].value_counts().values, width = 0.5, alpha = 0.5,color = ('pink','blue','brown','red','yellow'))
plt.legend(loc=0)
plt.xlabel('Race/Ethnicity')
plt.ylabel('Frequency')
plt.title('Show of Gender Bar Plot')
plt.show()
plt.subplots(figsize = (15,8))
sns.barplot(x = 'Parental_Level_of_Education', y= 'Reading_Score', hue= 'Gender',linewidth= 2,edgecolor= '.2' , data= data)
plt.legend()
plt.xlabel('Parental_Level_of_Education', size= 14)
plt.xticks(rotation=45)
plt.ylabel('Reading_Score', size=14)
plt.title('Parental_Level_of_Education - Reading_Score ',size= 18)
plt.show()
plt.subplots(figsize = (15,8))
sns.barplot(x = 'Parental_Level_of_Education', y= 'Writing_Score', hue= 'Gender',linewidth= 2,edgecolor= '.2' , data= data)
plt.legend()
plt.xlabel('Parental_Level_of_Education', size= 14)
plt.xticks(rotation=45)
plt.ylabel('Writing_Score', size=14)
plt.title('Parental_Level_of_Education - Writing_Score ',size= 18)
plt.show()
plt.subplots(figsize = (15,8))
sns.barplot(x = 'Parental_Level_of_Education', y= 'Math_Score', hue= 'Gender',linewidth= 2,edgecolor= '.2' , data= data)
plt.legend()
plt.xlabel('Parental_Level_of_Education', size= 14)
plt.xticks(rotation=45)
plt.ylabel('Math_Score', size=14)
plt.title('Parental_Level_of_Education - Math_Score ',size= 18)
plt.show()
plt.subplots(figsize = (24,9))
sns.barplot(x = 'Gender', y= 'Math_Score', hue= 'Parental_Level_of_Education', linewidth= 2,edgecolor= '.2' ,data= data)
plt.legend(loc=9)
plt.xlabel('Gender', size= 14)
plt.xticks(rotation=45)
plt.ylabel('Math_Score', size=14)
plt.title('Gender - Math Score', size =18)
plt.show()
plt.subplots(figsize = (24,9))
sns.barplot(x = 'Gender', y= 'Writing_Score', hue= 'Parental_Level_of_Education',linewidth= 2,edgecolor= '.2' , data= data)
plt.legend(loc=9)
plt.xlabel('Gender', size= 14)
plt.xticks(rotation=45)
plt.ylabel('Writing_Score', size=14)
plt.title('Gender - Writing_Score', size =18)
plt.show()
plt.subplots(figsize = (24,9))
sns.barplot(x = 'Gender', y= 'Reading_Score', hue= 'Parental_Level_of_Education', linewidth= 2,edgecolor= '.2' ,data= data)
plt.legend(loc=9)
plt.xlabel('Gender', size= 14)
plt.xticks(rotation=45)
plt.ylabel('Reading_Score', size=14)
plt.title('Gender - Reading_Score', size =18)
plt.show()
plt.subplots(figsize = (12,8))
sns.lineplot( x="Reading_Score", y="Writing_Score", hue="Gender",data=data)
plt.show()
plt.subplots(figsize = (12,8))
sns.scatterplot( x="Reading_Score", y="Writing_Score", hue="Gender",data=data)
plt.show()
data.boxplot(column= ('Reading_Score', 'Writing_Score','Math_Score'), by='Gender', figsize = (15,12))
plt.show
data_new = data.head(7)
data_new
melted = pd.melt(frame= data_new,id_vars = 'Gender', value_vars = ['Race/Ethnicity', 'Math_Score'] )
melted
data1 = data.head(7)
data2 =data.tail(7)
data_new = pd.concat([data1,data2],axis=0, ignore_index=True)
data_new
data1 = data['Reading_Score'].head(7)
data2 = data['Writing_Score'].head(7)
data3 = data['Writing_Score'].head(7)

conc_data = pd.concat([data1,data2,data3], axis = 1)
conc_data
data.info()
data['Writing_Score'].dropna(inplace = True)
assert data['Writing_Score'].notnull().all()
assert data['Math_Score'].notnull().all()
assert data['Reading_Score'].notnull().all()