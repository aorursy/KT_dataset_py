
# This is my first experience. I want to improve myself about data science.
# I look forward to your comments and suggestions. Thanks a lot. 

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/500_Person_Gender_Height_Weight_Index.csv') # read csv file
data.info() # preliminary information
data.corr() #information tables of correlation
# correlation between lines ploting
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True, linewidths=.18, fmt= '.2f',ax=ax)
plt.show()
data.head(5) # top 5 information
data.tail(5) # last 5 information
print(data.columns)
# Line Plot
# Female Height-Weight

Female = data[data.Gender =="Female"]
plt.plot(Female.Height, color = "purple", label = 'Height', linewidth = 2, alpha = 0.9, linestyle = '-')
plt.plot(Female.Weight, color = "green", label = 'Weight', linewidth = 2, alpha = 0.5, linestyle = ':')

plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('Values')              # label = name of label
plt.ylabel('Weight-Height')
plt.title('Female Height-Weight')  # title = title of plot
plt.show()

# Male Height-Weight
Male = data[data.Gender =="Male"]
plt.plot(Male.Height, color="red", label= "Height", linewidth = 2, alpha = 0.9, linestyle = '-')
plt.plot(Male.Weight, color="blue",label= "Weight", linewidth = 2, alpha = 0.5, linestyle = ':')

plt.legend(loc='upper right') # legend = puts label into plot
plt.xlabel("Values") # label = name of label
plt.ylabel("Weight-Height")
plt.title('Male Height-Weight') # title = title of plot
plt.show()
# Line Plot
# Female-Male Height

Female = data[data.Gender =="Female"]
Male = data[data.Gender =="Male"]
plt.plot(Female.Height, color = "blue", label = 'Female Height', linewidth = 2, alpha = 0.9, linestyle = '-')
plt.plot(Male.Height, color="red", label= 'Male Height', linewidth = 2, alpha = 0.5, linestyle = ':')

plt.legend(loc='upper right')
plt.xlabel("Values")
plt.ylabel("Height")
plt.title('Female-Male Height')
plt.show()

# Line Plot
# Female-Male Weight

Female = data[data.Gender =="Female"]
Male = data[data.Gender =="Male"]

plt.plot(Female.Weight, color = "blue", label = 'Female Weight', linewidth = 2, alpha = 0.9, linestyle = '-')
plt.plot(Male.Weight, color="red", label= 'Male Weight', linewidth = 2, alpha = 0.5, linestyle = ':')

plt.legend(loc='upper right')
plt.xlabel("Values")
plt.ylabel("Weight")
plt.title('Female-Male Weight')
plt.show()

# Scatter Plot
# Weight-index

data.plot(kind='scatter', x='Index', y='Weight', alpha = 0.5, color = 'purple')

plt.xlabel('Index')              
plt.ylabel('Weight')
plt.title('Weight-Index') 

# Scatter Plot
# Female Weight-index
index1= Female['Index'] >4
Female[index1]

plt.subplot(2,1,1)
plt.plot(Female[index1]['Weight'], color='black')
plt.title("Female Index- Weight")
plt.show()


# Scatter Plot
# Male Weight-index

index2= Male['Index'] >4
Male[index2]

plt.subplot(2,1,1)
plt.plot(Male[index2]['Weight'], color='blue')
plt.title("Male Index- Weight")
plt.show()


# Histogram
data.Height.plot(kind = 'hist',bins = 50,figsize = (10,10))  
plt.show()
for index,value in data[['Weight']][0:10].iterrows():
    print(index," : ",value)
data[np.logical_and(data['Height']>180, data['Weight']>100 )] # filtering of height and weight

data[np.logical_and(data['Height']>170, data['Weight']<55 )] # filtering of height and weight
data["condition"] = ["Obese Class 2" if i == 5 else "Obese Class 1" if i == 4 else "Overweight" 
                  if i == 3 else "Normal" if i == 2 else "Normal" if i == 1 else "Extremely" for i in data.Index]

data.loc[0:10,["Height","Weight","Index","condition"]]


data_height = [i/100 for i in data.Height]

data_bmi = np.zeros(500)  
i = 0
while i<len(data_bmi):
    data_bmi[i] = round( (data["Weight"][i] / (data_height[i]**2)) , 2 )  
    i = i + 1
data["Kg/m2"] = data_bmi

data.loc[:50,["Height","Weight","condition","Kg/m2"]]
data["Kg/m2"].plot(kind = 'line', color = 'purple',label = 'Kg/m2', linewidth=1, alpha = 2,grid = True, linestyle = '-')
plt.legend(loc = "upper right")
plt.ylabel("Kg/m2")
plt.title("BMI İnformation")
plt.show()