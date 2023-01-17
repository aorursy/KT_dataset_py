# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data= pd.read_csv('/kaggle/input/play-tennis/play_tennis.csv')
""" Calculating the Entropy of parent """

E_play = -((len(data[data['play']=='Yes'])/len(data))*math.log2(len(data[data['play']=='Yes'])/len(data))) - ((len(data[data['play']=='No'])/len(data))*math.log2(len(data[data['play']=='No'])/len(data)))

E_play
""" For outlook column """

out = data.groupby('outlook')



""" Creating different groups for every different Outlook Conditions """

out1 = out.get_group('Sunny')

out2 = out.get_group('Overcast')

out3 = out.get_group('Rain')



""" Calculating the Entropy for every different Outlook Conditions """

E_out1 = -((len(out1[out1['play']=='Yes'])/len(out1))*math.log2(len(out1[out1['play']=='Yes'])/len(out1))) - ((len(out1[out1['play']=='No'])/len(out1))*math.log2(len(out1[out1['play']=='No'])/len(out1)))

E_out2 = -((len(out2[out2['play']=='Yes'])/len(out2))*math.log2(len(out2[out2['play']=='Yes'])/len(out2)))

E_out3 = -((len(out3[out3['play']=='Yes'])/len(out3))*math.log2(len(out3[out3['play']=='Yes'])/len(out3))) - ((len(out3[out3['play']=='No'])/len(out3))*math.log2(len(out3[out3['play']=='No'])/len(out3)))



""" Calculating the Weighted Entropy """

WE_out = (len(out1)/len(data)*E_out1) + (len(out2)/len(data)*E_out2) + (len(out3)/len(data)*E_out3)



""" Canculating the Information Gain """

IG_out = E_play - WE_out

print("Information Gain of outlook is = ",round(IG_out,2))
""" For Temperature column """



temp = data.groupby('temp') 



""" Creating different groups for every different Temperatures """



temp1 = temp.get_group('Hot')

temp2 = temp.get_group('Mild')

temp3 = temp.get_group('Cool')



""" Calculating the Entropy for every different Temperatures """



E_temp1= -((len(temp1[temp1['play']=='Yes'])/len(temp1))*math.log2(len(temp1[temp1['play']=='Yes'])/len(temp1))) - ((len(temp1[temp1['play']=='No'])/len(temp1))*math.log2(len(temp1[temp1['play']=='No'])/len(temp1)))

E_temp2= -((len(temp2[temp2['play']=='Yes'])/len(temp2))*math.log2(len(temp2[temp2['play']=='Yes'])/len(temp2))) - ((len(temp2[temp2['play']=='No'])/len(temp2))*math.log2(len(temp2[temp2['play']=='No'])/len(temp2)))

E_temp3= -((len(temp3[temp3['play']=='Yes'])/len(temp3))*math.log2(len(temp3[temp3['play']=='Yes'])/len(temp3))) - ((len(temp3[temp3['play']=='No'])/len(temp3))*math.log2(len(temp3[temp3['play']=='No'])/len(temp3)))



""" Calculating the Weighted Entropy """



WE_temp= (len(temp1)/len(data)*E_temp1) + (len(temp2)/len(data)*E_temp2) + (len(temp3)/len(data)*E_temp3)



""" Canculating the Information Gain """



IG_temp = E_play-WE_temp

print("Information Gain of temp is = ",round(IG_temp,2))
# For humidity column

hum = data.groupby('humidity')



""" Creating different groups for every different Humidity Conditions """



hum1 = hum.get_group('High')

hum2 = hum.get_group('Normal')



""" Calculating the Entropy for every different Humidity Conditions """



E_hum1 = -((len(hum1[hum1['play']=='Yes'])/len(hum1))*math.log2(len(hum1[hum1['play']=='Yes'])/len(hum1))) - ((len(hum1[hum1['play']=='No'])/len(hum1))*math.log2(len(hum1[hum1['play']=='No'])/len(hum1)))

E_hum2 = -((len(hum2[hum2['play']=='Yes'])/len(hum2))*math.log2(len(hum2[hum2['play']=='Yes'])/len(hum2))) - ((len(hum2[hum2['play']=='No'])/len(hum2))*math.log2(len(hum2[hum2['play']=='No'])/len(hum2)))



""" Calculating the Weighted Entropy """



WE_hum = (len(hum1)/len(data)*E_hum1) + (len(hum2)/len(data)*E_hum2)



""" Canculating the Information Gain """



IG_hum = E_play-WE_hum

print("Information Gain of humidity is = ",round(IG_hum,2))
# For wind column

wind = data.groupby('wind')



# Creating different groups for every different Wind Conditions

wind1 = wind.get_group('Weak')

wind2 = wind.get_group('Strong')



# Calculating the Entropy for every different Wind Conditions

E_wind1 = -((len(wind1[wind1['play']=='Yes'])/len(wind1))*math.log2(len(wind1[wind1['play']=='Yes'])/len(wind1))) - ((len(wind1[wind1['play']=='No'])/len(wind1))*math.log2(len(wind1[wind1['play']=='No'])/len(wind1)))

E_wind2 = -((len(wind2[wind2['play']=='Yes'])/len(wind2))*math.log2(len(wind2[wind2['play']=='Yes'])/len(wind2))) - ((len(wind2[wind2['play']=='No'])/len(wind2))*math.log2(len(wind2[wind2['play']=='No'])/len(wind2)))



""" Calculating the Weighted Entropy """

WE_wind = (len(wind1)/len(data)*E_wind1) + (len(wind2)/len(data)*E_wind2)



""" Canculating the Information Gain """

IG_wind = E_play-WE_wind

print("Information Gain of wind is = ",round(IG_wind,2))
""" Acording to the values of Information Gain printing the name of the column 

    where the first split will take place """



if ((IG_out>IG_temp) and (IG_out>IG_hum) and (IG_out>IG_wind)):

    print("So the first split will take place on Outlook column")

    

elif ((IG_temp>IG_out) and (IG_temp>IG_hum) and (IG_temp>IG_wind)):

    print("so the first split will take place on Temp column")

    

elif ((IG_hum>IG_out) and (IG_hum>IG_temp) and (IG_hum>IG_wind)):

    print("so the first split will take place on humidity column")

    

elif ((IG_wind>IG_out) and (IG_wind>IG_hum) and (IG_wind>temp)): 

    print("so the first split will take place on Wind column")
df = pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')

df.sample(10)
""" Calculating the Entropy of parent """

E_species = -((len(df[df['species']=='Iris-setosa'])/len(df))*math.log2(len(df[df['species']=='Iris-setosa'])/len(df))) - ((len(df[df['species']=='Iris-virginica'])/len(df))*math.log2(len(df[df['species']=='Iris-virginica'])/len(df))) - ((len(df[df['species']=='Iris-versicolor'])/len(df))*math.log2(len(df[df['species']=='Iris-versicolor'])/len(df)))

E_species
""" Sorting the value of the column petal_length in ascending order """

df.sort_values(by=['petal_length'], ascending=True,inplace=True)
""" creating a list with the values of petal_length column """

list1=df['petal_length'].tolist()

l=len(list1)
""" Creating 2 Empty lists for future use """



list2=[]

list3=[]



"""

as I created a list with the values of petal_length column so I am using a loop where the value 

is starting from 0 to the length of the list

and Creating two different Sub-groups for each and every value of the list i.e. creating two 

different Sub-groups for each and every value of petal_length column

and then Calculating the Entropy for every Sub-Group 

"""

for i in range(l):

    df1=df[df['petal_length']<=list1[i]]

    df2=df[df['petal_length']>list1[i]]



#Calculating the Entropy for Sub-Group 1



#Calculating Entropy for Sub-Group 1 If all Species are present in the Sub-Group 



    if ((len(df1[df1['species']=='Iris-setosa'])!= 0) and (len(df1[df1['species']=='Iris-virginica'])!= 0) and (len(df1[df1['species']=='Iris-versicolor'])!= 0)):

        E_df1 = -((len(df1[df1['species']=='Iris-setosa'])/len(df1))*math.log2(len(df1[df1['species']=='Iris-setosa'])/len(df1))) - ((len(df1[df1['species']=='Iris-virginica'])/len(df1))*math.log2(len(df1[df1['species']=='Iris-virginica'])/len(df1))) - ((len(df1[df1['species']=='Iris-versicolor'])/len(df1))*math.log2(len(df1[df1['species']=='Iris-versicolor'])/len(df1)))



    # Calculating Entropy for Sub-Group 1 If Iris-versicolor is not present in the Sub-Group



    elif ((len(df1[df1['species']=='Iris-setosa'])!= 0) and (len(df1[df1['species']=='Iris-virginica'])!= 0) and (len(df1[df1['species']=='Iris-versicolor'])== 0)):

        E_df1 = -((len(df1[df1['species']=='Iris-setosa'])/len(df1))*math.log2(len(df1[df1['species']=='Iris-setosa'])/len(df1))) - ((len(df1[df1['species']=='Iris-virginica'])/len(df1))*math.log2(len(df1[df1['species']=='Iris-virginica'])/len(df1)))



    # Calculating Entropy for Sub-Group 1 If Iris-setosa is not present in the Sub-Group



    elif ((len(df1[df1['species']=='Iris-setosa'])== 0) and (len(df1[df1['species']=='Iris-virginica'])!= 0) and (len(df1[df1['species']=='Iris-versicolor'])!= 0)):

        E_df1 = - ((len(df1[df1['species']=='Iris-virginica'])/len(df1))*math.log2(len(df1[df1['species']=='Iris-virginica'])/len(df1))) - ((len(df1[df1['species']=='Iris-versicolor'])/len(df1))*math.log2(len(df1[df1['species']=='Iris-versicolor'])/len(df1)))





    # Calculating Entropy for Sub-Group 1 If Iris-virginica is not present in the Sub-Group



    elif ((len(df1[df1['species']=='Iris-setosa'])!= 0) and (len(df1[df1['species']=='Iris-virginica'])== 0) and (len(df1[df1['species']=='Iris-versicolor'])!= 0)):

        E_df1 = -((len(df1[df1['species']=='Iris-setosa'])/len(df1))*math.log2(len(df1[df1['species']=='Iris-setosa'])/len(df1))) - ((len(df1[df1['species']=='Iris-versicolor'])/len(df1))*math.log2(len(df1[df1['species']=='Iris-versicolor'])/len(df1)))



    # Calculating Entropy for Sub-Group 1 If and only if Iris-setosa present in the Sub-Group



    elif ((len(df1[df1['species']=='Iris-setosa'])!= 0) and (len(df1[df1['species']=='Iris-virginica'])== 0) and (len(df1[df1['species']=='Iris-versicolor'])== 0)):

        E_df1 = -((len(df1[df1['species']=='Iris-setosa'])/len(df1))*math.log2(len(df1[df1['species']=='Iris-setosa'])/len(df1)))





    # Calculating Entropy for Sub-Group 1 If and only if Iris-virginica present in the Sub-Group



    elif ((len(df1[df1['species']=='Iris-setosa'])== 0) and (len(df1[df1['species']=='Iris-virginica'])!= 0) and (len(df1[df1['species']=='Iris-versicolor'])== 0)):

        E_df1 = - ((len(df1[df1['species']=='Iris-virginica'])/len(df1))*math.log2(len(df1[df1['species']=='Iris-virginica'])/len(df1)))



    # Calculating Entropy for Sub-Group 1 If and only if Iris-versicolor present in the Sub-Group



    elif ((len(df1[df1['species']=='Iris-setosa'])== 0) and (len(df1[df1['species']=='Iris-virginica'])== 0) and (len(df1[df1['species']=='Iris-versicolor'])!= 0)):

        E_df1 = - ((len(df1[df1['species']=='Iris-versicolor'])/len(df1))*math.log2(len(df1[df1['species']=='Iris-versicolor'])/len(df1)))

            

        

        

# Applying the same formula for Sub-Group 2



    if ((len(df2[df2['species']=='Iris-setosa'])!= 0) and (len(df2[df2['species']=='Iris-virginica'])!= 0) and (len(df2[df2['species']=='Iris-versicolor'])!= 0)):

        E_df2 = -((len(df2[df2['species']=='Iris-setosa'])/len(df2))*math.log2(len(df2[df2['species']=='Iris-setosa'])/len(df2))) - ((len(df2[df2['species']=='Iris-virginica'])/len(df2))*math.log2(len(df2[df2['species']=='Iris-virginica'])/len(df2))) - ((len(df2[df2['species']=='Iris-versicolor'])/len(df2))*math.log2(len(df2[df2['species']=='Iris-versicolor'])/len(df2)))





    elif ((len(df2[df2['species']=='Iris-setosa'])!= 0) and (len(df2[df2['species']=='Iris-virginica'])!= 0) and (len(df2[df2['species']=='Iris-versicolor'])== 0)):

        E_df2= -((len(df2[df2['species']=='Iris-setosa'])/len(df2))*math.log2(len(df2[df2['species']=='Iris-setosa'])/len(df2))) - ((len(df2[df2['species']=='Iris-virginica'])/len(df2))*math.log2(len(df2[df2['species']=='Iris-virginica'])/len(df2)))





    elif ((len(df2[df2['species']=='Iris-setosa'])== 0) and (len(df2[df2['species']=='Iris-virginica'])!= 0) and (len(df2[df2['species']=='Iris-versicolor'])!= 0)):

        E_df2 = - ((len(df2[df2['species']=='Iris-virginica'])/len(df2))*math.log2(len(df2[df2['species']=='Iris-virginica'])/len(df2))) - ((len(df2[df2['species']=='Iris-versicolor'])/len(df2))*math.log2(len(df2[df2['species']=='Iris-versicolor'])/len(df2)))







    elif ((len(df2[df2['species']=='Iris-setosa'])!= 0) and (len(df2[df2['species']=='Iris-virginica'])== 0) and (len(df2[df2['species']=='Iris-versicolor'])!= 0)):

        E_df2 = -((len(df2[df2['species']=='Iris-setosa'])/len(df2))*math.log2(len(df2[df2['species']=='Iris-setosa'])/len(df2))) - ((len(df2[df2['species']=='Iris-versicolor'])/len(df2))*math.log2(len(df2[df2['species']=='Iris-versicolor'])/len(df2)))





    elif ((len(df2[df2['species']=='Iris-setosa'])!= 0) and (len(df2[df2['species']=='Iris-virginica'])== 0) and (len(df2[df2['species']=='Iris-versicolor'])== 0)):

        E_df2 = -((len(df2[df2['species']=='Iris-setosa'])/len(df2))*math.log2(len(df2[df2['species']=='Iris-setosa'])/len(df2)))







    elif ((len(df2[df2['species']=='Iris-setosa'])== 0) and (len(df2[df2['species']=='Iris-virginica'])!= 0) and (len(df2[df2['species']=='Iris-versicolor'])== 0)):

        E_df2 = - ((len(df2[df2['species']=='Iris-virginica'])/len(df2))*math.log2(len(df2[df2['species']=='Iris-virginica'])/len(df2)))





    elif ((len(df2[df2['species']=='Iris-setosa'])== 0) and (len(df2[df2['species']=='Iris-virginica'])== 0) and (len(df2[df2['species']=='Iris-versicolor'])!= 0)):

        E_df2 = - ((len(df2[df2['species']=='Iris-versicolor'])/len(df2))*math.log2(len(df2[df2['species']=='Iris-versicolor'])/len(df2)))

        

# Calculating the Weighted Entropy for every sub-group

    WE_df = (len(df1)/len(df)*E_df1) + (len(df2)/len(df)*E_df2)

    

# Calculating Information Gain for every sub-group

    IG_df = E_species-WE_df

    

# Storing every Information Gain  and their corrosponding petal_length value in an empty list we priviously created

    list2.append(IG_df)

    list3.append(list1[i])
""" Creating a dataframe with 2 columns Information Gain and petal_length"""

df5 = pd.DataFrame(list(zip(list2, list1)), columns =['Information Gain', 'petal_length']) 



""" Finding the maximum value of Information Gain in the new dataframe """

df5=df5[df5['Information Gain']==max(list2)]

list3=df5['petal_length'].tolist()



""" Printing the value of petal_length corrosponding to the maximum value of Information Gain """

print("The first split will take place where the value of petal_length is = ",list3[0])