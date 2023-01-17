# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import glob

from collections import Counter

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

     #for filename in filenames:

         #print(dirname,filename)



# Any results you write to the current directory are saved as output.
#read wxcel files

files=glob.glob(dirname+'/py*')

all_data = [pd.read_excel(f, sheet_name=None, ignore_index=True,sort=False,index_col=0) for f in files ]

total_examers=0

for i in all_data:    

    examer_amount=len(list(i.keys()))

    total_examers+=examer_amount

print("amount of examers: ",total_examers)
#%%dogru yanlis ve bos sayilari

py_mind_DYB=[all_data[0][i]['ogr.C'][20:23] for i in all_data[0]]

print("py_mind DYB Amounts:",py_mind_DYB)



sortedpy_mind=sorted(py_mind_DYB,key=lambda student:student[0])       

print("The most successful student in the py_mind:\n",sortedpy_mind[-1:])





py_opinion_DYB=[all_data[1][i]['ogr.C'][20:23] for i in all_data[1]]

print("py_opinion DYB Amounts:",py_opinion_DYB) 



sortedpy_opinion=sorted(py_opinion_DYB,key=lambda student:student[0])       

print("The most successful student in the py_opinion:\n",sortedpy_opinion[-1:])



  

py_science_DYB=[all_data[2][i]['ogr.C'][20:23] for i in all_data[2]]

print("py_science DYB Amounts:",py_science_DYB) 



sortedpy_science=sorted(py_science_DYB,key=lambda student:student[0])       

print("The most successful student in the py_science_DYB:\n",sortedpy_science[-1:])





py_sense_DYB=[all_data[3][i]['ogr.C'][20:23] for i in all_data[3]]

print("py_sense DYB Amounts:",py_sense_DYB)



sortedpy_sense=sorted(py_sense_DYB,key=lambda student:student[0])       

print("The most successful student in the py_sense_DYB:\n",sortedpy_sense[-1:])

  

       

         
#%%ortalama dogru sayisi   

py_mind_D=[all_data[0][i]['ogr.C'][20] for i in all_data[0]]

sortedpy_mind=sorted(py_mind_D)

print("py_mind D Average:",sum(py_mind_D)/10)

py_opinion_D=[all_data[1][i]['ogr.C'][20] for i in all_data[1]]

sortedpy_opinion=sorted(py_opinion_D) 



print("py_opinion D Average:",sum(py_opinion_D)/10)   

py_science_D=[all_data[2][i]['ogr.C'][20] for i in all_data[2]]

sortedpy_science=sorted(py_science_D)

print("py_science D Average:",sum(py_science_D)/8) 

py_sense_D=[all_data[3][i]['ogr.C'][20] for i in all_data[3]]

sortedpy_sense=sorted(py_sense_D)

print("py_sense D Average:",sum(py_sense_D)/10)

highest_scores=[sortedpy_mind[-1], sortedpy_opinion[-1],sortedpy_science[-1], sortedpy_sense[-1]]

print('the highest scores in four classes:\n ',highest_scores)

#%%ortalama yanlis sayisi   

py_mind_Y=[all_data[0][i]['ogr.C'][21] for i in all_data[0]]

print("py_mind Y Average:",sum(py_mind_Y)/10)

py_opinion_Y=[all_data[1][i]['ogr.C'][21] for i in all_data[1]]

print("py_opinion Y Average:",sum(py_opinion_Y)/10)   

py_science_Y=[all_data[2][i]['ogr.C'][21] for i in all_data[2]]

print("py_science Y Average:",sum(py_science_Y)/8) 

py_sense_Y=[all_data[3][i]['ogr.C'][21] for i in all_data[3]]

print("py_sense Y Average:",sum(py_sense_Y)/10)
#%%ortalama yanlis sayisi   

py_mind_B=[all_data[0][i]['ogr.C'][22] for i in all_data[0]]

print("py_mind B Average:",sum(py_mind_B)/10)

py_opinion_B=[all_data[1][i]['ogr.C'][22] for i in all_data[1]]

print("py_opinion B Average:",sum(py_opinion_B)/10)   

py_science_B=[all_data[2][i]['ogr.C'][22] for i in all_data[2]]

print("py_science B Average:",sum(py_science_B)/8) 

py_sense_B=[all_data[3][i]['ogr.C'][22] for i in all_data[3]]

print("py_sense B Average:",sum(py_sense_B)/10)
#%%en basarili 3 ogrenci

allStudents=[]

allStudents.extend(py_mind_DYB)

allStudents.extend(py_opinion_DYB)

allStudents.extend(py_science_DYB)

allStudents.extend(py_sense_DYB)

sortedallStudents=sorted(allStudents,key=lambda student:student[0], reverse=True) 

print("The most successful 3 students in the quiz2:\n",sortedallStudents[:3])
#%%en cok yapilan yanlislar

checklist=['D','A','D','B','E','B','A','B','D','D','C','A','B','B','A','C','D','B','A','D']

list_choices1=[]

for i in all_data[0].values():

    a=i['ogr.C'][:20]

    list_choices1.append(a)

list_choices2=[]

for i in all_data[1].values():

    a=i['ogr.C'][:20]

    list_choices2.append(a)

list_choices3=[]

for i in all_data[2].values():

    a=i['ogr.C'][:20]

    list_choices3.append(a)

list_choices4=[]

for i in all_data[3].values():

    a=i['ogr.C'][:20]

    list_choices4.append(a)



k1=[]

for i in list_choices1:

     for j in range(len(i)):

         if i[j]!=checklist[j]:

             k1.append(j)      



print('The most common wrong ques. in py_mind(index&amount)',Counter(k1).most_common(1))

k2=[]

for i in list_choices2:

     for j in range(len(i)):

         if i[j]!=checklist[j]:

             k2.append(j)            

     

print('The most common wrong ques. in py_opinion(index&amount)',Counter(k2).most_common(1))       

k3=[]

for i in list_choices3:

     for j in range(len(i)):

         if i[j]!=checklist[j]:

             k3.append(j)                 

print('The most common wrong ques. in py_science(index&amount)',Counter(k3).most_common(1))

k4=[]

for i in list_choices4:

     for j in range(len(i)):

         if i[j]!=checklist[j]:

             k4.append(j)                 

print('The most common wrong ques. in py_sense(index&amount)',Counter(k4).most_common(1)) 

k_all=[]

k_all.extend(k1)

k_all.extend(k2)

k_all.extend(k3)

k_all.extend(k4)



for i in list_choices4:

     for j in range(len(i)):

         if i[j]!=checklist[j]:

             k_all.append(j)                 

print('The most common wrong ques. in classes(index&amount)',Counter(k_all).most_common(1)) 
#%%py_mind basari

py_mind_list=list(all_data[0].keys())

x=np.array(py_mind_list)

y=np.array(py_mind_D)

plt.bar(x,y, color='#86bf91')

plt.title('bar plot py_mind', weight='bold')

plt.xlabel('Students',labelpad=10, size=12)

plt.ylabel('D amounts',size=12)

plt.show()
#%%py_opinion basari

py_opinion_list=list(all_data[1].keys())

x=np.array(py_opinion_list)

y=np.array(py_opinion_D)

plt.bar(x,y, color='#86bf91')

plt.title('bar plot py_opinion', weight='bold')

plt.xlabel('Students',labelpad=10, weight='bold', size=12)

plt.ylabel('D amounts',size=12)

plt.show()
#%%py_science basari

py_science_list=list(all_data[2].keys())

x=np.array(py_science_list)

y=np.array(py_science_D)

plt.bar(x,y, color='#86bf91')

plt.title('bar plot py_science', weight='bold')

plt.xlabel('Students',labelpad=10, size=12)

plt.ylabel('D amounts',size=12)

plt.show()
#%%py_opinion basari

py_sense_list=list(all_data[1].keys())

x=np.array(py_sense_list)

y=np.array(py_sense_D)

plt.bar(x,y, color='#86bf91')

plt.title('bar plot py_sense', weight='bold')

plt.xlabel('Students',labelpad=10, size=12)

plt.ylabel('D amounts',size=12)

plt.show()
#%%siniflarin basari durumu

py_all_list=[sum(py_mind_D)/10,sum(py_opinion_D)/10, sum(py_science_D)/8,sum(py_sense_D)/10]

classes_list=['py_mind','py_opinion','py_science','py_sense']

x=np.array(classes_list)

y=np.array(py_all_list)

plt.bar(x,y)

plt.title('bar plot all classes', weight='bold')

plt.xlabel('Students',labelpad=10, size=12)

plt.ylabel('D average',size=12)

plt.show()
#%%Sorulara gore dogru sayilarinin dagilimi

dicAllYs=dict(Counter(k_all))

dicValues=list(dicAllYs.values())

dicKeys=list(dicAllYs.keys())

dicKeysNew=list(map(lambda x: x+1,dicKeys))

dicValuesNew=list(map(lambda x: 38-x,dicValues))

x=np.array(dicKeysNew)

y=np.array(dicValuesNew)

plt.bar(x,y)

plt.title('bar plot D amounts', weight='bold')

plt.xlabel('Questions',labelpad=10, size=12)

plt.ylabel('D amount',size=12)

plt.show()
# %%tum siniflar plot

plt.subplot(4,1,1)

plt.plot(py_mind_D,color='red',label='mind')

plt.ylabel("mind-D")

plt.subplot(4,1,2)

plt.plot(py_opinion_D,color="green",label="opinion")

plt.ylabel("opinion-D")

plt.subplot(4,1,3)

plt.plot(py_science_D,color="blue",label="science")

plt.ylabel("science-D")

plt.subplot(4,1,4)

plt.plot(py_sense_D,color="blue",label="sense")

plt.ylabel("sense-D")

plt.xlabel("Student Amount")

plt.text(2.5,42,'All Classes D Graph',weight='bold',size=12)

plt.show()
# %% histogram



plt.hist(dicValuesNew,bins= 50)

plt.xlabel("soru bazinda dogru cevaplama sayisi")

plt.ylabel("frekans")

plt.title("hist")

plt.show()
#%% scatter plot



plt.scatter(py_all_list,highest_scores, color="red")

plt.xlabel("Averages")

plt.ylabel("the highest scores")

plt.title("The relationship between class averages and the most successful students of each class")

plt.show()


