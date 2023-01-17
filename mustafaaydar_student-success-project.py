#This is an assignment at Pycoders Coding School. 
#In the py-exams directory, there are 4 excel files-the exams of 4 classes(py_mind,py_opinion,py_science,py_sense)
#Each file has sheets named with student names.
#There are answers of the students in these sheets.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

files=[]
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        files.append(os.path.join(dirname, filename))#full path of the files
files

all_data = [pd.read_excel(file, sheet_name=None, index_col=0) for file in files ]
all_data   
#Finding total examiner amount
total_examiners=0
for i in all_data:
    examiner_amount=len(i.keys()) #Keys are sheetnames
    total_examiners+=examiner_amount
    #print("Examiner amount for each class:",examiner_amount)
    
print("Amount of Total Examiner: ",total_examiners)

#True/False/Emty amounts for each classes

py_science_TFE=[all_data[3][i]['ogr.C'][20:23] for i in all_data[3]]# True&False&Empty choices at 20-23 rows
print("Py_science TFE Amounts:\n",py_science_TFE, "\n")

py_opinion_TFE=[all_data[2][i]['ogr.C'][20:23] for i in all_data[2]]
print("Py_opinion TFE Amounts:\n\n",py_opinion_TFE, "\n")

py_mind_TFE=[all_data[1][i]['ogr.C'][20:23] for i in all_data[1]]
print("Py_mind TFE Amounts:\n",py_mind_TFE, "\n")

py_sense_TFE=[all_data[0][i]['ogr.C'][20:23] for i in all_data[0]]
print("Py_sense TFE Amounts:\n",py_sense_TFE, "\n") 

#The most succesful student in each classes with T/F/E amounts
sortedpy_opinion=sorted(py_opinion_TFE,key=lambda choice:choice[0]) #Created for sorting according to True/ascending order
print("The most successful student in Py_opinion with T/F/E amounts:\n",sortedpy_opinion[-1], "\n")

sortedpy_science=sorted(py_science_TFE,key=lambda choice:choice[0])       
print("The most successful student in Py_science with T/F/E amounts:\n",sortedpy_science[-1], "\n")

sortedpy_sense=sorted(py_sense_TFE,key=lambda choice:choice[0])       
print("The most successful student in Py_sense with T/F/E amounts:\n",sortedpy_sense[-1], "\n")

sortedpy_mind=sorted(py_mind_TFE,key=lambda choice:choice[0])       
print("The most successful student in Py_mind with T/F/E amounts:\n",sortedpy_mind[-1])
#Average of True answers for each classes&all course   
py_science_D=[all_data[3][i]['ogr.C'][20] for i in all_data[3]]#20st row is for True answers
print("Py_science True Answer Average:",np.array(py_science_D).mean()) 
py_opinion_D=[all_data[2][i]['ogr.C'][20] for i in all_data[2]] 
print("Py_opinion True Answer Average:",np.array(py_opinion_D).mean())
py_mind_D=[all_data[1][i]['ogr.C'][20] for i in all_data[1]]
print("Py_mind True Answer Average:",np.array(py_mind_D).mean())
py_sense_D=[all_data[0][i]['ogr.C'][20] for i in all_data[0]]
print("Py_sense True Answer Average:",np.array(py_sense_D).mean())
print("All Classes True Answer Average:",np.array(py_opinion_D+py_science_D+py_mind_D+py_sense_D).mean())
#The highest scores for each classes
sortedpy_opinion=sorted(py_opinion_D) 
sortedpy_science=sorted(py_science_D)
sortedpy_mind=sorted(py_mind_D)
sortedpy_sense=sorted(py_sense_D)

highest_scores=[sortedpy_mind[-1], sortedpy_opinion[-1],sortedpy_science[-1], sortedpy_sense[-1]]
print('The highest scores in all classes (PY/mind-opinion-science-sense):',highest_scores)
#Average of False answers for each class  
py_science_Y=[all_data[3][i]['ogr.C'][21] for i in all_data[3]]#21st row is for False answers
print("Py_science False Answer Average:",np.array(py_science_Y).mean()) 
py_opinion_Y=[all_data[2][i]['ogr.C'][21] for i in all_data[2]]
print("Py_opinion False Answer Average:",np.array(py_opinion_Y).mean()) 
py_mind_Y=[all_data[1][i]['ogr.C'][21] for i in all_data[1]] 
print("Py_mind False Answer Average:",np.array(py_mind_Y).mean())
py_sense_Y=[all_data[0][i]['ogr.C'][21] for i in all_data[0]]
print("Py_sense False Answer Average:",np.array(py_sense_Y).mean())
#Average of Empty answers for each class   
py_science_B=[all_data[3][i]['ogr.C'][22] for i in all_data[3]] #22nd row is for False answers
print("Py_science Empty Answer Average:",np.array(py_science_B).mean()) 
py_opinion_B=[all_data[2][i]['ogr.C'][22] for i in all_data[2]] 
print("Py_opinion Empty Answer Average:",np.array(py_opinion_B).mean()) 
py_mind_B=[all_data[1][i]['ogr.C'][22] for i in all_data[1]] 
print("Py_mind Empty Answer Average:",np.array(py_mind_B).mean())
py_sense_B=[all_data[0][i]['ogr.C'][22] for i in all_data[0]]
print("Py_sense Empty Answer Average:",np.array(py_sense_B).mean())
#The most successful 3 students in the course with T/F/E amounts
allStudents=[]
allStudents.extend(py_mind_TFE+py_opinion_TFE+py_science_TFE+py_sense_TFE)
sortedallStudents=sorted(allStudents,key=lambda choice:choice[0], reverse=True) 
print("The most successful 3 students in the course with T/F/E amounts:\n")
for i in sortedallStudents[:3]:
    print(i,"\n")
#The most common False questions
answer_key=['D','A','D','B','E','B','A','B','D','D','C','A','B','B','A','C','D','B','A','D'] #Answer key

choices_science=[all_data[3][i]['ogr.C'][:20] for i in all_data[3]]  #Choices - until 20st row 

choices_opinion=[all_data[2][i]['ogr.C'][:20] for i in all_data[2]]  
choices_mind=[all_data[1][i]['ogr.C'][:20] for i in all_data[1]]
choices_sense=[all_data[0][i]['ogr.C'][:20] for i in all_data[0]]

mind_falses=[]
for i in choices_mind:        #for each student
     for j in range(len(i)):  #for each choice
         if i[j]!=answer_key[j]:
             mind_falses.append(j) 
print('The most common false question in Py_mind(#Question&False Amount)',Counter(mind_falses).most_common(1))

opinion_falses=[]
for i in choices_opinion:
     for j in range(len(i)):
         if i[j]!=answer_key[j]:
             opinion_falses.append(j)            
print('The most common false question in Py_opinion(#Question&False Amount)',Counter(opinion_falses).most_common(1))     

science_falses=[]
for i in choices_science:
     for j in range(len(i)):
         if i[j]!=answer_key[j]:
             science_falses.append(j)  
                
print('The most common false question in Py_science(#Question&False Amount)',Counter(science_falses).most_common(1))

sense_falses=[]

for i in choices_sense:
     for j in range(len(i)):
         if i[j]!=answer_key[j]:
             sense_falses.append(j)                 
print('The most common false question in Py_sense(#Question&False Amount)',Counter(sense_falses).most_common(1)) 

k_all=mind_falses+opinion_falses+science_falses+sense_falses                
print('The most common false question in all classes(#Question&False Amount)',Counter(k_all).most_common(1)) 
#All Classes Success
py_all=[np.array(py_mind_D).mean(),np.array(py_opinion_D).mean(), np.array(py_science_D).mean(),np.array(py_sense_D).mean()]
class_names=['py_mind','py_opinion','py_science','py_sense']
plt.bar(class_names,py_all,color='yellow')
plt.title('True Average by Class', weight='bold',size=15)
plt.xlabel('Class')
plt.ylabel('True Average')
plt.show()
#Py_science Success
py_science_keys=all_data[3].keys()
plt.bar(py_science_keys,py_science_D, color='pink')
plt.title('Py_science True Amount by Student', weight='bold',size=15)
plt.xlabel('Student')
plt.ylabel('True Amount')
plt.show()
#Py_opinion Success
py_opinion_keys=all_data[2].keys()
plt.bar(py_opinion_keys,py_opinion_D, color='red')
plt.title('Py_opinion True Amount by Student', weight='bold',size=15)
plt.xlabel('Student')
plt.ylabel('True Amount')
plt.show()
#Py_mind Success
py_mind_keys=all_data[1].keys()
plt.bar(py_mind_keys,py_mind_D, color='#86bf91')
plt.title('Py_mind True Amount by Student', weight='bold',size=15)
plt.xlabel('Student')
plt.ylabel('True Amount')
plt.show()
#Py_sense Success
py_sense_keys=all_data[0].keys()
plt.bar(py_sense_keys,py_sense_D, color='blue')
plt.title('Py_sense True Amount by Student', weight='bold',size=15)
plt.xlabel('Student')
plt.ylabel('True Amount')
plt.show()
# Total True amounts of questions

false_answers=Counter(k_all)
false_answers_keys=false_answers.keys()
false_answers_values=false_answers.values()
new_keys=list(map(lambda x: x+1,false_answers_keys))
new_values=list(map(lambda x: 38-x,false_answers_values)) # For finding True's, subtracted from total student amount
plt.bar(new_keys,new_values,color='purple')
plt.title('Total True Amount by Question', weight='bold',size=15)
plt.xlabel('Question')
plt.ylabel('True Amount')
plt.show()
# All Classes Trues Plot

fig, ax = plt.subplots(2, 2)

ax[0, 0].plot(py_mind_D, 'r') #row=0, col=0
ax[0, 0].set_ylabel("Mind&Sense Trues")
ax[1, 0].plot(py_opinion_D, 'b') #row=1, col=0
ax[1, 0].set_ylabel("Opinion&Science Trues")
ax[0, 1].plot(py_sense_D, 'g') #row=0, col=1
ax[1, 1].plot(py_science_D, 'k') #row=1, col=1
plt.suptitle('All Classes Trues by Student',weight='bold',size=15)
ax[1, 0].set_xlabel('Student')
ax[1, 1].set_xlabel('Student')
plt.show()
# All classes True/False/Empty plot individually
plt.subplot(3,1,1)
plt.plot(py_mind_D,color='red',label='mind')
plt.ylabel("mind-T")
plt.subplot(3,1,2)
plt.plot(py_mind_Y,color='green',label='mind')
plt.ylabel("mind-F")
plt.subplot(3,1,3)
plt.plot(py_mind_B,color='blue',label='mind')
plt.ylabel("mind-E")
plt.xlabel("Student Amount")
plt.suptitle('Mind T/F/E by Student',weight='bold',size=15)
plt.show()

plt.subplot(3,1,1)
plt.plot(py_opinion_D,color='red',label='opinion')
plt.ylabel("opinion-T")
plt.subplot(3,1,2)
plt.plot(py_opinion_Y,color='green',label='opinion')
plt.ylabel("opinion-F")
plt.subplot(3,1,3)
plt.plot(py_opinion_B,color='blue',label='opinion')
plt.ylabel("opinion-E")
plt.xlabel("Student Amount")
plt.suptitle('Opinion T/F/E by Student',weight='bold',size=15)
plt.show()

plt.subplot(3,1,1)
plt.plot(py_science_D,color='red',label='science')
plt.ylabel("science-T")
plt.subplot(3,1,2)
plt.plot(py_science_Y,color='green',label='science')
plt.ylabel("science-F")
plt.subplot(3,1,3)
plt.plot(py_science_B,color='blue',label='science')
plt.ylabel("science-E")
plt.xlabel("Student Amount")
plt.suptitle('Science T/F/E by Student',weight='bold',size=15)
plt.show()

plt.subplot(3,1,1)
plt.plot(py_sense_D,color='red',label='sense')
plt.ylabel("sense-D")
plt.subplot(3,1,2)
plt.plot(py_sense_Y,color='green',label='sense')
plt.ylabel("sense-Y")
plt.subplot(3,1,3)
plt.plot(py_sense_B,color='blue',label='sense')
plt.ylabel("sense-B")
plt.xlabel("Student Amount")
plt.suptitle('Sense T/F/E by Student',weight='bold',size=15)
plt.show()
#Scatter plot with Trues of students
df1=pd.DataFrame(py_mind_D)
df2=pd.DataFrame(py_opinion_D)
df3=pd.DataFrame(py_sense_D)
df4=pd.DataFrame(py_science_D)
plt.scatter(df1.index ,df1, color="red", label="mind")
plt.scatter(df2.index,df2, color="blue", label="opinion")
plt.scatter(df3.index,df3, color="yellow", label="sense")
plt.scatter(df4.index, df4, color="green", label="science")
plt.legend()
plt.xlabel("Students")
plt.ylabel("True")
plt.suptitle('Scatterplot for Trues',weight='bold',size=15)
plt.show()