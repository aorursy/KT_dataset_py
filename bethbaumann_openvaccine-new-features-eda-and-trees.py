import os

import pickle

import re

import json

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline



import os

print(os.listdir("../input/stanford-covid-vaccine/"))

os.chdir("../input/stanford-covid-vaccine/")
with open('train.json') as f:

    data = json.loads("[" + 

        f.read().replace("}\n{", "},\n{") + "]")
structure_list=[]

for i in range(len(data)):

    structure_list.append(data[i]['structure'][0:68])

    

loop_type_list=[]

for i in range(len(data)):

    loop_type_list.append(data[i]['predicted_loop_type'][0:68])

    

sequence_list=[]

for i in range(len(data)):

    sequence_list.append(data[i]['sequence'][0:68]) 



deg50_Mg_list=[]

for i in range(len(data)):

    deg50_Mg_list.append(data[i]['deg_Mg_50C'][0:68]) 

    

reactivity_list=[]

for i in range(len(data)):

    reactivity_list.append(data[i]['reactivity'][0:68]) 



deg50_pH10_list=[]

for i in range(len(data)):

    deg50_pH10_list.append(data[i]['deg_Mg_pH10'][0:68]) 



deg50_pH10_error_list=[]

for i in range(len(data)):

    deg50_pH10_error_list.append(data[i]['deg_error_Mg_pH10'][0:68]) 



reactivity_error_list=[]

for i in range(len(data)):

    reactivity_error_list.append(data[i]['reactivity_error'][0:68]) 

    

deg50_Mg_error_list=[]

for i in range(len(data)):

    deg50_Mg_error_list.append(data[i]['deg_error_Mg_50C'][0:68]) 

    

seq_length_list=[]

for i in range(len(data)):

    seq_length_list.append([data[i]['seq_length']]*68)



# New feature - next base    

next_base_list=[]

for i in range(len(data)):

    next_base_list.append(sequence_list[i][1:]+'N')



# New feature - previous base

previous_base_list=[]

for i in range(len(data)):

    previous_base_list.append('N' + sequence_list[i][:-1])

    

# New feature - loop type at next base

next_loop_type_list=[]

for i in range(len(data)):

    next_loop_type_list.append(loop_type_list[i][1:]+'N')

    

# New feature - loop type at previous base

previous_loop_type_list=[]

for i in range(len(data)):

    previous_loop_type_list.append('N' + loop_type_list[i][:-1])



# New feature - total number of loop structures

loop_total_list=[]

for i in range(len(data)):

    loop_total_list.append([len(re.findall('\([.(]*\)', structure_list[i]))]*68)



# New feature - percent GC of sequence

total_GC_list=[]

for i in range(len(data)):

    total_GC_list.append([(sequence_list[i].count('G')+sequence_list[i].count('C'))/len(sequence_list[i])]*68)

    
def base_pairing_2(structure, sequence):

    '''

    finds the base paired to each base in a '(' or ')' structure

    creates a string the length of the sequence with the paired bases or 'N' for no pair

    '''

    base_pairing_list = ['N'] * len(sequence)

    open_counts=0

    close_counts=0

    open_positions=[]

    close_positions=[]

    open_positions_temp=[]

    close_positions_temp=[]

    for i , j in enumerate(structure):  

        if j == "(":

            open_counts += 1

            open_positions_temp.append(i)

        elif j == ")":

            close_counts += 1

            close_positions_temp.append(i)

        else:

            continue

        if open_counts == close_counts:

            open_positions.append(open_positions_temp)

            close_positions.append(close_positions_temp)

            open_positions_temp=[]

            close_positions_temp=[]

    for i in range(len(open_positions)):

        for j , k in zip(open_positions[i], close_positions[i][::-1]):

            base_pairing_list[k] = sequence[j]

            base_pairing_list[j] = sequence[k]

    return base_pairing_list

base_pairing_list = []

for i in range(len(data)):

    base_pairing_list.append(base_pairing_2(structure_list[i],sequence_list[i]))
def structure_type_length(structure):

    '''

    finds the length of each loop structure from opening '(' to closing ')'

    makes a string the length of the sequence with numbers representing the length of the current loop 

    '''

    structure=structure

    structure_length_list=[]

    prev_char=None

    length=None

    count=1

    for i in structure:

        if i != prev_char:

            if prev_char != None:

                length=count

                structure_length_list.extend([length] * length)

            count = 1

            prev_char= i

        else:

            count += 1

    if length:

        length=count

        structure_length_list.extend([length] * length)

    if not length:

        structure_length_list.extend([len(structure)]*len(structure))

    return structure_length_list

        
structure_type_length_list = []

for i in range(len(data)):

    structure_type_length_list.append(structure_type_length(structure_list[i]))
def splitintochars(a_list_of_lists):

    'splits lists of strings into lists of individual elements'

    final=[]

    for i in a_list_of_lists:

        final=final+[char for char in i]

    return final

    



def long_form(a_list_of_lists):

    'takes a list of sequences and converts it into a 1-dimensional array'

    temp=splitintochars(a_list_of_lists)

    temp=np.array(temp)

    temp = np.reshape(temp,[-1,1])

    temp = temp.flatten()

    return temp





def long_form2(a_list_of_lists):

    'takes a list of elements and converts it into a 1-dimensional array'

    long_list=[]

    for i in a_list_of_lists:

        for j in range(len(i)):

            long_list.append(i[j])

    temp=np.array(long_list)

    temp = np.reshape(temp,[-1,1])

    temp = temp.flatten()

    return temp

sequence_list=long_form(sequence_list)



structure_list=long_form(structure_list)



loop_total_list=long_form2(loop_total_list)



loop_type_list=long_form(loop_type_list)



seq_length_list=long_form2(seq_length_list)



next_base_list=long_form(next_base_list)



previous_base_list=long_form(previous_base_list)



deg50_Mg_error_list=long_form2(deg50_Mg_error_list)



deg50_pH10_list=long_form2(deg50_pH10_list)



reactivity_list=long_form2(reactivity_list)



deg50_pH10_error_list=long_form2(deg50_pH10_error_list)



reactivity_error_list=long_form2(reactivity_error_list)



deg50_Mg_list=long_form2(deg50_Mg_list)



next_loop_type_list=long_form(next_loop_type_list)



previous_loop_type_list=long_form(previous_loop_type_list)

   

total_GC_list=long_form2(total_GC_list)



base_pairing_list=long_form(base_pairing_list)



structure_type_length_list=long_form2(structure_type_length_list)       
len(sequence_list)
final_df=pd.DataFrame(data=({'base':sequence_list,'structure':structure_list,'total_loop':loop_total_list,'loop_type':loop_type_list \

                             ,'length':seq_length_list,'next_base':next_base_list,'previous_base':previous_base_list, \

                                 'deg50_Mg_error':deg50_Mg_error_list, 'deg50_Mg':deg50_Mg_list}), \

                          index=np.arange(len(sequence_list)))



final_df['position']=list(np.arange(1,69))*2400

final_df['structure_type_length']=structure_type_length_list

final_df['total_GC']=total_GC_list

final_df['base_pairing']=base_pairing_list

final_df['next_loop_type']=next_loop_type_list

final_df['previous_loop_type']=previous_loop_type_list    

final_df['reactivity']=reactivity_list

final_df['reactivity_error']=reactivity_error_list                         

final_df['deg50_pH10_error']=deg50_pH10_error_list

final_df['deg50_pH10']=deg50_pH10_list               
df=final_df

del final_df
df.columns
df.head()
df.groupby(['base','base_pairing'])['base'].count()
sns.distplot(df[df['deg50_Mg_error']<1]['deg50_Mg_error']);
df['base_context'] = df['previous_base'] + df['base'] + df['next_base']

df['loop_context'] = df['previous_loop_type'] + df['loop_type'] + df['next_loop_type']
df.drop(['base','next_base','previous_base','loop_type','next_loop_type','previous_loop_type'], axis=1, inplace=True)
df['structure']=df['structure'].astype('category')

df['base_pairing']=df['base_pairing'].astype('category')

df['base_context']=df['base_context'].astype('category')

df['loop_context']=df['loop_context'].astype('category')
plot_df=df[df['deg50_Mg_error']<1]
import seaborn as sns

mean_order = list(plot_df.groupby(['loop_context'])['deg50_Mg'].mean().sort_values().reset_index()['loop_context'])



plt.figure(figsize=(20,5))



sns.boxplot(x=plot_df['loop_context'],y=plot_df['deg50_Mg'],order=mean_order)



plt.xticks(rotation=90);
mean_order = list(plot_df.groupby(['base_context'])['deg50_Mg'].mean().sort_values().reset_index()['base_context'])



plt.figure(figsize=(20,5))



sns.boxplot(x=plot_df['base_context'],y=plot_df['deg50_Mg'],order=mean_order)



plt.xticks(rotation=90);
mean_order = list(plot_df.groupby(['base_pairing'])['deg50_Mg'].mean().sort_values().reset_index()['base_pairing'])



plt.figure(figsize=(20,5))



sns.boxplot(x=plot_df['base_pairing'],y=plot_df['deg50_Mg'],order=mean_order)



plt.xticks(rotation=90);
plt.figure(figsize=(20,5))



sns.boxplot(x=plot_df['structure'],y=plot_df['deg50_Mg']);
plt.figure(figsize=(20,5))



plt.scatter(plot_df['position'],plot_df['deg50_Mg']);
plt.figure(figsize=(20,5))



plt.scatter(x=plot_df['total_GC'],y=plot_df['deg50_Mg']);
df=pd.concat([df, pd.get_dummies(df['structure']), pd.get_dummies(df['base_pairing']), pd.get_dummies(df['loop_context']), pd.get_dummies(df['base_context'])],axis=1)

df=df.drop(['structure','base_pairing','loop_context','base_context'],axis=1)
df.head()
with open('test.json') as f:

    data = json.loads("[" + 

        f.read().replace("}\n{", "},\n{") + 

    "]")
id_list=[]

for i in range(len(data)):

    id_list.append([data[i]['id']]*len(data[i]['sequence']))



structure_list=[]

for i in range(len(data)):

    structure_list.append(data[i]['structure'][0:len(data[i]['sequence'])])

    

loop_type_list=[]

for i in range(len(data)):

    loop_type_list.append(data[i]['predicted_loop_type'][0:len(data[i]['sequence'])])

    

sequence_list=[]

for i in range(len(data)):

    sequence_list.append(data[i]['sequence'][0:len(data[i]['sequence'])])



seq_length_list=[]

for i in range(len(data)):

    seq_length_list.append([data[i]['seq_length']]*len(data[i]['sequence']))

    

next_base_list=[]

for i in range(len(data)):

    next_base_list.append(sequence_list[i][1:]+'N')

    

previous_base_list=[]

for i in range(len(data)):

    previous_base_list.append('N' + sequence_list[i][:-1])

    

loop_total_list=[]

for i in range(len(data)):

    loop_total_list.append([len(re.findall('\([.(]*\)', structure_list[i]))]*len(data[i]['sequence']))

    

position_list=[]

for i in range(len(data)):

    position_list.append([i for i in list(np.arange(1,len(data[i]['sequence'])+1))])

    

next_loop_type_list=[]

for i in range(len(data)):

    next_loop_type_list.append(loop_type_list[i][1:]+'N')

    

previous_loop_type_list=[]

for i in range(len(data)):

    previous_loop_type_list.append('N' + loop_type_list[i][:-1])

    

total_GC_list=[]

for i in range(len(data)):

    total_GC_list.append([(sequence_list[i].count('G')+sequence_list[i].count('C'))/len(sequence_list[i])]*len(data[i]['sequence']))



base_pairing_list = []

for i in range(len(data)):

    base_pairing_list.append(base_pairing_2(structure_list[i],sequence_list[i]))



structure_type_length_list = []

for i in range(len(data)):

    structure_type_length_list.append(structure_type_length(structure_list[i]))

sequence_list=long_form(sequence_list)

structure_list=long_form(structure_list)

loop_total_list=long_form2(loop_total_list)

loop_type_list=long_form(loop_type_list)

seq_length_list=long_form2(seq_length_list)

id_list=long_form(id_list)

next_base_list=long_form(next_base_list)

previous_base_list=long_form(previous_base_list)

next_loop_type_list=long_form(next_loop_type_list)

previous_loop_type_list=long_form(previous_loop_type_list)

total_GC_list=long_form(total_GC_list)

base_pairing_list=long_form(base_pairing_list)

structure_type_length_list=long_form2(structure_type_length_list)   

position_list=long_form2(position_list)
final_df=pd.DataFrame(data=({'base':sequence_list,'structure':structure_list,'total_loop':loop_total_list,'loop_type':loop_type_list \

                             ,'length':seq_length_list,'next_base':next_base_list,'previous_base':previous_base_list, 'id':id_list}), \

                          index=np.arange(len(sequence_list)))



final_df['structure_type_length']=structure_type_length_list

final_df['total_GC']=total_GC_list

final_df['base_pairing']=base_pairing_list

final_df['next_loop_type']=next_loop_type_list

final_df['previous_loop_type']=previous_loop_type_list    

final_df['position']=position_list 
test_df = final_df

del final_df
test_df.head()
test_df['base_context'] = test_df['previous_base'] + test_df['base'] + test_df['next_base']



test_df['loop_context'] = test_df['previous_loop_type'] + test_df['loop_type'] + test_df['next_loop_type']



test_df.drop(['base','next_base','previous_base','loop_type','next_loop_type','previous_loop_type'], axis=1, inplace=True)
test_df['structure']=test_df['structure'].astype('category')



test_df['base_pairing']=test_df['base_pairing'].astype('category')



test_df['base_context']=test_df['base_context'].astype('category')



test_df['loop_context']=test_df['loop_context'].astype('category')



test_df=pd.concat([test_df, pd.get_dummies(test_df['structure']), pd.get_dummies(test_df['base_pairing']), pd.get_dummies(test_df['loop_context']), pd.get_dummies(test_df['base_context'])],axis=1)



test_df=test_df.drop(['structure','base_pairing','loop_context','base_context'],axis=1)
test_id=test_df['id']



test_df=test_df.drop(['id'],axis=1)
extra_col_in_df=[i for i in list(df.columns) if i not in list(test_df.columns)]

extra_col_in_df
df = df.drop(extra_col_in_df[7:], axis=1)
test_df = test_df.drop(['length'], axis=1)



df = df.drop(['length'], axis=1)
[i for i in list(test_df.columns) if i not in list(df.columns)]



test_df = test_df.drop(['SEE', 'SSE'],axis=1)
test_df.shape



df.shape



[i for i in list(df.columns) if i not in list(test_df.columns)]
df=df.drop(['ISN'],axis=1)
from sklearn.experimental import enable_hist_gradient_boosting 

from sklearn.ensemble import HistGradientBoostingRegressor
df_deg50_Mg=df[df['deg50_Mg_error']<1]
df_deg50_Mg.columns
Y= df_deg50_Mg['deg50_Mg']
X = df_deg50_Mg.drop(['deg50_Mg', 'deg50_Mg_error', 'reactivity', 'reactivity_error', 'deg50_pH10', 'deg50_pH10_error' ],axis=1)
deg_50_Mg_tree=HistGradientBoostingRegressor()
deg_50_Mg_tree.fit(X, Y)
test_deg50_Mg = deg_50_Mg_tree.predict(test_df)
df_reactivity=df[df['reactivity_error']<1]



Y= df_reactivity['reactivity']



X = df_reactivity.drop(['deg50_Mg', 'deg50_Mg_error', 'reactivity', 'reactivity_error', 'deg50_pH10', 'deg50_pH10_error' ],axis=1)



deg_reactivity_tree=HistGradientBoostingRegressor()



deg_reactivity_tree.fit(X, Y)



test_reactivity = deg_reactivity_tree.predict(test_df)
df_deg50_pH10=df[df['deg50_pH10_error']<1]



Y= df_deg50_pH10['deg50_pH10']



X = df_deg50_pH10.drop(['deg50_Mg', 'deg50_Mg_error', 'reactivity', 'reactivity_error', 'deg50_pH10', 'deg50_pH10_error' ],axis=1)



deg_deg50_pH10_tree=HistGradientBoostingRegressor()



deg_deg50_pH10_tree.fit(X, Y)



test_deg50_pH10 = deg_deg50_pH10_tree.predict(test_df)
id_seqpos = [test_id[i] + '_' + str(test_df['position'][i]-1) for i in range(457953)]
len(id_seqpos)
id_seqpos[0]
deg_pH10 = np.random.rand(457953)
deg_50C = np.random.rand(457953)
final=pd.DataFrame({'id_seqpos':id_seqpos,'reactivity':test_reactivity, 'deg_Mg_pH10': test_deg50_pH10,'deg_pH10':deg_pH10, 'deg_Mg_50C':test_deg50_Mg, 'deg_50C':deg_50C})
final.head()
final.to_csv("beth_openvaccine_submission.csv", index=False)