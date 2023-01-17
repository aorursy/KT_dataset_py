import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from collections import Counter

import matplotlib.patches as mpatches
table=pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")

table.head(3)
#Attrition distribution

alpha=list()

%matplotlib inline

Attrition_counter=Counter(table['Attrition'])

print (Attrition_counter)

df = pd.DataFrame.from_dict(Attrition_counter, orient='index')

print (df)

fig0=df.plot(kind='bar')

fig0.legend(['Attrition'])
Attr_arg=list()

from collections import Counter

#table.iloc[:]['Attrition']

columns = ['Age','Attrition']

table2 = pd.DataFrame(table,columns=columns)



#Attrition counter

Attr_arg = table2['Attrition']

Attr_counter=Counter(Attr_arg)



#Age counter

Age_arg = table2['Age']

Age_counter=Counter(Age_arg)



#print letter_counter

df_Age = pd.DataFrame.from_dict(Age_counter, orient='index')

#print df

fig_Age=df_Age.plot(kind='bar')

fig_Age.legend(['Age'])
name_arg=dict()

name = table.iloc[:,6]

%matplotlib inline

for key in table.columns[1:35]:

    if key == table.columns[3]:

        continue

    if key == table.columns[9]:

        continue

    if key == table.columns[18]:

        continue

    if key == table.columns[19]:

        continue

    else:

        name_arg[key] = table[key]

        name_counter = Counter(name_arg[key])

        df_name = pd.DataFrame.from_dict(name_counter, orient='index')

    #print df

        fig_name=df_name.plot(kind='bar')

        fig_name.legend([key])

    #print name_arg['\xef\xbb\xbfAge']
#Creating a dataframe with 'Yes' Attrition values

yes_arg = dict()

list1=[]

dic=dict()

a=dict()

for j in table.columns[1:35]:

    for i in range(len(table)):

        if table.iloc[i]['Attrition'] == 'Yes':

            list1.append(table.iloc[i][j])

    yes_arg[j] = list1

    list1=[]

Yes_df = pd.DataFrame.from_dict(yes_arg)#,orient='index')
Yes_df.head(5)
#First look at the dependencies of Attrition values:

#Plotting distributions using the dataset where Attrition value is 'Yes' 

#vs the whole dataset where Attrition values is 'Yes/No' 

name_arg=dict()

%matplotlib inline

for key in table.columns[1:35]:

    if key == table.columns[3]:

        continue

    if key == table.columns[9]:

        continue

    if key == table.columns[18]:

        continue

    if key == table.columns[19]:

        continue

    else:

        #DF1: Dataframe with 'Yes/No' Attrition values



        name_arg[key] = table.iloc[:][key]

        name_counter = Counter(name_arg[key])

        df_name = pd.DataFrame.from_dict(name_counter, orient='index')



        #DF2: Dataframe with 'Yes' Attrition values

        yes_arg[key] = Yes_df[:][key]

        yes_counter = Counter(yes_arg[key])

        df_yes = pd.DataFrame.from_dict(yes_counter, orient='index')



        fig_name=df_name.plot(kind='bar',rot=90, align='center')

        fig_yes = df_yes.plot(kind='bar', ax=fig_name,rot=90, color='r')

        fig_name.set_ylabel([key])

        #fig_yes.set_ylabel('Yes')

        red_patch = mpatches.Patch(color='red', label='Attrition:Yes')

        blue_patch = mpatches.Patch(color='blue', label='Attrition:Yes/No')



        plt.legend(handles=[red_patch, blue_patch])
#For each parameter take the value with the highest frequency in both 'Attriion:Yes' and 'Attrition:Yes/No' datasets

yes_max_arg = dict()

name_max_arg = dict()

import operator

for key in table.columns[2:35]:

    #print key, Yes_df[:][key]

    if key == table.columns[9:10]:

        #print key

        continue

    yes_arg[key] = Yes_df[:][key]

    yes_counter = Counter(yes_arg[key])

    for k, v in yes_counter.items():

        if v == max(yes_counter.values()):

            max_keys_values_yes = (k,v)

    yes_max_arg[key] = max_keys_values_yes

    #print yes_max_arg

    name_arg[key] = table[:][key]

    name_counter = Counter(name_arg[key])

    max_keys = []

    max_values = []

    for k1,v1 in name_counter.items():

        if v1 == max(name_counter.values()):

            max_keys_values = (k1, v1)

    name_max_arg[key] = max_keys_values#[max_keys, max_values]

#print key, yes_max_arg, name_max_arg

#df_name = pd.DataFrame.from_dict(name_max_arg, orient='index')

#df_yes = pd.DataFrame.from_dict(yes_max_arg, orient='index')

    

    

    #print type(yes_counter)
#Collect only the parameters where the highest frequency value in dataset 'Attrition:Yes' is different from the one 

#in dataset 'Attrition:Yes/No'

yes_diff_arg = dict()

name_diff_arg = dict()

for i in range(len(name_max_arg)):

    if list(yes_max_arg.values())[i][0] != list(name_max_arg.values())[i][0]:

        yes_diff_arg[list(yes_max_arg.keys())[i]] = list(yes_max_arg.values())[i]

        name_diff_arg[list(name_max_arg.keys())[i]] = list(name_max_arg.values())[i]

#print yes_diff_arg, name_diff_arg
df_name_depend = pd.DataFrame.from_dict(name_diff_arg, orient='index')

df_yes_depend = pd.DataFrame.from_dict(yes_diff_arg, orient='index')
df_yes_depend.T.set_index(list(yes_diff_arg.keys()))

df_name_depend.T.set_index(list(name_diff_arg.keys()))
#Create a network with the variables that are connected with Attrition among employees, setting the edge width as the measure of how strong this dependence is (compared to the whole dataset). 

cols = [0,1,'Name_max', 'Name_freq']

df_yes_dependencies = pd.DataFrame(df_yes_depend,columns = cols)

df_yes_dependencies['Name_max'] = df_name_depend[0]

df_yes_dependencies['Name_freq'] = df_name_depend[1]

df_yes_dependencies.rename(columns={0: 'Yes_max', 1: 'Yes_freq'}, inplace=True)

df_yes_dependencies.head()
#Calculate the edge width

df_yes_dependencies['EdgeWidth'] = df_yes_dependencies['Yes_freq']/df_yes_dependencies['Name_freq']
df_yes_dependencies
#Create the network

i=0

nodes= [0]

edges= []

edgewidths = []

labels = {}

labels[0] = (df_yes_dependencies.index[0], df_yes_dependencies.Yes_max[i])

for i in range(len(df_yes_dependencies.index)+1):

    if i==len(df_yes_dependencies.index):

        nodes.append(i)

        edge=(0,i)

        edges.append(edge)

        labels[i]=('Attrition','Yes')

    elif i<len(df_yes_dependencies.index)-1:

        i=i+1

        edge = (i,len(df_yes_dependencies.index))

        edgewidth = df_yes_dependencies.EdgeWidth[i]

        edgewidths.append(edgewidth)

        edges.append(edge)

        nodes.append(i)

        labels[i]=(df_yes_dependencies.index[i], df_yes_dependencies.Yes_max[i])
import networkx as nx
#Plot network

G=nx.cycle_graph(13)

pos=nx.spring_layout(G)

nx.draw_networkx_nodes(G,pos,

                       nodelist=nodes,

                       node_color='r',alpha=0.8,

                       node_size=500)

nx.draw_networkx_edges(G,pos,

                       edgelist=edges,

                       width=edgewidths,alpha=1,edge_color='r')

nx.draw_networkx_labels(G,pos,labels,font_size=8)