

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import networkx as nx

import nltk, string

import matplotlib.pyplot as plt

%matplotlib inline



import os

print(os.listdir("../input"))



def headclean (head, notes, note,state): 

     

    if "SELECTION" in head:

        head = 'SELECTION PROCESS'

    if "REQUIREMENT" in head:

        head = 'REQUIREMENTS/ MIN QUALS'  

    if "REQUIREMENT" in head:

        head = 'REQUIREMENTS/ MIN QUALS'    

    if (head == 'NOTES:'):

        head = str(notes) + head

        notes += 1

    if (head == 'NOTE:'):

        head = str(note) + head

        note += 1

    if 'EQUAL EMPLOYMENT' in head:

        state = 0

    if 'EQUAL OPPORTUNITY' in head:

        state = 0

    if 'WHERE TO APPLY' in head:

        state = 0

    if 'REVISED' in head:

        state = 0

    if 'HTML' in head:

        state = 0

    if 'C+' in head:

        state = 0

    if 'AJAX' in head:

        state = 0

    if 'COBRA' in head:

        state = 0

    if 'INTERDEPARTMENTAL PROMOTIONAL BASIS' in head:

        state = 0

    if 'OPEN COMPETITIVE BASIS' in head:

        state = 0

    if 'DEPARTMENTAL PROMOTIONAL BASIS' in head:

        state = 0

    if 'INTERDEPARTMENTAL PROMOTIONAL' in head:

        state = 0

    if 'INTERDEPARMENTAL PROMOTIONAL' in head:

        state = 0

    if 'APPOINTMENT' in head:

        state = 0



    if 'ANNUALSALARY' in head:

        head = 'ANNUAL SALARY'

        state = 0

        

    return (head, notes, note,state)
bulletin_dir = "../input/cityofla/CityofLA/Job Bulletins"



df_bull = pd.DataFrame()

for filename in os.listdir(bulletin_dir):

    with open(bulletin_dir + "/" + filename, 'r', errors='ignore') as f:

        firstline = True

        data_list = []

        state = 0

        notes = 1

        note = 1

        body = ''

        for line in f.readlines():

            line = line.rstrip().lstrip()

            if line == 'BUREAU OF SANITATION':

                line = ''

            if (line != ''):

                if "DEPARTMENT OF PUBLIC WORKS" in line:

                    line = 'VOCATIONAL WORKER'   

                if "A LICENSE AS A CERTIFIED WELDER FOR STRUCTURAL" in line:

                    line =line.lower()

                if "ANNUAL SALARY" in line:

                    state = 0

                if "REVISED" in line:

                     state = 0

                if (firstline):

                    data_list.append(['TITLE', line])

                    firstline = False

                elif "Open Date:" in line:

                    job_bulletin_date = line.split("Open Date:")[1].split("(")[0].strip()

                    data_list.append(['OPEN DATE', job_bulletin_date])

                elif "Class Code:" in line:

                    classcode = line.split("Class Code:")[1].split("(")[0].strip()

                elif (line.isupper()):

                    if (state == 2):

                        #print (head,line, state)

                        data_list.append([head, body])

                        body = ''

                    state = 1

                    head, notes, note, state = headclean (line, notes, note, state)

                else:

                    body = body +line

                    if (state == 1):

                        state = 2

        data_list.append(['CLASS CODE', classcode])                

        df = pd.DataFrame(data_list)

        dfT = df.T

        dfT.columns = dfT.iloc[0]

        dfT.drop(dfT.index[0], inplace = True)

        df_bull = pd.concat([df_bull,dfT], axis=0, sort = False)

        
df_bull.shape
df_bull = df_bull.reset_index()

df_bull = df_bull.drop (['index'], axis=1)



#pd.options.display.max_colwidth = 100

#with pd.option_context("display.max_rows", 2000): display (df_bull)



df_bull.head()
df_bull_len = len(df_bull)

index = 0

df_bull['TITLE_CAT'] = df_bull['TITLE'].copy()

while index < df_bull_len:

    job = df_bull.iloc[index]['TITLE']

    job = job.replace('SENIOR ', 'SENIOR')

    df_bull.iloc[index]['TITLE_CAT'] = job.replace('WATER UTILITY OPERATOR SUPERVISOR', 

                                                   'WATER UTILITY OPERATORSUPERVISOR')

    #print (job, df_bull.iloc[index]['TITLE_CAT'])

    reqs = df_bull.iloc[index]['REQUIREMENTS/ MIN QUALS']

    reqs = reqs.replace('Senior ', 'Senior')

    df_bull.iloc[index]['REQUIREMENTS/ MIN QUALS'] = reqs.replace('Water Utility Operator Supervisor', 

                                                                  'Water Utility OperatorSupervisor')

    #print (reqs,df_bull.iloc[index]['REQUIREMENTS/ MIN QUALS'] )

    index += 1
def contains_word(s, w):

    return (' ' + w + ' ') in (' ' + s + ' ')



df_explicit = pd.DataFrame()

data_list = []

remove_punctuation_map = dict((ord(char), ' ') for char in string.punctuation)    



df_bull_len = len(df_bull)

index = 0

while index < df_bull_len:

    job = df_bull.iloc[index]['TITLE_CAT']

    reqs = df_bull.iloc[index]['REQUIREMENTS/ MIN QUALS']

    reqs =  reqs.upper().translate(remove_punctuation_map)

    

    for index2, row2 in df_bull.iterrows():

        experience = df_bull.iloc[index2]['TITLE_CAT']

        if (contains_word(reqs,experience) and (experience != job)):

            data_list.append([df_bull.iloc[index]['TITLE'], df_bull.iloc[index2]['TITLE']])

            #print ('Job: ',df_bull.iloc[index]['TITLE_CAT'], 'Exp needed', experience,'FOUND:',reqs)

    index += 1

df_explicit = pd.DataFrame(data_list)

df_explicit.columns = ["JOB", "EXPERIENCE"]

df_explicit.head()

           

    
df_explicit_g = df_explicit.groupby('JOB').count()



df_explicit_s_g = df_explicit_g.sort_values('EXPERIENCE',ascending = False)

df_explicit_s_g.head()


G = nx.Graph()



job = 'WATER UTILITY SUPERINTENDENT'

#G.add_node(job)

df_explicit_len = len(df_explicit)

index = 0

while index < df_explicit_len:

    if df_explicit.iloc[index]['JOB'] == job:

        print (job, ":   ",df_explicit.iloc[index]['EXPERIENCE'] )

        G.add_edge(job,df_explicit.iloc[index]['EXPERIENCE'])

    index +=1

plt.figure(figsize=(15, 15)) 

plt.axis('off')

nx.draw_networkx(G, with_labels=True, node_color='red', font_size=12, node_size=20000, width = 2)

plt.show()

def findsubsrecurse (job):

    G.add_node(job)

    df_explicit_len = len(df_explicit)

    index = 0

    while index < df_explicit_len:

        if df_explicit.iloc[index]['JOB'] == job:

            G.add_edge(df_explicit.iloc[index]['EXPERIENCE'],job)

            findsubsrecurse (df_explicit.iloc[index]['EXPERIENCE'])

        index +=1

    return

def findsubs (job):

   

    findsubsrecurse (job)

    plt.figure(figsize=(15, 15)) 

    plt.axis('off')

    pos = nx.circular_layout(G)

    nx.draw_networkx(G, pos,with_labels=True, node_color='red', 

                     font_size=12, node_size=2000, arrows = True, width = 2)

    plt.show()

    return
G = nx.DiGraph()

findsubs ('WATER UTILITY SUPERINTENDENT')


G = nx.DiGraph()

findsubs ('CHIEF INSPECTOR')


G = nx.DiGraph()

findsubs ('ELECTRICAL SERVICES MANAGER')

G = nx.DiGraph()

findsubs ('SENIOR LOAD DISPATCHER')
G = nx.DiGraph()

findsubs ('SENIOR SYSTEMS ANALYST')
G = nx.DiGraph()

findsubs ('UTILITY SERVICES MANAGER')

def findsuprecurse (experience):

    G.add_node(experience)

    df_explicit_len = len(df_explicit)

    index = 0

    while index < df_explicit_len:

        if df_explicit.iloc[index]['EXPERIENCE'] == experience:

            G.add_edge(experience,df_explicit.iloc[index]['JOB'])

            findsuprecurse (df_explicit.iloc[index]['JOB'])

        index +=1

    return

def findsup (job):

   

    findsuprecurse (job)

    plt.figure(figsize=(15, 15)) 

    plt.axis('off')

    pos = nx.circular_layout(G)

    nx.draw_networkx(G, pos,with_labels=True, node_color='red', 

                     font_size=12, node_size=2000, arrows = True, width = 2)

    plt.show()

    return
G = nx.DiGraph()

findsup ('SYSTEMS ANALYST')
G = nx.DiGraph()

findsup('SECRETARY')
G = nx.DiGraph()

findsup('PUBLIC RELATIONS SPECIALIST')
G = nx.DiGraph()

findsup('WELDER')
G = nx.DiGraph()

findsup('ELECTRICAL CRAFT HELPER')
#df.to_csv("competition_output.csv")#
#df["OPEN_DATE"] = df["OPEN_DATE"].astype('datetime64[ns]')
#df.describe()