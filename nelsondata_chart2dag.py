from IPython.display import Image

Image("../input/careerpathdata/PR.png")
'''

Script:  JBR_DAG.py

Purpose: Generate a directed acyclic graph for Career Paths and compute the next promotion for job class titles

Example: Public Relations Career Path

NOTICE:  Run this once after you run LAJobs2CSV.py because each time you run this, it will merge the DAG dataframe with the JOB BULLETIN CSV resulting in multiple indexes in the csv file

'''



import pandas as pd

import re

from graphviz import Digraph

from graphviz import Graph



def generate_edges(graph):

    ''' edges connect nodes '''

    edges = []

    for node in graph:

        for neighbour in graph[node]:

            edges.append((node, neighbour))

    return edges



def find_isolated_nodes(graph):

    """ returns a list of nodes that don't have any edges """

    isolated = []

    for node in graph:

        if not graph[node]:

            isolated += node

    return isolated



def create_graph(graph,edges,nodes,graphName):

    a = []

    dot = Digraph(format='png')

    for node in nodes:

        dot.node(node[0], node[1])

    for edge in edges:

        dot.edge(edge[0][0], edge[1][0])

    dot.render(graphName,view=True)



def no_letters(k,v):

    abcs = re.search('[a-zA-Z]',v)

    if not abcs:

        return int(k)

    else:

        return 0



chartNames = ['Accounting','Admin','Animal','Bind','PR']

chartNames = ['PR']

csvList = []



for chartName in chartNames:

        

    file = '../input/careerpathdata/'+chartName+'.txt'



    k = 1      # row counter

    row = {}   # dictionary with line and line number for each line in the OCR text

    csvRow = {}

    with open(file) as f:

        for line in f:

            line = line.strip("\n")

            row[str(k)] = line

            k += 1

            

    otherRows = [(no_letters(k,v)) for k,v in row.items()]

    textRows = [(str(i+1),row[str(i+1)]) for i,x in enumerate(otherRows) if x == 0]



    # Sometimes career path charts format node names across two lines.  In order to get the node name, these lines must be combined.  Fortunately, there are blank lines between node names

    prior = 1

    priorText = ''

    titles = {}

    for t in textRows:

        idx = t[0]

        priorIDX = str(int(t[0])-1)

        if (int(t[0]) - prior) == 1:

            titles[priorIDX] = priorText + " " + t[1]

        else:

            titles[idx] = t[1]

        prior = int(t[0])

        priorText = t[1]



    nodes = [(str(i+1), v) for i, v in enumerate(titles.values())]



    # create the DAG

    graph = {}

    priorNode = 1

    for node in nodes:

        if node[0] != priorNode:

            graph[str(priorNode)] = [str(node[0])]

        priorNode = node[0]



    edges = (generate_edges(graph))

    graphName = 'JBR_Output/dag'+ chartName +'.gv'

    #create_graph(graph,edges,nodes,graphName)



    print("\n")

    print("GRAPH", graph)

    print("\nNODES ", nodes)

    print("\nEDGES ", edges)

    print("\nISOLATED NODES: ", find_isolated_nodes(graph))



    # updating JOB BULLETIN.CSV with DAG in string format and png format.  adding the next promotion based on Job Class Title

    g = ",".join(("'{}':{}".format(*g) for g in graph.items()))

    g = "{" + g + "}"



    n = ",".join(("('{}':'{}')".format(*n) for n in nodes))

    n = "[" + n + "]"



    for node in nodes:

        nodeNum = int(node[0])

        csvRow['JOB_CLASS_TITLE'] = node[1].upper()

        csvRow["DAG_GRAPH"] = g

        csvRow["DAG_NODES"] = n

        csvRow["DAG_FILE"] = graphName

        if nodeNum != 1:

            promotion = [node[1] for node in nodes if int(node[0]) == nodeNum-1]

            csvRow["NEXT_PROMOTION"] = promotion[0].upper()

        csvList.append(csvRow)

        csvRow = {}



# DAG dataframe

df = pd.DataFrame(csvList)

df.index.name = 'IDX'

pd.options.display.max_columns=len(df)

df['NEXT_PROMOTION']=df.NEXT_PROMOTION.fillna('NONE')



# update Job Bulletin CSV file to add DAG and NEXT_PROMOTION

jb = pd.read_csv("../input/careerpathdata/JBR_Output/JobBulletin.csv")

df1 = pd.merge(jb, df, how='left', on='JOB_CLASS_TITLE')

df1.to_csv("JobBulletin.csv")

idx = df1.JOB_CLASS_TITLE[df1.JOB_CLASS_TITLE == 'PUBLIC RELATIONS SPECIALIST'].index.tolist()

print("\nIf your job class title is ", df1['JOB_CLASS_TITLE'].loc[idx[0]])

print("your next promotion is to ", df1["NEXT_PROMOTION"].loc[idx[0]])



idx = df1.JOB_CLASS_TITLE[df1.JOB_CLASS_TITLE == 'PRINCIPAL PUBLIC RELATIONS REPRESENTATIVE'].index.tolist()

print("\nIf your job class title is ", df1['JOB_CLASS_TITLE'].loc[idx[0]])

print("your next promotion is to ", df1["NEXT_PROMOTION"].loc[idx[0]])



idx = df1.JOB_CLASS_TITLE[df1.JOB_CLASS_TITLE == 'PUBLIC INFORMATION DIRECTOR'].index.tolist()

print("\nIf your job class title is ", df1['JOB_CLASS_TITLE'].loc[idx[0]])

print("your next promotion is to ", df1["NEXT_PROMOTION"].loc[idx[0]])

from IPython.display import Image

Image("../input/careerpathdata/dag.gv.png")