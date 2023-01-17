import numpy as np 

import pandas as pd

import networkx as nx

import itertools



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
survey = pd.read_csv('../input/survey_results_public.csv')

schema = pd.read_csv('../input/survey_results_schema.csv')

languages = survey['HaveWorkedLanguage']



G = nx.Graph()

#G.add_edges_from([(1,2),(1,3)])

edges = []

nodes = []

for row in languages:

    langs = row.split(';')

    if langs == 'NaN':

        pass

    if len(langs) == 1:

        nodes.append(langs)

    if len(langs) > 1:

        itNum = len(langs) - 1

        for L in range(0,len(langs)+1):

            for subset in itertools.combinations(langs,L):

                print(subset)

'''            

for L in range(0, len(stuff)+1):

  for subset in itertools.combinations(stuff, L):

    print(subset)

'''            

    
