!pip install fastcluster
#!pip install blackcellmagic

#%load_ext blackcellmagic
import numpy as np

import pandas as pd

from scipy.stats import entropy

from itertools import combinations, product

import seaborn as sns
dd0=pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")
def joint_entropy(dd, cols):

    return entropy(dd.groupby(cols).size())



def c(prefix, nums):

    return [prefix+str(n) for n in nums]
cols=c("bin_", range(5))+c("nom_", range(5))+c("ord_", range(5))+["day", "month"]
if False:

    N=10000

    R=5

    dd0=pd.DataFrame({"a":np.random.choice(range(R), size=N), "b":np.random.choice(range(R), size=N), "c":np.random.choice(range(R), size=N)})

    target=pd.DataFrame(list(map(list, product(dd0["a"].unique(), dd0["b"].unique()))), columns=["a", "b"])

    target["target"]=np.random.choice([0,1], size=len(target))

    dd0=dd0.merge(target, how="left")

    dd0.head()

    #dd0["target"]=np.random.choice([0,1], size=len(dd0))



    cols=["a", "b", "c"]
j_ent = {}



for col1, col2 in combinations(cols, 2):

    j_ent[frozenset([col1, col2])] = joint_entropy(dd0, [col1, col2])

    j_ent[frozenset(["target", col1, col2])] = joint_entropy(dd0, ["target", col1, col2])



for col in cols:

    j_ent[frozenset(["target", col])] = joint_entropy(dd0, ["target", col])

    j_ent[frozenset([col])] = joint_entropy(dd0, [col])

    

#avg=lambda x,y:(x+y)/2

avg=min



scores = pd.DataFrame(

    [

        (

            col1,

            col2,

            1 - (j_ent[frozenset(["target", col1, col2])] - j_ent[frozenset([col1, col2])])

            / (

                avg(

                    j_ent[frozenset(["target", col1])] + j_ent[frozenset([col2])],

                    j_ent[frozenset(["target", col2])] + j_ent[frozenset([col1])],

                )

                - j_ent[frozenset([col1, col2])]

            ),

        )

        for col1, col2 in product(cols, repeat=2)

        if col1 != col2

    ],

    columns=["col1", "col2", "score"],

)
sns.clustermap(scores.pivot("col1", "col2").fillna(scores["score"].min()), cmap="magma");