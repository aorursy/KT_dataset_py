import pandas as pd

import numpy as np



people_tmp= pd.read_csv( '../input/people.csv')

act_tmp= pd.read_csv('../input/act_train.csv')

tmp= people_tmp.columns.values

a=[tmp[0], tmp[4], tmp[2], tmp[1], tmp[3]]+ tmp[5:].tolist()

people= people_tmp[a]

#people.info()

p_col= ['people_id', 'p_date', 'p_group_1']

for i in range( 1, 39, 1):

    p_col.append( 'p_ch_'+str(i))

people.columns= p_col

# people.info()

for c in people.columns.values[2: ]:

    if type( people[c][0])== np.bool_ : # bool

        t1= people[c].values.flatten().tolist()

        t2= [1 if x else 0 for x in t1]

        people[c]= t2

    elif type(people[c][0])== type('1')   : # string

        t1= people[c].values.flatten()

        t2= list( map( lambda x: int( x.split()[1]), t1))

        people[c]= t2

    elif type( people[c][0])== type (1) or type( people[c][0])== type( .01):

        pass

# people.info()



# people.to_pickle(path= 'people')

# people.describe()
a_col= ['people_id', 'act_id', 'a_date', 'a_category']

for i in range( 1, 11, 1):

    a_col.append( 'a_ch_'+ str(i))

a_col.append( 'outcome')



act= act_tmp.copy( deep= True)

act.columns= a_col

# act.info()

for c in act.columns.values[3:14]:

    t1= act[c].values.flatten()

    t2= np.zeros(len(t1))

    for i in range( len(t1)):

        if type(t1[i])== type('1'):

            t2[i]= int( t1[i].split()[1])

        else:

            t2[i]= np.nan

            

    act[c]= t2

    

#act.to_pickle('../input/act')  

# act.info()
rn= act.shape[0]

A= [None]* rn

for i in range( rn):

    pID= act.ix[i]['people_id']

    pobs= people[people['people_id']== pID].values.flatten()[1:].tolist()

    A[i]= ( pobs)

A
