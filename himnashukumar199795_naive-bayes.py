import pandas as pd

import numpy as np

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.DataFrame(np.random.randint(0,2,(50,5)),columns=['x1','x2','x3','x4','y'])
y_proba=pd.DataFrame(df.groupby('y').count()['x1'])
y_proba.reset_index(inplace=True)
y_proba.columns=['y','freq']

y_proba
y_proba['proba']=[23/50,27/50]
y_proba
y_proba_x1=pd.crosstab(df['x1'],df['y'])

y_proba_x1.reset_index(inplace=True)

y_proba_x1.columns=['value of x1',0,1]

y_proba_x1
y_proba_x1['proba of 0']=[12/23,11/23]

y_proba_x1['proba of 1']=[11/27,16/27]

y_proba_x1
y_proba_x2=pd.crosstab(df['x2'],df['y'])

y_proba_x2.reset_index(inplace=True)

y_proba_x2.columns=['value of x2',0,1]

y_proba_x2
y_proba_x2['proba of 0']=[16/32,16/32]

y_proba_x2['proba of 1']=[7/18,11/18]

y_proba_x2
y_proba_x3=pd.crosstab(df['x3'],df['y'])

y_proba_x3.reset_index(inplace=True)

y_proba_x3.columns=['value of x3',0,1]

y_proba_x3
y_proba_x3['proba of 0']=[13/30,17/30]

y_proba_x3['proba of 1']=[10/20,10/20]

y_proba_x3
y_proba_x4=pd.crosstab(df['x4'],df['y'])

y_proba_x4.reset_index(inplace=True)

y_proba_x4.columns=['value of x4',0,1]

y_proba_x4
y_proba_x4['proba of 0']=[13/29,16/29]

y_proba_x4['proba of 1']=[10/21,11/21]

y_proba_x4
y_proba_x3
y_proba_x2
y_proba_x1
p_yes=0.592593*0.388889*0.5*0.47619*0.54

p_yes
p_no=0.478261*0.5*0.566667*0.448276*0.46

p_no
p_yes/(p_yes+p_no)
p_no/(p_yes+p_no)