import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb



plt.style.use('ggplot')

%matplotlib inline



# dipole_moments = pd.read_csv("../input/dipole_moments.csv")

# magnetic_shielding_tensors = pd.read_csv("../input/magnetic_shielding_tensors.csv")

# mulliken_charges = pd.read_csv("../input/mulliken_charges.csv")

# potential_energy = pd.read_csv("../input/potential_energy.csv")

# scalar_coupling_contributions = pd.read_csv("../input/scalar_coupling_contributions.csv")
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.sample(3)
test.sample(3)
structures = pd.read_csv("../input/structures.csv")

structures.sample(3)
molecule_1 = structures.iloc[:5,:]

molecule_1
import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go
trace1 = go.Scatter3d(

    x=molecule_1.x,

    y=molecule_1.y,

    z=molecule_1.z,

    mode='markers',

    marker=dict(

        size=10,

        color='rgb(255,0,0)',                # set color to an array/list of desired values      

    )

)



data = [trace1]

layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0  

    )

    

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
sb.countplot(train['type']);
plt.figure(figsize = (15,8))

sb.distplot(train.scalar_coupling_constant);
scalar_abv_sixty = train[train.scalar_coupling_constant > 60]

scalar_abv_sixty.sample(5)
scalar_abv_sixty.type.unique()
scalar_abv_sixty.type.value_counts()
train.type.value_counts()