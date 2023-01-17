import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/tzeva_adom_alerts.csv')
import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



vc = df['place'].value_counts()



data = [go.Bar(

            x=vc.index[0:25],

            y=vc.values[0:25]

    )]



layout = dict(

        title = 'Number of alerts in Israeli territory from palestinian rockets and mortars attacks <br> (25 most attacked areas)')

fig = dict(data=data, layout=layout )

py.iplot(fig, filename='IL')