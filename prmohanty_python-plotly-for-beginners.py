# Install plotly ...as Plotly is not installed as part of Python base package



!pip install plotly
# (*) Import plotly package

import plotly



# Check plotly package version

plotly.__version__ 
import plotly.graph_objects as go



Bar_Plot = go.Figure(

                    data=[go.Bar(y=[2, 1, 3])],

                    layout_title_text="Bar Plot using Plotly"

                    )



Bar_Plot.show()
import plotly.graph_objects as go



Scatter_Plot = go.Figure(

                    data=[go.Scatter(x=[0, 1, 2] , y=[2, 1, 3])],

                    layout_title_text="Scatter Plot using Plotly"

                    )



Scatter_Plot.show()