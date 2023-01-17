import plotly.graph_objects as go



fig = go.Figure(data=[go.Sankey(

    node = dict(pad = 15,thickness = 20,line = dict(color = "black", width = 0.5),

    label = ["Y1_Base", "Y1_Silver", "Y1_Gold", "Y2_Base", "Y2_Silver", "Y2_Gold"],

    color = ['#a6cee3','#fdbf6f','#fb9a99','#a6cee3','#fdbf6f','#fb9a99']

    ),

    link = dict(

      source = [0, 0,0, 1, 1,1, 2 ,2, 5], # indices correspond to source node wrt to label 

      target = [3, 4, 5, 3, 4, 5, 5,3,0], 

      value = [18, 8, 2, 1, 16, 4 , 8, 1,1],

      color = ['#a6cee3', '#a6cee3', '#a6cee3', '#fdbf6f','#fdbf6f', '#fdbf6f', '#fb9a99', '#fb9a99','#fb9a99']

  ))])



fig.update_layout(

    hovermode = 'x',

    title="Sankey Chart",

    font=dict(size = 10, color = 'white'),

)



fig.show()
import plotly.graph_objects as go

import urllib, json



url = 'https://raw.githubusercontent.com/plotly/plotly.js/master/test/image/mocks/sankey_energy.json'

response = urllib.request.urlopen(url)

data = json.loads(response.read())



# override gray link colors with 'source' colors

opacity = 0.4

# change 'magenta' to its 'rgba' value to add opacity

data['data'][0]['node']['color'] = ['rgba(255,0,255, 0.8)' if color == "magenta" else color for color in data['data'][0]['node']['color']]

data['data'][0]['link']['color'] = [data['data'][0]['node']['color'][src].replace("0.8", str(opacity))

                                    for src in data['data'][0]['link']['source']]



fig = go.Figure(data=[go.Sankey(

    valueformat = ".0f",

    valuesuffix = "TWh",

    # Define nodes

    node = dict(

      pad = 15,

      thickness = 15,

      line = dict(color = "black", width = 0.5),

      label =  data['data'][0]['node']['label'],

      color =  data['data'][0]['node']['color']

    ),

    # Add links

    link = dict(

      source =  data['data'][0]['link']['source'],

      target =  data['data'][0]['link']['target'],

      value =  data['data'][0]['link']['value'],

      label =  data['data'][0]['link']['label'],

      color =  data['data'][0]['link']['color']

))])



fig.update_layout(title_text="Energy forecast for 2050<br>Source: Department of Energy & Climate Change, Tom Counsell via <a href='https://bost.ocks.org/mike/sankey/'>Mike Bostock</a>",

                  font_size=10)

fig.show()