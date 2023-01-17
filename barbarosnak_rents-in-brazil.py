# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



import matplotlib.pyplot as plt



from wordcloud import WordCloud

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
rent1=pd.read_csv("/kaggle/input/brasilian-houses-to-rent/houses_to_rent.csv")

rent2=pd.read_csv("/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv")
rent1.head()
rent1.info()
rent2.head()
rent2.info()
# First we need to get rid of string values. Because of my not enough information about this subject I replaced sem info and Incluso as 0.

rent1.hoa.replace("Sem info","0",inplace=True)

rent1.hoa.replace("Incluso","0",inplace=True)

# In this part we changed object to integer.



rent1_hoa=[]

rent1_hoa=[each.replace("R$","0") for each in rent1.hoa]

rent1_hoa=[int(each.replace(",","")) for each in rent1_hoa]

# For rent amount we don't need to get rid of non-numerical values.

rent1_rent_amount=[each.replace("R$","") for each in rent1["rent amount"]]

rent1_rent_amount=[int(each.replace(",","")) for each in rent1_rent_amount]
# Again I changed Incluse to 0.

rent1["property tax"].replace("Incluso","0",inplace=True)



rent1_property_tax=[each.replace("R$","") for each in rent1["property tax"]]

rent1_property_tax=[int(each.replace(",","")) for each in rent1_property_tax]
# Fire insurance has no non-numerical values.

rent1_fire_insurance=[each.replace("R$","") for each in rent1["fire insurance"]]

rent1_fire_insurance=[int(each.replace(",","")) for each in rent1_fire_insurance]
#  Total has no non-numerical values.

rent1_total=[each.replace("R$","") for each in rent1.total]

rent1_total=[int(each.replace(",","")) for each in rent1_total]
# Floor values were object, because of that we changed them to integer.

rent1.floor.replace("-","0",inplace=True)

rent1.floor=rent1.floor.astype(int)



rent2.floor.replace("-","0",inplace=True)

rent2.floor=rent2.floor.astype(int)


y=rent1.total[:20]



trace1=go.Bar(

              x=rent1.rooms,

              y=y,

              name="Rooms",

              marker=dict(color="rgba(255,122,32,0.5)",

                          line=dict(color="rgb(0,0,0)",width=0.025)),

              text=rent1.rooms

)



trace2=go.Bar(

              x=rent1["parking spaces"],

              y=y,

              name="Parking Spaces",

              marker=dict(color="rgba(123,212,10,0.5)",

                        line=dict(color="rgb(0,0,0)",width=0.025)),

              text=rent1["parking spaces"]

    

)



data=[trace1,trace2]

layout=go.Layout(barmode="group")

fig=go.Figure(data=data,layout=layout)

iplot(fig)
df1=rent1.bathroom.iloc[:20]

df2=rent2.bathroom.iloc[:20]

df3=pd.concat([df1,df2],axis=0,ignore_index=True)



pie_list=[df3]

labels=df3.unique()





fig = { 

    "data": [

        {

            "values": pie_list,

            "labels": labels,

            "domain":{"x":[0,.5]},

            "name":"Number of Bathroom Rates",

            "hoverinfo":"label+percent+name",

            "hole":.3,

            "type":"pie",

            

        },

    ],

    "layout": {

        "title":"House's Numbers of Bathroom Rates",

        "annotations":[

            {"font":{"size":20},

             "showarrow":False,

             "text":"Number of Bathrooms",

             "x":0.2,

             "y":1

            },

        ]

    }

    

}





iplot(fig)
import plotly.figure_factory as ff



df=rent2.iloc[:100]

data=df.loc[:,["property tax (R$)","fire insurance (R$)","total (R$)"]]



data["index"]=np.arange(1,len(data)+1)



fig=ff.create_scatterplotmatrix(data,diag="box",index="index",colormap="Portland",colormap_type="cat",height=700,width=700)



iplot(fig)
rent2.head()
plt.subplots(figsize=(8,8))







wordcloud=WordCloud(

                     background_color="white",

                     width=512,

                     height=384,

                    ).generate("".join(rent2.city))







plt.imshow(wordcloud)

plt.axis("off")

plt.savefig("graph.png")



plt.show()
trace1=go.Scatter3d(

    x=rent1_total[:20],

    y=rent1_property_tax[:20],

    z=rent1_fire_insurance[:20],

    mode="markers",

    marker=dict(size=10,

                color="rgb(250,42,134)",))



data=[trace1]

layout=go.Layout(

margin=dict(

            l=0,

            r=0,

            b=0,

            t=0

    )

)



fig=go.Figure(data=data,layout=layout)



iplot(fig)