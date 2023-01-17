import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



%matplotlib inline
path = '../input/PIB.csv'



data1 = pd.read_csv(path)

df = data1.T

df.columns = data1['Country Name']

df = df[4:]

index = df.index.values.tolist()

argentina = df['Argentina'].fillna(method='backfill').values

mexico = df['México'].fillna(method='backfill').values

mundo = df['Mundo'].fillna(method='backfill').values

sudafrica = df['Sudáfrica'].fillna(method='backfill').values

eu = df['Estados Unidos'].fillna(method='backfill').values

españa = df['España'].fillna(method='backfill').values

china = df['China'].fillna(method='backfill').values

francia = df['Francia'].fillna(method='backfill').values

noruega = df['Noruega'].fillna(method='backfill').values



data_1 = pd.DataFrame(

    {

        'Argentina': argentina,

        'México': mexico,

        'Sudáfrica': sudafrica,

        'Estados Unidos': eu,

        'España': españa,

        'China': china,

        'Francia': francia,

        'Noruega': noruega,

        'Mundo': mundo

    }

)



import plotly

import plotly.graph_objs as go



plotly.offline.init_notebook_mode(connected=True)
trace1= go.Scatter(

    x= index,

    y= argentina,

    name= 'Argentina'

    #'name': 'Argentina',

)



trace2= go.Scatter(

    x= index,

    y= mexico,

    name= 'Mexico'

)



trace3= go.Scatter(

    x= index,

    y= mundo,

    name= 'Mundo'

)



trace4= go.Scatter(

    x= index,

    y= sudafrica,

    name= 'Sudáfrica'

)



trace5= go.Scatter(

    x= index,

    y= eu,

    name= 'EUA'

)



trace6= go.Scatter(

    x= index,

    y= españa,

    name= 'España'

)



trace7 = go.Scatter(

    x= index,

    y= china,

    name = 'China'

)



trace8 = go.Scatter(

    x= index,

    y= francia,

    name = 'Francia'

)



trace9 = go.Scatter(

    x= index,

    y= noruega,

    name = 'Noruega'

)



layout = go.Layout(

    title = 'Producto Interno Bruto',

    showlegend=True,

    legend=dict(

        orientation="h",

        traceorder='normal',

        font=dict(

            family='sans-serif',

            size=12,

            color='#000'

        ),

        bgcolor='#E2E2E2',

        bordercolor='#FFFFFF',

        borderwidth=2  

    )

    #xaxis = axis_x,

    #yaxis = axis_y,

    #sliders = sliders,    

)



data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9]

fig = go.Figure(data = data, layout=layout)

plotly.offline.iplot(fig)
def regresion_lineal(data, y, *args):

    

    import pandas as pd

    import numpy as np

    from sklearn.linear_model import LinearRegression

    from sklearn import metrics

    from sklearn.model_selection import train_test_split



    Y= data[y]

    X= data[args[0]]

    

    X_train, X_test, y_train, y_test= train_test_split(X, Y, test_size= 0.3, random_state= 2)

    

    # Regresión lineal

    lm= LinearRegression()

    lm.fit(X_train, y_train)

    print('Cuadrados: ', lm.score(X_train, y_train))

    prediccion= lm.predict(pd.DataFrame(X_test))

    

    SSD= sum((y_test - prediccion)**2)

    RSE= np.sqrt(SSD/(len(data)-2))

    media= np.mean(y_test)

    error= RSE/media

    # El error significa el porcentaje que el modelo no puede ser explicado

    print('Error: ', error)
y = 'Mundo'

x = ['México']

regresion_lineal(data_1, y, x)
y = 'Mundo'

x = ['Argentina']



regresion_lineal(data_1, y, x)
y = 'Mundo'

x = ['Argentina', 'México', 'Sudáfrica']



regresion_lineal(data_1, y, x)
y = 'Mundo'

x = ['Argentina', 'México', 'Sudáfrica', 'Estados Unidos']



regresion_lineal(data_1, y, x)