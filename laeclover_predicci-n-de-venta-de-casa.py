import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

resultados = pd.DataFrame({
    'Modelo':[],
    'Detalles':[],
    'R2 train':[],
    'R2 test':[],
    'MAE train':[],
    'MAE test':[]
})

def error_absoluto_medio(y,y_pred):
    v = np.abs(y - y_pred)
    return v.sum()/len(y)
    
def error_gen(y_train,y_pred_train,y_test,y_pred_test):
    MAE_TRAIN = error_absoluto_medio(y_train,y_pred_train)
    MAE_TEST = error_absoluto_medio(y_test,y_pred_test)
    return MAE_TRAIN,MAE_TEST
df = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
df.drop(['id'],axis=1,inplace=True)

df.head()
df.isna().sum()
from plotly.subplots import make_subplots
import numpy as np

xpoint, ypoint = np.meshgrid(range(4),range(5),indexing='ij')
cols = df.columns.values

fig2 = make_subplots(rows=4,cols=5)
for i,j,col in zip(xpoint.ravel(),ypoint.ravel(),cols):
    fig2.add_histogram(x=df[col],name=col,row=i+1,col=j+1)
    
fig2.show()
mapbox_access_token = "pk.eyJ1IjoibGFlY3MiLCJhIjoiY2tkZ3RveWozMjUzNTJ3anFxaGNydnpvYyJ9.Qv6zEMCrHgxm01Xqn4L8gw"
px.set_mapbox_access_token(mapbox_access_token)

fig = px.scatter_mapbox(df, lat="lat", lon="long",color="zipcode",
                        color_continuous_scale=px.colors.cyclical.IceFire,
                        title='Mapa organizado por código zip',
                        size_max=15, zoom=8.8,center=dict(lat=47.46,lon=-121.9))

fig.update_layout(
    autosize=False,
    height=800,
    width=1000,
    hovermode='closest',
)

fig.show()
mean_zip = df.groupby(['zipcode']).mean()['price'].sort_values(ascending=True)
code_zip = mean_zip.index.values.tolist()
val_zip = range(mean_zip.shape[0])

color = []
for code in df['zipcode']:
    ind = code_zip.index(code)
    color.append(val_zip[ind])

df['mean_price_sort'] = color

fig = px.scatter_mapbox(df, lat="lat", lon="long",color="mean_price_sort",
                        color_continuous_scale=px.colors.cyclical.IceFire, 
                        title='Mapa organizado por el precio medio de la zona',
                        size_max=15, zoom=8.8,center=dict(lat=47.46,lon=-121.9))

fig.update_layout(autosize=False,
                  height=800,
                  width=1000,
                  hovermode='closest')

fig.show()

df.drop('mean_price_sort',axis=1,inplace=True)
fig = px.scatter_matrix(df,
                       dimensions=['price','bedrooms','bathrooms','floors','waterfront']
                       )
fig.show()
fig = px.scatter_matrix(df,
                       dimensions=['price','sqft_living','sqft_living15','sqft_lot','sqft_lot15','sqft_above','sqft_basement']
                       )
fig.show()
fig = px.scatter_matrix(df,
                       dimensions=['price', 'view', 'condition', 'yr_built', 'yr_renovated', 'grade']
                       )
fig.show()
matrix_corr = df.corr()

heat = go.Heatmap(z=matrix_corr.values,
                  x=matrix_corr.index.values,
                  y=matrix_corr.columns.values)

layout = go.Layout(title='Matriz de correlación',
                   width=800, height=800,
                   xaxis_showgrid=False,
                   yaxis_showgrid=False,
                   yaxis_autorange='reversed')

fig=go.Figure(data=[heat],layout=layout)        
fig.show()
fig3 = make_subplots(rows=1,cols=2)
fig3.add_box(x=df['bathrooms'],y=df['price'],row=1,col=1,name='bathroom')
fig3.add_box(x=df['bedrooms'],y=df['price'],row=1,col=2,name='bedrooms')
fig3.show()

fig4 = px.box(df,x='grade',y='price',color='grade',title='Grade Box Plot')
fig4.show()
fig = px.scatter_3d(df, x='price', y='sqft_living', z='sqft_above',color='grade')
fig.show()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df2 = df.drop(['date'],axis=1)

def regresion_lineal(df,test_size=0.2,Prec_var='price'):
    x = df.drop(Prec_var,axis=1)
    y = df[Prec_var]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_size)
    lr = LinearRegression()
    lr.fit(x_train,y_train)

    r2_train = lr.score(x_train,y_train)
    r2_test = lr.score(x_test,y_test)
    
    y_train_pred = lr.predict(x_train)
    y_test_pred = lr.predict(x_test)
    
    MAE_train,MAE_test = error_gen(y_train,y_train_pred,y_test,y_test_pred)
    
    return r2_train,r2_test,MAE_train,MAE_test

r2_train,r2_test,MAE_train,MAE_test = regresion_lineal(df2)

final = resultados.shape[0]
resultados.loc[final] = ['Regresión Lineal','Sin procesar',r2_train,r2_test,MAE_train,MAE_test]
resultados
from sklearn.preprocessing import MinMaxScaler
aux1 =  df2.drop('price',axis=1)
col_esc = aux1.columns.values

scale = MinMaxScaler()

df3 = df2
df3[col_esc] = scale.fit_transform(df2[col_esc])
df3.head()
from sklearn.preprocessing import PowerTransformer

var_label1 = ['sqft_living', 'sqft_above', 'sqft_living15']
transf = PowerTransformer(method='box-cox',standardize=False)

var_transf = transf.fit_transform(df[var_label1])

df_tf = df3
df_tf[var_label1] = var_transf

xpoint, ypoint = np.meshgrid(range(1),range(3),indexing='ij')
fig2 =  make_subplots(rows=1,cols=3)
for i,j,col in zip(xpoint.ravel(),ypoint.ravel(),var_label1):
    fig2.add_histogram(x=df_tf[col],name=col,row=i+1,col=j+1)

fig2.update_layout(title_text="Histograma de las variables transformadas")
fig2.show()
r2_train,r2_test,MAE_train,MAE_test = regresion_lineal(df_tf)

final = resultados.shape[0]
resultados.loc[final] = ['Regresión Lineal','Transf. a gausiana',r2_train,r2_test,MAE_train,MAE_test]
resultados
from sklearn.preprocessing import KBinsDiscretizer

var_label2 = ['yr_built','yr_renovated']
agrup = KBinsDiscretizer(n_bins=7, encode='ordinal', strategy='uniform')

var_bin = agrup.fit_transform(df[var_label2])

df_bin = df3
df_bin[var_label2] = var_bin

xpoint, ypoint = np.meshgrid(range(1),range(2),indexing='ij')
fig2 =  make_subplots(rows=1,cols=2)
for i,j,col in zip(xpoint.ravel(),ypoint.ravel(),var_label2):
    fig2.add_histogram(x=df_bin[col],name=col,row=i+1,col=j+1)
fig2.show()
r2_train,r2_test,MAE_train,MAE_test = regresion_lineal(df_bin)

final = resultados.shape[0]
resultados.loc[final] = ['Regresión Lineal','Agrupamiento',r2_train,r2_test,MAE_train,MAE_test]
resultados
df_tf_bin = df3
df_tf_bin[var_label1] = var_transf
df_tf_bin[var_label2] = var_bin

r2_train,r2_test,MAE_train,MAE_test = regresion_lineal(df_tf_bin,test_size=0.2)

final = resultados.shape[0]
resultados.loc[final] = ['Regresión Lineal','Agrup. y transf.',r2_train,r2_test,MAE_train,MAE_test]
resultados
from sklearn.preprocessing import PolynomialFeatures

def regresion_poli(df,degree=2,test_size=0.2,Prec_var='price'):
    x = df.drop(Prec_var,axis=1)
    y = df[Prec_var]
    
    poly = PolynomialFeatures(degree=2,)
    x_poly = poly.fit_transform(x)
    
    x_train,x_test,y_train,y_test = train_test_split(x_poly,y,test_size=test_size)
        
    lr = LinearRegression()
    lr.fit(x_train,y_train)
    
    r2_train = lr.score(x_train,y_train)
    r2_test = lr.score(x_test,y_test)
    
    y_train_pred = lr.predict(x_train)
    y_test_pred = lr.predict(x_test)
    
    MAE_train,MAE_test = error_gen(y_train,y_train_pred,y_test,y_test_pred)
    

    return r2_train,r2_test,MAE_train,MAE_test

r2_train,r2_test,MAE_train,MAE_test = regresion_poli(df3,degree=3)
final = resultados.shape[0]
resultados.loc[final] = ['Regresión Polinómica','Sin procesar / Grado 3',r2_train,r2_test,MAE_train,MAE_test]

r2_train,r2_test,MAE_train,MAE_test = regresion_poli(df_bin,degree=3)
final = resultados.shape[0]
resultados.loc[final] = ['Regresión Polinómica','Agrupamiento / Grado 3',r2_train,r2_test,MAE_train,MAE_test]

r2_train,r2_test,MAE_train,MAE_test = regresion_poli(df_tf,degree=3)
final = resultados.shape[0]
resultados.loc[final] = ['Regresión Polinómica','Transf. a gausiana / Grado 3',r2_train,r2_test,MAE_train,MAE_test]

r2_train,r2_test,MAE_train,MAE_test = regresion_poli(df_tf_bin,degree=3)
final = resultados.shape[0]
resultados.loc[final] = ['Regresión Polinómica','Agrup. y transf. / Grado 3',r2_train,r2_test,MAE_train,MAE_test]
resultados
from sklearn.tree import DecisionTreeRegressor

def Arbol_Regresion(df,test_size=0.2,Prec_var='price'):
    x = df.drop(Prec_var,axis=1)
    y = df[Prec_var]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_size)
    
    tree = DecisionTreeRegressor(min_samples_split=30,min_samples_leaf=10,random_state=0)
    tree.fit(x_train,y_train)
    
    r2_train = tree.score(x_train,y_train)
    r2_test = tree.score(x_test,y_test)
    
    y_train_pred = tree.predict(x_train)
    y_test_pred = tree.predict(x_test)
    
    MAE_train,MAE_test = error_gen(y_train,y_train_pred,y_test,y_test_pred)
    
    return r2_train,r2_test, MAE_train,MAE_test
r2_train,r2_tes, MAE_train,MAE_testt = Arbol_Regresion(df3)
final = resultados.shape[0]
resultados.loc[final] = ['Árbol de regresión','Sin procesar',r2_train,r2_test, MAE_train,MAE_test]

r2_train,r2_test, MAE_train,MAE_test = Arbol_Regresion(df_bin)
final = resultados.shape[0]
resultados.loc[final] = ['Árbol de regresión','Agrupamiento',r2_train,r2_test, MAE_train,MAE_test]

r2_train,r2_test, MAE_train,MAE_test = Arbol_Regresion(df_tf)
final = resultados.shape[0]
resultados.loc[final] = ['Árbol de regresión','Transf. a gausiana',r2_train,r2_test, MAE_train,MAE_test]

r2_train,r2_test, MAE_train,MAE_test = Arbol_Regresion(df_tf_bin)
final = resultados.shape[0]
resultados.loc[final] = ['Árbol de regresión','Agrup. y transf.',r2_train,r2_test, MAE_train,MAE_test]
resultados
import tensorflow as tf
from tensorflow.keras import regularizers

def get_model_compile():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32,activation='tanh',input_shape=(18,)),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)   
    ])

    optimizer=tf.keras.optimizers.Adam(0.001)
    model.compile(loss='mse',
                 optimizer=optimizer,
                 metrics=['mae','mse'])
    return model
x = df_tf.drop('price',axis=1)
y = df_tf['price']

batch_size = 128
epochs=500

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

model = get_model_compile()

history = model.fit(x_train,y_train,
                    epochs=epochs,batch_size=batch_size,
                    validation_data=(x_test,y_test),
                    shuffle=True,
                    verbose=0)
evol_train = pd.DataFrame(history.history)
evol_train['epoch'] = history.epoch

trace1 = go.Scatter(x=evol_train['epoch'], y=evol_train['mae'], mode='lines',name='mae')
trace2 = go.Scatter(x=evol_train['epoch'], y=evol_train['val_mae'], mode='lines',name='val_mae')

trace3 = go.Scatter(x=evol_train['epoch'], y=evol_train['mse'], mode='lines',name='mse')
trace4 = go.Scatter(x=evol_train['epoch'], y=evol_train['val_mse'], mode='lines',name='val_mse')

fig = make_subplots(rows=1,cols=2)
fig.add_trace(trace1,row=1,col=1)
fig.add_trace(trace2,row=1,col=1)

fig.add_trace(trace3,row=1,col=2)
fig.add_trace(trace4,row=1,col=2)

fig.update_layout(title_text="Gráficas de MAE y MSE vs Época")
fig.show()

df_tf['price_pred'] = model.predict(x)
px.scatter(df_tf,x='price',y='price_pred',title='Gráfica de los precios reales vs los precios predichos ')
from sklearn.metrics import r2_score
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

r2_train = r2_score(y_train,y_pred_train)
r2_test = r2_score(y_test,y_pred_test)

MAE_TRAIN,MAE_TEST = error_gen(y_train,y_pred_train[:,0],y_test,y_pred_test[:,0])

final = resultados.shape[0]
resultados.loc[final] = ['ANN','Transf. a gausiana',r2_train,r2_test,MAE_TRAIN,MAE_TEST]
resultados