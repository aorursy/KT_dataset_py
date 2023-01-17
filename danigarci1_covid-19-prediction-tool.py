def PrepareTrainData(df,degree):

    X = []

    y= []

    N_train = df.shape[0]

    for k in range(N_train-1):

        sample = []

        for d in range(degree+1):

            sample.append(k**d)

        X.append(sample)

        y.append(np.log(df.total.iloc[k]).tolist())

    X = np.array(X)

    y = np.array(y).T

    return X, y
def PrepareTestData(size,degree):

    Xtst = []

    N_test = size

    for k in range(N_test):

        sample = []

        for d in range(degree+1):

            sample.append(k**d)

        Xtst.append(sample)

    Xtst = np.array(Xtst)

    return Xtst
from sklearn.preprocessing import MinMaxScaler

def Normalize(X,Xtst,y):

    minmaxx = MinMaxScaler(feature_range=(-1,1))

    minmaxx.fit(X)

    X_norm    = minmaxx.transform(X)

    Xtst_norm = minmaxx.transform(Xtst)

    minmaxy = MinMaxScaler(feature_range=(-1,1))

    minmaxy.fit(y[:,None])

    y_norm = minmaxy.transform(y[:,None]).ravel()

    return X_norm ,Xtst_norm, y_norm, minmaxy
from sklearn import linear_model

def TrainTestModelRidge(X_norm,y_norm,Xtst_norm,alpha):

    ytst_norm = []

    clf = linear_model.Ridge(alpha = alpha)

    clf.fit(X_norm,y_norm)

    ytst_norm = clf.predict(Xtst_norm)

    return ytst_norm
from sklearn import svm

def TrainTestModelSVM(X_norm,y_norm,Xtst_norm,ker,deg,c_value,ep):

    ytst_norm = []

    clf = svm.SVR(kernel = ker,degree = deg,C = c_value, epsilon=ep)

    clf.fit(X_norm,y_norm)

    ytst_norm = clf.predict(Xtst_norm)

    return ytst_norm
def Regression(df,modelType='Ridge',deg=3,alpha= 0.01,ker='rbf',c_value = 1,ep=0.1):

    train_size = df.shape[0]

    test_size = train_size *2

    

    #Create new lines

    X ,y = PrepareTrainData(df,deg)

    Xtst = PrepareTestData(test_size,deg)

    X_norm , Xtst_norm, y_norm, Scaler = Normalize(X, Xtst, y)

    if modelType == 'Ridge':

        ytst_norm = TrainTestModelRidge(X_norm,y_norm,Xtst_norm,alpha)

    elif modelType == 'SVR':

        ytst_norm = TrainTestModelSVM(X_norm,y_norm,Xtst_norm,ker,deg,c_value,ep)

    

    return y_norm, ytst_norm, Scaler
import bokeh.plotting.figure as bk_figure

from bokeh.io import curdoc, show

from bokeh.layouts import column,row

from bokeh.models import ColumnDataSource, Span, Label, Select, DataTable, TableColumn, DateFormatter

from bokeh.models.widgets import Slider, TextInput

from bokeh.io import output_notebook # enables plot interface in J notebook

import numpy as np

import pandas as pd

# init bokeh

from bokeh.application import Application

from bokeh.application.handlers import FunctionHandler



output_notebook()

infected = pd.read_csv('https://raw.githubusercontent.com/datadista/datasets/master/COVID%2019/ccaa_covid19_casos_long.csv')

altas = pd.read_csv('https://raw.githubusercontent.com/datadista/datasets/master/COVID%2019/ccaa_covid19_altas_long.csv')

death = pd.read_csv('https://raw.githubusercontent.com/datadista/datasets/master/COVID%2019/ccaa_covid19_fallecidos_long.csv')

hosp = pd.read_csv('https://raw.githubusercontent.com/datadista/datasets/master/COVID%2019/ccaa_covid19_hospitalizados_long.csv')

date_label_offset = 30

label_text_size = '8pt'

#Start at first infected

temp = infected[infected['CCAA']=='Total']

temp = temp[temp['total'] != 0]

first_date = temp.fecha.iloc[0]





train_size = temp.shape[0]

test_size = train_size *2

y_norm, ytst_norm, Scaler = Regression(temp[temp['CCAA']=='Total'],'Ridge',2,0.01,'rbf',1.0,0.1)



                                    # DATA FOR LOG PLOT

t1 = pd.date_range(start=first_date,periods= train_size)

y1 = Scaler.inverse_transform(y_norm[:,None])

t2 = pd.date_range(start=first_date,periods= test_size)

y2 = Scaler.inverse_transform(ytst_norm[:,None])

source11 = ColumnDataSource(data=dict(x=t1, y=y1))

source12 = ColumnDataSource(data=dict(x=t2, y=y2))





                                        # LOG PLOT

log_plot = bk_figure(plot_height=200, plot_width=800, title="COVID-19 Analysis (Log Scale)",

              tools="crosshair,pan,reset,save,wheel_zoom",x_axis_type = 'datetime')

log_plot.line('x', 'y', source=source12, line_width=3, line_alpha=0.6, legend_label= 'Predicted Infected')

log_plot.circle('x','y',source = source11, size = 2,color = 'red',legend_label='Real Infected')

log_plot.legend.click_policy="hide"

                                        #DATA FOR LINEAR PLOT

y11 = np.exp(y1)

y22 = np.exp(y2)

source21 = ColumnDataSource(data=dict(x=t1, y=y11))

source22 = ColumnDataSource(data=dict(x=t2, y=y22))



                                        #LINEAR PLOT

linear_plot = bk_figure(plot_height=200, plot_width=800, title="COVID-19 Analysis (Linear Scale)",

              tools="crosshair,pan,reset,save,wheel_zoom",x_axis_type = 'datetime')



linear_plot.line('x', 'y', source=source22, line_width=3, line_alpha=0.6,legend_label='Predicted Infected')

linear_plot.circle('x','y',source = source21, size = 2,color = 'red',legend_label='Real Infected')





position_of_max = np.argmax(source22.data['y'],axis = 0)

vline = Span(location = source22.data['x'][position_of_max][0],dimension = 'height', line_color = 'black', line_width = 2)

label_text =  'Predicted max: {} Fecha: {}'.format(int(np.max(source22.data['y'])),t2[position_of_max].strftime("%d-%m-%Y").values)

text = Label(x = t2[position_of_max-date_label_offset][0], y= source22.data['y'][position_of_max-5][0][0],text = label_text,text_font_size= label_text_size)

linear_plot.add_layout(vline)

linear_plot.add_layout(text)

linear_plot.legend.click_policy="hide"







# Set up widgets

ccaa = Select(title="CCAA:", value="Total", options=list(infected['CCAA'].unique()))

model = Select(title="Model:", value="Ridge", options=['Ridge','SVR'])

degree = Slider(title="degree", value=2, start=1, end=10, step=1)

alph = Slider(title="alpha (only Ridge)", value=.01, start=.000, end=1.000, step = .01)

kernel = Select(title="kernel (only SVR)", value="rbf", options=['rbf','linear'])

Cvalue = Slider(title="C (only SVR)", value=1, start=1, end=10, step = 1)

Epsilon = Slider(title="Epsilon (only SVR)", value=.1, start=0, end=1.000, step = .1)



source_table = ColumnDataSource(data =  dict(

        dates=infected[infected['CCAA']=='Total'].fecha,

        infected=infected[infected['CCAA']=='Total'].total,

        hospitalized=pd.concat([pd.Series(np.zeros(infected[infected['CCAA']=='Total'].shape[0]-hosp[hosp['CCAA']=='Total'].shape[0],dtype = int)),hosp[hosp['CCAA']=='Total'].total],ignore_index = True),

        altas=pd.concat([pd.Series(np.zeros(infected[infected['CCAA']=='Total'].shape[0]-altas[altas['CCAA']=='Total'].shape[0],dtype = int)),altas[altas['CCAA']=='Total'].total],ignore_index = True),

        fallecidos=pd.concat([pd.Series(np.zeros(infected[infected['CCAA']=='Total'].shape[0]-death[death['CCAA']=='Total'].shape[0],dtype = int)),death[death['CCAA']=='Total'].total],ignore_index = True),

    ))



columns = [

        TableColumn(field="dates", title="Date"),

        TableColumn(field="infected", title="Infected"),

        TableColumn(field="hospitalized", title="Hospitalized"),

        TableColumn(field="altas", title="Recovered"),

        TableColumn(field="fallecidos", title="Death"),

    ]

data_table = DataTable(source=source_table, columns=columns, width=600, height=380)



def update_data(attrname, old, new):

    # Get the current slider values

    deg = degree.value

    ca = ccaa.value

    a = alph.value

    ker = kernel.value

    c = Cvalue.value

    ep = Epsilon.value

    modelType = model.value

    

    temp = infected[infected['CCAA']==ca]

    temp = temp[temp['total'] != 0]

    

    first_date = temp['fecha'].iloc[0]

    train_size = temp.shape[0]

    test_size = train_size *2

    

    #Create new lines

    y_norm, ytst_norm, Scaler = Regression(temp[temp['CCAA']==ca],modelType,deg,a,ker,c,ep)

    

    #Update log plot

    t1 = pd.date_range(start=first_date,periods= y_norm.shape[0])

    y1 = Scaler.inverse_transform(y_norm[:,None])

    t2 = pd.date_range(start=first_date,periods= ytst_norm.shape[0])

    y2 = Scaler.inverse_transform(ytst_norm[:,None])

    source11.data=dict(x=t1, y=y1)

    source12.data=dict(x=t2, y=y2)

    

    #update linear plot

    y11 = np.exp(y1)

    y22 = np.exp(y2)

    source21.data=dict(x=t1, y=y11)

    source22.data=dict(x=t2, y=y22)

    

    #update vertical line

    position_of_max = np.argmax(source22.data['y'],axis = 0)

    vline.location = source22.data['x'][position_of_max][0]

    

    #update label

    label_text =  '    Predicted max: {} Fecha: {}'.format(int(np.max(source22.data['y'])),t2[position_of_max].strftime("%d-%m-%Y").values)

    text.text = label_text

    text.x = t2[position_of_max-date_label_offset][0]

    text.y = source22.data['y'][position_of_max-5][0][0]





    source_table.data = dict(

        dates=infected[infected['CCAA']==ca].fecha,

        infected=infected[infected['CCAA']==ca].total,

        hospitalized=pd.concat([pd.Series(np.zeros(infected[infected['CCAA']==ca].shape[0]-hosp[hosp['CCAA']==ca].shape[0],dtype = int)),hosp[hosp['CCAA']==ca].total],ignore_index = True),

        altas=pd.concat([pd.Series(np.zeros(infected[infected['CCAA']==ca].shape[0]-altas[altas['CCAA']==ca].shape[0],dtype = int)),altas[altas['CCAA']==ca].total],ignore_index = True),

        fallecidos=pd.concat([pd.Series(np.zeros(infected[infected['CCAA']==ca].shape[0]-death[death['CCAA']==ca].shape[0],dtype = int)),death[death['CCAA']==ca].total],ignore_index = True),

        )



for w in [ccaa, degree, alph,kernel,Cvalue,Epsilon,model]:

    w.on_change('value', update_data)





# Set up layouts and add to document

plots = column(log_plot,linear_plot)

plots_and_table = row(plots,data_table)

customizable_parameters = column(degree, alph,kernel, Cvalue,Epsilon,sizing_mode= 'stretch_both')

layout = column(model,ccaa,plots_and_table,

             customizable_parameters)





def modify_doc(doc):

    doc.add_root(row(layout, width=800))

    doc.title = "COVID-19 ANalysis Spain"





# handler = FunctionHandler(modify_doc)

# app = Application(handler)

# show(app)

curdoc().add_root(row(layout, width=800))

curdoc().title = "COVID19-Analysis"

show(layout,notebook_handle = True)