import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# ----------------------------> LECTURA DESDE PC <----------------------------
# mainpath = "C:/Users/Jason/Downloads/Curso Libre Introducción a Python/DATA/"
# filename = "ESTADISTICAS_EN_EDUCACION_BASICA_POR_MUNICIPIO.csv"
# data = pd.read_csv(mainpath+filename)
# --------------------------> LECTURA DESDE KAGGLE <--------------------------
data = pd.read_csv("/kaggle/input/educacionmunicipiocol/ESTADISTICAS_EN_EDUCACION_BASICA_POR_MUNICIPIO.csv")
data2 = data.copy()

pd.set_option("display.max_columns", 41)
data
data.shape  # Dimensión del dataset
data.dtypes
pd.DataFrame(data.isnull().sum(), columns = ["NA's"]).sort_values(by = "NA's")
# Número total de valores faltantes en el dataframe
pd.isnull(data).values.ravel().sum()
data.dropna(axis = 0, how = "all", inplace = True)
Departamentos = ['Cauca', 'Córdoba', 'Guainía', 'Guaviare', 'Vaupés', 'Vichada', 'Bogotá, D.C.']
Variables     = ['COBERTURA_NETA', 'TASA_MATRICULACIÓN_5_16', 'SEDES_CONECTADAS_A_INTERNET']

data.pivot_table(index = ["DEPARTAMENTO"],
                 values = Variables,
                 aggfunc = {'COBERTURA_NETA' : np.mean,
                            'TASA_MATRICULACIÓN_5_16' : np.mean,
                            'SEDES_CONECTADAS_A_INTERNET' : np.sum}).loc[Departamentos]
def Participacion(serie):
    Participacion = serie.sum()/len(serie)*0.01
    return(Participacion)

Punto1 = data.pivot_table(index = ["DEPARTAMENTO"],
                          values = Variables,
                          aggfunc = {'COBERTURA_NETA' : Participacion,
                                     'TASA_MATRICULACIÓN_5_16' : Participacion,
                                     'SEDES_CONECTADAS_A_INTERNET' : Participacion}).loc[Departamentos]
import plotly.express as px
Figura1 = go.Figure()

MyColors = ["#BF834E", "#F68118", "#F9CA00", "#AEF133", "#19EE9F", "#DE67B4", "#F13057"]
Temp = Punto1.transpose()
for i, color in zip(Temp.columns, MyColors):
    Figura1.add_bar(x = Temp.index, y = Temp[i], marker_color = color, name = i)

Figura1.update_layout(
    # Usando una tema por defecto (https://plotly.com/python/templates/)
    template = "plotly_dark",
    # Personalización de Títulos
    title_text = "PARTICIPACIÓN OBTENIDA POR DEPARTAMENTO",
    title_font_family = "Open Sans",
    title_font_color = "white",
    title_font_size = 24,
    # Personalización de Ejes
    xaxis = dict(
        tickmode = "array",
        tickvals = [0, 1, 2],
        ticktext = ["Cobertura Neta", "Conección a Internet", "Tasa de Matriculación"]
    )
)    
Figura1.update_xaxes(
    title_text = "Variable",
    title_font = dict(size = 20, family = "Balto", color = "#28F618"),
    # Tickes
    tickangle = 0,
    tickfont = dict(size = 16, family = "Overpass", color = "#27E54B"),
    # Línea Base
    showline = True, linewidth = 2, linecolor = "#FFFFFF",
)
Figura1.update_yaxes(
    title_text = "Porcentaje",
    tickformat = "%",
    title_font = dict(size = 20, family = "Balto", color = "#21ACF3"),
    # Tickes
    tickfont = dict(size = 16, family = "Overpass", color = "#23BFBB"),
    # Línea Base
    showline = True, linewidth = 2, linecolor = "#FFFFFF", mirror = True
)
Figura1.show()
Punto2 = data.groupby("MUNICIPIO").aggregate({'COBERTURA_NETA': np.mean,
                                              'TASA_MATRICULACIÓN_5_16': np.mean})
Punto2
Punto2.sort_values(by = "COBERTURA_NETA").head(4)
Punto2.sort_values(by = "TASA_MATRICULACIÓN_5_16").head(4)
Punto2.sort_values(by = "COBERTURA_NETA", ascending = False).head(4)
Punto2.sort_values(by = "TASA_MATRICULACIÓN_5_16", ascending = False).head(4)
MyColors = ["#D8122F", "#10ADBA"]
Figura2 = go.Figure()

for i,color in zip(Punto2.columns, MyColors):
    Figura2.add_bar(x = Punto2.index, y = Punto2[i], marker_color = color, name = i,
                    text = Punto2[i])

Figura2.update_layout(
    barmode = "stack", # (https://plotly.com/python/bar-charts/)
    template = "plotly_white",
    title_text = "COBERTURA Y TASA DE MATRICULACIÓN POR MUNICIPIO",
    title_font_family = "Open Sans",
    title_font_color = "#E02878",
    title_font_size = 22,
    legend = dict(orientation = "h", yanchor = "bottom", y = 1.02, xanchor = "right", x = 1)
)    
Figura2.update_xaxes(
    title_text = "Municipio",
    title_font = dict(size = 18, family = "Arial", color = "#FFA900"),
    tickangle = 90,
    tickfont = dict(size = 12, family = "Balto", color = "#F68118")
)
Figura2.update_yaxes(
    title_text = "Media/Promedio",
    title_font = dict(size = 18, family = "Arial", color = "#27E54B"),
    tickfont = dict(size = 16, family = "Balto", color = "#43AE00")
)
Figura2.update_traces(texttemplate = '%{text:.4s}', textposition = "outside")
Figura2.update_layout(uniformtext_minsize = 5, uniformtext_mode = "hide")

Figura2.show()
data["AÑO"]
Fechas2 = []
for i in data["AÑO"]:
    Fechas2.append( str(i)[:4] + "-01" )

pd.DataFrame(Fechas2)
Fechas2 = pd.to_datetime(Fechas2)
Fechas2
data.set_index(Fechas2, inplace = True)
data
data_Cobertura = data["COBERTURA_NETA"].resample("Y").mean()
data_Matriculacion = data["TASA_MATRICULACIÓN_5_16"].resample("Y").mean()

Resumen = pd.concat([data_Cobertura, data_Matriculacion], axis = "columns", keys = ["Cobertura Neta", "Tasa de Matriculación"])
# Resumen.iloc[:-1,:] = Resumen.iloc[:-1,:]
Resumen
Figura3 = go.Figure()
Figura3.add_scatter(
    x = Resumen["Cobertura Neta"].index,y=Resumen["Cobertura Neta"],
        mode = "lines", line =  dict(color = "#FF8826", width = 3),
        name = "Cobertura Neta", showlegend = True
)
Figura3.add_scatter(
    x = Resumen["Tasa de Matriculación"].index, y = Resumen["Tasa de Matriculación"],
        mode = "lines", line = dict(color = "#D40606", width = 3),
        name = "Tasa de Matriculación", showlegend = True
)

Figura3.update_traces(mode = "markers+lines", hovertemplate = None)
Figura3.update_layout(
    template = "simple_white",
    hovermode = "x", # (https://plotly.com/python/hover-text-and-formatting/)
    title_text = "EVOLUCIÓN HISTÓRICA DE LA COBERTURA Y LA TASA DE MATRICULACIÓN<BR>EN PROMEDIO DE 2011 A 2019",
    title_font_family = "Open Sans",
    title_font_color = "#21ACF3",
    title_font_size = 18,
    legend = dict(orientation = "h", yanchor = "bottom", y = 1.02, xanchor = "right", x = 1)
)    
Figura3.update_xaxes(
    title_text = "Año",
    title_font = dict(size = 20, family = "Rockwell", color = "#00C41F"),
    # Línea Base
    showline = True, linewidth = 2, linecolor = "#00A3BA",
)
Figura3.update_yaxes(
    title_text = "Tasa (%)",
    title_font = dict(size = 20, family = "Rockwell", color = "#A1009A"),
    # Línea Base
    showline = True, linewidth = 2, linecolor = "#00A3BA"
)
Figura3.show()
Departamentos = ['Bogotá, D.C.', 'Antioquia', 'Atlántico', 'Meta', 'Boyacá', 'Caldas', 'Huila']
Variables = ['DEPARTAMENTO', 'DESERCIÓN_TRANSICIÓN', 'DESERCIÓN_PRIMARIA', 'DESERCIÓN_SECUNDARIA', 'DESERCIÓN_MEDIA']
data2 = data2[Variables]
data2.set_index("DEPARTAMENTO", inplace = True)
data2 = data2.loc[Departamentos]
Figura4 = make_subplots(rows=2, cols=2, start_cell = "top-left")

Figura4.add_trace(go.Box(
    x = data2.index, y = data2["DESERCIÓN_TRANSICIÓN"],
    name = "Deserción Transición", marker_color = "#F13057", notched = True), row = 1, col = 1)
Figura4.add_trace(go.Box(
    x = data2.index, y = data2["DESERCIÓN_PRIMARIA"],
    name = "Deserción Primaria", marker_color = "#F68118", notched = True), row = 1, col = 2)
Figura4.add_trace(go.Box(
    x = data2.index, y = data2["DESERCIÓN_SECUNDARIA"],
    name = "Deserción Secundaria", marker_color = "#AEF133", notched = True), row = 2, col = 1)
Figura4.add_trace(go.Box(
    x = data2.index, y = data2["DESERCIÓN_MEDIA"],
    name = "Deserción Media", marker_color = "#19EE9F", notched = True), row = 2, col = 2)

#Figura4.update_layout(title_text="Box Plot Styling Outliers")

Figura4.update_layout(
    template = "plotly_dark",
    legend = dict(orientation = "h", yanchor = "bottom", y = 1.05, xanchor = "right", x = 1.05),
    # Personalización de Títulos
    title_text = "TASA DE DESERCIÓN EN LOS PRINCIPALES DEPARTAMENTOS",
    title_font_family = "Open Sans",
    title_font_color = "#FFFFFF",
    title_font_size = 22,
)
Figura4.update_xaxes(
    tickangle = 45,
    tickfont = dict(size = 9, family = "Arial", color = "#F9CA00"),
    showline = True, linewidth = 2, linecolor = "#FFFFFF",
)
Figura4.update_yaxes(
    tickfont = dict(size = 10, family = "Arial", color = "#21ACF3"),
    showline = True, linewidth = 2, linecolor = "#FFFFFF", mirror = True
)
Figura4.show()