import pandas as pd
import json
import os
import numpy as np
import math
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import io
from IPython.display import display, Markdown, HTML

init_notebook_mode()

input_path = '../input/'
file_paths = [input_path + filename for filename in os.listdir(input_path)]
df_file_paths = file_paths[:3] + file_paths[4:5] + file_paths[6:]


dataframes = [pd.read_json(file_path, encoding='utf8') for file_path in df_file_paths]
reports_df, epsas_df, indicators_df, variables_df, measurements_df  = dataframes

complete_reports_df = pd.merge(reports_df, epsas_df, left_on='epsa', right_on='url')
complete_measurements_df = pd.merge(measurements_df, epsas_df, left_on='epsa', right_on='url')


rdf = complete_reports_df
mdf = complete_measurements_df
vdf = variables_df
idf = indicators_df
edf = epsas_df

with io.open(input_path + 'supply_areas.json', encoding='utf8') as json_data:
    supply_json = json.load(json_data)
    
epsa_jsons = []
for feature in supply_json['features']:
    epsa_json = dict(
        type='FeatureCollection',
        features=[feature]
    )
    epsa_jsons.append(epsa_json)
    
with open('../input/points.json', encoding='utf8') as f:
    points_json = json.load(f)
    
ind_names = [
    'Rendimiento actual de la fuente', 'Uso eficiente del recurso',
    'Cobertura de muestras de agua potable',
    'Conformidad de los análisis de agua potable realizados',
    'Dotación', 'Continuidad por racionamiento', 'Continuidad por corte',
    'Cobertura del servicio de agua potable',
    'Cobertura del servicio de alcantarillado sanitario',
    'Cobertura de micromedición',
    'Incidencia extracción de agua cruda subterránea ',
    'Índice de tratamiento de agua residual', 'Control de agua residual',
    'Capacidad instalada de planta de tratamiento de agua potable',
    'Capacidad instalada de planta de tratamiento de agua residual ',
    'Presión del servicio de agua potable',
    'Índice de agua no contabilizada en producción',
    'Índice de agua no contabilizada en la red',
    'Densidad de fallas en tuberías de agua potable',
    'Densidad de fallas en conexiones de agua potable',
    'Densidad de fallas en tuberías de agua residual',
    'Densidad de fallas en conexiones de agua residual',
    'Índice de operación eficiente', 'Prueba ácida',
    'Eficiencia de recaudación', 'Índice de endeudamiento total', 'Tarifa media',
    'Costo unitario de operación', 'Índice de ejecución de inversiones',
    'Personal calificado', 'Número de empleados por cada 1000 conexiones',
    'Atención de reclamos'
]
ind_units = ['%', '%', '%', '%', 'l/hab/día', 'hr/día', '%', '%', '%', '%',
             '%', '%', '%', '%', '%', '%', '%', '%', 'fallas/100km',
             'fallas/1000conex.', 'fallas/100km', 'fallas/1000conex.', '%', '-',
             '%', '%', 'Bs.', 'Bs.', '%', '%', 'empleados/1000conex.', '%']

colors = [
    '#1f77b4','#ff7f0e','#2ca02c',
    '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22','#17becf'
]

def get_ind_name(ind):
    return ind_names[int(ind[3:])-1]

def get_ind_unit(ind):
    return ind_units[int(ind[3:])-1]
def hide_toggle():
    pass
from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
 <a href="javascript:code_toggle()">here</a>.''')
df = complete_measurements_df
val_year = 2017

def get_area(epsa_json):
    return epsa_json['features'][0]['properties']['area'].split(',')[0].split('.')[0]

sorted_epsa_jsons = sorted(epsa_jsons, key = lambda x: 0 if get_area(x) == '' else int(get_area(x)), reverse=True)

colors = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', 
    '#f58231', '#911eb4', '#46f0f0', '#f032e6', 
    '#bcf60c', '#fabebe', '#008080', '#e6beff', 
    '#9a6324', '#fffac8', '#800000', '#aaffc3', 
    '#808000', '#ffd8b1', '#808080',
] * 10    
 
def get_layer_code(layer):
    return layer['features'][0]['properties']['code'].split('-')[0]

codes = list(set([get_layer_code(layer) for layer in epsa_jsons]))

code_to_color = {code: '#000075' if code == 'SAGUAPAC' else colors[i] for i,code in enumerate(codes)}

val_label = 'Agua Potable'
val_label2= 'Alcantarillado'
val_label3= 'Micromedición'
val_unit = '%'

def interpolate(c1, c2, f):
    r = c1[0] + (c2[0]-c1[0]) * f
    g = c1[1] + (c2[1]-c1[1]) * f
    b = c1[2] + (c2[2]-c1[2]) * f
    return [r,g,b]



def get_layer_color(epsa_code, min_value):
    if epsa_code in list(df.code):
        percentage = df[(df.code==epsa_code)&(df.year==val_year)].ind8.iloc[0]
        if np.isnan(percentage):
            percentage = 0 
    else:
        percentage = 0
        
    factor = (percentage - min_value)/(100 - min_value) # Biased factor for more contrast
    
    color1 = min_colors[0]
    color2 = min_colors[1]
    real_factor = 0.0
    for range_min, range_max, c_min, c_max in zip(min_ranges, max_ranges, min_colors, max_colors):
        if range_min <= factor < range_max:
            color1 = c_min
            color2 = c_max
            real_factor = (factor - range_min) / (range_max - range_min)
    
    c = interpolate(color1, color2, real_factor)
    color_string = f'rgb({str(c[0])[:6]},{str(c[1])[:6]},{str(c[2])[:6]})'
    return(color_string)

def get_display_value(epsa_code, ind):
    if epsa_code in list(df.code):
        val = df[(df.code==epsa_code)&(df.year==val_year)][ind].iloc[0]
        if np.isnan(val):
            val = 0.0
    else:
        val = 0.0
    return val

layers = [dict(
    sourcetype = 'geojson',
    source = layer,
    type = 'fill',   
    color = code_to_color[get_layer_code(layer)],
    opacity = 0.8
) for layer in sorted_epsa_jsons]

p_lats = []
p_lons = []
p_texts = []
p_colors = []

for code in list(points_json.keys()):
    epsa_p = points_json[code]
    n = len(epsa_p['lats'])
    p_lats += epsa_p['lats']
    p_lons += epsa_p['lons'] 
    
    clean_code = 'COSPELCAR' if code == 'El\xa0' or code =='Área' else  code.split('-')[0]
    clean_code2 = 'LA PORTEÑA' if clean_code == 'LA_PORTEÑA' else clean_code
    clean_code3 = 'COAPAS VINTO' if clean_code2 == 'COAPAS' else clean_code2
    value = str(df[(df.code == 'COOSPELCAR') & (df.year == 2017)]['ind8'].iloc[0]).replace('.',',') if clean_code3 == 'COSPELCAR' else str(get_display_value(clean_code3,'ind8')).replace('.',',')
    value2= str(df[(df.code == 'COOSPELCAR') & (df.year == 2017)]['ind9'].iloc[0]).replace('.',',') if clean_code3 == 'COSPELCAR' else str(get_display_value(clean_code3,'ind9')).replace('.',',')
    value3= str(df[(df.code == 'COOSPELCAR') & (df.year == 2017)]['ind10'].iloc[0]).replace('.',',') if clean_code3 == 'COSPELCAR' else str(get_display_value(clean_code3,'ind10')).replace('.',',')
    
    p_colors += [code_to_color[clean_code]] * n
    
    
    
    epsa_cat = 'B' if clean_code2 in ['EPSA', 'COSPELCAR'] else epsas_df[epsas_df.code == clean_code3].category.iloc[0]
    
    
    ap_opt = dict(A= '>90%', B='>90%', C='>80%', D='>70%')[epsa_cat]
    alc_opt = '>65%'
    micro_opt = '>80%' if epsa_cat == 'D' else '>90%'
    
#     display_text = f'<b>Agua Potable:</b> Cobertura del servicio de agua potable (parámetro óptimo: {ap_opt})<br><b>Alcantarillado:</b> Cobertura del servicio de alcantarillado sanitario (parámetro óptimo: {alc_opt})<br><b>Micromedición:</b> Cobertura de micromedición (parámetro óptimo: {micro_opt})'
    display_text = f'{clean_code3}<br>{val_label}: {value} {val_unit} (óptimo: {ap_opt})<br>{val_label2}: {value2} {val_unit} (óptimo: {alc_opt})<br>{val_label3}: {value3} {val_unit} (óptimo: {micro_opt})'
    p_texts += [display_text for k in range(n)]
    
scatter_dict = dict(
    type = 'scattermapbox',
    mode = 'markers',
    
    lat = p_lats, 
    lon = p_lons,
    text = p_texts,
    
    marker = dict(
        size = 50,
        opacity = 0,
#         color= 'rgb(248, 248, 255)'
        color = p_colors,
    ),
    showlegend = False,
    hoverinfo = 'text',
)

mapbox_access_token = 'pk.eyJ1Ijoic2VyZ2lvLWNodW1hY2Vyby1maSIsImEiOiJjamswOTUzeHkwMDk0M3dvNnJoeTByZGlpIn0.3mmjpLwDrIUcdJTowlCd1A'

layout = dict(
    title = '<b>Agua Potable:</b> Cobertura del servicio de agua potable<br><b>Alcantarillado:</b> Cobertura del servicio de alcantarillado sanitario<br><b>Micromedición:</b> Cobertura de micromedición',
    titlefont = dict(
        family='Helvetica LT Std',
        size=18
    ),
    autosize = False,
    width = 1000,
    height = 600,
    hovermode = 'closest',
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',

    mapbox = dict(
        accesstoken = mapbox_access_token,
        layers = layers,
        bearing = 0,
        center = dict( 
            lat = -17.610907366555434, 
            lon = -63.13396632812757,
        ),
        pitch = 0,
        zoom = 8,
        style = 'light'
    ) 
)

fig = dict(data=[scatter_dict], layout=layout)

iplot(fig, filename='mapa-coberturas-new', link_text='')


hide_toggle()
crdf = complete_reports_df
cmdf = complete_measurements_df

categories = list(crdf.category.sort_values().unique())
year = 2017

selected_vars = ['v22', 'v23', 'v24']
trace_names = {'v22': 'Total', 'v23': 'Abastecida', 'v24': 'Servida'}

v22_descr = f'<br>Población Total (Del Área de Servicio Autorizado) (hab.)'
v23_descr = f'<br>Población Abastecida(hab.)'
v24_descr = f'<br>Población Servida (hab.)'

def make_title(cat):
    base_title = f'<b>Población - 2017 - Categoría {cat}</b>'
    return base_title + v22_descr + v23_descr + v24_descr
    

data = []

for cat_i, category in enumerate(categories):
    visible = True if cat_i == 0 else False 
    for ind_i, indicator in enumerate(selected_vars):
        
        fdf = crdf[(crdf.category == category) & (crdf.year == year)]
        filtered_df = fdf.sort_values(indicator)
                    
        trace = go.Bar(
            x=filtered_df.code,
            y=filtered_df[indicator],
            name=trace_names[indicator],
            text=['{:,.0f}'.format(y).replace(',','_').replace('.',',').replace('_','.') + ' hab.' for y in filtered_df[indicator]],
            opacity=0.8,
            hoverinfo='name+text+x',
            visible=visible,
            textfont=dict(
                color='black',
            )
        )

        data.append(trace)
            
def get_visible_list(category):
    category_map = dict(A=0, B=1, C=2, D=3)    
    base_array = [False for i in range(4)]
    base_array[category_map[category]] = True
    fat_array = [[x] * 3 for x in base_array]
    return [x for l in fat_array for x in l]

def make_x_axis():
    return dict(
        title=f'EPSA',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        )
    )

def make_y_axis():
    return dict(
        title='habitantes',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        ),
        tickformat=",.",
        tickangle=-40,
    )

updatemenus = [dict(
    type='buttons',
    showactive=True,
    active=0,
    xanchor='left',
    yanchor='top',
    direction='right',
    x= 0.3,
    y= -0.5,
    buttons = [dict(
        label=f'Categoría {cat}',
        method='update', # modify both data and layout
        args = [
            {'visible': get_visible_list(cat)}, # data modification
            # layout modification
            dict(
                title= make_title(cat),
#                 xaxis= make_x_axis(cat),
#                 yaxis= make_y_axis(ind),
#                 shapes= make_shapes(ind),
            ),
        ],
    ) for cat in categories]
)]

layout = go.Layout(
    title= make_title('A'),
    titlefont= dict(
        family='Helvetica LT Std',
        size=18,
    ),
    xaxis= make_x_axis(),
    yaxis= make_y_axis(),
    updatemenus = updatemenus,
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, link_text='')

hide_toggle()
crdf = complete_reports_df[complete_reports_df.state == 'SC']
cmdf = complete_measurements_df[complete_measurements_df.state == 'SC']

categories = list(crdf.category.sort_values().unique())
year = 2017

selected_vars = ['v22', 'v23', 'v24']
trace_names = {'v22': 'Total', 'v23': 'Abastecida', 'v24': 'Servida'}

v22_descr = f'<br>Población Total (Del Área de Servicio Autorizado) (hab.)'
v23_descr = f'<br>Población Abastecida(hab.)'
v24_descr = f'<br>Población Servida (hab.)'

def make_title(cat):
    base_title = f'<b>Población - Santa Cruz - 2017 - Categoría {cat}</b>'
    return base_title + v22_descr + v23_descr + v24_descr
    

data = []

for cat_i, category in enumerate(categories):
    visible = True if cat_i == 0 else False 
    for ind_i, indicator in enumerate(selected_vars):
        
        fdf = crdf[(crdf.category == category) & (crdf.year == year)]
        filtered_df = fdf.sort_values(indicator)
                    
        trace = go.Bar(
            x=filtered_df.code,
            y=filtered_df[indicator],
            name=trace_names[indicator],
            text=['{:,.0f}'.format(y).replace(',','_').replace('.',',').replace('_','.') + ' hab.' for y in filtered_df[indicator]],
            opacity=0.8,
            hoverinfo='name+text+x',
            visible=visible,
            textfont=dict(
                color='black',
            )
        )

        data.append(trace)
            
def get_visible_list(category):
    category_map = dict(A=0, B=1, C=2, D=3)    
    base_array = [False for i in range(4)]
    base_array[category_map[category]] = True
    fat_array = [[x] * 3 for x in base_array]
    return [x for l in fat_array for x in l]

def make_x_axis():
    return dict(
        title=f'EPSA',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        )
    )

def make_y_axis():
    return dict(
        title='habitantes',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        ),
        tickformat=",.",
        tickangle=-40,
    )

updatemenus = [dict(
    type='buttons',
    showactive=True,
    active=0,
    xanchor='left',
    yanchor='top',
    direction='right',
    x= 0.3,
    y= -0.5,
    buttons = [dict(
        label=f'Categoría {cat}',
        method='update', # modify both data and layout
        args = [
            {'visible': get_visible_list(cat)}, # data modification
            # layout modification
            dict(
                title= make_title(cat),
#                 xaxis= make_x_axis(cat),
#                 yaxis= make_y_axis(ind),
#                 shapes= make_shapes(ind),
            ),
        ],
    ) for cat in categories]
)]

layout = go.Layout(
    title= make_title('A'),
    titlefont= dict(
        family='Helvetica LT Std',
        size=18,
    ),
    xaxis= make_x_axis(),
    yaxis= make_y_axis(),
    updatemenus = updatemenus,
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, link_text='')

hide_toggle()
years = list(crdf.year.sort_values().unique())
cdf = rdf
epsa_codes = list(cdf.code.unique())


categories = ['A', 'B', 'C', 'D']
variable = 'v22'

var_name = variables_df[variables_df.var_id == int(variable[1:])].name.iloc[0]
var_unit = variables_df[variables_df.var_id == int(variable[1:])].unit.iloc[0]

base_title = '<b>Tendencias en Población</b>'

def get_label(var):
    return dict(
        v22='Población Total',
        v23='Población Abastecida',
        v24='Población Servida'
    )[var]

data = [] 

def show_diff(vals):
    
#     vals0 = vals[:-1]
    vals1 = vals[1:]
    return [0] + [y-list(vals)[0] for y in vals1]

def get_var_list(code):
    fdf = rdf[rdf.code == code]
    res = []
    for year in years:
        x = fdf[fdf.year == year].v22
        res.append(math.nan if len(list(x)) == 0 else x.iloc[0])
    return res

def get_diff_list(code):
    var_list = [0 if math.isnan(x) else x for x in get_var_list(code)]
    res = [0]
    for i in range(3):
        res.append(var_list[i+1] - var_list[i])
    return res

def is_of_cat(code, cat):
    return code in list(epsas_df[epsas_df.category == cat].code)

for i, cat in enumerate(categories):
    visible= True if i == 0 else False
    for code in [code for code in epsa_codes if is_of_cat(code, cat)]:
        var_list = get_var_list(code)
        diff_list = get_diff_list(code)
        data.append(go.Scatter(
            x= years,
            y= var_list,
            mode='lines+markers',
            name=code,
            visible= visible,
            text=['{:,.0f}'.format(x).replace(',','_').replace('.',',').replace('_','.') + f' hab.<br>diferencia al año previo: ' + '{:,.0f}'.format(y).replace(',','_').replace('.',',').replace('_','.') + ' hab.'  for x,y in zip(var_list, diff_list)],
            textfont=dict(
                family='Helvetica LT Std',
                size=18,
                color='black',
            ),
            hoverinfo='name+text+x',
        ))

def cat_number(cat):
    return len(list(epsas_df[(epsas_df.category == cat) & (epsas_df.state == 'SC')].code))
        
num_a = len([code for code in epsa_codes if is_of_cat(code, 'A')])
num_b = len([code for code in epsa_codes if is_of_cat(code, 'B')])
num_c = len([code for code in epsa_codes if is_of_cat(code, 'C')])
num_d = len([code for code in epsa_codes if is_of_cat(code, 'D')]) 

def get_visible_list(cat):
    cat_map = {cat: i for i,cat in enumerate(categories)}
    base_array = [False for i in range(len(categories))]
    base_array[cat_map[cat]] = True
    fat_array = [[base_array[cat_map[cat]]] * dict(A=num_a,B=num_b,C=num_c,D=num_d)[cat] for cat in categories]
    return [x for l in fat_array for x in l]


def make_title(cat): 
    return base_title + f'<br>{var_name} ({var_unit})'

        
updatemenus = [dict(
    type='buttons',
    showactive=True,
    active=0,
    xanchor='left',
    yanchor='top',
    direction='right',
    x= 0.3,
    y= -0.3,
    buttons = [dict(
        label=f'Categoría {cat}',
        method='update', # modify both data and layout
        args = [
            {'visible': get_visible_list(cat)}, # data modification
            # layout modification
            dict(
                title= make_title(cat),
#                 xaxis= make_x_axis(cat),
#                 yaxis= make_y_axis(ind),
#                 shapes= make_shapes(ind),
            ),
        ],
    ) for cat in categories]
)]

def make_y_axis():
    return dict(
        title='Población Total (hab.)',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        ),
        tickformat=",.",
        tickangle=-40,
    )

def make_x_axis():
    return dict(
        title='Año',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f',
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        ),
        range=[2013.8, 2017.2],
        autorange= False,
        dtick=1,
    )
    
layout = dict(
    title= make_title('v22'),
    titlefont=dict(
        family='Helvetica LT Std',
        size=18,
    ),
    xaxis= make_x_axis(),
    yaxis= make_y_axis(),
    
    updatemenus= updatemenus,
    hovermode='closest',
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, link_text='')

hide_toggle()
years = list(crdf.year.sort_values().unique())
cdf = rdf[rdf.state == 'SC']
epsa_codes = list(cdf.code.unique())


categories = ['A', 'B', 'C', 'D']
variable = 'v22'

var_name = variables_df[variables_df.var_id == int(variable[1:])].name.iloc[0]
var_unit = variables_df[variables_df.var_id == int(variable[1:])].unit.iloc[0]

base_title = '<b>Tendencias en Población - Santa Cruz</b>'

def get_label(var):
    return dict(
        v22='Población Total',
        v23='Población Abastecida',
        v24='Población Servida'
    )[var]

data = [] 

def show_diff(vals):
    
#     vals0 = vals[:-1]
    vals1 = vals[1:]
    return [0] + [y-list(vals)[0] for y in vals1]

def get_var_list(code):
    fdf = rdf[rdf.code == code]
    res = []
    for year in years:
        x = fdf[fdf.year == year].v22
        res.append(math.nan if len(list(x)) == 0 else x.iloc[0])
    return res

def get_diff_list(code):
    var_list = [0 if math.isnan(x) else x for x in get_var_list(code)]
    res = [0]
    for i in range(3):
        res.append(var_list[i+1] - var_list[i])
    return res

def is_of_cat(code, cat):
    return code in list(epsas_df[epsas_df.category == cat].code)

for i, cat in enumerate(categories):
    visible= True if i == 0 else False
    for code in [code for code in epsa_codes if is_of_cat(code, cat)]:
        var_list = get_var_list(code)
        diff_list = get_diff_list(code)
        data.append(go.Scatter(
            x= years,
            y= var_list,
            mode='lines+markers',
            name=code,
            visible= visible,
            text=['{:,.0f}'.format(x).replace(',','_').replace('.',',').replace('_','.') + f' hab.<br>diferencia al año previo: ' + '{:,.0f}'.format(y).replace(',','_').replace('.',',').replace('_','.') + ' hab.'  for x,y in zip(var_list, diff_list)],
            textfont=dict(
                family='Helvetica LT Std',
                size=18,
                color='black',
            ),
            hoverinfo='name+text+x',
        ))

def cat_number(cat):
    return len(list(epsas_df[(epsas_df.category == cat) & (epsas_df.state == 'SC')].code))
        
num_a = len([code for code in epsa_codes if is_of_cat(code, 'A')])
num_b = len([code for code in epsa_codes if is_of_cat(code, 'B')])
num_c = len([code for code in epsa_codes if is_of_cat(code, 'C')])
num_d = len([code for code in epsa_codes if is_of_cat(code, 'D')]) 

def get_visible_list(cat):
    cat_map = {cat: i for i,cat in enumerate(categories)}
    base_array = [False for i in range(len(categories))]
    base_array[cat_map[cat]] = True
    fat_array = [[base_array[cat_map[cat]]] * dict(A=num_a,B=num_b,C=num_c,D=num_d)[cat] for cat in categories]
    return [x for l in fat_array for x in l]


def make_title(cat): 
    return base_title + f'<br>{var_name} ({var_unit})'

        
updatemenus = [dict(
    type='buttons',
    showactive=True,
    active=0,
    xanchor='left',
    yanchor='top',
    direction='right',
    x= 0.3,
    y= -0.3,
    buttons = [dict(
        label=f'Categoría {cat}',
        method='update', # modify both data and layout
        args = [
            {'visible': get_visible_list(cat)}, # data modification
            # layout modification
            dict(
                title= make_title(cat),
#                 xaxis= make_x_axis(cat),
#                 yaxis= make_y_axis(ind),
#                 shapes= make_shapes(ind),
            ),
        ],
    ) for cat in categories]
)]

def make_y_axis():
    return dict(
        title='Población Total (hab.)',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        ),
        tickformat=",.",
        tickangle=-40,
    )

def make_x_axis():
    return dict(
        title='Año',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f',
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        ),
        range=[2013.8, 2017.2],
        autorange= False,
        dtick=1,
    )
    
layout = dict(
    title= make_title('v22'),
    titlefont=dict(
        family='Helvetica LT Std',
        size=18,
    ),
    xaxis= make_x_axis(),
    yaxis= make_y_axis(),
    
    updatemenus= updatemenus,
    hovermode='closest',
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, link_text='')

hide_toggle()
crdf = complete_reports_df
cmdf = complete_measurements_df

categories = list(crdf.category.sort_values().unique())
selected_year = 2017

selected_vars = ['v17', 'v18']
default_cat = 'A'

ydf = crdf[crdf.year == selected_year]

vdf = variables_df
v1_name = vdf[vdf.var_id == int(selected_vars[0][1:])].name.iloc[0]
v1_unit = vdf[vdf.var_id == int(selected_vars[0][1:])].unit.iloc[0]
v2_name = vdf[vdf.var_id == int(selected_vars[1][1:])].name.iloc[0]
v2_unit = vdf[vdf.var_id == int(selected_vars[1][1:])].unit.iloc[0]

base_title = '<b>Conexiones de Agua Potable y Alcantarillado</b>'
v1_description = f'<br><b>Agua Potable:</b> {v1_name} ({v1_unit})'
v2_description = f'<br><b>Alcantarillado:</b> {v2_name} ({v2_unit})'

data = []
    

for i, category in enumerate(categories):
    fdf = ydf[ydf.category == category].fillna(0)
    total_col = fdf[selected_vars[0]] + fdf[selected_vars[1]]
    fdf = fdf.assign(total=total_col).sort_values('total')
    visible = True if i == 0 else False
    
    percentages = fdf[selected_vars[1]]/fdf[selected_vars[0]]

    data.append(go.Bar(
        x=list(fdf.code),
        y=list(fdf[selected_vars[0]]),
        text= ['{:,.0f}'.format(val).replace(',','_').replace('.',',').replace('_','.') for val in list(fdf[selected_vars[0]])],
        textfont = dict(
            family='Helvetica LT Std',
            size=18,
            color='black',
        ),
        opacity=0.8,  
        textposition='auto',
        hoverinfo='text+x+name',
        visible=visible,
        name='Agua Potable',
    ))
    
    data.append(go.Bar(
        x=list(fdf.code),
        y=list(fdf[selected_vars[1]]),
        text= ['{:,.0f}'.format(val).replace(',','_').replace('.',',').replace('_','.')  for val in  fdf[selected_vars[1]]],
        textfont = dict(
            family='Helvetica LT Std',
            size=18,
            color='black',
        ),
        opacity=0.8,  
        textposition='auto',
        hoverinfo='text+x+name',
        visible=visible,
        name='Alcantarillado',
    ))

def get_visible_list(category):
    category_map = dict(A=0, B=1, C=2, D=3)
    base_array = [False for i in range(4)]
    base_array[category_map[category]] = True
    fat_array = [[x] * 2 for x in base_array]
    return [x for l in fat_array for x in l]

def make_x_axis(category):
    return dict(
        title=f'EPSAs Categoría {category}',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        ),
    )

def make_y_axis():
    return dict(
        title='Número de Conexiones',
        titlefont = dict(
            family= 'Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        ),
        tickformat=",.",
        tickangle=-40,
    )

updatemenus = [dict(
    type= 'buttons',
    showactive=True,
    active=0,
    xanchor='left',
    yanchor='top',
    direction='right',
    x= 0.25,
    y= -0.3,
    buttons = [dict(
        label= f'Categoría {cat}',
        method='update',
        args = [
            {'visible': get_visible_list(cat)},
            dict(
                title= f'Presión del Servicio de Agua Potable - Categoría {cat}',
                xaxis= make_x_axis(cat),
            ),
        ],
    ) for cat in categories]
)]

layout = go.Layout(
    title= base_title + '<b> - Categoría A</b>' + v1_description + v2_description,
    titlefont=dict(
        family='Helveticat LT Std',
        size=18,
    ),
    xaxis= make_x_axis('A'),
    yaxis= make_y_axis(),
    updatemenus=updatemenus,
    barmode='stack',
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, link_text='')

hide_toggle()
crdf = complete_reports_df[complete_reports_df.state == 'SC']
cmdf = complete_measurements_df

categories = list(crdf.category.sort_values().unique())
selected_year = 2017

selected_vars = ['v17', 'v18']
default_cat = 'A'

ydf = crdf[crdf.year == selected_year]

vdf = variables_df
v1_name = vdf[vdf.var_id == int(selected_vars[0][1:])].name.iloc[0]
v1_unit = vdf[vdf.var_id == int(selected_vars[0][1:])].unit.iloc[0]
v2_name = vdf[vdf.var_id == int(selected_vars[1][1:])].name.iloc[0]
v2_unit = vdf[vdf.var_id == int(selected_vars[1][1:])].unit.iloc[0]

base_title = '<b>Conexiones de Agua Potable y Alcantarillado</b>'
v1_description = f'<br><b>Agua Potable:</b> {v1_name} ({v1_unit})'
v2_description = f'<br><b>Alcantarillado:</b> {v2_name} ({v2_unit})'

data = []
    

for i, category in enumerate(categories):
    fdf = ydf[ydf.category == category].fillna(0)
    total_col = fdf[selected_vars[0]] + fdf[selected_vars[1]]
    fdf = fdf.assign(total=total_col).sort_values('total')
    visible = True if i == 0 else False
    
    percentages = fdf[selected_vars[1]]/fdf[selected_vars[0]]

    data.append(go.Bar(
        x=list(fdf.code),
        y=list(fdf[selected_vars[0]]),
        text= ['{:,.0f}'.format(val).replace(',','_').replace('.',',').replace('_','.') for val in list(fdf[selected_vars[0]])],
        textfont = dict(
            family='Helvetica LT Std',
            size=18,
            color='black',
        ),
        opacity=0.8,  
        textposition='auto',
        hoverinfo='text+x+name',
        visible=visible,
        name='Agua Potable',
    ))
    
    data.append(go.Bar(
        x=list(fdf.code),
        y=list(fdf[selected_vars[1]]),
        text= ['{:,.0f}'.format(val).replace(',','_').replace('.',',').replace('_','.')  for val in  fdf[selected_vars[1]]],
        textfont = dict(
            family='Helvetica LT Std',
            size=18,
            color='black',
        ),
        opacity=0.8,  
        textposition='auto',
        hoverinfo='text+x+name',
        visible=visible,
        name='Alcantarillado',
    ))

def get_visible_list(category):
    category_map = dict(A=0, B=1, C=2, D=3)
    base_array = [False for i in range(4)]
    base_array[category_map[category]] = True
    fat_array = [[x] * 2 for x in base_array]
    return [x for l in fat_array for x in l]

def make_x_axis(category):
    return dict(
        title=f'EPSAs Categoría {category}',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        ),
    )

def make_y_axis():
    return dict(
        title='Número de Conexiones',
        titlefont = dict(
            family= 'Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        ),
        tickformat=",.",
        tickangle=-40,
    )

updatemenus = [dict(
    type= 'buttons',
    showactive=True,
    active=0,
    xanchor='left',
    yanchor='top',
    direction='right',
    x= 0.25,
    y= -0.3,
    buttons = [dict(
        label= f'Categoría {cat}',
        method='update',
        args = [
            {'visible': get_visible_list(cat)},
            dict(
                title= f'Presión del Servicio de Agua Potable - Categoría {cat}',
                xaxis= make_x_axis(cat),
            ),
        ],
    ) for cat in categories]
)]

layout = go.Layout(
    title= base_title + '<b> - Categoría A</b>' + v1_description + v2_description,
    titlefont=dict(
        family='Helveticat LT Std',
        size=18,
    ),
    xaxis= make_x_axis('A'),
    yaxis= make_y_axis(),
    updatemenus=updatemenus,
    barmode='stack',
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, link_text='')

hide_toggle()
crdf = complete_reports_df

category = 'A'
state = 'SC'

variables = ['v1', 'v2', 'v3']
states = ['LP', 'SC', 'CO', 'PO', 'BE', 'TA', 'CH', 'PA', 'OR']

state_name_map = {
    'LP': 'La Paz',
    'SC': 'Santa Cruz',
    'CO': 'Cochabamba',
    'PO': 'Potosí',
    'BE': 'Beni',
    'TA': 'Tarija',
    'CH': 'Chuquisaca',
    'PA': 'Pando',
    'OR': 'Oruro', 
}

def get_domain(var):
    if var == 'v1':
        return {"x": [0, .30]}
    if var == 'v2':
        return {"x": [.32, .62]}
    else:
        return {"x": [.64, .94]}
        
def get_visible_list(state):
    state_map = {'LP': 0, 'SC': 1, 'CO': 2, 
                 'PO': 3, 'BE': 4, 'TA': 5,
                 'CH': 6, 'PA': 7, 'OR': 8}
    
    offset = state_map[state]
    
    base_array = [0 for i in range(9)]
    base_array[offset] = 1
    
    big_true = [True for i in range(3)]
    big_false = [False for i in range(3)]
    
    nested_array = [big_true if x == 1 else big_false for x in base_array]
    
    return [x for l in nested_array for x in l]

data = []

for state in states:
    filtered_df = crdf[(crdf.state == state) & (crdf.year == 2017)]
    visible = True if state == 'LP' else False
    for var in variables:
        data.append(dict(
            values=list(filtered_df[var]),
            labels=list(filtered_df.code),
            type='pie',
            hole=.4,
            textposition='inside',
            name=var,
            domain=get_domain(var),
            hoverinfo='label+value+percent+name',
            visible=visible,
            textfont=dict(
                color='black',
            )
        ))


var1 = variables_df[variables_df.var_id == 1]
var2 = variables_df[variables_df.var_id == 2]
var3 = variables_df[variables_df.var_id == 3]

v1name = 'Volumen de agua cruda extraído en fuente superficial'
v2name = 'Volumen de agua cruda extraído en fuente subterránea'
v3name = 'Volumen de agua potabilizada (Planta de tratamiento y/o tanque de desinfección)'

v1unit = var1.unit.iloc[0]
v2unit = var2.unit.iloc[0]
v3unit = var3.unit.iloc[0]

def make_annotations(state):
    sdf = crdf[(crdf.state == state) & (crdf.year == 2017)]
    return [dict(
        font=dict(size=20),
        showarrow=False,
        text='V1',
        x=0.13,
        y=0.5,
    ), dict(
        font=dict(size=20),
        showarrow=False,
        text='V2',
        x=0.47,
        y=0.5,
    ), dict(
        font=dict(size=20),
        showarrow=False,
        text='V3',
        x=0.81,
        y=0.5,
    ), dict(
        font=dict(size=20),
        showarrow=False,
        text='Fuentes<br>Superficiales',
        x=0.07,
        y=0,
    ), dict(
        font=dict(size=20),
        showarrow=False,
        text='Fuentes<br>Subterráneas',
        x=0.48,
        y=0,
    ), dict(
        font=dict(size=20),
        showarrow=False,
        text='Agua<br>Potabilizada',
        x=0.87,
        y=0,
    ), dict(
        font=dict(size=20),
        showarrow=False,
        text='{:,.0f}'.format(sdf.v1.sum()).replace(',','.'),
        x=0.05,
        y=-0.1,
    ), dict(
        font=dict(size=20),
        showarrow=False,
        text='{:,.0f}'.format(sdf.v2.sum()).replace(',','.'),
        x=0.45,
        y=-0.1,
    ), dict(
        font=dict(size=20),
        showarrow=False,
        text='{:,.0f}'.format(sdf.v3.sum()).replace(',','.'),
        x=0.9,
        y=-0.1,
    ), dict(
        font=dict(size=20),
        showarrow=False,
        text='Total:',
        x=-0.1,
        y=-0.1,
    ),]

updatemenus = [dict(
    active=0,
    xanchor='left',
    yanchor='top',
    direction='up',
    x= .4,
    y= -0.15,
    buttons = [dict(
        label=f'{state_name_map[state]}',
        method='update', # modify both data and layout
        args = [
            {'visible': get_visible_list(state)}, # data modification
            # layout modification
            dict(
                annotations=make_annotations(state)
#                 title= make_title(ind),
#                 xaxis= make_x_axis(cat),
#                 yaxis= make_y_axis(ind),
#                 shapes= make_shapes(ind),
            ),
        ],
    ) for state in states]
)]


layout = dict(
    updatemenus = updatemenus,
    title = f'<b>Variables Volumen 1 - 2017</b><br><b>V1</b>: {v1name} (m3/año)<br><b>V2</b>: {v2name} (m3/año)<br><b>V3</b>: {v3name} (m3/año)',
    titlefont = dict(
        family='Helvetica LT Std',
        size=18
    ),
    annotations=make_annotations('LP'), 
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, link_text='')

hide_toggle()
crdf = complete_reports_df

category = 'A'
state = 'SC'

variables = ['v3', 'v5', 'v6']
states = ['LP', 'SC', 'CO', 'PO', 'BE', 'TA', 'CH', 'PA', 'OR']

state_name_map = {
    'LP': 'La Paz',
    'SC': 'Santa Cruz',
    'CO': 'Cochabamba',
    'PO': 'Potosí',
    'BE': 'Beni',
    'TA': 'Tarija',
    'CH': 'Chuquisaca',
    'PA': 'Pando',
    'OR': 'Oruro', 
}

def get_domain(var):
    if var == 'v3':
        return {"x": [0, .30]}
    if var == 'v5':
        return {"x": [.32, .62]}
    else:
        return {"x": [.64, .94]}
        
def get_visible_list(state):
    state_map = {'LP': 0, 'SC': 1, 'CO': 2, 
                 'PO': 3, 'BE': 4, 'TA': 5,
                 'CH': 5, 'PA': 7, 'OR': 8}
    
    offset = state_map[state]
    
    base_array = [0 for i in range(9)]
    base_array[offset] = 1
    
    big_true = [True for i in range(3)]
    big_false = [False for i in range(3)]
    
    nested_array = [big_true if x == 1 else big_false for x in base_array]
    
    return [x for l in nested_array for x in l]

data = []

for state in states:
    filtered_df = crdf[(crdf.state == state) & (crdf.year == 2017)]
    visible = True if state == 'LP' else False
    for var in variables:
        data.append(dict(
            values=list(filtered_df[var]),
            labels=list(filtered_df.code),
            type='pie',
            hole=.4,
            textposition='inside',
            name=var,
            domain=get_domain(var),
            hoverinfo='label+value+percent+name',
            visible=visible,
            textfont=dict(
                color='black',
            )
        ))


var3 = variables_df[variables_df.var_id == 3]
var5 = variables_df[variables_df.var_id == 5]
var6 = variables_df[variables_df.var_id == 6]

v3name = 'Volumen de agua potabilizada (Planta de tratamiento y/o tanque de desinfección)'
v5name = var5.name.iloc[0]
v6name = var6.name.iloc[0]

v3unit = var3.unit.iloc[0]
v5unit = var5.unit.iloc[0]
v6unit = var6.unit.iloc[0]

def make_annotations(state):
    sdf = crdf[(crdf.state == state) & (crdf.year == 2017)]
    return [dict(
        font=dict(size=20),
        showarrow=False,
        text='V3',
        x=0.13,
        y=0.5,
    ), dict(
        font=dict(size=20),
        showarrow=False,
        text='V5',
        x=0.47,
        y=0.5,
    ), dict(
        font=dict(size=20),
        showarrow=False,
        text='V6',
        x=0.81,
        y=0.5,
    ), dict(
        font=dict(size=20),
        showarrow=False,
        text='Agua<br>Potabilizada',
        x=0.07,
        y=0,
    ), dict(
        font=dict(size=20),
        showarrow=False,
        text='AP<br>Facturado',
        x=0.48,
        y=0,
    ), dict(
        font=dict(size=20),
        showarrow=False,
        text='AR<br>Tratada',
        x=0.87,
        y=0,
    ), dict(
        font=dict(size=20),
        showarrow=False,
        text='{:,.0f}'.format(sdf.v3.sum()).replace(',','.'),
        x=0.05,
        y=-0.1,
    ), dict(
        font=dict(size=20),
        showarrow=False,
        text='{:,.0f}'.format(sdf.v5.sum()).replace(',','.'),
        x=0.45,
        y=-0.1,
    ), dict(
        font=dict(size=20),
        showarrow=False,
        text='{:,.0f}'.format(sdf.v6.sum()).replace(',','.'),
        x=0.9,
        y=-0.1,
    ), dict(
        font=dict(size=20),
        showarrow=False,
        text='Total:',
        x=-0.1,
        y=-0.1,
    ),]
    

updatemenus = [dict(
    active=0,
    xanchor='left',
    yanchor='top',
    direction='up',
    x= .4,
    y= -0.15,
    buttons = [dict(
        label=f'{state_name_map[state]}',
        method='update', # modify both data and layout
        args = [
            {'visible': get_visible_list(state)}, # data modification
            # layout modification
            dict(
                annotations=make_annotations(state),
#                 title= make_title(ind),
#                 xaxis= make_x_axis(cat),
#                 yaxis= make_y_axis(ind),
#                 shapes= make_shapes(ind),
            ),
        ],
    ) for state in states]
)]


layout = dict(
    updatemenus = updatemenus,
    title = f'<b>Variables Volumen 2</b><br><b>V3</b>: {v3name}(m3/año)<br><b>V5</b>: {v5name}(m3/año)<br><b>V6</b>: {v6name}(m3/año)',
    titlefont = dict(size=15),
    annotations = make_annotations('LP'),
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, link_text='')

hide_toggle()
crdf = complete_reports_df
cmdf = complete_measurements_df

categories = list(crdf.category.sort_values().unique())
year = 2017

selected_vars = ['v3', 'v5']
trace_names = {'v3': 'Agua Potabilizada', 'v5': 'AP Facturado'}

data = []

for cat_i, category in enumerate(categories):
    visible = True if cat_i == 0 else False 
    for ind_i, indicator in enumerate(selected_vars):
        
        fdf = crdf[(crdf.category == category) & (crdf.year == year)]
        filtered_df = fdf.sort_values(indicator)
                    
        trace = go.Bar(
            x=filtered_df.code,
            y=filtered_df[indicator],
            name=trace_names[indicator],
            text=['{:,.0f}'.format(y).replace(',','_').replace('.',',').replace('_','.') + ' m3/año' for y in filtered_df[indicator]],
            opacity=0.8,
            hoverinfo='name+text+x',
            visible=visible,
            textfont=dict(
                color='black',
            )
        )

        data.append(trace)
            
def get_visible_list(category):
    category_map = dict(A=0, B=1, C=2, D=3)    
    base_array = [False for i in range(4)]
    base_array[category_map[category]] = True
    fat_array = [[x] * 2 for x in base_array]
    return [x for l in fat_array for x in l]

def make_x_axis():
    return dict(
        title=f'EPSA',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        )
    )

def make_y_axis():
    return dict(
        title=f'm3/año',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        ),
        tickformat=",.",
        tickangle=-40,
    )

updatemenus = [dict(
    type='buttons',
    showactive=True,
    active=0,
    xanchor='left',
    yanchor='top',
    direction='right',
    x= 0.3,
    y= -0.5,
    buttons = [dict(
        label=f'Categoría {cat}',
        method='update', # modify both data and layout
        args = [
            {'visible': get_visible_list(cat)}, # data modification
            # layout modification
            dict(
#                 title= make_title(ind),
#                 xaxis= make_x_axis(cat),
#                 yaxis= make_y_axis(ind),
#                 shapes= make_shapes(ind),
            ),
        ],
    ) for cat in categories]
)]

layout = go.Layout(
    title = f'<b>Comparación Volumen 2017: Agua Potabilizada vs. AP Facturado</b><br><b>Agua Potabilizada:</b> Volumen de Agua Potabilizada (m3/año)<br><b>AP Facturado:</b> Volumen de Agua Potable Facturado (m3/año)',
    titlefont= dict(
        family='Helvetica LT Std',
        size=18,
    ),
    xaxis= make_x_axis(),
    yaxis= make_y_axis(),
    updatemenus = updatemenus,
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, link_text='')

hide_toggle()
crdf = complete_reports_df[complete_reports_df.state == 'SC']
cmdf = complete_measurements_df

categories = list(crdf.category.sort_values().unique())
year = 2017

selected_vars = ['v3', 'v5']
trace_names = {'v3': 'Agua Potabilizada', 'v5': 'AP Facturado'}

data = []

for cat_i, category in enumerate(categories):
    visible = True if cat_i == 0 else False 
    for ind_i, indicator in enumerate(selected_vars):
        
        fdf = crdf[(crdf.category == category) & (crdf.year == year)]
        filtered_df = fdf.sort_values(indicator)
                    
        trace = go.Bar(
            x=filtered_df.code,
            y=filtered_df[indicator],
            name=trace_names[indicator],
            text=['{:,.0f}'.format(y).replace(',','_').replace('.',',').replace('_','.') + ' m3/año' for y in filtered_df[indicator]],
            opacity=0.8,
            hoverinfo='name+text+x',
            visible=visible,
            textfont=dict(
                color='black',
            )
        )

        data.append(trace)
            
def get_visible_list(category):
    category_map = dict(A=0, B=1, C=2, D=3)    
    base_array = [False for i in range(4)]
    base_array[category_map[category]] = True
    fat_array = [[x] * 2 for x in base_array]
    return [x for l in fat_array for x in l]

def make_x_axis():
    return dict(
        title=f'EPSA',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        )
    )

def make_y_axis():
    return dict(
        title=f'm3/año',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        ),
        tickformat=",.",
        tickangle=-40,
    )

updatemenus = [dict(
    type='buttons',
    showactive=True,
    active=0,
    xanchor='left',
    yanchor='top',
    direction='right',
    x= 0.3,
    y= -0.5,
    buttons = [dict(
        label=f'Categoría {cat}',
        method='update', # modify both data and layout
        args = [
            {'visible': get_visible_list(cat)}, # data modification
            # layout modification
            dict(
#                 title= make_title(ind),
#                 xaxis= make_x_axis(cat),
#                 yaxis= make_y_axis(ind),
#                 shapes= make_shapes(ind),
            ),
        ],
    ) for cat in categories]
)]

layout = go.Layout(
    title = f'<b>Comparación Volumen 2017: Agua Potabilizada vs. AP Facturado - Santa Cruz</b><br><b>Agua Potabilizada:</b> Volumen de Agua Potabilizada (m3/año)<br><b>AP Facturado:</b> Volumen de Agua Potable Facturado (m3/año)',
    titlefont= dict(
        family='Helvetica LT Std',
        size=18,
    ),
    xaxis= make_x_axis(),
    yaxis= make_y_axis(),
    updatemenus = updatemenus,
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, link_text='')

hide_toggle()
years = list(crdf.year.sort_values().unique())
cdf = rdf
epsa_codes = list(cdf.code.unique())


categories = ['A', 'B', 'C', 'D']
variable = 'v3'

var_name = variables_df[variables_df.var_id == int(variable[1:])].name.iloc[0][:33]
var_unit = variables_df[variables_df.var_id == int(variable[1:])].unit.iloc[0]

base_title = '<b>Tendencias en Volumen</b>'

data = [] 

def get_var_list(code):
    fdf = rdf[rdf.code == code]
    res = []
    for year in years:
        x = fdf[fdf.year == year][variable]
        res.append(math.nan if len(list(x)) == 0 else x.iloc[0])
    return res

def is_of_cat(code, cat):
    return code in list(edf[edf.category == cat].code)

for i, cat in enumerate(categories):
    visible= True if i == 0 else False
    for code in [code for code in epsa_codes if is_of_cat(code, cat)]:
        var_list = get_var_list(code)
        data.append(go.Scatter(
            x= years,
            y= var_list,
            mode='lines+markers',
            name=code,
            visible= visible,
            text=['{:,.0f}'.format(x).replace(',','_').replace('.',',').replace('_','.') + 'm3/año' for x in var_list],
            textfont=dict(
                family='Helvetica LT Std',
                size=18,
                color='black',
            ),
            hoverinfo='name+text+x',
        ))

def cat_number(cat):
    return len(list(epsas_df[(epsas_df.category == cat) & (epsas_df.state == 'SC')].code))
 
num_a = len([code for code in epsa_codes if is_of_cat(code, 'A')])
num_b = len([code for code in epsa_codes if is_of_cat(code, 'B')])
num_c = len([code for code in epsa_codes if is_of_cat(code, 'C')])
num_d = len([code for code in epsa_codes if is_of_cat(code, 'D')])

def get_visible_list(cat):
    cat_map = {cat: i for i,cat in enumerate(categories)}
    base_array = [False] * 4
    base_array[cat_map[cat]] = True
    fat_array = [[base_array[cat_map[cat]]] * dict(A=num_a, B=num_b, C=num_c, D=num_d)[cat] for cat in categories]
    return [x for l in fat_array for x in l]


def make_title(cat): 
    return base_title + f'<b> - Categoría {cat}</b>' + f'<br>Volumen de agua potabilizada (m3/año)'

        
updatemenus = [dict(
    type='buttons',
    showactive=True,
    active=0,
    xanchor='left',
    yanchor='top',
    direction='right',
    x= 0.3,
    y= -0.3,
    buttons = [dict(
        label=f'Categoría {cat}',
        method='update',
        args = [
            {'visible': get_visible_list(cat)},
            dict(
                title= make_title(cat),
            ),
        ],
    ) for cat in categories]
)]

def make_y_axis():
    return dict(
        title='Volumen AP Producido (m3/año)',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        ),
        tickformat=",.",
        tickangle=-40,
    )

def make_x_axis():
    return dict(
        title='Año',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f',
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        ),
        range=[2013.8, 2017.2],
        autorange= False,
        dtick=1,
    )
    
layout = dict(
    titlefont=dict(
        family='Helvetica LT Std',
        size=18,
    ),
    xaxis= make_x_axis(),
    yaxis= make_y_axis(),
    title= make_title('A'),
    updatemenus= updatemenus,
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, link_text='')

hide_toggle()
years = list(crdf.year.sort_values().unique())
cdf = rdf[rdf.state=='SC']
epsa_codes = list(cdf.code.unique())


categories = ['A', 'B', 'C', 'D']
variable = 'v3'

var_name = variables_df[variables_df.var_id == int(variable[1:])].name.iloc[0][:33]
var_unit = variables_df[variables_df.var_id == int(variable[1:])].unit.iloc[0]

base_title = '<b>Tendencias en Volumen - Santa Cruz</b>'

data = [] 

def get_var_list(code):
    fdf = rdf[rdf.code == code]
    res = []
    for year in years:
        x = fdf[fdf.year == year][variable]
        res.append(math.nan if len(list(x)) == 0 else x.iloc[0])
    return res

def is_of_cat(code, cat):
    return code in list(edf[edf.category == cat].code)

for i, cat in enumerate(categories):
    visible= True if i == 0 else False
    for code in [code for code in epsa_codes if is_of_cat(code, cat)]:
        var_list = get_var_list(code)
        data.append(go.Scatter(
            x= years,
            y= var_list,
            mode='lines+markers',
            name=code,
            visible= visible,
            text=['{:,.0f}'.format(x).replace(',','_').replace('.',',').replace('_','.') + 'm3/año' for x in var_list],
            textfont=dict(
                family='Helvetica LT Std',
                size=18,
                color='black',
            ),
            hoverinfo='name+text+x',
        ))

def cat_number(cat):
    return len(list(epsas_df[(epsas_df.category == cat) & (epsas_df.state == 'SC')].code))
 
num_a = len([code for code in epsa_codes if is_of_cat(code, 'A')])
num_b = len([code for code in epsa_codes if is_of_cat(code, 'B')])
num_c = len([code for code in epsa_codes if is_of_cat(code, 'C')])
num_d = len([code for code in epsa_codes if is_of_cat(code, 'D')])

def get_visible_list(cat):
    cat_map = {cat: i for i,cat in enumerate(categories)}
    base_array = [False] * 4
    base_array[cat_map[cat]] = True
    fat_array = [[base_array[cat_map[cat]]] * dict(A=num_a, B=num_b, C=num_c, D=num_d)[cat] for cat in categories]
    return [x for l in fat_array for x in l]


def make_title(cat): 
    return base_title + f'<b> - Categoría {cat}</b>' + f'<br>Volumen de agua potabilizada (m3/año)'

        
updatemenus = [dict(
    type='buttons',
    showactive=True,
    active=0,
    xanchor='left',
    yanchor='top',
    direction='right',
    x= 0.3,
    y= -0.3,
    buttons = [dict(
        label=f'Categoría {cat}',
        method='update',
        args = [
            {'visible': get_visible_list(cat)},
            dict(
                title= make_title(cat),
            ),
        ],
    ) for cat in categories]
)]

def make_y_axis():
    return dict(
        title='Volumen AP Producido (m3/año)',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        ),
        tickformat=",.",
        tickangle=-40,
    )

def make_x_axis():
    return dict(
        title='Año',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f',
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        ),
        range=[2013.8, 2017.2],
        autorange= False,
        dtick=1,
    )
    
layout = dict(
    titlefont=dict(
        family='Helvetica LT Std',
        size=18,
    ),
    xaxis= make_x_axis(),
    yaxis= make_y_axis(),
    title= make_title('A'),
    updatemenus= updatemenus,
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, link_text='')

hide_toggle()
crdf = complete_reports_df
cmdf = complete_measurements_df

colors = [
    '#1f77b4','#ff7f0e','#2ca02c',
    '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22','#17becf'
]

categories = list(crdf.category.sort_values().unique())
selected_year = 2017

selected_inds = ['ind17', 'ind18']
default_cat = 'A'

ydf = cmdf[cmdf.year == selected_year]

traces = []

for category in categories:
    fdf1 = ydf[ydf.category == category]
    fdf = fdf1[np.isfinite(fdf1[selected_inds[1]])].sort_values(selected_inds[1])

    trace0 = go.Bar(
        x=list(fdf.code),
        y=list(fdf[selected_inds[0]]),
        text= [f'{code}<br>' + str(val).replace('.',',') + get_ind_unit(selected_inds[1]) for code,val in zip(list(fdf.code),list(fdf[selected_inds[0]]))],
        opacity=0.8,  
        textposition='auto',
        hoverinfo='text',
        visible=True,
        name= f'{category} - ANC en Prod.',
        textfont=dict(
            color='black',
        )
    )

    trace1 = go.Bar(
        x=list(fdf.code),
        y=list(fdf[selected_inds[1]]),
        xaxis='x2',
        yaxis='y1',
        text= [f'{code}<br>' + str(val).replace('.',',') + get_ind_unit(selected_inds[0]) for code,val in zip(list(fdf.code),list(fdf[selected_inds[1]]))],
        opacity=0.8,  
        textposition='auto',
        hoverinfo='text',
        visible=True,
        name= f'{category} - ANC en Red',
        textfont=dict(
            color='black',
        )
    )
    
    traces += [trace0, trace1]

from plotly import tools

def get_visible_list(category):
    category_map = dict(A=0, B=1, C=2, D=3)
    base_array = [False for i in range(4)]
    base_array[category_map[category]] = True
    fat_array = [[x] * 2 for x in base_array]
    return [x for l in fat_array for x in l]


def make_y_axis1():
    return dict(
        title='ANC en la Red (%)',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        ),
        dtick=5,
        autorange=True,
    )

def make_y_axis2():
    return dict(
        title=f'ANC en Prod. (%)',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        ),
        dtick=5,
        autorange=False,
        range=[0,34],
    )


base_title = '<b>Agua no Contabilizada (ANC) - 2017</b>'
ind_descr1 = f'<br><b>ANC en Prod:</b> {get_ind_name(selected_inds[1])}'
ind_descr2 = f'<br><b>ANC en Red:</b> {get_ind_name(selected_inds[0])}'

def make_shapes(cat):
    cat_to_y1 = dict(A=5, B=10, C=10, D=15, TODAS=0)
    cat_to_y12 = dict(A=30, B=30, C=30, D=30, TODAS=0)
    code_to_y0 = dict(ind17=0, ind18=0)
    
    return [dict(
        type= 'rect',
        xref= 'paper', yref= yr,
        x0=0, y0=code_to_y0[ind_code],
        x1=1, y1= cat_to_y1[cat] if yr == 'y2' else cat_to_y12[cat],
        fillcolor= colors[0],
        opacity= 0.2,
        line= {'width': 0,}
    ) for yr, ind_code in zip(['y', 'y2'], ['ind18', 'ind17'])]

def make_title(cat):
    cat_title = '' if cat == 'TODAS' else f'<b> - Categoría {cat}</b>'
    par_1 = '' if cat == 'TODAS' else f' (rango óptimo: {dict(A="<5%",B="<10%",C="<10%",D="<15%")[cat]})'
    par_2 = '' if cat == 'TODAS' else ' (rango_óptimo: <30%)' 
    return base_title + cat_title + ind_descr1 + par_1 + ind_descr2 + par_2

def make_annotations(cat):
    formulas = [dict(
        font=dict(size=15),
        showarrow=False,
        text='$ANCP=\\left(1−\\frac{\\text{Volumen de AP producida}}{\\text{Volumen extraído de fuentes}}\\right)\\times 100$',
        xref='paper', yref='paper',
        x=1.15, y=1.2,
    ), dict(
        font=dict(size=15),
        showarrow=False,
        text='$ANCR=\\left(1−\\frac{\\text{Volumen de AP facturado}}{\\text{Volumen de AP producido}}\\right)\\times 100$',
        xref='paper', yref='paper',
        x=1.15, y=1.1,
    )] if cat == 'TODAS' else []
    
    return [dict(
        yref="paper", 
        xref="paper", 
        y=1.15, 
        x=0, 
        text=make_title(cat), 
        showarrow=False, 
        font=dict(size=17)
    )] + formulas
     
updatemenus = [dict(
    type= 'buttons',
    showactive=True,
    active=0,
    xanchor='left',
    yanchor='top',
    direction='right',
    x= 0.25,
    y= -0.2,
    buttons = [dict(
        label= f'Todas',
        method='update',
        args = [
            {'visible': [True] * 8},
            dict(
#                 title= base_title,
                annotations= make_annotations('TODAS'),
                shapes=make_shapes('TODAS')
            ),
        ],
    )] + [dict(
        label= f'Categoría {cat}',
        method='update',
        args = [
            {'visible': get_visible_list(cat)},
            dict(
#                 title= make_title(cat),
                shapes= make_shapes(cat),
                annotations= make_annotations(cat),
            ),
        ],
    ) for cat in categories]
)]    

layout = go.Layout(
#     title= make_title('TODAS'),
    titlefont=dict(
        family='Helvetica LT Std',
        size=18,
    ),
    hovermode='closest',
    updatemenus=updatemenus,
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
    height=700,
    yaxis= make_y_axis1(),
    yaxis2=make_y_axis2(),
    shapes=make_shapes('TODAS'),
    annotations = make_annotations('TODAS'),
)




fig = tools.make_subplots(rows=2, cols=1, specs=[[{}], [{}]], shared_xaxes=True,
                          shared_yaxes=False, vertical_spacing=0.001, print_grid=False)

for i, trace in enumerate(traces):
    x = 2 if (i % 2) == 0 else 1 
    fig.append_trace(trace, x, 1)

fig['layout'].update(layout)
iplot(fig, link_text='')

hide_toggle()
crdf = complete_reports_df
cmdf = complete_measurements_df[complete_measurements_df.state == 'SC']

colors = [
    '#1f77b4','#ff7f0e','#2ca02c',
    '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22','#17becf'
]

categories = list(crdf.category.sort_values().unique())
selected_year = 2017

selected_inds = ['ind17', 'ind18']
default_cat = 'A'

ydf = cmdf[cmdf.year == selected_year]

traces = []

for category in categories:
    fdf1 = ydf[ydf.category == category]
    fdf = fdf1[np.isfinite(fdf1[selected_inds[1]])].sort_values(selected_inds[1])

    trace0 = go.Bar(
        x=list(fdf.code),
        y=list(fdf[selected_inds[0]]),
        text= [f'{code}<br>' + str(val).replace('.',',') + get_ind_unit(selected_inds[1]) for code,val in zip(list(fdf.code),list(fdf[selected_inds[0]]))],
        opacity=0.8,  
        textposition='auto',
        hoverinfo='text',
        visible=True,
        name= f'{category} - ANC en Prod.',
        textfont=dict(
            color='black',
        )
    )

    trace1 = go.Bar(
        x=list(fdf.code),
        y=list(fdf[selected_inds[1]]),
        xaxis='x2',
        yaxis='y1',
        text= [f'{code}<br>' + str(val).replace('.',',') + get_ind_unit(selected_inds[0]) for code,val in zip(list(fdf.code),list(fdf[selected_inds[1]]))],
        opacity=0.8,  
        textposition='auto',
        hoverinfo='text',
        visible=True,
        name= f'{category} - ANC en Red',
        textfont=dict(
            color='black',
        )
    )
    
    traces += [trace0, trace1]

from plotly import tools

def get_visible_list(category):
    category_map = dict(A=0, B=1, C=2, D=3)
    base_array = [False for i in range(4)]
    base_array[category_map[category]] = True
    fat_array = [[x] * 2 for x in base_array]
    return [x for l in fat_array for x in l]


def make_y_axis1():
    return dict(
        title='ANC en la Red (%)',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        ),
        dtick=5,
        autorange=True,
    )

def make_y_axis2():
    return dict(
        title=f'ANC en Prod. (%)',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        ),
        dtick=5,
        autorange=False,
        range=[0,34],
    )


base_title = '<b>Agua no Contabilizada (ANC) - Santa Cruz - 2017</b>'
ind_descr1 = f'<br><b>ANC en Prod:</b> {get_ind_name(selected_inds[1])}'
ind_descr2 = f'<br><b>ANC en Red:</b> {get_ind_name(selected_inds[0])}'

def make_shapes(cat):
    cat_to_y1 = dict(A=5, B=10, C=10, D=15, TODAS=0)
    cat_to_y12 = dict(A=30, B=30, C=30, D=30, TODAS=0)
    code_to_y0 = dict(ind17=0, ind18=0)
    
    return [dict(
        type= 'rect',
        xref= 'paper', yref= yr,
        x0=0, y0=code_to_y0[ind_code],
        x1=1, y1= cat_to_y1[cat] if yr == 'y2' else cat_to_y12[cat],
        fillcolor= colors[0],
        opacity= 0.2,
        line= {'width': 0,}
    ) for yr, ind_code in zip(['y', 'y2'], ['ind18', 'ind17'])]

def make_title(cat):
    cat_title = '' if cat == 'TODAS' else f'<b> - Categoría {cat}</b>'
    par_1 = '' if cat == 'TODAS' else f' (rango óptimo: {dict(A="<5%",B="<10%",C="<10%",D="<15%")[cat]})'
    par_2 = '' if cat == 'TODAS' else ' (rango óptimo: <30%)' 
    return base_title + cat_title + ind_descr1 + par_1 + ind_descr2 + par_2

def make_annotations(cat):
    formulas = [dict(
        font=dict(size=15),
        showarrow=False,
        text='$ANCP=\\left(1−\\frac{\\text{Volumen de AP producida}}{\\text{Volumen extraído de fuentes}}\\right)\\times 100$',
        xref='paper', yref='paper',
        x=1.15, y=1.2,
    ), dict(
        font=dict(size=15),
        showarrow=False,
        text='$ANCR=\\left(1−\\frac{\\text{Volumen de AP facturado}}{\\text{Volumen de AP producido}}\\right)\\times 100$',
        xref='paper', yref='paper',
        x=1.15, y=1.1,
    )] if cat == 'TODAS' else []
    
    return [dict(
        yref="paper", 
        xref="paper", 
        y=1.15, 
        x=0, 
        text=make_title(cat), 
        showarrow=False, 
        font=dict(size=17)
    )] + formulas
     
updatemenus = [dict(
    type= 'buttons',
    showactive=True,
    active=0,
    xanchor='left',
    yanchor='top',
    direction='right',
    x= 0.25,
    y= -0.2,
    buttons = [dict(
        label= f'Todas',
        method='update',
        args = [
            {'visible': [True] * 8},
            dict(
#                 title= base_title,
                annotations= make_annotations('TODAS'),
                shapes=make_shapes('TODAS'),
            ),
        ],
    )] + [dict(
        label= f'Categoría {cat}',
        method='update',
        args = [
            {'visible': get_visible_list(cat)},
            dict(
#                 title= make_title(cat),
                shapes= make_shapes(cat),
                annotations= make_annotations(cat),
            ),
        ],
    ) for cat in categories]
)]    

layout = go.Layout(
#     title= make_title('TODAS'),
    titlefont=dict(
        family='Helvetica LT Std',
        size=18,
    ),
    hovermode='closest',
    updatemenus=updatemenus,
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
    height=700,
    yaxis= make_y_axis1(),
    yaxis2=make_y_axis2(),
    shapes=make_shapes('TODAS'),
    annotations = make_annotations('TODAS'),
)




fig = tools.make_subplots(rows=2, cols=1, specs=[[{}], [{}]], shared_xaxes=True,
                          shared_yaxes=False, vertical_spacing=0.001, print_grid=False)

for i, trace in enumerate(traces):
    x = 2 if (i % 2) == 0 else 1 
    fig.append_trace(trace, x, 1)

fig['layout'].update(layout)
iplot(fig, link_text='')

hide_toggle()
crdf = complete_reports_df
cmdf = complete_measurements_df

categories = list(crdf.category.sort_values().unique()) # ['A', 'B', 'C', 'D']
selected_year = 2017

selected_ind = 'ind16'
default_cat = 'A'

ydf = cmdf[cmdf.year == selected_year]

base_title = '<b>Presión del Servicio de Agua Potable - Categoría A</b>'
ind_description = f'<br>{get_ind_name(selected_ind)} ({get_ind_unit(selected_ind)})'


data = []

for i, category in enumerate(categories):
    fdf1 = ydf[ydf.category == category]
    fdf = fdf1[np.isfinite(fdf1[selected_ind])].sort_values(selected_ind)
    visible = True if i == 0 else False
    data.append(go.Bar(
        x=list(fdf.code),
        y=list(fdf[selected_ind]),
        text= [f'{code}<br>' + str(val).replace('.',',') + get_ind_unit(selected_ind) for code,val in zip(list(fdf.code),list(fdf[selected_ind]))],
        opacity=0.8,  
        textposition='auto',
        hoverinfo='text',
        visible=visible,
        textfont=dict(
            color='black',
        )
    ))

def get_visible_list(category):
    category_map = dict(A=0, B=1, C=2, D=3)
    base_array = [False for i in range(4)]
    base_array[category_map[category]] = True
    return base_array

def make_x_axis(category):
    return dict(
        title=f'EPSAs Categoría {category}',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        )
    )

def make_y_axis():
    return dict(
        title=f'{get_ind_unit(selected_ind)}',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        )
    )

def make_title(cat):
    cat_title = '' if cat == 'TODAS' else f'<b> - Categoría {cat}</b>'
    cat_param = '' if cat == 'TODAS' else '(óptimo: >95%)'
    base_title = f'<b>Presión del Servicio de Agua Potable (PAP)</b>{cat_title}'
    ind_descr = f'<br>{get_ind_name(selected_ind)} {cat_param}'
    return base_title + ind_descr

def make_annotations(cat):
    formulas = [dict(
        font=dict(size=15),
        showarrow=False,
        text='$PAP = \\frac{\\text{Número de puntos con presión entre 13 a 70 mca}}{\\text{Número de puntos de muestreo de presión}} \\times 100$',
        xref='paper', yref='paper',
        x=1, y=1.2,
    )] if cat == 'TODAS' else []
    
    return [dict(
        yref="paper", 
        xref="paper", 
        y=1.15, 
        x=0 if cat == 'TODAS' else 0.5, 
        text=make_title(cat), 
        showarrow=False, 
        font=dict(size=17)
    )] + formulas

updatemenus = [dict(
    type= 'buttons',
    showactive=True,
    active=0,
    xanchor='left',
    yanchor='top',
    direction='right',
    x= 0.25,
    y= -0.3,
    buttons = [dict(
        label= f'Categoría {cat}',
        method='update', # modify both data and layout
        args = [
            {'visible': get_visible_list(cat)}, # data modification
            dict(
#                 title= make_title(cat),
                xaxis= make_x_axis(cat),
                annotations=make_annotations(cat),
            ),
        ],
    ) for cat in categories]
)]

def make_shapes():
    return [dict(
        type= 'rect',
        xref= 'paper', yref= 'y',
        x0=0, y0=95,
        x1=1, y1=100,
        fillcolor= colors[0],
        opacity= 0.2,
        line= {'width': 0,}
    )]    

layout = go.Layout(
#     title= make_title('A'),
    titlefont=dict(
        family='Helvetica LT Std',
        size=18,
    ),
    xaxis= make_x_axis('A'),
    yaxis= make_y_axis(),
    hovermode='closest',
    updatemenus=updatemenus,
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
    shapes=make_shapes(),
    annotations=make_annotations('TODAS'),
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, link_text='')

hide_toggle()
crdf = complete_reports_df
cmdf = complete_measurements_df[complete_measurements_df.state == 'SC']

categories = list(crdf.category.sort_values().unique()) # ['A', 'B', 'C', 'D']
selected_year = 2017

selected_ind = 'ind16'
default_cat = 'A'

ydf = cmdf[cmdf.year == selected_year]

base_title = '<b>Presión del Servicio de Agua Potable - Santa Cruz - Categoría A</b>'
ind_description = f'<br>{get_ind_name(selected_ind)} ({get_ind_unit(selected_ind)})'


data = []

for i, category in enumerate(categories):
    fdf1 = ydf[ydf.category == category]
    fdf = fdf1[np.isfinite(fdf1[selected_ind])].sort_values(selected_ind)
    visible = True if i == 0 else False
    data.append(go.Bar(
        x=list(fdf.code),
        y=list(fdf[selected_ind]),
        text= [f'{code}<br>' + str(val).replace('.',',') + get_ind_unit(selected_ind) for code,val in zip(list(fdf.code),list(fdf[selected_ind]))],
        opacity=0.8,  
        textposition='auto',
        hoverinfo='text',
        visible=visible,
        textfont=dict(
            color='black',
        )
    ))

def get_visible_list(category):
    category_map = dict(A=0, B=1, C=2, D=3)
    base_array = [False for i in range(4)]
    base_array[category_map[category]] = True
    return base_array

def make_x_axis(category):
    return dict(
        title=f'EPSAs Categoría {category}',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        )
    )

def make_y_axis():
    return dict(
        title=f'{get_ind_unit(selected_ind)}',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        )
    )

def make_title(cat):
    cat_title = '' if cat == 'TODAS' else f'<b> - Categoría {cat}</b>'
    cat_param = '' if cat == 'TODAS' else '(óptimo: >95%)'
    base_title = f'<b>Presión del Servicio de Agua Potable (PAP) - SC</b>{cat_title}'
    ind_descr = f'<br>{get_ind_name(selected_ind)} {cat_param}'
    return base_title + ind_descr

def make_annotations(cat):
    formulas = [dict(
        font=dict(size=15),
        showarrow=False,
        text='$PAP = \\frac{\\text{Número de puntos con presión entre 13 a 70 mca}}{\\text{Número de puntos de muestreo de presión}} \\times 100$',
        xref='paper', yref='paper',
        x=1, y=1.2,
    )] if cat == 'TODAS' else []
    
    return [dict(
        yref="paper", 
        xref="paper", 
        y=1.15, 
        x=0 if cat == 'TODAS' else 0.5, 
        text=make_title(cat), 
        showarrow=False, 
        font=dict(size=17)
    )] + formulas

updatemenus = [dict(
    type= 'buttons',
    showactive=True,
    active=0,
    xanchor='left',
    yanchor='top',
    direction='right',
    x= 0.25,
    y= -0.3,
    buttons = [dict(
        label= f'Categoría {cat}',
        method='update', # modify both data and layout
        args = [
            {'visible': get_visible_list(cat)}, # data modification
            dict(
#                 title= make_title(cat),
                xaxis= make_x_axis(cat),
                annotations=make_annotations(cat),
            ),
        ],
    ) for cat in categories]
)]

def make_shapes():
    return [dict(
        type= 'rect',
        xref= 'paper', yref= 'y',
        x0=0, y0=95,
        x1=1, y1=100,
        fillcolor= colors[0],
        opacity= 0.2,
        line= {'width': 0,}
    )]    

layout = go.Layout(
#     title= make_title('A'),
    titlefont=dict(
        family='Helvetica LT Std',
        size=18,
    ),
    xaxis= make_x_axis('A'),
    yaxis= make_y_axis(),
    hovermode='closest',
    updatemenus=updatemenus,
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
    shapes=make_shapes(),
    annotations=make_annotations('TODAS'),
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, link_text='')

hide_toggle()
years = list(rdf.year.sort_values().unique()) # [2014, 2015, 2016, 2017]
categories = list(rdf.category.sort_values().unique()) # ['A', 'B', 'C', 'D']

plane_dims = ['v22', 'v23']
size_dim = 'v22'

xi = int(plane_dims[0][1:]) # 17
yi = int(plane_dims[1][1:]) # 18

xname = vdf[vdf.var_id==xi].name.iloc[0] # 'Número total de conexiones de agua potable activas medidas y no medidas'
yname = vdf[vdf.var_id==yi].name.iloc[0] # 'Número total de conexiones de alcantarillado sanitario activas '
xunit = vdf[vdf.var_id==xi].unit.iloc[0] # 'conex.'
yunit = vdf[vdf.var_id==yi].unit.iloc[0] # 'conex.'

code_to_cat = {code: cat for code,cat in zip(epsas_df.code, epsas_df.category)}

cat_to_y = {cat: len(categories) - i for i,cat in enumerate(categories)}

grid_data = {}

for year in years:
    for category in categories:
        frdf = rdf[(rdf.year == year) & (rdf.category == category)]
                       
        percentages = [x/y * 100 for x,y in zip(frdf.v23, frdf.v22)]
        
        grid_data[f'{year}_{category}_text'] = [f'{code}<br>{"%.2f"%(p)}%<br>Población Abastecida: {a}<br>Población Total: {b}' for code,a,b,p in zip(frdf.code, frdf.v23, frdf.v22,percentages)]
        grid_data[f'{year}_{category}_x'] = percentages
        grid_data[f'{year}_{category}_y'] = [cat_to_y[code_to_cat[code]] for code in frdf.code]
        
def create_trace(year, category):
    return dict(
        x=[0.0 if math.isnan(x) else x for x in grid_data[f'{year}_{category}_x']],
        y=grid_data[f'{year}_{category}_y'],
        marker= dict(
            symbol='line-ns',
            size=25,
            opacity=0.7,
            line = dict(
              color = colors[dict(A=0, B=1, C=2, D=3)[category]],
              width = 2
            )
        ),
        mode= 'markers',
        text= grid_data[f'{year}_{category}_text'],
        name= 'Categoría: ' + category,
        hoverinfo = 'text',
    )

base_data = [create_trace(years[0], category) for category in categories]

def create_frame(year):
    frame_data = [create_trace(year, category) for category in categories]
    return dict(data=frame_data, name=str(year))

frames = [create_frame(year) for year in years]

animation_settings = dict(
    frame = dict(duration=1200, redraw=False),
    fromcurrent = False,
    transition = dict(duration=1200, easing='cubic-in-out'),
)

def make_step(year):
    return dict(
        method = 'animate',  
        args = [[year], animation_settings],
        label= year
    )
steps = [make_step(str(year)) for year in years] 

sliders = [dict(
    active = 1,
    currentvalue = {
        'prefix': 'Año: ',
        'font': {'size': 20},
        'visible': True,
        'xanchor': 'right'
    },
    steps = steps,
    yanchor= 'top',
    xanchor= 'left',
    pad= {'b': 10, 't': 50},
    len= 0.9,
    x= 0.1,
    y= 0,
)]

updatemenus = [dict(
    buttons= [dict(
        args= [[str(y) for y in years], animation_settings],
        label= 'Animar',
        method= 'animate',
    )],
    direction= 'left',
    pad= dict(r=10, t=87),
    showactive= False,
    type= 'buttons',
    x= 0.1,
    y= 0,
    xanchor= 'right',
    yanchor= 'top',
)]

layout = go.Layout(
    title='Población Abastecida',
    hovermode='x',
    width=1000,
    legend=dict(x=.1, y=1.1, orientation='h'),
    xaxis=dict(
        title='Porcentaje (%): Población Abastecida / Población Total del Área de Servicio * 100',
        range=[35,100],
        autorange=False,
        tickmode='linear',
        tick0 = 35,
        dtick = 5,
    ),
    yaxis=dict(visible=False),
#     xaxis={'title':f'Variable {str(xi)}: {xname} ({xunit})'},
#     yaxis={'title':f'Variable {str(yi)}: {yname} ({yunit})'},
    plot_bgcolor='#dfe8f3',
    sliders= sliders,
    updatemenus = updatemenus,
)

figure = go.Figure(data=base_data, frames=frames, layout=layout)
iplot(figure)

hide_toggle()
num_indicators = 32
cmdf = complete_measurements_df
ind_list = [f'ind{i+1}' for i in range(num_indicators)]

def get_data_rating(epsa_code, year):
    values = list(cmdf[(cmdf.year == year) & (cmdf.code == epsa_code)][ind_list].iloc[0])
    num_reported = sum([0 if math.isnan(val) else 1 for val in values])
    return num_reported / num_indicators * 100

selected_category = 'B'
cmdf = complete_measurements_df
filtered_df = cmdf[cmdf.category == selected_category]
epsa_codes = list(filtered_df.code.unique())
years = list(filtered_df.sort_values('year').year.unique())

scat1 = go.Scatter(
    x= [2014] * len(epsa_codes),
    y= list(filtered_df[filtered_df.year == 2014].ind1),
    text= epsa_codes,
    mode='markers+text',
    showlegend=False,
)

data = [scat1]

for epsa_code in epsa_codes:
    line_trace = go.Scatter(
        x= years,
        y= [get_data_rating(epsa_code, year) for year in years],
#         y= list(filtered_df[filtered_df.code == epsa_code].sort_values('year').ind1),
        mode='lines',
        name=epsa_code,
#         hoverinfo='none',
    )
    data.append(line_trace)
    
layout = dict(
    xaxis= dict(
        title='Años',
        range=[2013.8, 2017.2],
        autorange= False,
        dtick=1,
    ),
    yaxis= dict(
        title='Porcentaje de variables reportadas (%)',
#         dtick=1,
#         range=[50,100],
#         autorange=False,
    ),
    title= f'Ranking EPSAS - Porcentaje de variables reportadas<br>Categoría {selected_category}',
    updatemenus= [dict(
        x=0,
        y=0,
        type= 'buttons',
        buttons=[dict(
            label='Animar',
            method='animate',
            args= [None],
        )],
    )],
    hovermode='closest',
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)

colors = [
    '#1f77b4', '#ff7f0e','#2ca02c',
    '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22','#17becf'
]

colors = colors * 3

frames = [{'data': [go.Scatter(
    x=[year]*len(epsa_codes),
    y=[get_data_rating(epsa_code, year) for epsa_code in epsa_codes],
#     y=list(filtered_df[filtered_df.year == year].ind1),
    text=epsa_codes,
    textposition='top center',
    mode='markers',
    marker=dict(symbol='circle', size=20, color=colors[1:]),
)]} for year in years]

fig = go.Figure(data=data, layout=layout, frames=frames)
iplot(fig)

hide_toggle()
crdf = complete_reports_df
cmdf = complete_measurements_df

colors = [
    '#1f77b4','#ff7f0e','#2ca02c',
    '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22','#17becf'
]

categories = list(crdf.category.sort_values().unique()) # ['A', 'B', 'C', 'D']
years = list(crdf.year.sort_values().unique()) # [2014, 2015, 2016, 2017]

selected_inds = ['ind23','ind27', 'ind28', 'ind25']

default_cat = 'A'
default_ind = 'ind23'

data = []

for cat_i, category in enumerate(categories):
    for ind_i, indicator in enumerate(selected_inds):
        visible = True if cat_i == 0 and ind_i == 0 else False
        
        for year_i, year in enumerate(years):
        
            filtered_df = cmdf[(cmdf.category == category) & (cmdf.year == year)]
                    
            trace = go.Bar(
                x=filtered_df.code,
                y=filtered_df[indicator],
                name=str(year),
                text=[f'{str(y).replace(".",",")} {get_ind_unit(indicator)}' for y in filtered_df[indicator]],
                hoverinfo='name+text+x',
                opacity=0.8,
                marker=dict(color=colors[year_i]),
                visible=visible,
                textfont=dict(
                    color='black',
                )
            )
            
            data.append(trace)
            
def get_visible_list(category, indicator):
    category_map = dict(A=0, B=1, C=2, D=3)
    indicator_map = dict(ind23=0, ind27=1, ind28=2, ind25=3)
    
    cat_offset = category_map[category]
    ind_offset = indicator_map[indicator]
    
    offset = (cat_offset * 4) + ind_offset
    
    base_array = [0 for i in range(16)]
    base_array[offset] = 1
    
    big_true = [True for i in range(4)]
    big_false = [False for i in range(4)]
    
    nested_array = [big_true if x == 1 else big_false for x in base_array]
    
    return [x for l in nested_array for x in l]

def make_x_axis(category):
    return dict(
        title=f'EPSAs Categoría {category}',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        ),
    )

def make_y_axis(ind_code):
    return dict(
        title=f'{get_ind_unit(ind_code)}',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        ),
    )

def make_title(ind_code, cat):
    code_map = dict(ind23='IOE', ind27='TM', ind28='CUO', ind25='ER')
    ind_name = get_ind_name(ind_code)
    ind_unit = get_ind_unit(ind_code)
    if ind_code == 'ind23':
        ind_param = '(óptimo: 65%-75%)'
    if ind_code == 'ind25':
        ind_param ='(óptimo: >90%)'
    if ind_code == 'ind27' and cat == 'A':
        ind_param = '(óptimo: >30% al CUO)'
    if ind_code == 'ind27' and cat in ['B', 'C', 'D']:
        ind_param = '(óptimo: >CUO)'
    if ind_code == 'ind28' and cat == 'A':
        ind_param = '(óptimo: <30% al TM)'
    if ind_code == 'ind28' and cat in ['B', 'C', 'D']:
        ind_param = '(óptimo: <TM)'

    return f'<b>Indicadores Económicos - Categoría {cat}</b><br><b>{code_map[ind_code]}:</b> {ind_name} {ind_param}'
    

def make_shapes(ind_code):
    
    special_shapes1 = [dict(
        type= 'rect',
        xref= 'paper',
        yref= 'y',
        x0= 0,
        y0= 65,
        x1=1,
        y1= 75,
        fillcolor= colors[0],
        opacity= 0.2,
        line= {'width': 0,}
    )]
    
    special_shapes2 = [dict(
        type= 'rect',
        xref= 'paper',
        yref= 'y',
        x0= 0,
        y0= 90,
        x1=1,
        y1= 100,
        fillcolor= colors[0],
        opacity= 0.2,
        line= {'width': 0,}
    )]
    
    if ind_code == 'ind23':
        return special_shapes1
    if ind_code == 'ind25':
        return special_shapes2
    else:
        return []

    
drop_x_map = dict(
    ind23=0,
    ind27=0.25,
    ind28=0.5,
    ind25=0.75,
)

def get_label(ind, cat):
    name_list = ['IOE', 'TM', 'CUO', 'ER']
    ind_map = dict(ind23=0, ind27=1, ind28=2, ind25=3)
    return f'{name_list[ind_map[ind]]} - Category {cat}'


def make_annotations(ind, cat):
    ind_formulas = dict(
        ind23='$IOE = \\frac{\\text{Costos operativos del servicio}}{\\text{Ingresos operativos del servicio}} \\times 100$',
        ind27='$TM = \\frac{\\text{Ingresos por servicios}}{\\text{Volumen de AP facturado}}$',
        ind28='$CUO = \\frac{\\text{Costos operativos totales}}{\\text{Volumen de AP facturado}}$',
        ind25='$ER = \\left(1-\\frac{\\text{Ctas por cobrar de facturación gestión actual}}{\\text{Ingresos por servicios}}\\right)\\times 100$',
    )
        
    formulas = [dict(
        font=dict(size=15),
        showarrow=False,
        text=ind_formulas[ind],
        xref='paper', yref='paper',
        x=1, y=1.15,
    )]
    
    return [dict(
        yref="paper", 
        xref="paper", 
        y=1.15, 
        x=0, 
        text=make_title(ind, cat), 
        showarrow=False, 
        font=dict(
            family='Helvetica LT Std',
            size=17,
        ),
    )] + formulas
    

updatemenus = [dict(
    showactive=True,
    active=0,
    xanchor='left',
    yanchor='top',
    direction='up',
    x= drop_x_map[ind],
    y= -0.5,
    buttons = [dict(
        label=get_label(ind, cat),
        method='update', # modify both data and layout
        args = [
            {'visible': get_visible_list(cat, ind)}, # data modification
            # layout modification
            dict(
#                 title= make_title(ind, cat),
                xaxis= make_x_axis(cat),
                yaxis= make_y_axis(ind),
                shapes= make_shapes(ind),
                annotations= make_annotations(ind, cat),
            ),
        ],
    ) for cat in categories]
) for ind in selected_inds]

layout = go.Layout(
#     title= make_title(default_ind, 'A'),
    titlefont=dict(
        family='Helvetica LT Std',
        size=18,
    ),
    xaxis= make_x_axis(default_cat),
    yaxis= make_y_axis(default_ind),
    shapes=make_shapes(default_ind),
#     hovermode='closest',
    updatemenus = updatemenus,
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
    annotations=make_annotations(default_ind, 'A')
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, link_text='')

hide_toggle()
crdf = complete_reports_df
cmdf = complete_measurements_df[complete_measurements_df.state == 'SC']

colors = [
    '#1f77b4','#ff7f0e','#2ca02c',
    '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22','#17becf'
]

categories = list(crdf.category.sort_values().unique()) # ['A', 'B', 'C', 'D']
years = list(crdf.year.sort_values().unique()) # [2014, 2015, 2016, 2017]

selected_inds = ['ind23','ind27', 'ind28', 'ind25']

default_cat = 'A'
default_ind = 'ind23'

data = []

for cat_i, category in enumerate(categories):
    for ind_i, indicator in enumerate(selected_inds):
        visible = True if cat_i == 0 and ind_i == 0 else False
        
        for year_i, year in enumerate(years):
        
            filtered_df = cmdf[(cmdf.category == category) & (cmdf.year == year)]
                    
            trace = go.Bar(
                x=filtered_df.code,
                y=filtered_df[indicator],
                name=str(year),
                text=[f'{str(y).replace(".",",")} {get_ind_unit(indicator)}' for y in filtered_df[indicator]],
                hoverinfo='name+text+x',
                opacity=0.8,
                marker=dict(color=colors[year_i]),
                visible=visible,
                textfont=dict(
                    color='black',
                )
            )
            
            data.append(trace)
            
def get_visible_list(category, indicator):
    category_map = dict(A=0, B=1, C=2, D=3)
    indicator_map = dict(ind23=0, ind27=1, ind28=2, ind25=3)
    
    cat_offset = category_map[category]
    ind_offset = indicator_map[indicator]
    
    offset = (cat_offset * 4) + ind_offset
    
    base_array = [0 for i in range(16)]
    base_array[offset] = 1
    
    big_true = [True for i in range(4)]
    big_false = [False for i in range(4)]
    
    nested_array = [big_true if x == 1 else big_false for x in base_array]
    
    return [x for l in nested_array for x in l]

def make_x_axis(category):
    return dict(
        title=f'EPSAs Categoría {category}',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        ),
    )

def make_y_axis(ind_code):
    return dict(
        title=f'{get_ind_unit(ind_code)}',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica LT Std',
            size=16,
        ),
    )

def make_title(ind_code, cat):
    code_map = dict(ind23='IOE', ind27='TM', ind28='CUO', ind25='ER')
    ind_name = get_ind_name(ind_code)
    ind_unit = get_ind_unit(ind_code)
    if ind_code == 'ind23':
        ind_param = '(óptimo: 65%-75%)'
    if ind_code == 'ind25':
        ind_param ='(óptimo: >90%)'
    if ind_code == 'ind27' and cat == 'A':
        ind_param = '(óptimo: >30% al CUO)'
    if ind_code == 'ind27' and cat in ['B', 'C', 'D']:
        ind_param = '(óptimo: >CUO)'
    if ind_code == 'ind28' and cat == 'A':
        ind_param = '(óptimo: <30% al TM)'
    if ind_code == 'ind28' and cat in ['B', 'C', 'D']:
        ind_param = '(óptimo: <TM)'

    return f'<b>Indicadores Económicos - SC - Categoría {cat}</b><br><b>{code_map[ind_code]}:</b> {ind_name} {ind_param}'
    

def make_shapes(ind_code):
    
    special_shapes1 = [dict(
        type= 'rect',
        xref= 'paper',
        yref= 'y',
        x0= 0,
        y0= 65,
        x1=1,
        y1= 75,
        fillcolor= colors[0],
        opacity= 0.2,
        line= {'width': 0,}
    )]
    
    special_shapes2 = [dict(
        type= 'rect',
        xref= 'paper',
        yref= 'y',
        x0= 0,
        y0= 90,
        x1=1,
        y1= 100,
        fillcolor= colors[0],
        opacity= 0.2,
        line= {'width': 0,}
    )]
    
    if ind_code == 'ind23':
        return special_shapes1
    if ind_code == 'ind25':
        return special_shapes2
    else:
        return []

    
drop_x_map = dict(
    ind23=0,
    ind27=0.25,
    ind28=0.5,
    ind25=0.75,
)

def get_label(ind, cat):
    name_list = ['IOE', 'TM', 'CUO', 'ER']
    ind_map = dict(ind23=0, ind27=1, ind28=2, ind25=3)
    return f'{name_list[ind_map[ind]]} - Category {cat}'


def make_annotations(ind, cat):
    ind_formulas = dict(
        ind23='$IOE = \\frac{\\text{Costos operativos del servicio}}{\\text{Ingresos operativos del servicio}} \\times 100$',
        ind27='$TM = \\frac{\\text{Ingresos por servicios}}{\\text{Volumen de AP facturado}}$',
        ind28='$CUO = \\frac{\\text{Costos operativos totales}}{\\text{Volumen de AP facturado}}$',
        ind25='$ER = \\left(1-\\frac{\\text{Ctas por cobrar de facturación gestión actual}}{\\text{Ingresos por servicios}}\\right)\\times 100$',
    )
        
    formulas = [dict(
        font=dict(size=15),
        showarrow=False,
        text=ind_formulas[ind],
        xref='paper', yref='paper',
        x=1, y=1.15,
    )]
    
    return [dict(
        yref="paper", 
        xref="paper", 
        y=1.15, 
        x=0, 
        text=make_title(ind, cat), 
        showarrow=False, 
        font=dict(
            family='Helvetica LT Std',
            size=17,
        ),
    )] + formulas
    

updatemenus = [dict(
    showactive=True,
    active=0,
    xanchor='left',
    yanchor='top',
    direction='up',
    x= drop_x_map[ind],
    y= -0.5,
    buttons = [dict(
        label=get_label(ind, cat),
        method='update', # modify both data and layout
        args = [
            {'visible': get_visible_list(cat, ind)}, # data modification
            # layout modification
            dict(
#                 title= make_title(ind, cat),
                xaxis= make_x_axis(cat),
                yaxis= make_y_axis(ind),
                shapes= make_shapes(ind),
                annotations= make_annotations(ind, cat),
            ),
        ],
    ) for cat in categories]
) for ind in selected_inds]

layout = go.Layout(
#     title= make_title(default_ind, 'A'),
    titlefont=dict(
        family='Helvetica LT Std',
        size=18,
    ),
    xaxis= make_x_axis(default_cat),
    yaxis= make_y_axis(default_ind),
    shapes=make_shapes(default_ind),
#     hovermode='closest',
    updatemenus = updatemenus,
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
    annotations=make_annotations(default_ind, 'A')
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, link_text='')

hide_toggle()
crdf = complete_reports_df
cmdf = complete_measurements_df

categories = list(crdf.category.sort_values().unique()) # ['A', 'B', 'C', 'D']
year = 2017

selected_inds = ['ind27', 'ind28']
trace_names = {'ind27': 'TM', 'ind28': 'CUO'}

data = []

for cat_i, category in enumerate(categories):
    visible = True if cat_i == 0 else False 
    for ind_i, indicator in enumerate(selected_inds):
        filtered_df = cmdf[(cmdf.category == category) & (cmdf.year == year)]
                    
        trace = go.Bar(
            x=filtered_df.code,
            y=filtered_df[indicator],
            name=trace_names[indicator],
            text=[f'{str(y).replace(".",",")} Bs.' for y in filtered_df[indicator]],
            hoverinfo='name+text+x',
            opacity=0.8,
            visible=visible,
            textfont=dict(
                color='black',
            )
        )

        data.append(trace)
            
def get_visible_list(category):
    category_map = dict(A=0, B=1, C=2, D=3)    
    base_array = [False for i in range(4)]
    base_array[category_map[category]] = True
    fat_array = [[x] * 2 for x in base_array]
    return [x for l in fat_array for x in l]


def make_x_axis():
    return dict(
        title='EPSA',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica Lt Std',
            size=16,
        ),
    )

def make_y_axis():
    return dict(
        title='Bolivianos',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica Lt Std',
            size=16,
        ),
    )

def make_title(cat):
    cat_title = '' if cat == 'TODAS' else f'<b> - Categoría {cat}</b>'
    base_title=f'<b>Tarifa Media vs. Costo Unitario de Operación</b>{cat_title}'
    par_text1= '' if cat == 'TODAS' else f'(óptimo: {dict(A=">30% al CUO",B=">CUO",C=">CUO",D=">CUO")[cat]})'
    par_text2= '' if cat == 'TODAS' else f'(óptimo: {dict(A="<30% al TM",B="<TM",C="<TM",D="<TM")[cat]})'
    ind_descr1= f'<br><b>TM:</b> {get_ind_name("ind27")} (Bs.) {par_text1}'
    ind_descr2= f'<br><b>CUO:</b> {get_ind_name("ind28")} (Bs.) {par_text2}'
    return base_title + ind_descr1 + ind_descr2

def make_annotations(cat):
    formulas = [dict(
        font=dict(size=15),
        showarrow=False,
        text='$TM = \\frac{\\text{Ingresos por servicios}}{\\text{Volumen de AP facturado}}$',
        xref='paper', yref='paper',
        x=0.9, y=1.30,
    ), dict(
        font=dict(size=15),
        showarrow=False,
        text='$CUO = \\frac{\\text{Costos operativos totales}}{\\text{Volumen de AP facturado}}$',
        xref='paper', yref='paper',
        x=0.9, y=1.15,
    )] if cat == 'TODAS' else []
    
    return [dict(
        yref="paper", 
        xref="paper", 
        y=1.30, 
        x=0 if cat == 'TODAS' else 0.5, 
        text=make_title(cat), 
        showarrow=False, 
        font=dict(size=17)
    )] + formulas

updatemenus = [dict(
    type='buttons',
    showactive=True,
    active=0,
    xanchor='left',
    yanchor='top',
    direction='right',
    x= 0.3,
    y= -0.5,
    buttons = [dict(
        label=f'Categoría {cat}',
        method='update', # modify both data and layout
        args = [
            {'visible': get_visible_list(cat)}, # data modification
            # layout modification
            dict(
#                 title= make_title(cat),
                annotations=make_annotations(cat),
#                 xaxis= make_x_axis(cat),
#                 yaxis= make_y_axis(ind),
#                 shapes= make_shapes(ind),
            ),
        ],
    ) for cat in categories]
)]

layout = go.Layout(
#     title = make_title('A'),
    titlefont = dict(
        family='Helvetica LT Std',
        size=18,
    ),
    xaxis= make_x_axis(),
    yaxis= make_y_axis(),
    updatemenus = updatemenus,
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
    annotations= make_annotations('TODAS'),
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, link_text='')

hide_toggle()
crdf = complete_reports_df
cmdf = complete_measurements_df[complete_measurements_df.state == 'SC']

categories = list(crdf.category.sort_values().unique()) # ['A', 'B', 'C', 'D']
year = 2017

selected_inds = ['ind27', 'ind28']
trace_names = {'ind27': 'TM', 'ind28': 'CUO'}

data = []

for cat_i, category in enumerate(categories):
    visible = True if cat_i == 0 else False 
    for ind_i, indicator in enumerate(selected_inds):
        filtered_df = cmdf[(cmdf.category == category) & (cmdf.year == year)]
                    
        trace = go.Bar(
            x=filtered_df.code,
            y=filtered_df[indicator],
            name=trace_names[indicator],
            text=[f'{str(y).replace(".",",")} Bs.' for y in filtered_df[indicator]],
            hoverinfo='name+text+x',
            opacity=0.8,
            visible=visible,
            textfont=dict(
                color='black',
            )
        )

        data.append(trace)
            
def get_visible_list(category):
    category_map = dict(A=0, B=1, C=2, D=3)    
    base_array = [False for i in range(4)]
    base_array[category_map[category]] = True
    fat_array = [[x] * 2 for x in base_array]
    return [x for l in fat_array for x in l]


def make_x_axis():
    return dict(
        title='EPSA',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica Lt Std',
            size=16,
        ),
    )

def make_y_axis():
    return dict(
        title='Bolivianos',
        titlefont=dict(
            family='Helvetica LT Std',
            size=18,
            color='#7f7f7f'
        ),
        tickfont=dict(
            family='Helvetica Lt Std',
            size=16,
        ),
    )

def make_title(cat):
    cat_title = '' if cat == 'TODAS' else f'<b> - Categoría {cat}</b>'
    base_title=f'<b>Tarifa Media vs. Costo Unitario de Operación - SC</b>{cat_title}'
    par_text1= '' if cat == 'TODAS' else f'(óptimo: {dict(A=">30% al CUO",B=">CUO",C=">CUO",D=">CUO")[cat]})'
    par_text2= '' if cat == 'TODAS' else f'(óptimo: {dict(A="<30% al TM",B="<TM",C="<TM",D="<TM")[cat]})'
    ind_descr1= f'<br><b>TM:</b> {get_ind_name("ind27")} (Bs.) {par_text1}'
    ind_descr2= f'<br><b>CUO:</b> {get_ind_name("ind28")} (Bs.) {par_text2}'
    return base_title + ind_descr1 + ind_descr2

def make_annotations(cat):
    formulas = [dict(
        font=dict(size=15),
        showarrow=False,
        text='$TM = \\frac{\\text{Ingresos por servicios}}{\\text{Volumen de AP facturado}}$',
        xref='paper', yref='paper',
        x=0.9, y=1.30,
    ), dict(
        font=dict(size=15),
        showarrow=False,
        text='$CUO = \\frac{\\text{Costos operativos totales}}{\\text{Volumen de AP facturado}}$',
        xref='paper', yref='paper',
        x=0.9, y=1.15,
    )] if cat == 'TODAS' else []
    
    return [dict(
        yref="paper", 
        xref="paper", 
        y=1.30, 
        x=0 if cat == 'TODAS' else 0.5, 
        text=make_title(cat), 
        showarrow=False, 
        font=dict(size=17)
    )] + formulas

updatemenus = [dict(
    type='buttons',
    showactive=True,
    active=0,
    xanchor='left',
    yanchor='top',
    direction='right',
    x= 0.3,
    y= -0.5,
    buttons = [dict(
        label=f'Categoría {cat}',
        method='update', # modify both data and layout
        args = [
            {'visible': get_visible_list(cat)}, # data modification
            # layout modification
            dict(
#                 title= make_title(cat),
                annotations=make_annotations(cat),
#                 xaxis= make_x_axis(cat),
#                 yaxis= make_y_axis(ind),
#                 shapes= make_shapes(ind),
            ),
        ],
    ) for cat in categories]
)]

layout = go.Layout(
#     title = make_title('A'),
    titlefont = dict(
        family='Helvetica LT Std',
        size=18,
    ),
    xaxis= make_x_axis(),
    yaxis= make_y_axis(),
    updatemenus = updatemenus,
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
    annotations= make_annotations('TODAS'),
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, link_text='')

hide_toggle()