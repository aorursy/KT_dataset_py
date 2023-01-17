!pip install plotly
#Helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #Viz

import seaborn as sns #Viz

import plotly.express as px #Viz



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel('/kaggle/input/productos-consumo-masivo/output - Kaggle.xlsx', decimal=',')

# Se muestran los primeros 3 valores

df.head(3)
df_clean = df[['prod_id','prod_brand', 'tags', 'prod_unit_price']]

df_clean.head(3)
#Agrupamos nuestra tabla anterior por el prod_id. Buscamos tener para cada producto:

# prod_id: Identificador del producto en la base de datos

# NumVentasPorProducto: Veces que se vendió ese producto

# VentasTotalesPorProducto = Sumatoria total de las ventas de ese producto

# PrecioPromedioPorProducto = Promedio del precio de ese producto a lo largo del tiempo (Es válido porque solo es 1 año)

# Tags: sector comercial al que pertenece el producto

# Brand: nombre de la marca comercial que fabricó el producto



NumVentasPorProducto = df_clean.groupby(['prod_id'])[['prod_unit_price']].count()

VentasTotalesPorProducto = df_clean.groupby(['prod_id'])[['prod_unit_price']].sum()

PrecioPromedioPorProducto = df_clean.groupby(['prod_id'])[['prod_unit_price']].mean()

tags = df_clean.groupby(['prod_id'])[['tags']].first()

prod_brand = df_clean.groupby(['prod_id'])[['prod_brand']].first()

NumVentasPorProducto.columns = ['NumVentasPorProducto']

VentasTotalesPorProducto.columns = ['VentasTotalesPorProducto']

PrecioPromedioPorProducto.columns = ['PrecioPromedioPorProducto']

tags.columns = ['tags']

prod_brand.columns = ['prod_brand']



# Ahora unimos todos estos vectores en un data frame, uniendolos a través del prod_id

df_compl = prod_brand.merge(tags,left_on='prod_id', right_on='prod_id')

df_compl = df_compl.merge(PrecioPromedioPorProducto,left_on='prod_id', right_on='prod_id')

df_compl = df_compl.merge(VentasTotalesPorProducto,left_on='prod_id', right_on='prod_id')

df_compl = df_compl.merge(NumVentasPorProducto,left_on='prod_id', right_on='prod_id')

df_compl.head(3)

#Agrupamos df_compl por prod_brand. Buscamos tener para cada prod_brand o Marca tener:

# prod_brand: nombre de la marca comercial que fabricó el producto

# PorMarca_PrecioProductos: precio promedio de los productos que ofrece la Marca

# PorMarca_Total_NumProductos: # productos distintos que ofrece cada marca

# PorMarca_Total_DiversificacionPrecios: desv_est de cada producto distinto (Su promedio) respecto del promedio de los productos de la marca

# PorMarca_Total_NumVentas: # total de ventas por cada marca. Sumatoria del número de ventas de todos los productos de la marca

# PorMarca_Total_ValorVentas: valor total de todas las ventas dentro de cada sector comercial. Sumatoria del valor de todas las ventas en el sector



PorMarca_Total_NumProductos = df_compl.groupby(['prod_brand'])[['PrecioPromedioPorProducto']].count()

PorMarca_Total_DiversificacionPrecios = df_compl.groupby(['prod_brand'])[['PrecioPromedioPorProducto']].std()

PorMarca_Total_NumVentas = df_compl.groupby(['prod_brand'])[['NumVentasPorProducto']].sum()

PorMarca_Total_ValorVentas = df_compl.groupby(['prod_brand'])[['VentasTotalesPorProducto']].sum()



PorMarca_Total_NumProductos.columns = ['PorMarca_Total_NumProductos']

PorMarca_Total_DiversificacionPrecios.columns = ['PorMarca_Total_DiversificacionPrecios']

PorMarca_Total_NumVentas.columns = ['PorMarca_Total_NumVentas']

PorMarca_Total_ValorVentas.columns = ['PorMarca_Total_ValorVentas']



# Ahora unimos todos estos vectores en un data frame, uniendolos a través del prod_id

df_ByBrand = PorMarca_Total_NumProductos.merge(PorMarca_Total_DiversificacionPrecios,left_on='prod_brand', right_on='prod_brand')

df_ByBrand = df_ByBrand.merge(PorMarca_Total_NumVentas,left_on='prod_brand', right_on='prod_brand')

df_ByBrand = df_ByBrand.merge(PorMarca_Total_ValorVentas,left_on='prod_brand', right_on='prod_brand')

df_ByBrand.head(3)
#Agrupamos df_compl por prod_brand y Tags, para un posterior análisis

# tags: sector comercial al que pertenece el producto

# prod_brand: nombre de la marca comercial que fabricó el producto

# PrecioProductos: precio promedio de los productos para cada marca intersectada con el sector comercial

# NumProductos: # productos distintos para cada marca intersectada con el sector comercial

# PrecioProductos_Desviacion: desv_est de los precios respecto del promedio para cada marca intersectada con el sector comercial

# NumVentas: # total de ventas para cada marca intersectada con el sector comercial

# ValorVentas: valor total de todas las ventas para cada marca intersectada con el sector comercial



PrecioProductos = df_compl.groupby(['prod_brand','tags'])[['PrecioPromedioPorProducto']].mean()

NumProductos = df_compl.groupby(['prod_brand','tags'])[['PrecioPromedioPorProducto']].count()

NumVentas = df_compl.groupby(['prod_brand','tags'])[['NumVentasPorProducto']].sum()

ValorVentas = df_compl.groupby(['prod_brand','tags'])[['VentasTotalesPorProducto']].sum()



PrecioProductos.columns = ['PrecioProductos']

NumProductos.columns = ['NumProductos']

NumVentas.columns = ['NumVentas']

ValorVentas.columns = ['ValorVentas']



# Ahora unimos todos estos vectores en un data frame, uniendolos a través del prod_id

df_BrandAndTags = PrecioProductos.merge(NumProductos,on=['prod_brand','tags'])

df_BrandAndTags = df_BrandAndTags.merge(NumVentas,on=['prod_brand','tags'])

df_BrandAndTags = df_BrandAndTags.merge(ValorVentas,on=['prod_brand','tags'])

df_BrandAndTags.head(3)
#Agrupamos df_BrandAndTags por prod_brand, para conseguir datos de las Marcas en el contexto de los sectores comerciales

# prod_brand: nombre de la marca comercial que fabricó el producto

# Marca_NumSectoresComerciales: # Tags o sectores comerciales en los que participa una Marca

# PorMarca_DiversificacionGananciasEntreSectores: desv_est dentro de cada marca para los ganancias totales en cada sector comercial respecto del promedio

# PorMarca_DiversificacionProductosPorSector: desv_est dentro de cada marca para los precios por producto en cada sector comercial respecto del promedio

# PorMarca_DiversificacionNumeroVentasPorSector: desv_est dentro de cada marca para el número de ventas en cada sector comercial respecto del promedio



PorMarca_NumSectoresComerciales = df_BrandAndTags.groupby(['prod_brand'])[['ValorVentas']].count()

PorMarca_DiversificacionProductosPorSector = df_BrandAndTags.groupby(['prod_brand'])[['NumProductos']].std()

PorMarca_DiversificacionNumeroVentasPorSector = df_BrandAndTags.groupby(['prod_brand'])[['NumVentas']].std()



PorMarca_NumSectoresComerciales.columns = ['PorMarca_NumSectoresComerciales']

PorMarca_DiversificacionProductosPorSector.columns = ['PorMarca_DiversificacionProductosPorSector']

PorMarca_DiversificacionNumeroVentasPorSector.columns = ['PorMarca_DiversificacionNumeroVentasPorSector']



# Ahora unimos todos estos vectores en un data frame, uniendolos a través del prod_id

df_BrandAndTagsByBrand = PorMarca_NumSectoresComerciales.merge(PorMarca_DiversificacionProductosPorSector,on=['prod_brand'])

df_BrandAndTagsByBrand = df_BrandAndTagsByBrand.merge(PorMarca_DiversificacionNumeroVentasPorSector,on=['prod_brand'])

df_BrandAndTagsByBrand.head(3)

#Agrupamos df_BrandAndTags por tags, para conseguir datos de los sectores comerciales en el contexto de las brands

# tags: sector comercial al que pertenece el producto

# PorTag_InequidadGananciasPorSector: desv_est dentro de cada sector comercial para los precios por producto por cada marca respecto del promedio

# PorTag_DiversificacionProductosPorSector: desv_est dentro de cada sector comercial para el número de productos por cada marca respecto del promedio

# PorTag_DiversificacionPreciosPorSector: desv_est dentro de cada sector comercial para los precios por producto por cada marca respecto del promedio



PorTag_NumMarcas = df_BrandAndTags.groupby(['tags'])[['ValorVentas']].count()

PorTag_InequidadGananciasPorSector = df_BrandAndTags.groupby(['tags'])[['ValorVentas']].std()

PorTag_DiversificacionProductosPorSector = df_BrandAndTags.groupby(['tags'])[['NumProductos']].std()

PorTag_DiversificacionPreciosPorSector = df_BrandAndTags.groupby(['tags'])[['PrecioProductos']].std()



PorTag_NumMarcas.columns = ['PorTag_NumMarcas']

PorTag_InequidadGananciasPorSector.columns = ['PorTag_InequidadGananciasPorSector']

PorTag_DiversificacionProductosPorSector.columns = ['PorTag_DiversificacionProductosPorSector']

PorTag_DiversificacionPreciosPorSector.columns = ['PorTag_DiversificacionPreciosPorSector']



# Ahora unimos todos estos vectores en un data frame, uniendolos a través del prod_id

df_BrandAndTagsByTag = PorTag_NumMarcas.merge(PorTag_InequidadGananciasPorSector,on=['tags'])

df_BrandAndTagsByTag = df_BrandAndTagsByTag.merge(PorTag_DiversificacionProductosPorSector,on=['tags'])

df_BrandAndTagsByTag = df_BrandAndTagsByTag.merge(PorTag_DiversificacionPreciosPorSector,on=['tags'])

df_BrandAndTagsByTag.head(3)
#Ahora unimos toda la información de las marcas en el df_BrandAndTags

df_BrandAndTagsWithBrands = df_BrandAndTags.copy()

df_BrandAndTagsWithBrands = df_BrandAndTagsWithBrands.reset_index()

df_BrandAndTagsWithBrands = df_BrandAndTagsWithBrands.merge(df_ByBrand,on=['prod_brand'])

df_BrandAndTagsWithBrands = df_BrandAndTagsWithBrands.merge(df_BrandAndTagsByBrand,on=['prod_brand'])

df_BrandAndTagsWithBrands = df_BrandAndTagsWithBrands.fillna(0)

df_BrandAndTagsWithBrands.head(3)
#Ahora calculamos qué tanto ha invertido cada marca de su # ventas, diversidad de productos y ganancia en cada sector

InversionEnSector_NumProductos = df_BrandAndTagsWithBrands['NumProductos'] / df_BrandAndTagsWithBrands['PorMarca_Total_NumProductos']

InversionEnSector_NumVentas = df_BrandAndTagsWithBrands['NumVentas'] / df_BrandAndTagsWithBrands['PorMarca_Total_NumVentas']

InversionEnSector_ValorVentas = df_BrandAndTagsWithBrands['ValorVentas'] / df_BrandAndTagsWithBrands['PorMarca_Total_ValorVentas']



# Y posteriormente calculamos 

df_BrandSummary = df_BrandAndTagsWithBrands.copy()

df_BrandSummary['PorMarca_InversionEnSector'] = (InversionEnSector_NumProductos+InversionEnSector_NumVentas+InversionEnSector_ValorVentas)/3

df_BrandSummary['PorMarca_Potencia'] = df_BrandAndTagsWithBrands['PorMarca_Total_NumVentas'] * df_BrandAndTagsWithBrands['PorMarca_Total_ValorVentas']

df_BrandSummary['PorMarca_Diversificacion'] = df_BrandAndTagsWithBrands['PorMarca_Total_NumProductos'] * df_BrandAndTagsWithBrands['PorMarca_NumSectoresComerciales'] 

df_BrandSummary['PorMarca_NoEspecializacion'] = (df_BrandAndTagsWithBrands['PorMarca_DiversificacionProductosPorSector'] + df_BrandAndTagsWithBrands['PorMarca_DiversificacionNumeroVentasPorSector'])/2

df_BrandSummary = df_BrandSummary[['prod_brand','tags','PorMarca_InversionEnSector','PorMarca_Potencia','PorMarca_Diversificacion','PorMarca_NoEspecializacion']]



df_BrandSummary.head(3)
# Ahora calculamos el impacto de la potencia, diversificacion y estabilidad de cada marca para cada sector

# Ya que si una marca está en 10 sectores, no va a ser tan relevante su impacto en el área

df_BrandSummary_adjustedByInversion = df_BrandSummary.copy()

df_BrandSummary_adjustedByInversion['PorMarca_Ajustado_Potencia'] = df_BrandSummary_adjustedByInversion['PorMarca_Potencia'] * df_BrandSummary_adjustedByInversion['PorMarca_InversionEnSector']

df_BrandSummary_adjustedByInversion['PorMarca_Ajustado_Diversificacion'] = df_BrandSummary_adjustedByInversion['PorMarca_Diversificacion'] * df_BrandSummary_adjustedByInversion['PorMarca_InversionEnSector']

df_BrandSummary_adjustedByInversion['PorMarca_Ajustado_NoEspecializacion'] = df_BrandSummary_adjustedByInversion['PorMarca_NoEspecializacion'] * df_BrandSummary_adjustedByInversion['PorMarca_InversionEnSector']

df_BrandSummary_adjustedByInversion = df_BrandSummary_adjustedByInversion[['prod_brand','tags','PorMarca_Ajustado_Potencia','PorMarca_Ajustado_Diversificacion','PorMarca_Ajustado_NoEspecializacion']]

df_BrandSummary_adjustedByInversion.head(3)


# Y posteriormente resumimos las estadisticas de las marcas para los tags



PorTag_PotenciaDeMarcas = df_BrandSummary_adjustedByInversion.groupby(['tags'])[['PorMarca_Ajustado_Potencia']].mean()

PorTag_DiversificacionDeMarcas = df_BrandSummary_adjustedByInversion.groupby(['tags'])[['PorMarca_Ajustado_Diversificacion']].mean()

PorTag_NoEspecializacionDeMarcas = df_BrandSummary_adjustedByInversion.groupby(['tags'])[['PorMarca_Ajustado_NoEspecializacion']].mean()

PorTag_Variedad_PotenciaDeMarcas = df_BrandSummary_adjustedByInversion.groupby(['tags'])[['PorMarca_Ajustado_Potencia']].std()

PorTag_Variedad_DiversificacionDeMarcas = df_BrandSummary_adjustedByInversion.groupby(['tags'])[['PorMarca_Ajustado_Diversificacion']].std()

PorTag_Variedad_NoEspecializacionDeMarcas = df_BrandSummary_adjustedByInversion.groupby(['tags'])[['PorMarca_Ajustado_NoEspecializacion']].std()



PorTag_PotenciaDeMarcas.columns = ['PorTag_PotenciaDeMarcas']

PorTag_DiversificacionDeMarcas.columns = ['PorTag_DiversificacionDeMarcas']

PorTag_NoEspecializacionDeMarcas.columns = ['PorTag_NoEspecializacionDeMarcas']

PorTag_Variedad_PotenciaDeMarcas.columns = ['PorTag_Variedad_PotenciaDeMarcas']

PorTag_Variedad_DiversificacionDeMarcas.columns = ['PorTag_Variedad_DiversificacionDeMarcas']

PorTag_Variedad_NoEspecializacionDeMarcas.columns = ['PorTag_Variedad_NoEspecializacionDeMarcas']



# Ahora unimos todos estos vectores en un data frame, uniendolos a través del prod_id

df_BrandStatsByTag = PorTag_PotenciaDeMarcas.merge(PorTag_DiversificacionDeMarcas,on=['tags'])

df_BrandStatsByTag = df_BrandStatsByTag.merge(PorTag_NoEspecializacionDeMarcas,on=['tags'])

df_BrandStatsByTag = df_BrandStatsByTag.merge(PorTag_Variedad_PotenciaDeMarcas,on=['tags'])

df_BrandStatsByTag = df_BrandStatsByTag.merge(PorTag_Variedad_DiversificacionDeMarcas,on=['tags'])

df_BrandStatsByTag = df_BrandStatsByTag.merge(PorTag_Variedad_NoEspecializacionDeMarcas,on=['tags'])

df_BrandStatsByTag = df_BrandStatsByTag.fillna(0)

df_BrandStatsByTag.head(3)
#Agrupamos df_compl por Tags. Buscamos tener para cada tag o sector comercial tener:

# tags: sector comercial al que pertenece el producto

# PorTag_PrecioProductos: precio promedio de los productos dentro de cada sector comercial

# PorTag_NumProductos: # productos distintos dentro de cada sector comercial

# PorTag_PrecioProductos_Desviacion: desv_est de cada producto distinto (Su promedio) respecto del promedio de los productos en el sector comercial

# PorTag_NumVentas: # total de ventas dentro de cada sector comercial. Sumatoria del número de ventas de todos los productos en el sector

# PorTag_ValorVentas: valor total de todas las ventas dentro de cada sector comercial. Sumatoria del valor de todas las ventas en el sector



PorTag_PrecioProductos = df_compl.groupby(['tags'])[['PrecioPromedioPorProducto']].mean()

PorTag_NumProductos = df_compl.groupby(['tags'])[['PrecioPromedioPorProducto']].count()

PorTag_DiversificacionPrecios = df_compl.groupby(['tags'])[['PrecioPromedioPorProducto']].std()

PorTag_NumVentas = df_compl.groupby(['tags'])[['NumVentasPorProducto']].sum()

PorTag_ValorVentas = df_compl.groupby(['tags'])[['VentasTotalesPorProducto']].sum()



PorTag_PrecioProductos.columns = ['PorTag_PrecioProductos']

PorTag_NumProductos.columns = ['PorTag_NumProductos']

PorTag_DiversificacionPrecios.columns = ['PorTag_DiversificacionPrecios']

PorTag_NumVentas.columns = ['PorTag_NumVentas']

PorTag_ValorVentas.columns = ['PorTag_ValorVentas']



# Ahora unimos todos estos vectores en un data frame, uniendolos a través del prod_id

df_Bytags = PorTag_PrecioProductos.merge(PorTag_NumProductos,left_on='tags', right_on='tags')

df_Bytags = df_Bytags.merge(PorTag_DiversificacionPrecios,left_on='tags', right_on='tags')

df_Bytags = df_Bytags.merge(PorTag_NumVentas,left_on='tags', right_on='tags')

df_Bytags = df_Bytags.merge(PorTag_ValorVentas,left_on='tags', right_on='tags')

df_Bytags.head(3)
# Ahora unimos las estadísticas de cada tag junto con las estadísticas de las marcas, por tag

df_BytagsWithBrandStats = df_Bytags.merge(df_BrandStatsByTag,left_on='tags', right_on='tags')

df_BytagsWithBrandStats.head(3)
#Aquí calculamos las variables que vamos a presentar finalmente

df_finalTagStats = df_BytagsWithBrandStats.copy()

df_finalTagStats['BajoCostoEntradaMercado'] = 1/(df_finalTagStats['PorTag_PrecioProductos']-(df_finalTagStats['PorTag_DiversificacionPrecios']/2))

df_finalTagStats['FacilidadVenta'] = df_finalTagStats['PorTag_NumVentas']

df_finalTagStats['FlujoDinero'] = df_finalTagStats['PorTag_ValorVentas']

df_finalTagStats['DebilidadMarcasRivales'] = 1/(df_finalTagStats['PorTag_PotenciaDeMarcas']+1)

df_finalTagStats['MercadoNoEspecializado'] = df_finalTagStats['PorTag_NoEspecializacionDeMarcas']+df_finalTagStats['PorTag_Variedad_DiversificacionDeMarcas']

df_finalTagStats = df_finalTagStats[['BajoCostoEntradaMercado','FacilidadVenta','FlujoDinero','DebilidadMarcasRivales','MercadoNoEspecializado']]

df_finalTagStats.head(3)
#Definimos la formula para normalizar

def NormalizeData(data):

    return (data - np.min(data)) / (np.max(data) - np.min(data))
# Normalización y pesos de los atributos

Peso_BajoCostoEntradaMercado = 3 #Es muy importante que los productos no sean tan caros que se necesite ser millonario para entrar

Peso_FacilidadVenta = 1

Peso_FlujoDinero = 2 # Tiene que ser fácil ganar dinero

Peso_DebilidadMarcasRivales = 1

Peso_MercadoNoEspecializado = 3 #Si el mercado es muy especializado, al punto que las marcas solo tienen esa línea de productos,

                                #Como en la fabricación de vino. Es muy difícil entrar al mercado y habrán regulaciones estrictas



df_finalTagStats_norm = df_finalTagStats.copy()

df_finalTagStats_norm['BajoCostoEntradaMercado'] =  NormalizeData(df_finalTagStats_norm['BajoCostoEntradaMercado']**Peso_BajoCostoEntradaMercado)

df_finalTagStats_norm['FacilidadVenta'] = NormalizeData(df_finalTagStats_norm['FacilidadVenta']**Peso_FacilidadVenta)

df_finalTagStats_norm['FlujoDinero'] = NormalizeData(df_finalTagStats_norm['FlujoDinero']**Peso_FlujoDinero)

df_finalTagStats_norm['DebilidadMarcasRivales'] = NormalizeData(df_finalTagStats_norm['DebilidadMarcasRivales']**Peso_DebilidadMarcasRivales)

df_finalTagStats_norm['MercadoNoEspecializado'] = NormalizeData(df_finalTagStats_norm['MercadoNoEspecializado']**Peso_MercadoNoEspecializado)

df_finalTagStats_norm.head(3)
#Calculo final

df_finalTagStats_norm['Clasificador de mejor inversión'] = df_finalTagStats_norm['BajoCostoEntradaMercado'] * df_finalTagStats_norm['FacilidadVenta'] * df_finalTagStats_norm['FlujoDinero'] * df_finalTagStats_norm['DebilidadMarcasRivales'] * df_finalTagStats_norm['MercadoNoEspecializado']

df_finalTagStats_norm['Clasificador de mejor inversión'] = NormalizeData(df_finalTagStats_norm['Clasificador de mejor inversión'])

df_finalTagStats_norm=df_finalTagStats_norm.reset_index()

df_finalTagStats_norm['Sector Comercial'] = df_finalTagStats_norm['tags']

df_finalTagStats_norm = df_finalTagStats_norm.drop(['tags'],axis=1)

df_finalTagStats_norm=df_finalTagStats_norm.set_index('Sector Comercial')

df_finalTagStats_norm = df_finalTagStats_norm.sort_values(['Clasificador de mejor inversión'], ascending = (False))

df_result = df_finalTagStats_norm.head(20)



df_result
from matplotlib import cm

import matplotlib.colors as colors

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):

    new_cmap = colors.LinearSegmentedColormap.from_list(

    'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),

    cmap(np.linspace(minval, maxval, n)))

    return new_cmap  



cmap = plt.get_cmap('Blues')

new_cmap = truncate_colormap(cmap, 0.3, 0.9)



df_result[['BajoCostoEntradaMercado','FacilidadVenta','FlujoDinero','DebilidadMarcasRivales','MercadoNoEspecializado']].plot(kind='bar', stacked=True, cmap=new_cmap,title ='Los 20 mejores sectores para emprender').set_ylabel('Sumatoria de scores')

df_result['Clasificador de mejor inversión'].plot(secondary_y=True, legend =True,color='red', rot =90).set_ylabel('Score geométrico ponderado\n(Clasificador de inversión)')
[]