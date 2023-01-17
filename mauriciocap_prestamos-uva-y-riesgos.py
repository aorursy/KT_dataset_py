import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt # plots
#definamos funciones en python para calcular la cuota y el credito
def cuota(deudaAntes,tasaUnPeriodo,cantCuotasRestantes):
    "devuelve la cuota fija para el resto de un prestamo sistema frances"
    return (deudaAntes*tasaUnPeriodo/(1-((1+tasaUnPeriodo)**(-cantCuotasRestantes))))

def prestamo(totalPrestado,tasaNominalAnual,anios):
    "devuelve un diccionario con todos los datos y calculos del prestamo, \
    asi se puede ver lindo como output pero tambien usar en otras expresiones"
    cuotas= anios*12
    tasaMensual= tasaNominalAnual/12
    vcuota= cuota(totalPrestado,tasaMensual,cuotas)
    totalPagado= vcuota*cuotas
    cuotasCapital= totalPrestado/vcuota
    return {'totalPrestado': totalPrestado, 'tasaNominalAnual':tasaNominalAnual,'anios':anios, 'cuotas': cuotas, 'tasaMensual': tasaMensual, 'cuota': vcuota, 'totalPagado': totalPagado, 'prestadoVsPagado': totalPagado/totalPrestado, 'cuotasQuePodriaHaberAhorradoYComprar': totalPrestado/vcuota}
p35= prestamo(50000,.035,30) #50000 UVA a 30 años tasa 3.5%
p35
p50= prestamo(50000,.05,30) #50000 UVA a 30 años tasa 5%
p50
p70= prestamo(50000,.07,30) #50000 UVA a 30 años tasa 7%
p70
def sigoDebiendo(nroCuota,valorCuota,tasaNominalAnual,anios):
    "devuelve el capital vivo, ver [E5]"
    tasaMensual= tasaNominalAnual/12
    return (valorCuota*(1-((1+tasaMensual)**-((anios*12)-nroCuota)))/tasaMensual)

def detallePorMes(prestamo):
    "fabrico un dataframe/series con lo que sigo debiendo en cada mes"
    idx= range(0,prestamo['anios']*12+1)
    return pd.Series(map(lambda nroCuota: sigoDebiendo(nroCuota,prestamo['cuota'],prestamo['tasaNominalAnual'],prestamo['anios']),idx),index=idx).to_frame(name='debo')
    
#calculo para el prestamo mas barato y el mas caro
detalle35= detallePorMes(p35)
detalle70= detallePorMes(p70)
detalle35.head()
ax=detalle35.plot(legend="pepe",color='blue')
detalle70.plot(ax=ax,legend=True,color='orange')
ax.legend(["TNA=3.5%","TNA=7%"])
plt.title("Cuanto CAPITAL sigo debiendo despues de cada cuota");
tasaInflacionAnual=15/100 #Una inflacion "moderada" para Argentina en los proximos años
tasaMensual= (1+tasaInflacionAnual)**(1/12) #para que UVA_mes_5=UVA_mes_4*(1+tasaMensual)
def ff(row):
    return row['debo']*(tasaMensual**row.name)
detalle70ajustadoInflacion= detalle70.apply(ff,axis=1)
detalle70ajustadoInflacion.head()
detalle70ajustadoInflacion.plot()
plt.title("Cuanto CAPITAL sigo debiendo despues de cada cuota ajustado por inflacion");