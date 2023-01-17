#Number of generators : 5 
#Numero de generadores : 5
n_gen = 5
# Consumption in L/h for 1600Kw
# Consumo en L/h para generador de 1600Kw
consumption = 370
# Endurance in days
# Autonomía en días
endurance = 60
# Consumption in Tn/day
# Consumo en Toneladas/dia
CPD = (consumption*24/1000)*n_gen * 0.8
print('El Consumo de combustible es de: ' + str(round(CPD, 1)) + ' Toneladas/dia')
#Tons of Fuel needed for 90 days
#Toneladas de Combustible necesarias para 90 dias
TOF = 1.5*endurance*CPD
print('Toneladas de Combustible necesarias para 90 dias: '+ str(round(TOF, 1)) + ' Tons')
# Price of MDO € per Ton
# Precio en € del Diesel Marino por Tonelada
POF = 600 
# Fuel cost per 90 days
# Coste del combustible para los 90 días
COF = POF*TOF
print('Coste de Combustible para 90 dias: '+ str(round(COF, 1)) + ' €')
# Crew
# Tripulación
crew = 50
# Yearly median salary for qualified staff in €
# Salario medio anual para personal cualificado en €
salary = 60000
# Staff cost for 3 months
# Costes de personal para 3 meses(dos de operación, medio de ida y medio de vuelta)
COS = salary*crew/4
print('Coste de Personal para 3 meses: '+ str(round(COS, 1)) + ' €')
# Food 12€ per person & day
# Alimentación 12€ por persona y día
COA = 12*crew*90
print('Coste de Alimentación para 3 meses: '+ str(round(COA, 1)) + ' €')
# Total Cost for now
# Coste total calculado hasta ahora
CT = COF+COS+COA
print('Coste Total en combustible, personal y comida : '+ str(round(CT, 1)) + ' €')
