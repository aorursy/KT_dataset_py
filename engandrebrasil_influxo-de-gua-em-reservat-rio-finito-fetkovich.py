import pandas as pd #Manipulação de dados, biblioteca analoga ao excel.
import numpy as np # Calculos cientificos.
ra = 4000 # Raio do aquífero.
ro = 500 # Raio do reservatório.
h = 10.3 # Espessura da formação.
Φ = 0.13 # Porosidade.
k = 20 # Permeabilidade.
μ = 1.0 # Viscosidade da água.
ct = 93.9*10**(-6) # Compressilidade da Formação.
π = np.pi # Pi = 3,14.
p0 = 200 # Pressão inicial do reservatório.
p1 = 180 # Pressão do reservatório ao longo do primeiro ano de produção.
p2 = 150 # Pressão do reservatório após entre o primeiro e o segundo ano de produção.
p3 = 20 # Pressão do reservatório ao longo do primeiro ano de produção.
f = (0.05255)/(2*π) # Como a questão não trás o angulo de invasão, supus o este valor.
Δt1 = 365 # Intervalo de tempo do primeiro ano de produção -> 365 dias.
Δt2 = 730 # Intervalo de tempo do segundo ano de produção -> 730 dias.
Wi = π*(ra**2-ro**2)*h*Φ # Volume inicial do aquífero.
Wei = ct*Wi*p0 # Influxo máximo do aquífero
J = (2 * π * f * k * h)/(μ * (np.log( ra / ro )-( 3 / 4 ))) # Indice de produtividade do aquífero.
print(" O volume inicial, influxo máximo (Wei) e o índice de produtividade do aquífero são respectivamente: Wi = ",  Wi, "m³,"," Wei =", Wei, "m³,","J = ", J, "m³/dia/kgf") # Apresenta os valores de Wi, Wei e J.
We1 = (Wei/p0)*(p0-p1)*(1-np.exp((-J*p0/Wei)*Δt1)) # Influxo acumulado de água ao final do primeiro ano de produção.
print ("O influxo acumulado e água ao final do primeiro ano de produção é:", We1) # Apresenta o valor de We1.
pressao_media_aquifero = p0*(1-(We/Wei)) # Pressão média do aquífero.
pressao_media_contato_oa = (p1+p2)/2 # Média das pressões no contato óleo água no intervalo Δt2, (180-150)/2.
We2 = (Wei/p0)*(pressao_media_aquifero-pressao_media_contato_oa)*(1-(np.exp((-J*p0*Δt2)/Wei))) # Influxo acumulado do aquífero no intervalo intervalo de tempo Δt2.
We1_2 = We2 + We1 # Influxo acumulado ao final do segundo ano de produção.
print ("O influxo acumulado e água ao final do segundo ano de produção é:", We1_2, "m³") # Apresenta o valor de We1_2.
We_max = (Wei/p0)*(p0-p3) # Quantidade máxima de água ofericida pela aquífero.
print ("A quantidade máxima de água que pode ser oferecida pelo aquífero é:", We_max, "m³") # Apresenta o valor de We_max.
resultados_df = pd.DataFrame(data = [[47900,We1],[146700,We1_2],[1120000,We_max]], columns = ['Resultado Esperado', 'Resultado Calculado'], index =['We 1° ano','We 2° ano','We max']) #Cria DataFrame com os resultados esperado e calculados
resultados_df['Erro Absoluto %'] = (abs(resultados_df['Resultado Esperado']-resultados_df['Resultado Calculado'])/(resultados_df['Resultado Esperado']))*100 #Cria a coluna "Erro Absoluto"
resultados_df