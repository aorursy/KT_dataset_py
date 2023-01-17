import pandas as pd #Manipulação de dados, biblioteca analoga ao excel.

import numpy as np #Calculos cientificos

import seaborn as sns #Plotar gráficos

from sklearn.linear_model import LinearRegression #Regressão linear

from sklearn.model_selection import train_test_split #Dividir dados para treinamento e teste
ro = 1200 #Raio do reservatório

h = 10 #Espessura da formação

Φ = 0.15 #Porosidade

k = 50 #Permeabilidade

μ = 1.0 #Viscosidade da água

cf = 55*10**(-6) #Compressilidade da Formação

cw = 45*10**(-6) #Compressilidade da Formação

π = np.pi #pi

f = (2*π)/(2*π) #Ângulo de invasão
hist_pressao_df = pd.DataFrame(data = [[0,220],[1,180],[2,160],[3,160],[4,160]], columns = ['Tempo [ano]', 'Pressão [kgf/cm²]']) #[t=0,P=220],[t=1,P=180],[t=2,P=160],[t=3,P=160],[t=4,P=160]



hist_pressao_df #Apresenta o DataFrame
ct = cw + cf # Calcula a compressibilidade total

U = 2*π*f*Φ*ct*h*ro**2 #Calcula a constante de influxo do aquífero
def tempo_adimensional(t): #Cria a função para o calculo do tempo admensional

    tempo_admensional = (0.0003484*24*365*k)*t/(Φ*μ*ct*ro**2)

    return tempo_admensional
tabela_aquifero_infinito = pd.read_csv("../input/infinityaquifer/infinity aquifer.csv", sep = ";") #Importa dos dados de td vs wd
fig = sns.lmplot('td', 'wd', data = tabela_aquifero_infinito, fit_reg=False) #Cria gráfico td vs wd
x = tabela_aquifero_infinito.iloc[0:100, :-1].values #vincula a variavel x aos dados de td

y = tabela_aquifero_infinito.iloc[0:100, 1].values #vincula a variavel y aos dados de wd

linearRegressor = LinearRegression() #Cria uma variável para armazenar a regressão

linearRegressor.fit(x, y) #Ajusta a regressão com as variáveis de treinamento

print ("O R² da regressão linear utilizada é", linearRegressor.score(x,y)) #Apresenta o R², que indica a representatividade da curva em relação aos dados
def wd(td): #Cria a função de previsão de wd.

    if td <= 0: #Para valores de td <= 0 o wd será 0.

        wd = 0

    else:

        wd = float(linearRegressor.predict(np.array([[td]])))

    return wd
ΔPi = (hist_pressao_df.iloc[0,1] - hist_pressao_df.iloc[1,1])/2 #Calcula o ΔP para 1 ano de produção
wedi = U * ΔPi * wd(7.0648) #Calcula o influxo acumulado ao final do 1° ano de produção

print ("O influxo acumulado de água ao final do 1° ano de produção é", wedi, "m³.")
hist_pressao_df['Influxo admensional'] = [wd(tempo_adimensional(hist_pressao_df.iloc[4,0]-hist_pressao_df.iloc[0,0])),# wd(td=(4 anos - 0 ano))

                                          wd(tempo_adimensional(hist_pressao_df.iloc[4,0]-hist_pressao_df.iloc[1,0])),# wd(td=(4 anos - 1 ano))

                                          wd(tempo_adimensional(hist_pressao_df.iloc[4,0]-hist_pressao_df.iloc[2,0])),# wd(td=(4 anos - 2 anos))

                                          wd(tempo_adimensional(hist_pressao_df.iloc[4,0]-hist_pressao_df.iloc[3,0])),# wd(td=(4 anos - 3 anos))

                                          wd(tempo_adimensional(hist_pressao_df.iloc[4,0]-hist_pressao_df.iloc[4,0]))]# wd(td=(4 anos - 4 anos))

                                          # para um intervalo de tempo maior, utilize uma estrutura de repetição.

hist_pressao_df['ΔPj'] = [(hist_pressao_df.iloc[0,1]-hist_pressao_df.iloc[1,1])/2, #(ΔP0-ΔP1)/2

                          (hist_pressao_df.iloc[0,1]-hist_pressao_df.iloc[2,1])/2, #(ΔP0-ΔP2)/2

                          (hist_pressao_df.iloc[1,1]-hist_pressao_df.iloc[3,1])/2, #(ΔP1-ΔP3)/2

                          (hist_pressao_df.iloc[2,1]-hist_pressao_df.iloc[4,1])/2, #(ΔP2-ΔP4)/2

                          (hist_pressao_df.iloc[4,1]-hist_pressao_df.iloc[4,1])/2] #(ΔP3-ΔP3)/2

hist_pressao_df #Apresenta o DataFrame
hist_pressao_df['ΔPj*Wd'] = hist_pressao_df['Influxo admensional']*hist_pressao_df['ΔPj'] #Cria a coluna "ΔPj*Wd"
hist_pressao_df #Apresenta o Data Frame com a coluna "ΔPj*Wd"
We = U*hist_pressao_df['ΔPj*Wd'].sum()#Calcula o influxo acumulado

print ("O influxo acumulado de água ao final do 4° ano de produção é", We, "m³.")
resultados_df = pd.DataFrame(data = [[156892,wedi],[1087117,We]], columns = ['Resultado Esperado', 'Resultado Calculado'], index =['Wed 1° ano','Wed 4° ano']) #Cria DataFrame com os resultados esperado e calculados
resultados_df['Erro Absoluto'] = (resultados_df['Resultado Esperado']-resultados_df['Resultado Calculado'])/(resultados_df['Resultado Esperado']) #Cria a coluna "Erro Absoluto"

resultados_df #Apresenta DataFrame Final