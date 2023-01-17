# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
with open('/kaggle/input/pesquisa-data-hackers-2019/data_dictionary.txt') as file:

    content = file.read()

    print(content)

    
data = pd.read_csv('/kaggle/input/pesquisa-data-hackers-2019/datahackers-survey-2019-anonymous-responses.csv')

pd.set_option("display.max_columns", 200)

data.head()
data["('P5', 'living_state')"].value_counts()
data["('D3', 'anonymized_degree_area')"].value_counts()
data["('P8', 'degreee_level')"].value_counts()
data["('P18', 'time_experience_before')"].value_counts()
data["('P2', 'gender')"].value_counts()
data["('P22', 'most_used_proggraming_languages')"].value_counts()
data["('D1', 'living_macroregion')"].value_counts()
data["('P10', 'job_situation')"].value_counts()
# Solução por Lucas Trajano https://www.kaggle.com/lucastrajano Mais rápido que meu regex...

data['meanSal'] = data["('P16', 'salary_range')"].fillna('$ 0/').apply(lambda x: #get the mean in ranges

                    int( #transform all in int in the end

                    (int(str(x)[str(x).rfind(' ')+1:str(x).rfind('/')].replace('.','')) # Get max in range 

                    +

                    int(str(x)[str(x).find('$')+2:str(x).find('/')].replace('.','')) #Get min

                    )/2)) #divide by 2
data['meanSal']
data["('P5', 'living_state')"].value_counts().plot.bar(color='#16003d', title='Regiões que possuem mais Entendedores de Dados')

plt.ylabel("Número de pessoas")

plt.show()
data["('P2', 'gender')"].value_counts().plot.bar(color=['#f5005a', 'blue'], title='Homens vs Mulheres na área',)

plt.xticks(rotation=0)

plt.show()
data["('P22', 'most_used_proggraming_languages')"].value_counts().nlargest(3).plot.bar(color=['green', 'orange', 'blue'])

plt.title("Linguagens mais usadas por 'Data Understanders'")

plt.xticks(rotation=0)

plt.show()
data["('P8', 'degreee_level')"].value_counts().plot.barh(title='Nível de Ensino', color='#16003d')

plt.show()
data["('P10', 'job_situation')"].value_counts().plot.barh(title='Situação de Trabalho', color='#16003d')

plt.show()
data["('D3', 'anonymized_degree_area')"].value_counts().plot.barh(color='#16003d')

plt.title('Área de Graduação dos Partcipantes')

plt.show()
data.groupby("('P10', 'job_situation')")['meanSal'].mean()
data.groupby("('P10', 'job_situation')")['meanSal'].mean().nlargest(6).plot.barh(color='#16003d')

plt.title("Situação de Trabalho vs Média Salarial")

plt.ylabel("")

plt.show()
data.groupby("('P5', 'living_state')")['meanSal'].mean()
data.groupby("('P5', 'living_state')")['meanSal'].mean().plot.bar(color='#16003d')

plt.title("Média Salarial nas regiões brasileiras")

plt.xlabel("")

plt.show()
data.groupby("('P5', 'living_state')")['meanSal'].sum()
data.groupby("('P5', 'living_state')")['meanSal'].sum().plot.bar(color='#16003d')

plt.title("'PIB' Salarial nas regiões brasileiras")

plt.xlabel("")

plt.ylabel("EM R$ X 10⁶")

plt.show()
# Cerca de 1231R$ a mais para os homens

data.groupby("('P2', 'gender')")['meanSal'].mean()
data.groupby("('P2', 'gender')")['meanSal'].mean().plot.bar(color=['#f5005a', 'blue'])

plt.title("Comparativo entre os salários de Homens e Mulheres")

plt.xlabel("")

plt.ylabel("Média Salarial em R$")

plt.xticks(rotation=0)

plt.show()
data.groupby(["('P2', 'gender')", "('P5', 'living_state')"])['meanSal'].mean()
# Diminuindo a Frase pra ficar melhor no gráfico

col_ad = "('P17', 'time_experience_data_science')"

data.loc[data[col_ad] == 'Não tenho experiência na área de dados', col_ad] = 'Sem experiência'



men = data[data.loc[:, "('P2', 'gender')"] == 'Masculino'] #Linhas com sexo Masculino

men_state_mean = men.groupby("('P5', 'living_state')")['meanSal'].mean() #Média Masculina por região



woman = data[data.loc[:, "('P2', 'gender')"] == 'Feminino'] #Linhas com sexo Feminino

woman_state_mean = woman.groupby("('P5', 'living_state')")['meanSal'].mean() #Média Feminina por região
menxp = men.groupby("('P17', 'time_experience_data_science')")['meanSal'].mean()

womanxp = woman.groupby("('P17', 'time_experience_data_science')")['meanSal'].mean()
plt.barh(men_state_mean.index, men_state_mean.values, color='blue')

plt.barh(woman_state_mean.index, woman_state_mean.values, color='#f5005a')



plt.xlabel("Média Salarial em R$")

plt.ylabel("Estados")



plt.title("Comparativo entre o salário de Homens e Mulheres por Região")



plt.legend(['Homens', 'Mulheres'])

plt.show()
plt.barh(menxp.index, menxp.values, color='blue')

plt.barh(womanxp.index, womanxp.values, color='#f5005a')



plt.xlabel("Média Salarial em R$")

plt.ylabel("Experiência na área de dados")

plt.legend(['Homens', 'Mulheres'])



plt.title("Comparativo entre o salário de Homens e Mulheres com mesma experiência")

plt.show()
col_xp = "('P18', 'time_experience_before')"

data.loc[data[col_xp] == 'Não tive experiência na área de TI/Engenharia de Software antes de começar a trabalhar na área de dados', col_xp] = 'Sem xp em TI antes'
data.groupby("('P18', 'time_experience_before')")['meanSal'].mean().plot.barh(color='red') #Vermelho pra variar

plt.title("Experiência em TI antes vs Salário na área de Dados")

plt.ylabel("")

plt.xlabel("Salário")

plt.show()