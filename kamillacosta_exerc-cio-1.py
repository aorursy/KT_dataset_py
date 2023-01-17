import numpy as np # Importando biblioteca de Algebra linera

import pandas as pd # Importando biblioteca pandas responsável por explorar dados

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier #Importanto biblioteca responsável pelo modelo 

from sklearn.model_selection import train_test_split



import os

print(os.listdir("../input")) # Listando todos os arquivos existentes do dataset



# Salvando caminho de arquivo para facilitar acesso a variável

train_data = '../input/train.csv'

# Lendo e salvando os dados no DataFrame indicado como data

data = pd.read_csv(train_data)

# Imprimindo resumo dos dados

data.describe()
# Transformando strings em 0

data = data.apply(pd.to_numeric, errors = 'coerse')

data = data.replace(np.NaN, 0)

data
#Concatenando arquivos de validação e teste 

test_path = '../input/test.csv'

test =  pd.read_csv(test_path)



valid_path = '../input/valid.csv'

valid =  pd.read_csv(valid_path)



general_data = pd.concat([test, valid])

general_data.to_csv('general_data.csv', index = None)

general_data
# Transformando strings em 0

general_data = general_data.apply(pd.to_numeric, errors = 'coerse')

general_data = general_data.replace(np.NaN, 0)

general_data
#Selecionando colunas que serão treinadas

y = data.sale_price

features = ['lot','gross_square_feet','land_square_feet']    

X = data[features]



# Dividindo dados de treinamento e validação

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state = 1)



#Definindo modelo

model = RandomForestRegressor(random_state=1, max_depth= 5, n_jobs= -1)



#Adaptando modelo aos dados

model.fit(X_train, y_train)

valid = model.predict(X_valid)
test = model.predict(general_data[features])

output = pd.DataFrame({'sale_id': general_data.sale_id,

                       'sale_price': test})

output.to_csv('predicao_imoveis', index=False)