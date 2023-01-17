# Comecei importando as bibliotecas do Pandas e scikit-learn
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
# Agora importarei os dados de treino e de teste em respectivas variáveis
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.head()
#Utilizando o método drop(),eu removi as colunas. Perceba que, 
#como preciso remover mais de uma coluna ao mesmo tempo, eu devo passá-las na forma de uma lista, 
#que no Python é caracterizada pelos colchetes []. Em seguida, eu informo o argumento axis=1, 
#que indica que eu quero retirar a coluna inteira, e não apenas uma linha. 
#Por fim, informo o argumento inplace=True, que salva a alteração direto no dataset, 
#sem que eu tenha que armazenar o valor na variável.
#_______________________________
# Para remover os dados irrelevantes será usado o código abaixo:
train.drop(['Name', 'Ticket','Cabin'], axis=1, inplace=True)
# Para o test também deve ser retirado, pois ao gerar o modelo de ML pode ocorrer erros
test.drop(['Name', 'Ticket','Cabin'], axis=1, inplace=True)
# Agora será usado o método get_dummies() do pandas para transformar os caracteres em numeros.
# Este método usará o OneHotEncoder para isso.
# A partir disso será criado um novo data-frame para armazenar os dados em termos numéricos para executar o ML
new_data_train = pd.get_dummies(train)
new_data_test = pd.get_dummies(test)



new_data_train.head()
new_data_test.head()
# O próximo passo é verificar se há valores nulos
# new_data_train indica o DataFrame utilizado;
# .isnull() é uma função que retorna todos os valores nulos encontrados;
# .sum() irá somar todas as ocorrências e agrupá-las;
# .sort_values(ascending=False) ordenará os dados. Ao passar o argumento ascending=False eu indico querer ordenar do maior para o menor.
# Para isso será utilizado o código abaixo:
new_data_train.isnull().sum().sort_values(ascending=False).head(10)
new_data_train['Age'].fillna(new_data_train['Age'].mean(), inplace=True)
new_data_test['Age'].fillna(new_data_test['Age'].mean(), inplace=True)
# Basicamente o que foi feito:
# .fillna(). Como argumento, eu informo o que deve ser inserido nos campos que não tiveram um valor definido. 
# Nesse caso, foi inserido  a média de idades da coluna Age utilizando o código new_data_train["Age"].mean().
# O argumento inplace=True irá inserir e salvar a informação direto no DataFrame.
new_data_test.isnull().sum().sort_values(ascending=False).head(10)
# Foi feito o mesmo para o cabeçalho Fare
new_data_test['Fare'].fillna(new_data_test['Fare'].mean(), inplace=True)
new_data_test.isnull().sum().sort_values(ascending=False).head(10)
# como Feature será usado a variável passenger_datas para saber os dados dos passageiros
# como Target se o mesmo sobreviveu dai a varável disembodied ele irá receber os dados da coluna 'Survived'
passenger_datas = new_data_train.drop('Survived', axis=1)
disembodied = new_data_train['Survived']
# tree = que armazenará uma instância do objeto DecisionTreeClassifier()
# max_depth = argumento que define a profundidade máxima da árvore
# fit() = cria o modelo passando os argumentos criados
tree = DecisionTreeClassifier(max_depth = 18, random_state = 0)
tree.fit(passenger_datas, disembodied )
# Score() = verifica os resultados baseado na pergunta
tree.score(passenger_datas, disembodied)
# O arquivo deve ser enviado na extesão .csv com os cabeçalhos PassengerId e Survived.
# pd.DataFrame()= irá criar um novo DataFrame e armazená-lo variável 'submission', 
# que será exportada em formato de arquivo .csv
submission = pd.DataFrame()
submission['PassengerId'] = new_data_test['PassengerId']
submission['Survived'] = tree.predict(new_data_test)
submission.to_csv('submission.csv', index=False)