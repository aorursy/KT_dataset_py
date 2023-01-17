import pandas as pd
from sklearn.tree import DecisionTreeClassifier
train_data = pd.read_csv('../input/train.csv', sep=',')
'''a função .head(n=5) mostra os n primeiros valores da tabela. Por padrão, caso omitido, n será n=5.'''

train_data.head()
'''a função .describe() gera uma estatística descritiva que resume a tendência, dispersão
e a forma da distribuição de um conjunto de dados excluindo os NaN values.'''

train_data.describe()
clean_data = train_data.copy()
'''
.isnull() retorna True se o conteúdo do atributo for NaN, e False caso contrário.
.any() retorna True se ao menos um elemento do conjunto de dados analisados conter True e retorna False 
caso contrário.
'''

clean_data.isnull().any()
del clean_data['Cabin']
del clean_data['Embarked']
'''
    dropna() irá descartar todas as linhas com dados omissos. O parâmetro subset=['Age'] especifica que
    a função .dropna() deverá descartar apenas as linhas com a omissão de dados na coluna Age, ou seja,
    qualquer outro NaN value continuará nos dados se não estiver na coluna Age.
'''

clean_data = clean_data.dropna(subset=['Age'])
'''
    clean_data['Sex'] == 'male' irá gerar uma lista com True e False para todos os valores do atributo
    clean_data['Sex'].
    Quando multiplicamos um valor Boleano por 1, estamos dizendo que todos os valores boleano True convertam
    para o valor inteiro 1 e False se converta no inteiro 0.
'''


clean_data['Sex'] = (clean_data['Sex'] == 'male')*1
clean_data.head()
y = clean_data[['Survived']].copy()
y.head()
data_features = ['Pclass', 'Sex', 'Age', 'Fare']
X = clean_data[data_features].copy()
X.head()
y.head()
survivor_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
survivor_classifier.fit(X, y);
test_data = pd.read_csv('../input/test.csv', sep=',')
test_data.head()
test_data.isnull().any()
test_data['Age'].fillna(test_data.Age.mean(), inplace=True)
test_data['Fare'].fillna(test_data.Fare.mean(), inplace=True)
test_data['Sex'] = (test_data['Sex'] == 'male')*1
y_pred = survivor_classifier.predict(test_data[data_features])
subimission = pd.DataFrame({
        "PassengerId": test_data['PassengerId'],
        "Survived": y_pred
    })
subimission.head()
subimission.to_csv('titanic_n.csv', index=False)