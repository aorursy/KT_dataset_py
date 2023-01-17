import pandas as pd 
from sklearn.neural_network import MLPRegressor

# nahrani ucicich date
df = pd.read_csv('../input/train.csv')

print("\njak vypada 'ucici' soubor... (neni tam is_old, a treba je dulezity :D)")
print(df.head(5))

# data pro uceni
X = df[['km']] 
y = df['price'] 

# create model and learn
model = MLPRegressor(max_iter=25, solver='lbfgs');
model.fit(X, y);

# nahrani dat ktera mame predikovat
df_test = pd.read_csv('../input/test.csv')

print("\njak vypadaji data pro predikci...")
print(df_test.head(5))

# predikce
X_test = df_test[['km']] 
df_test['price'] = model.predict(X_test)

# zahozeni dat ktera nepotrebuju do vysledneho souboru
result = df_test.drop(['km', 'is_old'], axis=1)

print("\njak vypada vysledny 'soutezni' soubor... nahrajte ho na kaggle...")
print(result.head(5))

# tenhle vysledek je potreba nahrat na kaggle
result.to_csv('result.csv', index=False)

