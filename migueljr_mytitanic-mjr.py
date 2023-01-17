import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
%matplotlib inline
# Reading data
dados = pd.read_csv('./input/titanic/train.csv')

# Use at the end to compare
y_comparar_fim = dados[['Survived']]
# Survivors
dados.hist('Survived');
dados.head(6)
# Convert Sex Column to number 
# Male = 0
# Female = 1

def sex(x):
    if x.lower() == 'female':
        return 1
    elif x.lower() == 'male':
        return 0

# Apply
dados['Sex'] = dados['Sex'].apply(sex)
# Check
dados.head(6)
RETIRAR = ['PassengerId','Name','Ticket','Cabin']

dados.drop(RETIRAR,axis=1,inplace = True)
# Check
dados.head()
#convert string to number
dados = pd.get_dummies(dados)
# Correlation graph between columns
plt.figure(figsize=(8,6))
sns.heatmap(dados.corr(),annot = True);
#Columns will be used
UTILIZADOS = ['Survived','Pclass','Sex','Age','Fare']

dados = dados[UTILIZADOS]
# Check
dados.head(4)
# Search for null values
sns.heatmap(dados.isnull(),cbar = False);

sns.boxplot(x='Pclass', y='Age', data = dados)
plt.grid()
def age(x):
    age = x[0]
    pclass = x[1]
    
    if np.isnan(age) == True:
        
        if pclass == 1:
            return 37
        elif pclass == 2:
            return 29
        elif pclass == 3:
            return 24
    else:
        return age

dados['Age'] = dados[['Age','Pclass']].apply(age,axis=1)
# Check
dados.head()
X = dados.drop('Survived',axis=1)
y = dados[['Survived']]
X.shape,y.shape

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size = 0.1,
    random_state = 0,
    shuffle = True,
)
X_train.shape,y_train.shape
# Model - n is a multiple used to change neural network
def creat_model(n):
    
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(
      units = n*2,
      activation = 'relu',
      input_shape = (X_train.shape[1],)
    ))
    

    model.add(tf.keras.layers.Dropout(0.1))
    
    model.add(tf.keras.layers.Dense(
      units = n*3,
      activation = 'relu',

    ))
    
    model.add(tf.keras.layers.Dropout(0.1))
    
    model.add(tf.keras.layers.Dense(
      units = 1,
      activation = 'sigmoid',
    ))
    
    # Compile
    model.compile(loss = 'binary_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])
    
    return model

# Train
def treinar(model, epochs):

    history = model.fit(
      X_train,
      y_train,
      validation_split=0.1,
      epochs = epochs,
      shuffle = True,
      )
    return history


#This function separates the values of predictions
#into 0 and 1
def cond(x):
    if x >= 0.5:
      return 1
    else:
      return 0

def predict(X,y,model):

    pred_train = model.predict(X)
    pred_df = pd.DataFrame()
    pred_df['pred'] = pred_train[:,0]
    pred_df['pred'] = pred_df['pred'].apply(cond)
    y = y.reset_index().drop('index',axis=1)
    y_train_pred = y.join(pred_df)
    k = 0
    total = len(y_train_pred['pred'])
    for x,y in zip(y_train_pred['pred'],y_train_pred['Survived']):
      if x == y:
          k = k+1
    p = k/total
    p_ = np.around(p*100,2)
    print(f"Accuracy porcentage {p_}%")
    return p_

def plot(history):

  plt.figure()
  plt.plot(history.history['accuracy'],'or', label = 'acc',alpha = 0.3)
  plt.plot(history.history['val_accuracy'], 'ob', label = 'acc_val',alpha = 0.3)
  plt.legend()
  plt.grid()

  plt.figure()
  plt.plot(history.history['loss'],'or', label = 'loss',alpha = 0.3)
  plt.plot(history.history['val_loss'],'ob', label = 'loss_val',alpha = 0.3)
  plt.legend()
  plt.grid()
v ={}
# n (number of neurons defined in the function above).
for n in range(200,301,20):
    # Can varied the epochs
    for epochs in range (40,41,1):
        
        #Apply Model and Training functions
        model = creat_model(n)
        history= treinar(model,epochs)
        
        #v [0] = Percentage of correct training data
        #v [1] = Percentage of correct validation data
        #v [2] = History (used to make the plot)
        #v [3] = Model (used to choose which trained model I will apply)
        v[str(n)+'_'+str(epochs)] = [predict(X_train,y_train,model),predict(X_val,y_val,model),history,model]
for k in v.keys():
    print(k)
    predict(X,y_comparar_fim,v[k][3])
    
# neuron_epochs
# Example of accuracy and loss data depending on the epochs
plot(v['300_40'][2])
dados_test = pd.read_csv('./input/titanic/test.csv')

dados_x_test = dados_test[UTILIZADOS[1:]]# take out survived 
# Check
dados_x_test.head()
dados_x_test['Sex'] = dados_x_test['Sex'].apply(sex)
sns.heatmap(dados_x_test.isnull(),cbar=False);
sns.boxplot(x='Pclass', y='Age', data = dados_x_test);
plt.grid()
def age_new(x):
    age = x[0]
    pclass = x[1]
    
    if np.isnan(age) == True:
        
        if pclass == 1:
            return 42
        elif pclass == 2:
            return 27
        elif pclass == 3:
            return 24
    else:
        return age
    
dados_x_test['Age'] = dados_x_test[['Age','Pclass']].apply(age_new,axis=1)
dados_x_test = scaler.fit_transform(dados_x_test)
prev_env = v['300_40'][3].predict(dados_x_test)
dados_env = pd.DataFrame()
dados_env['PassengerId'] = dados_test['PassengerId']
dados_env['Survived'] = prev_env[:,0]
dados_env['Survived'] = dados_env['Survived'].apply(cond)
#Save data
#dados_env.to_csv('./Titanic_Predictions.csv',index=False)