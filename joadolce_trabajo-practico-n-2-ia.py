import numpy as np



import matplotlib

%matplotlib inline



import matplotlib.pyplot as plt



import warnings

warnings.simplefilter('ignore')



# Establecemos la aleatoridad de los numeros mediante un "seed".

np.random.seed(0)
# Creamos un par de metodos auxiliares que nos permitirán crear los datos las redes neuronales.



def create_data(func, sample_size, std, domain=[0, 1]):

    x = np.linspace(domain[0], domain[1], sample_size)

    np.random.shuffle(x)

    t = func(x) + np.random.normal(scale=std, size=x.shape)

    return x, t



def sinusoidal(x):

    return np.sin(2 * np.pi * x)



def create_sinusoidal_data():

    x_train, y_train = create_data(sinusoidal, 10, 0.2)

    x_test = np.linspace(0., 1.0, 50)

    y_test = sinusoidal(x_test)

    return x_train, y_train, x_test, y_test



def polynomial_features(x, M):

    if np.isscalar(x):

        return x**np.arange(1, M+1)

    elif (isinstance(x, (list, tuple)) or

          (isinstance(x, (np.ndarray,)) and x.ndim == 1)):

        return np.vstack([x_**np.arange(1, M+1) for x_ in x])

    else:

        raise TypeError("wrong type")
x_train, y_train, x_test, y_test = create_sinusoidal_data()



plt.figure(figsize=(20,5))

plt.suptitle('Mini dataset de Prueba', fontsize = 20)

plt.plot(x_test, y_test)

plt.scatter(x_train, y_train, color='red')

plt.ylabel('$\sin(2 \pi x)$')

plt.xlabel('$x$')

plt.grid()

plt.show()
# Importamos las librerías de Torch, para ser utilizadas.

import torch



# Numero de parámetros

P = 3



# Si es posible, se enviará el código a alguna GPU, y de esta forma acelerar el proceso de ejecución.

# De no estar disponible, se utilizará el CPU.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Generamos los datos de train y test para la ejecución de los ejercicios.

x_train_vec = polynomial_features(x_train, P)

y_train_vec = y_train.reshape(-1, 1)



x_test_vec = polynomial_features(x_test, P)

y_test_vec = y_test.reshape(-1, 1)



# Trasladamos estos vectores generados previamente a "tensores", utilizados en Pytorch.

x_train_tensor = torch.from_numpy(np.float32(x_train_vec)).to(device)

y_train_tensor = torch.from_numpy(np.float32(y_train_vec)).to(device)

n_train = len(x_train_tensor)



x_test_tensor = torch.from_numpy(np.float32(x_test_vec)).to(device)

y_test_tensor = torch.from_numpy(np.float32(y_test_vec)).to(device)

n_test = len(x_test_tensor)
# Importamos "torch.nn" para poder crear las capas de la red.

from torch import nn



class LinearRegression(nn.Module):

    def __init__(self, num_features):

        super().__init__() 

        

        # La capa que aplica una transformación lineal a los datos entrantes.

        self.fc = nn.Linear(in_features=num_features, out_features=1, bias=True)

        

    def forward(self, x):

        out = self.fc(x)

        return out
from prettytable import PrettyTable

from torch.optim import SGD



# Creamos la tabla que nos permitirá mostrar los resultados obtenidos.

resultados = PrettyTable()

resultados.field_names = ['Regularización', 'Learning Rate', 'Weight Decay', 'Epochs', 'Min Loss']



# Inicializamos las variables.

# Generamos un diccionario donde guardar los mejores parámetros obtenidos del algoritmo.

parametros_L1 = {"lr": 0, "wd": 0, "epochs": 0}



# Variable creada para guardar el modelo optimo, es decir, aquel en el que se obtenga menor "Loss".

min_loss = 999999999



# Establecemos ejemplos de parámetros para realizar la búsqueda en grilla.

LEARNING_RATE = [0.001, 0.01, 0.1]

WEIGHT_DECAY = [0.001, 0.01, 0.1]

EPOCHS = [500, 1000, 5000, 10000]



# Creamos un "criterio" que mide el error absoluto medio (MAE).

criterion = nn.L1Loss(reduction='sum')



for lr in LEARNING_RATE:

    for wd in WEIGHT_DECAY:        

        for epochs in EPOCHS:

            

            # Instanciamos el modelo

            model = LinearRegression(num_features=P)

            

            # Lo enviamos al dispoitivo disponible (GPU/CPU).

            model.to(device)

            

            # Implementamos "descenso de gradiente estocástico".

            optimizer = SGD(model.parameters(), lr=lr, weight_decay=wd)

            

            # Seteamos el modelo para entrenar.

            model.train()

            

            train_loss = []

            test_loss = []



            for epoch in range(epochs):

                

                # Limpiamos todos los gradientes.

                optimizer.zero_grad()

                

                # Realizamos la predicción con el modelo (forward pass).

                y_pred = model(x_train_tensor)

                

                # Calculamos la pérdida (loss) y la dividmos por el numero de obs.

                loss = criterion(y_pred, y_train_tensor) / n_train

                

                # Calculamos los gradientes (backward pass).

                loss.backward()

                

                # Realizamos el paso de optimización, actualizando los pesos.

                optimizer.step()                                    

                train_loss.append(loss.detach().item())



                with torch.no_grad():

                    y_pred = model(x_test_tensor)

                    loss = criterion(y_pred, y_test_tensor) / n_test

                    test_loss.append(loss.item())

            

            # Seteamos el modelo para predecir, sin variar sus pesos.

            model.eval()

            y_pred = model(x_test_tensor)

            after_train = criterion(y_pred, y_test_tensor) / n_test



            if after_train.item() < min_loss:

                

                # Actualizamos los valores del diccionario.

                parametros_L1.update({"lr": lr,"wd": wd,"epochs": epochs})

                min_loss = after_train.item()

                

                # Obtenemos una instancia del modelo, con sus respectivos pesos y características.

                best_model_L1 = type(model)(num_features=P)

                best_model_L1.load_state_dict(model.state_dict())

                

                best_train_loss_L1 = train_loss

                best_test_loss_L1 = test_loss



# Guardamos los mejores resultados e imprimimos los mejores parametros.

resultados.add_row(['L1', str(parametros_L1.get('lr')), str(parametros_L1.get('wd')), str(parametros_L1.get('epochs')), str(round(min_loss, 3))])            

print("Mejores parámetros:",parametros_L1)
# Inicializamos las variables.

# Generamos un diccionario donde guardar los mejores parámetros obtenidos del algoritmo.

parametros = {"lr": 0, "wd": 0, "epochs": 0}



# Variable creada para guardar el modelo optimo, es decir, aquel en el que se obtenga menor "Loss".

min_loss = 99999



# Establecemos ejemplos de parámetros para realizar la búsqueda en grilla.

LEARNING_RATE = [0.5, 0.3, 0.2, 0.1]

WEIGHT_DECAY = [0.1, 0.01, 0.001, 0.0001]

EPOCHS = [500, 1000, 5000, 10000]



# Creamos un criterio que mide el error cuadrático medio (MSE).

criterion = nn.MSELoss(reduction='sum')



for lr in LEARNING_RATE:

    for wd in WEIGHT_DECAY:

        for epochs in EPOCHS:

            

            # Instanciamos el modelo

            model = LinearRegression(num_features=P)

            # Lo enviamos al dispoitivo disponible (GPU/CPU).

            model.to(device)

            

            # Implementamos "descenso de gradiente estocástico".

            optimizer = SGD(model.parameters(), lr=lr, weight_decay=wd)

            

            # Seteamos el modelo para entrenar.

            model.train()

            

            train_loss = []

            test_loss = []



            for epoch in range(epochs):

                

                # Limpiamos todos los gradientes.

                optimizer.zero_grad()

                

                # Realizamos la predicción con el modelo (forward pass).

                y_pred = model(x_train_tensor)

                

                # Calculamos la pérdida (loss) y la dividmos por el numero de obs.

                loss = criterion(y_pred, y_train_tensor) / n_train

                

                # Calculamos los gradientes (backward pass).

                loss.backward()

                

                # Realizamos el paso de optimización, actualizando los pesos.

                optimizer.step()



                train_loss.append(loss.detach().item())



                with torch.no_grad():

                    y_pred = model(x_test_tensor)

                    loss = criterion(y_pred, y_test_tensor) / n_test

                    test_loss.append(loss.item())

            

            # Seteamos el modelo para predecir, sin variar sus pesos.

            model.eval()

            y_pred = model(x_test_tensor)

            after_train = criterion(y_pred, y_test_tensor) / n_test



            if after_train.item() < min_loss:

                

                # Actualizamos los valores del diccionario.

                parametros.update({"lr": lr,"wd": wd,"epochs": epochs})

                min_loss = after_train.item()

                

                # Obtenemos una instancia del modelo, con sus respectivos pesos y características.

                best_model = type(model)(num_features=P)

                best_model.load_state_dict(model.state_dict())

                

                best_train_loss = train_loss

                best_test_loss = test_loss



# Guardamos los mejores resultados e imprimimos los mejores parámetros.

resultados.add_row(['L2', str(parametros.get('lr')), str(parametros.get('wd')), str(parametros.get('epochs')), str(round(min_loss, 3))])  

print("Mejores parámetros: ",parametros)
# Imprimimos los valores optimos previamente obtenidos de ambos modelos.

print(resultados)



# Graficamos los resultados obtenidos previamente.

plt.figure(figsize=(20,10))

plt.suptitle('L1 vs L2', fontsize = 20)



plt.subplot(221)

plt.title('Valores de Loss (L1)', fontsize= 15)

plt.plot(np.arange(parametros_L1["epochs"]), best_train_loss_L1)

plt.plot(np.arange(parametros_L1["epochs"]), best_test_loss_L1)

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.grid()



# Obtenemos los valores predichos por el modelo.

best_model_L1.to(device)

best_model_L1.eval()  

y_pred = best_model_L1(x_test_tensor)

y_pred = y_pred.cpu().detach().numpy().squeeze()



plt.subplot(222)

plt.title('Real vs Predicción (L1)', fontsize= 15)

plt.plot(x_test, y_test)

plt.plot(x_test, y_pred, color='orange')

plt.scatter(x_train, y_train, color='red')

plt.ylabel('$\sin(2 \pi x)$')

plt.xlabel('$x$')

plt.legend(['real', 'predict', 'train'])

plt.grid()



plt.subplot(223)

plt.title('Valores de Loss (L2)', fontsize= 15)

plt.plot(np.arange(parametros["epochs"]), best_train_loss)

plt.plot(np.arange(parametros["epochs"]), best_test_loss)

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.grid()



# Obtenemos los valores predichos por el modelo.

best_model.to(device)

best_model.eval()  

y_pred = best_model(x_test_tensor)

y_pred = y_pred.cpu().detach().numpy().squeeze()



plt.subplot(224)

plt.title('Real vs Predicción (L2)', fontsize= 15)

plt.plot(x_test, y_test)

plt.plot(x_test, y_pred, color='orange')

plt.scatter(x_train, y_train, color='red')

plt.ylabel('$\sin(2 \pi x)$')

plt.xlabel('$x$')

plt.legend(['real', 'predict', 'train'])

plt.grid()



plt.show()
class FeedforwardNet(nn.Module):

    

        def __init__(self, input_size, hidden_size):

            super(FeedforwardNet, self).__init__()

            self.input_size = input_size

            self.hidden_size  = hidden_size

            

            self.fc1 = nn.Linear(self.input_size, self.hidden_size)

            self.fc2 = nn.Linear(self.hidden_size, 1)

            

        def forward(self, x):

            x = self.fc1(x)

            output = self.fc2(x)

            return output
# Generamos la tabla para guardar los resultados.

resultados = PrettyTable()

resultados.field_names = ['Learning Rate', 'Nº de Neuronas', 'Test Loss']



parametros = {"lr": 0, "hs": 0}

min_loss = 99999



# "HIDEEN_SIZE" representa el Nº de neuronas.

HIDEEN_SIZE = [2, 5, 10, 20]

LEARNING_RATE = [0.001, 0.01, 0.1]

epochs = 10000



criterion = nn.MSELoss(reduction='sum')



for h in HIDEEN_SIZE:    

    for lr in LEARNING_RATE:

        model = FeedforwardNet(input_size = P, hidden_size = h)

        model.to(device)

        model.train()

        

        optimizer = SGD(model.parameters(), lr=lr)



        train_loss = []

        test_loss = []



        for epoch in range(epochs):

            

            # Limpiamos todos los gradientes.

            optimizer.zero_grad()

                

            # Realizamos la predicción con el modelo (forward pass).

            y_pred = model(x_train_tensor)

                

            # Calculamos la pérdida (loss) y la dividmos por el numero de obs.

            loss = criterion(y_pred, y_train_tensor) / n_train

                

            # Calculamos los gradientes (backward pass).

            loss.backward()

                

            # Realizamos el paso de optimización, actualizando los pesos.

            optimizer.step()



            train_loss.append(loss.detach().item())



            with torch.no_grad():

                y_pred = model(x_test_tensor)

                loss = criterion(y_pred, y_test_tensor) / n_test

                test_loss.append(loss.item())

                    

        model.eval()

        y_pred = model(x_test_tensor)

        after_train = criterion(y_pred, y_test_tensor) / n_test

        

        if after_train.item() < min_loss:

            parametros.update({"lr": lr,"hs": h})

            min_loss = after_train.item()

            

            best_model = type(model)(input_size=P, hidden_size=h)

            best_model.load_state_dict(model.state_dict())

            

            best_train_loss = train_loss

            best_test_loss = test_loss

            

        resultados.add_row([str(lr), str(h), str(round(after_train.item(), 3))])

        

print("Mejores parámetros: ",parametros)
# Ordenamos la tabla por la columna "Test Loss"

resultados.sortby = "Test Loss"

print(resultados)
plt.figure(figsize=(20,5))

plt.suptitle('FeedForwardNet', fontsize = 20)



plt.subplot(121)

plt.title('Valores de Loss', fontsize= 15)

plt.plot(np.arange(epochs), best_train_loss)

plt.plot(np.arange(epochs), best_test_loss)

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.grid()



best_model.to(device)

best_model.eval()  

y_pred = best_model(x_test_tensor)

y_pred = y_pred.cpu().detach().numpy().squeeze()



plt.subplot(122)

plt.title('Real vs Predicción', fontsize= 15)

plt.plot(x_test, y_test)

plt.plot(x_test, y_pred, color='orange')

plt.scatter(x_train, y_train, color='red')

plt.ylabel('$\sin(2 \pi x)$')

plt.xlabel('$x$')

plt.legend(['real', 'predict', 'train'])

plt.grid()



plt.show()
class FeedforwardNet_Relu_Tahn(nn.Module):

        def __init__(self, input_size, hidden_size):

            super(FeedforwardNet_Relu_Tahn, self).__init__()

            self.input_size = input_size

            self.hidden_size = hidden_size



            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)

            self.relu = torch.nn.ReLU()

            self.fc2 = torch.nn.Linear(self.hidden_size, 1)

            self.tanh = torch.nn.Tanh()  

            

        def forward(self, x):

            x = self.fc1(x)

            x = self.relu(x)

            x = self.fc2(x)

            x = self.tanh(x)

            return x
resultados = PrettyTable()

resultados.field_names = ['Learning Rate', 'Nº de neuronas', 'Test Loss']



parametros = {"lr": 0, "hs": 0}

min_loss = 99999



HIDEEN_SIZE = [2, 5, 10, 20]

LEARNING_RATE = [0.001, 0.01, 0.1]

epochs = 10000



criterion = nn.MSELoss(reduction='sum')



for h in HIDEEN_SIZE:  

    for lr in LEARNING_RATE:

        

        model = FeedforwardNet_Relu_Tahn(input_size=P, hidden_size=h)

        model.to(device)

        model.train()

        

        optimizer = SGD(model.parameters(), lr=lr)



        train_loss = []

        test_loss = []



        for epoch in range(epochs):

            

            # Limpiamos todos los gradientes.

            optimizer.zero_grad()

                

            # Realizamos la predicción con el modelo (forward pass).

            y_pred = model(x_train_tensor)

                

            # Calculamos la pérdida (loss) y la dividmos por el numero de obs.

            loss = criterion(y_pred, y_train_tensor) / n_train

                

            # Calculamos los gradientes (backward pass).

            loss.backward()

                

            # Realizamos el paso de optimización, actualizando los pesos.

            optimizer.step()



            train_loss.append(loss.detach().item())



            with torch.no_grad():

                y_pred = model(x_test_tensor)

                loss = criterion(y_pred, y_test_tensor) / n_test

                test_loss.append(loss.item())

                    

        model.eval()

        y_pred = model(x_test_tensor)

        after_train = criterion(y_pred, y_test_tensor) / n_test



        if after_train.item() < min_loss:

            parametros.update({"lr": lr,"hs": h})

            min_loss = after_train.item()

            

            best_model = type(model)(input_size=P, hidden_size=h)

            best_model.load_state_dict(model.state_dict())

            

            best_train_loss = train_loss

            best_test_loss = test_loss

        

        resultados.add_row([str(lr), str(h), str(round(after_train.item(), 3))])

        

print("Mejores parámetros: ",parametros)
# Ordenamos la tabla por la columna "Test Loss"

resultados.sortby = "Test Loss"

print(resultados)
plt.figure(figsize=(20,5))

plt.suptitle('Net con funciones de activación', fontsize = 20)



plt.subplot(121)

plt.title('Valores de Loss', fontsize= 15)

plt.plot(np.arange(epochs), best_train_loss)

plt.plot(np.arange(epochs), best_test_loss)

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.grid()



best_model.to(device)

best_model.eval()  

y_pred = best_model(x_test_tensor)

y_pred = y_pred.cpu().detach().numpy().squeeze()



plt.subplot(122)

plt.title('Real vs Predicción', fontsize= 15)

plt.plot(x_test, y_test)

plt.plot(x_test, y_pred, color='orange')

plt.scatter(x_train, y_train, color='red')

plt.ylabel('$\sin(2 \pi x)$')

plt.xlabel('$x$')

plt.legend(['real', 'predict', 'train'])

plt.grid()



plt.show()
# Importamos la libreria "data" que nos permitirá generar "DataLoaders" y asi iterar sobre los datos.

from torch.utils import data



#Inicializamos las variables.

# Utilizamos las neuronas optimas del ejercicio anterior.

h = parametros.get('hs')



resultados = PrettyTable()

resultados.field_names = ['Learning Rate', 'Batch Size', 'Test Loss']



parametros = {"lr": 0, "bs": 0}

min_loss = 99999

epochs = 10000

BATCH_SIZE = [2, 10, 50, 100]

LEARNING_RATE = [0.001, 0.01, 0.1]



# Generamos los dataset de trein y test a partir de los tensores.

train_data = data.TensorDataset(x_train_tensor, y_train_tensor)

test_data = data.TensorDataset(x_test_tensor, y_test_tensor)



for batch in BATCH_SIZE:

    

    # Generamos los iterables sobre de los dos datasets previamente creados. 

    train_loader = data.DataLoader(dataset=train_data, batch_size=batch, shuffle=True, num_workers=0,) 

    test_loader = data.DataLoader(dataset=test_data, batch_size=batch, shuffle=False, num_workers=0,)

    

    for lr in LEARNING_RATE:

        train_loss, test_loss = [], []

        model = FeedforwardNet_Relu_Tahn(input_size=P, hidden_size=h)

        model.to(device)

        model.train()

        

        optimizer = SGD(model.parameters(), lr=lr)

        

        for epoch in range(epochs):

            train_loss_acc = 0.

            

            # Aquí es donde realizamos la iteración sobre los elementos del dataset, segun el "batch_size" elegido.

            for x, y in train_loader:

                x.to(device, non_blocking=True)

                y.to(device, non_blocking=True)



                optimizer.zero_grad()

                y_pred = model(x)

                loss = criterion(y_pred, y)

                loss.backward()

                optimizer.step()



                train_loss_acc += loss.detach().item()

            train_loss.append(train_loss_acc / len(train_data))

            

            with torch.no_grad():

                test_loss_acc = 0.

                for x, y in test_loader:        

                    x.to(device, non_blocking=True)

                    y.to(device, non_blocking=True)        



                    y_pred = model(x)      

                    loss = criterion(y_pred, y)



                    test_loss_acc += loss.item()    

                test_loss.append(test_loss_acc / len(test_data))



        model.eval()

        y_pred = model(x_test_tensor)

        after_train = criterion(y_pred, y_test_tensor) / len(y_test_tensor)



        if after_train.item() < min_loss:

            parametros.update({"lr": lr,"bs": batch})

            min_loss = after_train.item()

            

            best_model = type(model)(input_size=P, hidden_size=h)

            best_model.load_state_dict(model.state_dict())

            

            best_train_loss = train_loss

            best_test_loss = test_loss

        

        resultados.add_row([str(lr), str(batch), str(round(after_train.item(), 3))])



print("Mejores parámetros: ",parametros)
print("Utilizando ", h, " Nº de neuronas" )

# Ordenamos la tabla por la columna "Test Loss"

resultados.sortby = "Test Loss"

print(resultados)
plt.figure(figsize=(20,5))



plt.subplot(121)

plt.title('Valores de Loss', fontsize= 15)

plt.plot(np.arange(epochs), best_train_loss)

plt.plot(np.arange(epochs), best_test_loss)

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.grid()



best_model.to(device)

best_model.eval()  

y_pred = best_model(x_test_tensor)

y_pred = y_pred.cpu().detach().numpy().squeeze()



plt.subplot(122)

plt.title('Real vs Predicción', fontsize= 15)

plt.plot(x_test, y_test)

plt.plot(x_test, y_pred, color='orange')

plt.scatter(x_train, y_train, color='red')

plt.ylabel('$\sin(2 \pi x)$')

plt.xlabel('$x$')

plt.legend(['real', 'predict', 'train'])

plt.grid()



plt.show()
# Inicializamos las variables.

parametros = {"lr": 0, "bs": 0}



# Utilizamos las neuronas optimas del ejercicio anterior.

min_loss = 99999

epochs = 10000

BATCH_SIZE = [2, 4, 6, 8]

LEARNING_RATE = [0.001, 0.01, 0.1]



criterion = nn.MSELoss(reduction='sum')



for batch in BATCH_SIZE:

    

    # Generamos los iterables sobre de los dos datasets previamente creados.

    train_loader = data.DataLoader(dataset=train_data, batch_size=batch, shuffle=True, num_workers=0,) 

    test_loader = data.DataLoader(dataset=test_data, batch_size=batch, shuffle=False, num_workers=0,)

    

    for lr in LEARNING_RATE:

        train_loss, test_loss = [], []

        model = FeedforwardNet_Relu_Tahn(input_size=P, hidden_size=h)

        model.to(device)

        model.train()

        

        optimizer = SGD(model.parameters(), lr=lr)

        

        for epoch in range(epochs):

            train_loss_acc = 0.

            

            # Limpiamos los gradientes.

            optimizer.zero_grad()

            

            # Aquí es donde realizamos la iteración sobre los elementos del dataset, segun el "batch_size" elegido.

            for x, y in train_loader:

                x.to(device, non_blocking=True)

                y.to(device, non_blocking=True)



                y_pred = model(x)

                loss = criterion(y_pred, y) / batch

                loss.backward()

                

                train_loss_acc += loss.detach().item()

            train_loss.append(train_loss_acc / len(train_data))

            

            # Actualizamos utilizando el gradiente sobre la función de costo acumulada, tras recorrer todo el dataset.

            optimizer.step()

            

            with torch.no_grad():

                test_loss_acc = 0.

                for x, y in test_loader:        

                    x.to(device, non_blocking=True)

                    y.to(device, non_blocking=True)        



                    y_pred = model(x)      

                    loss = criterion(y_pred, y) / batch



                    test_loss_acc += loss.item()    

                test_loss.append(test_loss_acc / len(test_data))



        model.eval()

        y_pred = model(x_test_tensor)

        after_train = criterion(y_pred, y_test_tensor) / len(y_test_tensor)



        if after_train.item() < min_loss:

            parametros.update({"lr": lr,"bs": batch})

            min_loss = after_train.item()

            

            best_model = type(model)(input_size=P, hidden_size=h)

            best_model.load_state_dict(model.state_dict())

            

            best_train_loss = train_loss

            best_test_loss = test_loss

        

        resultados.add_row([str(lr), str(batch), str(round(after_train.item(), 3))])

        

print("Mejores parámetros: ", parametros)
print("Utilizando un Nº de ", h, " neuronas" )

# Ordenamos la tabla por la columna "Test Loss"

resultados.sortby = "Test Loss"

print(resultados)
plt.figure(figsize=(20,5))



plt.subplot(121)

plt.title('Valores de Loss', fontsize= 15)

plt.plot(np.arange(epochs), best_train_loss)

plt.plot(np.arange(epochs), best_test_loss)

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.grid()



best_model.to(device)

best_model.eval()  

y_pred = best_model(x_test_tensor)

y_pred = y_pred.cpu().detach().numpy().squeeze()



plt.subplot(122)

plt.title('Real vs Predicción', fontsize= 15)

plt.plot(x_test, y_test)

plt.plot(x_test, y_pred, color='orange')

plt.scatter(x_train, y_train, color='red')

plt.ylabel('$\sin(2 \pi x)$')

plt.xlabel('$x$')

plt.legend(['real', 'predict', 'train'])

plt.grid()



plt.show()
import torch

from torch.autograd import Variable

from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

from torch.optim import Adam

from sklearn.metrics import accuracy_score

import imageio

import torch.nn.functional as F

import os



import torchvision

from torchvision.datasets import ImageFolder

from torchvision.utils import make_grid

from torchvision import transforms

from torch.utils.data import DataLoader



# Generamos algunas trasnformaciones que van a ser aplicadas al train set.

transform_train = transforms.Compose([

    

    # Por ejemplo algunos recortes aleatorios.

    transforms.RandomCrop(32, padding=4),

    

    # Y tambien algunos giros de ejes al azar.

    transforms.RandomHorizontalFlip(),

    

    # Para finalmente enviarlo al tensor y normalizarlo.

    transforms.ToTensor(),

    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

])



# Normalizamos el conjunto de test igual que el conjunto de train sin augmentation.

transform_test = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

])
# Descargando el dataset y aplicando las transformaciones.

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform = transform_train)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform = transform_test)



# Trasladamos los sets a su respectivo dataloader.

trainloader = torch.utils.data.DataLoader(trainset, batch_size= 128, shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=2)



classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# Función auxiliar que nos permitirá mostrar las imagenes.

def imshow(img):

    img = img / 2 + 0.5

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()
fig = plt.figure(figsize=(20,12)) 

fig.suptitle('Ejemplos del dataset:',fontsize=20)



dataiter = iter(trainloader)

images, labels = dataiter.next()

i = 1



for im, lab in zip(images, labels):

    sub = plt.subplot(4, 8, i)

    img = im / 2 + 0.5

    npimg = img.numpy()

    

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.xlabel(classes[lab])

    plt.xticks([])

    plt.yticks([])

    

    if i == 32:

        break

    i += 1
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

    

        self.conv_layer = nn.Sequential(



            # Bloque 1

            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),



            # Bloque 2

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout2d(p=0.05),



            # Bloque 3

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

        )



        self.fc_layer = nn.Sequential(

            nn.Dropout(p=0.1),

            nn.Linear(4096, 1024),

            nn.ReLU(inplace=True),

            nn.Linear(1024, 512),

            nn.ReLU(inplace=True),

            nn.Dropout(p=0.1),

            nn.Linear(512, 10)

        )



    def forward(self, x):

        x = self.conv_layer(x)

        x = x.view(x.size(0), -1)

        x = self.fc_layer(x)

        return x
net = Net()

net.to(device)



batch = 32

step = len(trainset) // batch



criterion = nn.CrossEntropyLoss()

optimizer = Adam(net.parameters(), lr = 0.000075, weight_decay=0.0001)



net.train()



trainloader = torch.utils.data.DataLoader(trainset, batch_size= batch, shuffle=True, num_workers=2)



for epoch in range(40):



    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):

        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)

        

        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()



        running_loss += loss.item()



print("Entrenamiento Finalizado")
net.eval()

correct = 0

total = 0



with torch.no_grad():

    for data in testloader:

        images, labels = data

        images, labels = images.to(device), labels.to(device)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels.cuda()).sum().item()



print('Exactitud de la red para las 10000 imagenes del test: %d %%' % (100 * correct / total))
fig = plt.figure(figsize=(20,15)) 

fig.suptitle('Ejemplos de predicciones:',fontsize=15)



dataiter = iter(testloader)

with torch.no_grad():

    for i in range(1,33):

        images, _ = dataiter.next()

        _, predicted = torch.max(net(images.cuda()), 1)

        sub = plt.subplot(4, 8, i)

        img = images[0] / 2 + 0.5

        npimg = img.numpy()

        plt.imshow(np.transpose(npimg, (1, 2, 0)))

        plt.xlabel(classes[predicted])

        i += 1
import torchvision.models as models

from torch.optim import lr_scheduler

from collections import OrderedDict



model = models.vgg11(pretrained=True)



for param in model.parameters():

    param.requires_grad = False

    

fc = nn.Sequential(OrderedDict([

    ('0', nn.Linear(4096,1024)),

    ('1', nn.ReLU(inplace=True)),

    ('2', nn.Dropout(p=0.25)),

    ('3', nn.Linear(1024,500)),

    ('4', nn.ReLU(inplace=True)),

    ('5', nn.Dropout(p=0.2)),

    ('6', nn.Linear(500,10)),

    ('7', nn.LogSoftmax())

]))



model.classifier[6] = fc

model.to(device)



criterion = nn.CrossEntropyLoss()

optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)

scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
batch = 32

step = len(trainset) // batch



trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=4)

testloader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=4)



model.train()

running_loss = 0.0



for epoch in range(40):

    loss = 0

    for i, data in enumerate(trainloader, 0):

        inputs, labels = data

        inputs = inputs.to(device)

        labels = labels.to(device)

        

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

          

    scheduler.step()

    

print("Entrenamiento Finalizado")
model.eval()

correct = 0

total = 0

with torch.no_grad():

    for data in testloader:

        images, labels = data

        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()



print('Exactitud de la red para las 10000 imagenes del test: %d %%' % (100 * correct / total))
fig = plt.figure(figsize=(20,15)) 

fig.suptitle('Ejemplos de predicciones:',fontsize=20)



dataiter = iter(testloader)

with torch.no_grad():

    for i in range(1,33):

        images, _ = dataiter.next()

        _, predicted = torch.max(model(images.cuda()), 1)

        sub = plt.subplot(4, 8, i)

        img = images[0] / 2 + 0.5

        npimg = img.numpy()

        plt.imshow(np.transpose(npimg, (1, 2, 0)))

        plt.xlabel(classes[predicted])

        i += 1