import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
train_data_path = "../input/ex1_train_data.csv"
train_data = pd.read_csv(train_data_path, delimiter = ";")
# Implemente aqui, a leitura dos arquivos csv
train_data.head(5)
def plot_points(data):
    x = data["x"]
    y = data["y"]
    plt.scatter(x, y, color = 'b', alpha = 0.3)
    plt.title("m² x $")
    plt.xlabel("Metragem")
    plt.ylabel("Custo (Milhares)")
plot_points(train_data)
def calculate_mse(Y, Y_pred):
    mse = 1/len(Y) * (sum( (Y - Y_pred) ** 2 ))
    
    return mse
best_a = None
best_b = None
best_error = None
# Implemente aqui, sua lógica de busca pelo melhores valores de a e b
for a in range(1, 100):
    for b in range(1, 100):
        Y_pred = train_data['x'] * a + b
        
        new_mse = calculate_mse(train_data["y"], Y_pred)
        
        if best_error is None or new_mse < best_error:
            best_error = new_mse
            best_a = a
            best_b = b
print("Best mse => {}, Best a => {}, Best b => {}".format(best_error, best_a, best_b))
def plot_points_and_lreg(data, a, b):
    x = data["x"]
    y = data["y"]
    plt.scatter(x, y, color = 'b', alpha = 0.3)
    plt.title("m² x $")
    plt.xlabel("Metragem")
    plt.ylabel("Custo (Milhares)")
    plt.plot(x, a * x + b, color = 'r')
plot_points_and_lreg(train_data, best_a, best_b)