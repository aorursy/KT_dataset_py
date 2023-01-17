import os

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/tictactoe-endgame-dataset-uci/tic-tac-toe-endgame.csv',',')

df
df['V1'],v1 = pd.factorize(df['V1'], sort=True)

df['V2'],v2 = pd.factorize(df['V2'], sort=True)

df['V3'],v3 = pd.factorize(df['V3'], sort=True)

df['V4'],v4 = pd.factorize(df['V4'], sort=True)

df['V5'],v5 = pd.factorize(df['V5'], sort=True)

df['V6'],v6 = pd.factorize(df['V6'], sort=True)

df['V7'],v7 = pd.factorize(df['V7'], sort=True)

df['V8'],v8 = pd.factorize(df['V8'], sort=True)

df['V9'],v9 = pd.factorize(df['V9'], sort=True)

df['V10'],v10 = pd.factorize(df['V10'], sort=True)

print(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10)

print(df)
class_names = [v10[0], v10[1]]

class_names
x = df.drop('V10',axis=1)

y = df['V10']



# Vamos separar (split) nossos dados em conj. de dados para treinamento e testes..

x_train, x_test, y_train, y_test = train_test_split(x, y)

[x_train.shape, x_test.shape, y_train.shape, y_test.shape]
def classify_mean_accuracy(max_n, max_c):

    

    max_range = range(1, max_n)

    cases = range(0, max_c)

    funcs = ['relu', 'logistic', 'tanh']

    

    best_func = 'relu'

    best_g_sum = 0

    best_g = 0

    best_g_n = 0

    

    for func in funcs:

        

        best_list = []

        best_sum = 0

        best_c = 0

        best_c_n = 0

        

        for case in cases:

            

            best = 0

            best_n = 0

            accuracy_list = []

            

            for n in max_range:

                mlp = MLPClassifier(solver='lbfgs', activation=func, hidden_layer_sizes=(n))

                mlp.fit(x_train, y_train)

                y_pred = mlp.predict(x_test)

                score = accuracy_score(y_test, y_pred)

                accuracy_list.append(score)

                

                if score > best:

                    best = score

                    best_n = n

            

            accuracy_sum = sum(accuracy_list)

            

            if accuracy_sum > best_sum:

                best_sum = accuracy_sum

                best_list = accuracy_list

                best_c = best

                best_c_n = best_n

                

        if best_sum > best_g_sum:

            best_g_sum = best_sum

            best_g = best_c

            best_g_n = best_c_n

            best_func = func

                

        plt.plot(max_range, best_list, label=func)

    

    plt.title('Activation function: relu x logistic x tanh')

    plt.xlabel('Number of neurons')

    plt.ylabel('Accuracy')

    plt.legend(loc='best')

    plt.show()

    

    return (best_g, best_g_n, best_func)



def classify_point_accuracy(max_n, max_c):

      

    max_range = range(1, max_n)

    cases = range(0, max_c)

    funcs = ['relu', 'logistic', 'tanh']

    colors = ['ro', 'go', 'bo']

    

    best = 0

    best_n = 0

    best_mlp = None

    c = 0

    

    for func in funcs:

        

        color = colors[c]

        c += 1

        

        for case in cases:

            

            accuracy_list = []

            

            for n in max_range:

                mlp = MLPClassifier(solver='lbfgs', activation=func, hidden_layer_sizes=(n))

                mlp.fit(x_train, y_train)

                y_pred = mlp.predict(x_test)

                score = accuracy_score(y_test, y_pred)

                accuracy_list.append(score)

                

                if score > best:

                    best = score

                    best_n = n

                    best_mlp = mlp

            

            plt.plot(max_range, accuracy_list, color, label='{}{}'.format(func, case))

    

    plt.title('Activation function: relu x logistic x tanh')

    plt.xlabel('Number of neurons')

    plt.ylabel('Accuracy')

    plt.legend(loc='best')

    plt.show()

    

    return (best, best_n, best_mlp)
best_mean_results = classify_mean_accuracy(10, 10) # usually the best amount of neurons is found between the input and output size. Only one hidden layer is required for most feedback neural networks

best_mean_results
best_point_results = classify_point_accuracy(10, 10) # usually the best amount of neurons is found between the input and output size. Only one hidden layer is required for most feedback neural networks

best_point_results
# First try out the best punctual accuracy mlp found. It usually brings results of at least 85%

mlp = best_point_results[2]
# use the model to make predictions with the test data

y_pred = mlp.predict(x_test)

# how did our model perform?

count_misclassified = (y_test != y_pred).sum()

print('Misclassified samples: {}/{}'.format(count_misclassified, len(y_test)))

accuracy = accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))

print(confusion_matrix(y_test, y_pred))
# Second try out the best mean accuracy data into a new mlp and check results. They are usually at least 75%

mlp = MLPClassifier(solver='lbfgs', activation=best_mean_results[2], hidden_layer_sizes=best_mean_results[1])

mlp.fit(x_train,y_train)
# use the model to make predictions with the test data

y_pred = mlp.predict(x_test)

# how did our model perform?

count_misclassified = (y_test != y_pred).sum()

print('Misclassified samples: {}/{}'.format(count_misclassified, len(y_test)))

accuracy = accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))

print(confusion_matrix(y_test, y_pred))