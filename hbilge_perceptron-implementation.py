import numpy as np

import matplotlib.pyplot as plt



def plot(w_star, w, X, y):



    positive_points = X[y == 1, :]

    negative_points = X[y == -1, :]



    bias = 0.01



    plt.figure(figsize=(8, 6))



    plt.style.use("ggplot")



    plt.scatter(positive_points[:, 0], positive_points[:, 1], marker="o", color="blue", s=10, label="positive class")

    plt.scatter(negative_points[:, 0], negative_points[:, 1], marker="o", color="green", s=10, label="negative class")



    # f function on the hyperplane (target line)

    line_x = np.linspace(-2, 2, 50)

    line_y = -(bias + w[0] * line_x) / w[1]     # bias + w0*x + w1*y = 0



    # g function on the hyperplane (which approximates f - chosen from hypothesis set H)

    line_x1 = np.linspace(-2, 2, 50)

    line_y1 = -(get_trained_bias() + w_star[0] * line_x1) / w_star[1]



    plt.plot(line_x, line_y, color= "black", label="f function")

    plt.plot(line_x1, line_y1, color= "red", label="g function")

    plt.legend(loc = 1, prop= {'size': 7})



    plt.xlim(-2, 2)

    plt.ylim(-2, 2)



    plt.xlabel("x1")

    plt.ylabel("x2")

    plt.show()

    return



def train_perceptron(X, y):



    w = np.zeros(X.shape[1])



    global iteration

    iteration = 1



    global b

    b = float(0)



    while True:

        results = np.sign(np.dot(X, w) + b)

        misclassified_indices = np.where(y != results)[0]

        if len(misclassified_indices) == 0:

            break

        picked_misclassified = np.random.choice(misclassified_indices)

        w += y[picked_misclassified] * X[picked_misclassified]

        b += y[picked_misclassified]

        iteration += 1

    return w



# return number of iteration from train perceptron function

def get_iteration():

    return iteration



# return bias from train perceptron function

def get_trained_bias():

    return b



#  Generate a linearly separable data set of size 20, 100, and 1000.

if __name__ == "__main__":



    range = np.array([20, 100, 1000])

    for x in range:

        weight = np.array([0.3,0.2])

        bias = 0.01

        input_X = np.random.randn(x,2)

        output_Y = np.sign(np.dot(input_X,weight)+bias)

        w_star = train_perceptron(input_X, output_Y)

        plot(w_star, weight, input_X, output_Y)

        iteration = get_iteration()

        print("The number of iteration for", x, "data point is: " , str(iteration))