import pandas as pd

data = pd.read_csv("../input/advertising.csv")

data.head()



def predict_sales(radio, weight, bias):

    return weight*radio + bias
def cost_function(radio, sales, weight, bias):

    companies = len(radio)

    total_error = 0.0

    for i in range(companies):

        total_error += (sales[i] - (weight*radio[i] + bias))**2

    return total_error / companies
def update_weights(radio, sales, weight, bias, learning_rate):

    weight_deriv = 0

    bias_deriv = 0

    companies = len(radio)



    for i in range(companies):

        # Calculate partial derivatives

        # -2x(y - (mx + b))

        weight_deriv += -2*radio[i] * (sales[i] - (weight*radio[i] + bias))



        # -2(y - (mx + b))

        bias_deriv += -2*(sales[i] - (weight*radio[i] + bias))



    # We subtract because the derivatives point in direction of steepest ascent

    weight -= (weight_deriv / companies) * learning_rate

    bias -= (bias_deriv / companies) * learning_rate



    return weight, bias
def train(radio, sales, weight, bias, learning_rate, iters):

    cost_history = []



    for i in range(iters):

        weight,bias = update_weights(radio, sales, weight, bias, learning_rate)



        #Calculate cost for auditing purposes

        cost = cost_function(radio, sales, weight, bias)

        cost_history.append(cost)



        # Log Progress

        if i % 10 == 0:

            print("iter={:d}    weight={:.2f}    bias={:.4f}    cost={:.2}".format(i, weight, bias, cost))



    return weight, bias, cost_history
radio = data['Radio'].values

sales = data['Sales'].values

weight = 0

bias = 0

lr = 0.01

iters = 100

train(radio,sales,weight,bias,lr,iters)