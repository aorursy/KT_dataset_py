import csv



from math import sqrt



# Load CSV file, remove headers if any are dfound, 

# convert values to floats, and remove rows that 

# don't have x and y values

def load_csv(filename):

    dataset = list()

    with open(filename) as csvfile:

        reader = csv.reader(csvfile)

        has_header = csv.Sniffer().has_header(csvfile.read(1024))

        csvfile.seek(0)

        if has_header:

            headers = reader.__next__()

        for row in reader:

            if len(row) != 2:

                continue

            for i in range(len(row)):

                row[i] = float(row[i].strip())

            dataset.append(row)

    return dataset



# Compute the root mean squared error

def rmse_metric(actual, predicted):

    error_sum = 0.0

    for i in range(len(actual)):

        difference = predicted[i] - actual[i]

        error_sum = error_sum + (difference ** 2)

    return sqrt(error_sum / len(actual))



# Returns a test set with the Y values stripped out, 

# and a separate list of actual values that can

# be compared to the predicted values.

def create_test_and_actual(test):

    actual = list()

    stripped = list()

    for row in test:

        c = list(row)

        actual.append(row[-1])

        c[-1] = None

        stripped.append(c)

    return stripped, actual



# Evaluates the supplied algorithm on the test

# and train data and returns actual values

# along with predicted and the root mean squared error

# of the actual and predicted sets

def evaluate(train, test, algorithm, *args):

    test_set, actual = create_test_and_actual(test)

    predicted = algorithm(train, test_set, *args)

    rmse = rmse_metric(actual, predicted)

    return actual, predicted, rmse



def mean(values):

    return sum(values) / float(len(values))



def covariance(x, mean_x, y, mean_y):

    covar = 0.0

    for i in range(len(x)):

        covar += (x[i] - mean_x) * (y[i] - mean_y)

    return covar



def variance(values, mean):

    return sum([(x - mean) ** 2 for x in values])



def coefficients(dataset):

    x = [row[0] for row in dataset]

    y = [row[1] for row in dataset]

    x_mean = mean(x)

    y_mean = mean(y)

    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)

    b0 = y_mean - b1 * x_mean

    return b0, b1



def linear_regression(train, test):

    predictions = list()

    b0, b1 = coefficients(train)

    for row in test:

        predictions.append(b0 + b1 * row[0])

    return predictions



def zero_rule_regression(train, test):

    values = [row[-1] for row in train]

    prediction = mean(values)

    predicted = [prediction for i in range(len(test))]

    return predicted



test = load_csv('../input/test.csv')

train = load_csv('../input/train.csv')
import matplotlib.pyplot as plot



x_test = [row[0] for row in test]

y_test = [row[1] for row in test]

x_train = [row[0] for row in train]

y_train = [row[1] for row in train]



lr_actual, lr_predicted, lr_rmse = evaluate(train, test, linear_regression)



zr_actual, zr_predicted, zr_rmse = evaluate(train, test, zero_rule_regression)



plot.title('Predicted vs Actual (linear regression)')

plot.scatter(x_test, lr_actual, color='black')

plot.scatter(x_test, lr_predicted, color='red')

plot.show()

print('RMSE of linear regression: ' + str(lr_rmse))



plot.title('Predicted vs Actual (zero rule regression)')

plot.scatter(x_test, zr_actual, color='black')

plot.scatter(x_test, zr_predicted, color='red')

plot.show()

print('RMSE of zero rule regression: ' + str(zr_rmse))
