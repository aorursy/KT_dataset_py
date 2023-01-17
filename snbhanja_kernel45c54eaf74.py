import os
print(os.listdir("../input/lstm-talk-master/lstm-talk-master/"))

# Any results you write to the current directory are saved as output.
import sys
sys.path.insert(0, "../input/lstm-talk-master/lstm-talk-master/")

import os
import numpy as np
import pandas as pd
from lstm import create_train_and_test, create_model
from plot import analyze_and_plot_results

import matplotlib as plt
%matplotlib inline
# For reproducibility
np.random.seed(1234)
# Hyper-parameters
sequence_length = 48  # 1 day = 48 * 30m
duplication_ratio = 0.04
epochs = 10
batch_size = 50
split_index = 336  # 7 days = 336 * 30m
split_index += sequence_length  # Pad split index by sequence_length
# LSTM layers
layers = {
    'input': 1,
    'hidden1': 64,
    'hidden2': 256,
    'hidden3': 100,
    'output': 1
}
# Make sure output dir exists
output_dir = '/results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Prep data
input_csv = 'nyc_taxi.csv'  # Data sampled every 30m.
df = pd.read_csv(os.path.join("../input/lstm-talk-master/lstm-talk-master/data", input_csv))
df.head()
df.dtypes
data = df.value.values.astype("float64")
y_true = data[split_index + sequence_length:]
timestamps = df.timestamp.values[split_index + sequence_length:]
y_true.shape
timestamps.shape
X_train, y_train, X_test, y_test = create_train_and_test(
    data=data, sequence_length=sequence_length,
    duplication_ratio=duplication_ratio, split_index=split_index)
# Create LSTM model
model = create_model(sequence_length=sequence_length, layers=layers)
# Train model
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                    validation_split=0.0)
# Save history.
df = pd.DataFrame(data={'epochs': range(epochs),
                        'loss': history.history['loss']})
history_csv_path = os.path.join(output_dir, 'history.csv')
df.to_csv(history_csv_path, index=False)
# Predict values.
print("Predicting...")
y_pred = model.predict(X_test)
print("Reshaping...")
y_pred = np.reshape(y_pred, (y_pred.size,))

# Save results.
print("Saving...")
results_csv_path = os.path.join(output_dir, input_csv)
df = pd.DataFrame(data={'timestamps': timestamps, 'y_true': y_true,
                        'y_test': y_test, 'y_pred': y_pred})
df.to_csv(results_csv_path, index=False)

!ls -la
# Extract anomalies from predictions and plot results.
print("Plotting...")
analyze_and_plot_results(results_csv_path, history_csv_path)
print("Done! Results saved in:", output_dir)
!ls -la
y_true.shape
y_pred.shape
#y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred)

#i * np.log(j) + (1 - i) * np.log(1 - j)

log_likelyhood = []
for i,j in zip(y_test, y_pred):
    log_likelyhood.append(i * np.log(j) + (1 - i) * np.log(1 - j))
y_test
y_pred
df.head()
log_likelyhood
type(y_test)
sum(np.isnan(y_test))
sum(np.isnan(log_likelyhood))
np.isnan(log_likelyhood)
-5.37142495e-01 * np.log(-0.0537142495)
in_array = [1, -3, 5, 2**8] 
print ("Input array : ", in_array) 
out_array = np.log(in_array) 
print ("Output array : ", out_array) 
#error
def compute_scores(y_test, y_pred, normalize=False):
    # Errors
    errors = np.array((y_test - y_pred) ** 2)
    if normalize:
        errors = errors / float(errors.max() - errors.min())
    return errors

errors = compute_scores(y_test, y_pred, normalize=True)
#import math

#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(np.log(y_pred[i] + 1) - np.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    #return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5
    return terms_to_sum
rmsle_values = rmsle(y_test, y_pred) 
errors
rmsle_values
max(errors)
np.log(errors)
array = np.random.normal(0.0, 1.0, 5) 
print("1D Array filled with random values "
      "as per gaussian distribution : \n", array) 
y_test
y_pred
errors
min(errors)
max(errors)
sum(errors > 0.8)
type(errors)
!pip install lsanomaly
from lsanomaly import LSAnomaly

# At train time lsanomaly calculates parameters rho and sigma
lsanomaly = LSAnomaly()
# or alternatively
lsanomaly = LSAnomaly(sigma=3, rho=0.1)
import numpy as np
lsanomaly.fit(np.array([[1],[2],[3],[1],[2],[3]]))
lsanomaly.predict(np.array([[0]]))
lsanomaly.predict_proba(np.array([[0]]))
lsanomaly.fit(y_train.reshape(-1, 1))
lsanomaly.predict(np.array([[0]]))
errors
np.log(errors)
min(np.log(errors))
max(np.log(errors))
from scipy.stats import norm
norm.pdf(3, 7, 2)
from collections import deque 

def moving_averages(data, size, rampUp=True, rampDown=True):
    """Slide a window of <size> elements over <data> to calc an average

    First and last <size-1> iterations when window is not yet completely
    filled with data, or the window empties due to exhausted <data>, the
    average is computed with just the available data (but still divided
    by <size>).
    Set rampUp/rampDown to False in order to not provide any values during
    those start and end <size-1> iterations.
    Set rampUp/rampDown to functions to provide arbitrary partial average
    numbers during those phases. The callback will get the currently
    available input data in a deque. Do not modify that data.
    """
    d = deque()
    running_sum = 0.0

    data = iter(data)
    # rampUp
    for count in range(1, size):
        try:
            val = next(data)
        except StopIteration:
            break
        running_sum += val
        d.append(val)
        #print("up: running sum:" + str(running_sum) + "  count: " + str(count) + "  deque: " + str(d))
        if rampUp:
            if callable(rampUp):
                yield rampUp(d)
            else:
                yield running_sum / size

    # steady
    exhausted_early = True
    for val in data:
        exhausted_early = False
        running_sum += val
        #print("st: running sum:" + str(running_sum) + "  deque: " + str(d))
        yield running_sum / size
        d.append(val)
        running_sum -= d.popleft()

    # rampDown
    if rampDown:
        if exhausted_early:
            running_sum -= d.popleft()
        for (count) in range(min(len(d), size-1), 0, -1):
            #print("dn: running sum:" + str(running_sum) + "  deque: " + str(d))
            if callable(rampDown):
                yield rampDown(d)
            else:
                yield running_sum / size
            running_sum -= d.popleft()
mv_avg = list(moving_averages(errors, 10, rampUp=False))
len(mv_avg)
len(errors)
mv_avg
min(mv_avg)
max(mv_avg)
from matplotlib.pyplot import plot
plot(mv_avg)
np.std(errors)
error_df = pd.DataFrame({'timestamp' : timestamps, 'errors':errors})
error_df.head()
error_df.dtypes
error_df["timestamp"] = pd.to_datetime(error_df["timestamp"])
error_df.head()
error_df.index = error_df["timestamp"]
error_df.drop(columns = ["timestamp"], inplace=True)
error_df.head()
r = error_df.errors.rolling('1D').agg(['mean', 'std'])

r.columns
r.head()
from scipy.stats import norm

def calculate_pdf_moving(error, mean, std):
    return norm.pdf(error, mean, std)
    
error_df.head()
error_df["mean"] = r["mean"]
error_df["std"] = r["std"]
error_df["y_test"] = y_test
error_df.head()
error_df["pdf"]  = np.vectorize(calculate_pdf_moving)(error_df["errors"], error_df['mean'], error_df['std'])
error_df.head()
plot(errors)
plot(error_df["mean"])
plot(error_df["std"])
plot(error_df["pdf"])
plot(-np.log(error_df["pdf"]))
error_df.head()
error_df["pdf"] = error_df["pdf"]/100
plot(error_df["pdf"])
probs = np.exp(error_df["errors"]) / (np.exp(error_df["errors"])).sum()
probs
sum(probs)
min(probs)
max(probs)
probs.sum()
log_probs = np.log(probs)
log_probs
probabilities = np.exp(log_probs)
probabilities
plot(probabilities)
# Define gaussian model function
def gaussian_model(x, mu, sigma):
    coeff_part = 1/(np.sqrt(2 * np.pi * sigma**2))
    exp_part = np.exp( - (x - mu)**2 / (2 * sigma**2) )
    return coeff_part*exp_part
# Compute sample statistics
mean = np.mean(error_df["y_test"])
stdev = np.std(error_df["y_test"])
# Model the population using sample statistics
population_model = gaussian_model(error_df["y_test"], mu=mean, sigma=stdev)
population_model
# Guess parameters
mu_guess = np.mean(y_pred)

sigma_guess = np.std(y_pred)
# For each sample point, compute a probability
probabilities = np.zeros(len(y_pred))
for n, distance in enumerate(y_pred):
    probabilities[n] = gaussian_model(distance, mu=mu_guess, sigma=sigma_guess)
probabilities
likelihood = np.product(probabilities)
loglikelihood = np.sum(np.log(probabilities))
likelihood
loglikelihood
# Create an array of mu guesses 
low_guess = mu_guess - 2*sigma_guess 
high_guess = mu_guess + 2*sigma_guess 
mu_guesses = np.linspace(low_guess, high_guess, 101) 
# Compute the loglikelihood for each guess 
loglikelihoods = np.zeros(len(mu_guesses)) 
for n, mu_guess in enumerate(mu_guesses):     
    loglikelihoods[n] = compute_loglikelihood(y_test, mu=mu_guess, sigma=sigma_guess)

y_test
errors
import matplotlib.pyplot as plt
%matplotlib inline
sample_mean = np.mean(y_test)
sample_stdev = np.std(y_test)

def gaussian_model(x, mu, sigma):
    return 1/(np.sqrt(2 * np.pi * sigma**2)) * np.exp( - (x - mu)**2 / (2 * sigma**2) )

def compute_loglikelihood(samples, mu, sigma=250):
    probs = np.zeros(len(samples))
    for n, sample in enumerate(samples):
        probs[n] = gaussian_model(sample, mu, sigma)
    loglikelihood = np.sum(np.log(probs))
    return loglikelihood

def plot_loglikelihoods(mu_guesses, loglikelihoods):
    max_loglikelihood = np.max(loglikelihoods)
    max_index = np.where(loglikelihoods==max_loglikelihood)
    max_guess = mu_guesses[max_index][0]
    font_options = {'family' : 'Arial', 'size'   : 16}
    plt.rc('font', **font_options)
    fig, axis = plt.subplots(figsize=(10,6))
    axis.plot(mu_guesses, loglikelihoods)
    axis.plot(max_guess, max_loglikelihood, marker="o", color="red")
    axis.grid()
    axis.set_ylabel('Log Likelihoods')
    axis.set_xlabel('Guesses for Mu')
    axis.set_title('Max Log Likelihood = {:0.1f} \n was found at Mu = {:0.1f}'.format(max_loglikelihood, max_guess))
    fig.tight_layout()
    plt.show()
    return fig

# Create an array of mu guesses, centered on sample_mean, spread out +/- by sample_stdev
low_guess = sample_mean - 2*sample_stdev
high_guess = sample_mean + 2*sample_stdev
mu_guesses = np.linspace(low_guess, high_guess, 101)

# Compute the loglikelihood for each model created from each guess value
loglikelihoods = np.zeros(len(mu_guesses))
for n, mu_guess in enumerate(mu_guesses):
    loglikelihoods[n] = compute_loglikelihood(y_test, mu=mu_guess, sigma=sample_stdev)

# Find the best guess by using logical indexing, the print and plot the result
best_mu = mu_guesses[loglikelihoods==np.max(loglikelihoods)]
print('Maximum loglikelihood found for best mu guess={}'.format(best_mu))
fig = plot_loglikelihoods(mu_guesses, loglikelihoods)
loglikelihoods
import matplotlib.pyplot as plt
%matplotlib inline
sample_mean = np.mean(y_pred)
sample_stdev = np.std(y_pred)

def gaussian_model(x, mu, sigma):
    return 1/(np.sqrt(2 * np.pi * sigma**2)) * np.exp( - (x - mu)**2 / (2 * sigma**2) )

def compute_loglikelihood(samples, mu, sigma=250):
    probs = np.zeros(len(samples))
    for n, sample in enumerate(samples):
        probs[n] = gaussian_model(sample, mu, sigma)
    loglikelihood = np.sum(np.log(probs))
    return loglikelihood

def plot_loglikelihoods(mu_guesses, loglikelihoods):
    max_loglikelihood = np.max(loglikelihoods)
    max_index = np.where(loglikelihoods==max_loglikelihood)
    max_guess = mu_guesses[max_index][0]
    font_options = {'family' : 'Arial', 'size'   : 16}
    plt.rc('font', **font_options)
    fig, axis = plt.subplots(figsize=(10,6))
    axis.plot(mu_guesses, loglikelihoods)
    axis.plot(max_guess, max_loglikelihood, marker="o", color="red")
    axis.grid()
    axis.set_ylabel('Log Likelihoods')
    axis.set_xlabel('Guesses for Mu')
    axis.set_title('Max Log Likelihood = {:0.1f} \n was found at Mu = {:0.1f}'.format(max_loglikelihood, max_guess))
    fig.tight_layout()
    plt.show()
    return fig

# Create an array of mu guesses, centered on sample_mean, spread out +/- by sample_stdev
low_guess = sample_mean - 2*sample_stdev
high_guess = sample_mean + 2*sample_stdev
mu_guesses = np.linspace(low_guess, high_guess, 101)

# Compute the loglikelihood for each model created from each guess value
loglikelihoods = np.zeros(len(mu_guesses))
for n, mu_guess in enumerate(mu_guesses):
    loglikelihoods[n] = compute_loglikelihood(y_pred, mu=mu_guess, sigma=sample_stdev)

# Find the best guess by using logical indexing, the print and plot the result
best_mu = mu_guesses[loglikelihoods==np.max(loglikelihoods)]
print('Maximum loglikelihood found for best mu guess={}'.format(best_mu))
fig = plot_loglikelihoods(mu_guesses, loglikelihoods)
loglikelihoods

min(probs)
max(probs)
sum(probs)
