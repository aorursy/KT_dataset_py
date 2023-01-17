import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
%matplotlib inline
from ipywidgets import interactive, HBox, VBox

def generate_signal(function, domain_min, domain_max, n_datapoints, random_seed=42):
    np.random.seed(random_seed)
    data = np.zeros((n_datapoints, 2))
    data[:,0] = np.random.uniform(domain_min, domain_max, n_datapoints)
    vfunc = np.vectorize(function)
    data[:,1] = vfunc(data[:,0])
    return data

def generate_noisy_data(function, domain_min, domain_max, n_datapoints, stdev=1, random_seed=42):
    np.random.seed(random_seed)
    data = generate_signal(function, domain_min, domain_max, n_datapoints, random_seed)
    data[:,1] += np.random.randn(n_datapoints) * stdev
    return data
domain_min = -2*np.pi
domain_max = 2*np.pi
true_function = np.sin
def plot_data(ax, X, y):
    ax.scatter(X,y,s=10, edgecolor="black", c="darkorange", label="data")
    
def plot_true(ax, X_sample, true_function):
    ax.plot(X_sample, true_function(X_sample), label='True signal')
    
def plot_model(ax, X_sample, y_pred):
    ax.plot(X_sample, y_pred, label="Model's estimate")
def create_interactive_demo(model, sample_size, noise_stdev, **params):
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(12,12))
    
    # Generating a training set.
    train_data = generate_noisy_data(true_function, domain_min, domain_max, sample_size, noise_stdev, 0)
    X_train = train_data[:,[0]]
    y_train = train_data[:,1]
    plot_data(ax1, X_train, y_train)
    
    # Generating a test set whose size is 20% of the training set.
    test_data = generate_noisy_data(true_function, domain_min, domain_max, int(0.2*sample_size), noise_stdev, 1)
    X_test = test_data[:,[0]]
    y_test = test_data[:,1]
    
    # Plotting the true function
    X_sample = np.linspace(domain_min, domain_max, sample_size)[:, np.newaxis]
    y_true = true_function(X_sample)
    plot_true(ax1, X_sample, true_function)
    
    estimator = model(**params, random_state=42)
    estimator.fit(X_train, y_train)
    y_pred_sample = estimator.predict(X_sample)
    plot_model(ax1, X_sample, y_pred_sample)
    
    train_error = mean_squared_error(y_train, estimator.predict(X_train))
    train_spread = mean_squared_error(y_train, true_function(X_train))
    test_error = mean_squared_error(y_test, estimator.predict(X_test))
    test_spread = mean_squared_error(y_test, true_function(X_test))
    true_error = mean_squared_error(y_true, y_pred_sample)
    bayes_error = noise_stdev**2
    
    width = 0.7
    ax2.bar([1,1+width/2, 2, 2+width/2], [train_error, train_spread, test_error, test_spread], width/2,
           color=["blue", "lightblue", "orange", 'yellow'])
    ax2.bar([3,4], [bayes_error, true_error], color=["red", "green"])
    ax2.set_xticks([1,2,3,4])
    ax2.set_xticklabels(["Train error/spread:"+str(round(train_error,3))+'/'+str(round(train_spread,3)), 
                         "Test error/spread:"+str(round(test_error,3))+'/'+str(round(test_spread,3)), 
                         "Bayes error:"+str(round(bayes_error,3)), 
                         "True error:"+str(round(true_error,3))])
    plt.legend()
    plt.show()
    
def draw_forest_plot(sample_size=1000, noise_stdev=1, **params):
    create_interactive_demo(RandomForestRegressor, sample_size, noise_stdev, **params)

parameters = {'n_estimators':(1,50,2), 'max_depth':(1,20), 'min_samples_split':(2,600,3),
             'min_samples_leaf':(1,300,3)}
interactive_plot = interactive(draw_forest_plot, sample_size=(100, 10000), noise_stdev=(0.1,2.0), **parameters)

vb1 = VBox(interactive_plot.children[:-1])
vb2 = VBox(interactive_plot.children[-1:])
hb = HBox([vb1,vb2])
interactive_plot.children = [hb]

initial_values = [1, 20, 2, 1]
for i in range(4):
    vb1.children[i+2].value = initial_values[i]

hb.layout.display = 'flex'
# You might need to mess around with these values if the slides and figures don't fit your screen well.
vb1.layout.width = '30%'
interactive_plot.layout.width = '3000px'
display(interactive_plot)

def plot_true_and_test_errors(ax, sample_size=1000, noise_stdev=1.0, varying_parameter='max_depth', interval=np.arange(2,20), n_average=None):    
    # Generating a training set.
    train_data = generate_noisy_data(true_function, domain_min, domain_max, sample_size, noise_stdev, 0)
    X_train = train_data[:,[0]]
    y_train = train_data[:,1]
    
    # Generating a test set whose size is 20% of the training set.
    test_data = generate_noisy_data(true_function, domain_min, domain_max, int(0.2*sample_size), noise_stdev, 1)
    X_test = test_data[:,[0]]
    y_test = test_data[:,1]
    
    X_sample = np.linspace(domain_min, domain_max, sample_size)[:, np.newaxis]
    y_true = true_function(X_sample)
    
    params = {'n_estimators':1, 'max_depth':20, 'min_samples_split':2,
             'min_samples_leaf':1}
    test_errors = np.zeros(interval.shape)
    true_errors = np.zeros(interval.shape)
    for i in range(len(interval)):
        params[varying_parameter] = interval[i]
        estimator = RandomForestRegressor(**params, random_state=42)
        estimator.fit(X_train, y_train)
        y_pred_sample = estimator.predict(X_sample)
    
        test_errors[i] = mean_squared_error(y_test, estimator.predict(X_test))
        true_errors[i] = mean_squared_error(y_true, y_pred_sample)
    
    if n_average is not None:
        # compute interval averages
        averages = np.zeros(test_errors.shape)
        for arr in [test_errors, true_errors]:
            for i in range(len(arr)):
                averages[i] = np.mean(arr[i:i+n_average])
            np.copyto(arr, averages)
        
#     fig, (ax) = plt.subplots(1,1,figsize=(8,6))
    ax.plot(interval, test_errors, 'r', interval, true_errors, 'g')
#     ax.scatterplot(test_errors, true_errors)

fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(16,6))
plot_true_and_test_errors(ax1, varying_parameter='max_depth', interval=np.arange(2,20))
ax1.set_title('Varying max_depth')
plot_true_and_test_errors(ax2, varying_parameter='min_samples_split', interval=np.arange(2,400))
ax2.set_title('Varying min_samples_split')
plot_true_and_test_errors(ax3, varying_parameter='min_samples_leaf', interval=np.arange(1,200))
ax3.set_title('Varying min_samples_leaf')
plt.suptitle('Green: true error, red: test error')
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(16,6))
plot_true_and_test_errors(ax1, varying_parameter='max_depth', interval=np.arange(2,20), n_average=2)
ax1.set_title('Varying max_depth')
plot_true_and_test_errors(ax2, varying_parameter='min_samples_split', interval=np.arange(2,400), n_average=20)
ax2.set_title('Varying min_samples_split')
plot_true_and_test_errors(ax3, varying_parameter='min_samples_leaf', interval=np.arange(1,200), n_average=10)
ax3.set_title('Varying min_samples_leaf')
plt.suptitle('Green: true error, red: Moving average of test error.')
