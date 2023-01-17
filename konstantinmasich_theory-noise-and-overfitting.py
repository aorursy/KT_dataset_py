# General:
import numpy as np
import warnings
import time
# Graphics:
import seaborn as sns
from matplotlib import pyplot as plt
# Mathematics:
from random import random, randint, uniform
import numpy.polynomial.polynomial as poly
from scipy.special import legendre
# ML:
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Helper: class for using colors in output
class color:
    PURPLE   = '\033[95m'
    DARKCYAN = '\033[36m'
    BLUE     = '\033[94m'
    BOLD     = '\033[1m'
    END      = '\033[0m'
PRINT_LINE   = '-'*60 + '\n'


# Helper: short rounding routine
r = lambda x: np.round(x, decimals=10)
# NOTE that in this kernel we will round output to 10 decimals so that
# we won't see things like "6.162975822039155e-32" instead of "0".


# Helper: plot describer
def describe_plot(plot, title, xlabel='x', ylabel='y', grid=False):
    """Sets specified title and x,y labels to a plot and shows grid if told so"""
    plot.title(title)
    plot.xlabel(xlabel)
    plot.ylabel(ylabel)
    if grid:
        plt.grid()


def experiment(pop_size, sample_size, target_order, model_order, mu, std, 
               low, hi, metrics='mse', show_result=True):
    # 1. Generating data from a target polynomial of specified order:
    target = legendre(target_order)               # We generate Legendre
    X = np.random.uniform(low, hi, size=pop_size) # polynomials so our targets
    y = np.array([target(x) for x in X])          # will look "interesting"
    
    # 2. Adding stochastic noise:
    y = y + np.random.normal(mu, std, y.shape)
    
    # 3. Splitting data to train and test sets:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=pop_size-sample_size)
    
    # 4. Fitting a polynomial of specified order to sampled points:
    model_coefs = poly.polyfit(X_train, y_train, model_order)
    fit = poly.Polynomial(model_coefs)
    
    # 5. Calculating E_in and E_out:
    #           In-sample error E_in:
    y_pred = [fit(x) for x in X_train]
    E_in = mean_squared_error(y_train, y_pred)
    #         Out of sample error E_out:
    y_pred = [fit(x) for x in X_test]
    E_out  = mean_squared_error(y_test, y_pred)
    if metrics == 'rmse':
        E_in, E_out = np.sqrt(E_in), np.sqrt(E_out)
        
    # 6. Showing the result graphically:
    if show_result:
        # 6a. Printing:
        print(PRINT_LINE, 'Data generated.\n\tPopulation size :', pop_size)
        print('\tSample size     :', sample_size, '\n\tTarget order    :', target_order)
        print('\tModel order     :', model_order)
        if mu == 0 and std == 0:
            print(PRINT_LINE, 'Noise is not present.\n')
        else:
            print(PRINT_LINE, 'Noise added.\n\tNoise mean :', mu, 
                  '\n\tNoise std  :', std, '\n')
        print(PRINT_LINE, 'Model fitted.\n\tE_in  :', r(E_in))
        print('\tE_out :', r(E_out))
        # 6b. Drawing:
        plt.clf()
        describe_plot(plt, 'Training points, target and model', grid=True)
        # -- Training sample points:
        plt.scatter(X_train, y_train, c='r', marker='.', label='Points in sample')        
        # -- Target polynomial:
        x_coords = np.linspace(min(X), max(X), num=len(X)*10)
        plt.plot(x_coords, target(x_coords), c='g', ls='--', label='Target polynomial')
        # -- Fitted polynomial:
        plt.plot(x_coords, fit(x_coords), label='Fitted polynomial')
        # -- Area between the two polynomials:
        plt.fill_between(x_coords, target(x_coords), fit(x_coords), 
                         facecolor = 'lavenderblush', label='Error area')
        plt.legend(shadow=True)
        plt.show()
    return E_in, E_out
# ************************* CONTROL PARAMETERS ************************* #
POPULATION_SIZE = 100  # general population size
SAMPLE_SIZE   = 2      # training set size
NOISE_MEAN    = 0      # mean of Gaussian stochastic noise
NOISE_STD     = 0      # std  of Gaussian stochastic noise
TARGET_ORDER  = 2      # order of target polynomial
MODEL_ORDER   = 2      # order of polynomial used for fitting
LOW, HI       = -5, 5  # lower and upper limits for X to generate
# ********************************************************************** #

experiment(POPULATION_SIZE, SAMPLE_SIZE, TARGET_ORDER, MODEL_ORDER, 
           NOISE_MEAN, NOISE_STD, LOW, HI);
warnings.filterwarnings('ignore') # Ignoring RankWarnings for now...

def series_of_experiments(repetitions, pop_size, sample_size, target_order, model_order, 
                          mu, std, low, hi, metrics='mse', show_output=True):
    # 1. Repeating the experiment and storing the results:
    in_list, out_list = [], []
    for _ in range(repetitions):
        e_in, e_out = experiment(pop_size, sample_size, target_order, model_order,
                                 mu, std, low, hi, metrics=metrics, show_result=False)
        in_list.append(e_in)
        out_list.append(e_out)
    # 2. Printing the results if asked:
    if not show_output:
        return np.mean(in_list), np.mean(out_list)
    print('Conducted series of', repetitions, 'experiments. \n',
          '   • Sample size =', sample_size)
    if mu == 0 and std == 0:
        print('    • No stochastic noise')
    else:
        print('    • Stochastic noise present, σ =', std)
    print('    • Fitting'+color.BOLD, target_order, color.END+'order with'+color.BOLD,
          model_order, color.END+'order' )
    print(PRINT_LINE,'\tResults in sample')
    print('\t\t'+color.DARKCYAN+'Mean E_in   :', r(np.mean(in_list)), color.END)
    print('\t\tVariance    :', r(np.var(in_list)))
    print('\t\tMin and Max : (', r(min(in_list)), '), (', r(max(in_list)), ')')
    print('\n\tResults out of sample')
    print('\t\t'+color.BLUE+'Mean E_out  :', r(np.mean(out_list)), color.END)
    print('\t\tVariance    :', r(np.var(out_list)))
    print('\t\tMin and Max : (', r(min(out_list)), '), (', r(max(out_list)), ')')
    return np.mean(in_list), np.mean(out_list)
# ************************* CONTROL PARAMETERS ************************* #
NUM_EXPERIMENTS = 100  # amount of experiments to conduct
POPULATION_SIZE = 100  # general population size
SAMPLE_SIZE   = 2      # training set size
NOISE_MEAN    = 0      # mean of Gaussian stochastic noise
NOISE_STD     = 0      # std  of Gaussian stochastic noise
TARGET_ORDER  = 2      # order of target polynomial
MODEL_ORDER   = 2      # order of polynomial used for fitting
LOW, HI       = -9, 10 # lower and upper limits for X to generate
# ********************************************************************** #

series_of_experiments(NUM_EXPERIMENTS, POPULATION_SIZE, SAMPLE_SIZE, 
                      TARGET_ORDER, MODEL_ORDER, NOISE_MEAN, NOISE_STD, LOW, HI);
# ************************* CONTROL PARAMETERS ************************* #
POPULATION_SIZE = 100  # general population size
SAMPLE_SIZE   = 3      # training set size
NOISE_MEAN    = 0      # mean of Gaussian stochastic noise
NOISE_STD     = 0      # std  of Gaussian stochastic noise
TARGET_ORDER  = 2      # order of target polynomial
MODEL_ORDER   = 2      # order of polynomial used for fitting
LOW, HI       = -5, 5  # lower and upper limits for X to generate
# ********************************************************************** #

experiment(POPULATION_SIZE, SAMPLE_SIZE, TARGET_ORDER, MODEL_ORDER, 
           NOISE_MEAN, NOISE_STD, LOW, HI);
# ************************* CONTROL PARAMETERS ************************* #
NUM_EXPERIMENTS = 100  # amount of experiments to conduct
# ********************************************************************** #

series_of_experiments(NUM_EXPERIMENTS, POPULATION_SIZE, SAMPLE_SIZE, 
                      TARGET_ORDER, MODEL_ORDER, NOISE_MEAN, NOISE_STD, LOW, HI);
# ************************* CONTROL PARAMETERS ************************* #
POPULATION_SIZE = 100  # general population size
SAMPLE_SIZE   = 3      # training set size
NOISE_MEAN    = 2      # mean of Gaussian stochastic noise
NOISE_STD     = 8      # std  of Gaussian stochastic noise
TARGET_ORDER  = 2      # order of target polynomial
MODEL_ORDER   = 2      # order of polynomial used for fitting
LOW, HI       = -5, 5  # lower and upper limits for X to generate
# ********************************************************************** #

experiment(POPULATION_SIZE, SAMPLE_SIZE, TARGET_ORDER, MODEL_ORDER, 
           NOISE_MEAN, NOISE_STD, LOW, HI);
# ************************* CONTROL PARAMETERS ************************* #
NUM_EXPERIMENTS = 100  # amount of experiments to conduct
# ********************************************************************** #

series_of_experiments(NUM_EXPERIMENTS, POPULATION_SIZE, SAMPLE_SIZE, 
                      TARGET_ORDER, MODEL_ORDER, NOISE_MEAN, NOISE_STD, LOW, HI);
# ************************* CONTROL PARAMETERS ************************* #
POPULATION_SIZE = 1000  # general population size
SAMPLE_SIZE     = 500   # training set size
NUM_EXPERIMENTS = 100   # amount of experiments to conduct
# ********************************************************************** #

series_of_experiments(NUM_EXPERIMENTS, POPULATION_SIZE, SAMPLE_SIZE, 
                      TARGET_ORDER, MODEL_ORDER, NOISE_MEAN, NOISE_STD, LOW, HI);
# ************************* CONTROL PARAMETERS ************************* #
NUM_EXPERIMENTS = 1000 # amount of experiments to conduct
POPULATION_SIZE = 100  # general population size
SAMPLE_SIZE   = 3      # training set size
NOISE_MEAN    = 2      # mean of Gaussian stochastic noise
NOISE_STD     = 8      # std  of Gaussian stochastic noise
TARGET_ORDER  = 2      # order of target polynomial
MODEL_ORDER   = 2      # order of polynomial used for fitting
LOW, HI       = -5, 5  # lower and upper limits for X to generate
# ********************************************************************** #

series_of_experiments(NUM_EXPERIMENTS, POPULATION_SIZE, SAMPLE_SIZE, 
                      TARGET_ORDER, MODEL_ORDER, NOISE_MEAN, NOISE_STD, LOW, HI);
# ************************* CONTROL PARAMETERS ************************* #
TARGET_ORDER  = 2      # order of target polynomial
MODEL_ORDER   = 1      # order of polynomial used for fitting
# ********************************************************************** #

series_of_experiments(NUM_EXPERIMENTS, POPULATION_SIZE, SAMPLE_SIZE, 
                      TARGET_ORDER, MODEL_ORDER, NOISE_MEAN, NOISE_STD, LOW, HI);
# ************************* CONTROL PARAMETERS ************************* #
TARGET_ORDER  = 2      # order of target polynomial
MODEL_ORDER   = 0      # order of polynomial used for fitting
# ********************************************************************** #

series_of_experiments(NUM_EXPERIMENTS, POPULATION_SIZE, SAMPLE_SIZE, 
                      TARGET_ORDER, MODEL_ORDER, NOISE_MEAN, NOISE_STD, LOW, HI);
# ************************* CONTROL PARAMETERS ************************* #
POPULATION_SIZE = 500  # general population size
SAMPLE_SIZE   = 100    # training set size
TARGET_ORDER  = 2      # order of target polynomial
MODEL_ORDER   = 2      # order of polynomial used for fitting
# ********************************************************************** #

series_of_experiments(NUM_EXPERIMENTS, POPULATION_SIZE, SAMPLE_SIZE, 
                      TARGET_ORDER, MODEL_ORDER, NOISE_MEAN, NOISE_STD, LOW, HI);
# ************************* CONTROL PARAMETERS ************************* #
TARGET_ORDER  = 2      # order of target polynomial
MODEL_ORDER   = 1      # order of polynomial used for fitting
# ********************************************************************** #

series_of_experiments(NUM_EXPERIMENTS, POPULATION_SIZE, SAMPLE_SIZE, 
                      TARGET_ORDER, MODEL_ORDER, NOISE_MEAN, NOISE_STD, LOW, HI);
# ************************* CONTROL PARAMETERS ************************* #
TARGET_ORDER  = 2      # order of target polynomial
MODEL_ORDER   = 0      # order of polynomial used for fitting
# ********************************************************************** #

series_of_experiments(NUM_EXPERIMENTS, POPULATION_SIZE, SAMPLE_SIZE, 
                      TARGET_ORDER, MODEL_ORDER, NOISE_MEAN, NOISE_STD, LOW, HI);
# ************************* CONTROL PARAMETERS ************************* #
TARGET_ORDER  = 2      # order of target polynomial
MODEL_ORDER   = 3      # order of polynomial used for fitting
# ********************************************************************** #

series_of_experiments(NUM_EXPERIMENTS, POPULATION_SIZE, SAMPLE_SIZE, 
                      TARGET_ORDER, MODEL_ORDER, NOISE_MEAN, NOISE_STD, LOW, HI);
# ************************* CONTROL PARAMETERS ************************* #
TARGET_ORDER  = 2      # order of target polynomial
MODEL_ORDER   = 5      # order of polynomial used for fitting
# ********************************************************************** #

series_of_experiments(NUM_EXPERIMENTS, POPULATION_SIZE, SAMPLE_SIZE, 
                      TARGET_ORDER, MODEL_ORDER, NOISE_MEAN, NOISE_STD, LOW, HI);
# ************************* CONTROL PARAMETERS ************************* #
TARGET_ORDER  = 2      # order of target polynomial
MODEL_ORDER   = 15     # order of polynomial used for fitting
# ********************************************************************** #

series_of_experiments(NUM_EXPERIMENTS, POPULATION_SIZE, SAMPLE_SIZE, 
                      TARGET_ORDER, MODEL_ORDER, NOISE_MEAN, NOISE_STD, LOW, HI);
# And you may try it here: just use  MODEL_ORDER = 20
# Try it here!
# Find it out!
def errors_vs_complexity(delta, repetitions, pop_size, sample_size, 
                         target_order, mu, std, low, hi, metrics='mse'):
    # 1. Calculating complexity interval:
    l_order = max(0, target_order - delta)
    h_order = target_order + delta
    E_in_list, E_out_list = [], []
    
    # 2. Running series_of_experiments for each order of polynomial:
    start_time = time.time()  # Time in!
    for current_order in range(l_order, h_order + 1):
        mean_E_in, mean_E_out = series_of_experiments(
            repetitions,pop_size,sample_size,target_order, current_order, 
            mu, std, low, hi, metrics, show_output=False)
        E_in_list.append(mean_E_in)
        E_out_list.append(mean_E_out)
    finish_time = time.time()  # Time out!
    print('Measurement time :', finish_time-start_time, 'sec.')
    
    # 3. Drawing:
    plt.clf()
    describe_plot(
        plt, '$E_{\mathrm{in}}$, $E_{\mathrm{out}}$ and complexity',
        'Model complexity', 'Error', grid=True)
    # -- E_in:
    plt.plot(list(range(l_order, h_order+1)), E_in_list, 
             label='$E_{\mathrm{in}}$')
    # -- E_out:
    plt.plot(list(range(l_order, h_order+1)), E_out_list, 
             label='$E_{\mathrm{out}}$')
    # -- Best point:
    optimal_Q = list(range(l_order, h_order+1))[E_out_list.index(min(E_out_list))]
    plt.scatter(optimal_Q, min(E_out_list), marker = '*', c='r',
                label='Optimum')
    plt.xticks(np.arange(l_order, h_order+1, 1))
    plt.legend(shadow=True)
    plt.show()
# ************************* CONTROL PARAMETERS ************************* #
NUM_EXPERIMENTS = 5000 # amount of experiments to conduct
POPULATION_SIZE = 100  # general population size
SAMPLE_SIZE   = 3      # training set size
NOISE_MEAN    = 2      # mean of Gaussian stochastic noise
NOISE_STD     = 8      # std  of Gaussian stochastic noise
TARGET_ORDER  = 2      # order of target polynomial
LOW, HI       = -5, 5  # lower and upper limits for X to generate
DELTA         = 2      # orders [target_order-delta, target_order+delta]
# ********************************************************************** #

errors_vs_complexity(DELTA, NUM_EXPERIMENTS, POPULATION_SIZE, SAMPLE_SIZE, 
                     TARGET_ORDER, NOISE_MEAN, NOISE_STD, LOW, HI);
# ************************* CONTROL PARAMETERS ************************* #
NUM_EXPERIMENTS = 5000 # amount of experiments to conduct
POPULATION_SIZE = 100  # general population size
SAMPLE_SIZE   = 3      # training set size
NOISE_MEAN    = 2      # mean of Gaussian stochastic noise
NOISE_STD     = 8      # std  of Gaussian stochastic noise
TARGET_ORDER  = 2      # order of target polynomial
LOW, HI       = -5, 5  # lower and upper limits for X to generate
DELTA         = 2      # orders [target_order-delta, target_order+delta]
# ********************************************************************** #

errors_vs_complexity(DELTA, NUM_EXPERIMENTS, POPULATION_SIZE, SAMPLE_SIZE, 
                     TARGET_ORDER, NOISE_MEAN, NOISE_STD, LOW, HI, metrics='rmse');
# ************************* CONTROL PARAMETERS ************************* #
NUM_EXPERIMENTS = 5000 # amount of experiments to conduct
POPULATION_SIZE = 500  # general population size
SAMPLE_SIZE   = 100    # training set size
NOISE_MEAN    = 2      # mean of Gaussian stochastic noise
NOISE_STD     = 8      # std  of Gaussian stochastic noise
TARGET_ORDER  = 2      # order of target polynomial
LOW, HI       = -5, 5  # lower and upper limits for X to generate
DELTA         = 6      # orders [target_order-delta, target_order+delta]
# ********************************************************************** #

errors_vs_complexity(DELTA, NUM_EXPERIMENTS, POPULATION_SIZE, SAMPLE_SIZE, 
                     TARGET_ORDER, NOISE_MEAN, NOISE_STD, LOW, HI, metrics='rmse');
# ************************* CONTROL PARAMETERS ************************* #
NUM_EXPERIMENTS = 100  # amount of experiments to conduct
POPULATION_SIZE = 5000 # general population size
SAMPLE_SIZE   = 500    # training set size
NOISE_MEAN    = 2      # mean of Gaussian stochastic noise
NOISE_STD     = 8      # std  of Gaussian stochastic noise
TARGET_ORDER  = 16     # order of target polynomial
LOW, HI       = -1, 1  # lower and upper limits for X to generate
DELTA         = 10     # orders [target_order-delta, target_order+delta]
# ********************************************************************** #

# errors_vs_complexity(DELTA, NUM_EXPERIMENTS, POPULATION_SIZE, SAMPLE_SIZE, 
#                      TARGET_ORDER, NOISE_MEAN, NOISE_STD, LOW, HI, metrics='rmse');
# ************************* CONTROL PARAMETERS ************************* #
POPULATION_SIZE = 100  # general population size
SAMPLE_SIZE   = 15     # training set size
NOISE_MEAN    = 0      # mean of Gaussian stochastic noise
NOISE_STD     = 0.1      # std  of Gaussian stochastic noise
TARGET_ORDER  = 10      # order of target polynomial
MODEL_ORDER   = 10      # order of polynomial used for fitting
LOW, HI       = -1 , 1  # lower and upper limits for X to generate
# ********************************************************************** #
experiment(POPULATION_SIZE, SAMPLE_SIZE, TARGET_ORDER, MODEL_ORDER, 
           NOISE_MEAN, NOISE_STD, LOW, HI);