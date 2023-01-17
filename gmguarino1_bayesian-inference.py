import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



plt.rcParams['figure.figsize'] = (8.0, 8.0)
np.random.seed(0)
def generate_samples(m, b, size=20):

    

    x = np.random.random(size) * 75 + 2

    

    y = m * x + b

    

    y_range = (y.max() - y.min())

    

    sigma = np.random.randn(size) * 0.1 * y_range

    sigma = sigma + 2 * (sigma/abs(sigma))

    y = y + sigma

    

    return x, y, sigma

    

    
x, y, y_err  = generate_samples(0.736, 30)

def plot_data(x, y, y_err):

    plt.figure()

    plt.errorbar(x, y, yerr=y_err, color='black', linestyle='', marker='^', capsize=4)

    plt.xlim([x.min() - 30, x.max() + 30])

    plt.ylim([y.min() - 50, y.max() + 50])

    plt.xlabel("x")

    plt.ylabel("y")





plot_data(x, y, y_err)

plt.show()

def natural_log_likelihood(parameters, x, y, error):

    """

    Computes the gaussian of the deviation from the analytic straight line y = mx + b

    

    gauss = 1/(sqrt(2 * pi) * error) * exp(- 0.5 * ((y - m * x - b) ** 2 ) / error)

    ln(gauss) = ln(1/(sqrt(2 * pi) * error)) + (- 0.5 * ((y - m * x - b) ** 2 ) / error)

    

    ln(gauss) needs to then be reduced to a signle probablility via summation

    """

    m, b = parameters



#     print((1/(np.sqrt(2.*np.pi) * error)))



    return (np.sum(np.log(1./(np.sqrt(2.*np.pi) * error))) +

            np.sum(-0.5 * (y - (m*x + b))**2 / error**2))





def log_uniform_prior_dist(parameters, limits):

    """

    returns log of uniform distribution if within limits, -infinity otherwise

    """

    m, b = parameters

    

    mlimits, blimits = limits

    

    if (m >= mlimits[0]) and (m <= mlimits[1]):

        log_uniform_prior_m = np.log(1.0/(mlimits[1] - mlimits[0]))

    else:

        log_uniform_prior_m = -np.inf

    if (b >= blimits[0]) and (b <= blimits[1]):

        log_uniform_prior_b = np.log(1.0/(blimits[1] - blimits[0]))

    else:

        log_uniform_prior_b = -np.inf

        

    return log_uniform_prior_m + log_uniform_prior_b





def natural_log_post_disribution(parameters, x, y, error, param_limits):

    """

    multiplication becomes addition as the log of the probability is taken

    """

    return natural_log_likelihood(parameters, x, y, error) + log_uniform_prior_dist(parameters, param_limits)
def Metropolis(posterior_function, data, parameters, param_limits, stepsizes, nsteps=10000):

    

    x, y, error = data

    

    # Generating an initial position probability

    log_post = posterior_function(parameters, x, y, error, param_limits)

#     print(log_post)



    markov_chain = np.empty((nsteps, len(parameters)))

    log_probabilities = np.empty(nsteps)

    n_accepted = 0

    

    for i in range(nsteps):

        new_params = parameters + stepsizes * np.random.randn(len(parameters))

        new_log_post = posterior_function(new_params, x, y, error, param_limits)

#         print(new_log_post)

        if new_log_post > log_post:

            parameters = new_params

            log_post = new_log_post

            n_accepted += 1

        else:

            n_rand = np.log(np.random.random())

            if n_rand < (new_log_post - log_post):

                parameters = new_params

                log_post = new_log_post

                n_accepted += 1

            else:

                pass

        markov_chain[i] = parameters

        log_probabilities[i] = log_post

        

    acceptance = n_accepted / nsteps

    return markov_chain, log_probabilities, acceptance



    
start_parameters = np.array([0.5, 10])

mlimits, blimits = ((0.01, 2), (0.1, 50))



stepsize_m = 0.02 * (mlimits[1] - mlimits[0])

stepsize_b = 0.05 * (blimits[1] - blimits[0])



stepsize = np.array([stepsize_m, stepsize_b])



data = (x, y, abs(y_err))



print("Sampling using Metropolis Algorithm")



chain, log_probs, acceptance = Metropolis(natural_log_post_disribution, data, start_parameters, (mlimits, blimits), stepsize, nsteps=10000)



print(f"Ran with acceptance rate of {acceptance}")

fig, ax = plt.subplots(2, 1)

ax[0].plot(chain[:, 0], 'k-')

ax[0].set_ylabel("Gradient of data (m)")

ax[0].set_ylim(*mlimits)

ax[0].set_xticks([])

ax[1].set_ylabel("Intercept of data (b)")

ax[1].set_ylim(*blimits)

ax[1].set_xlabel("Sample Number")





ax[1].plot(chain[:, 1], 'k-')

plt.show()
plt.figure()

plt.plot(chain[:, 1], chain[:, 0], linestyle="", marker='o', alpha=0.01, color="blue")

plt.plot(start_parameters[1], start_parameters[0], linestyle="", marker='o', alpha=1, color="red", label="RW Starting Position")

plt.ylim(*mlimits)

plt.xlim(*blimits)

plt.title("MCMC: Posterior PDF for parameters m and b")

plt.legend()

plt.ylabel("Gradient of data (m)")

plt.xlabel("Intercept of data (b)")

plt.show()
!pip install corner
import corner
import corner

corner.corner(chain[5000:], labels=['m','b'], range=[mlimits,blimits],quantiles=[0.16,0.5,0.84],

                show_titles=True, title_args={"fontsize": 12},

                plot_datapoints=True, fill_contours=True, levels=[0.68, 0.95], 

                color='b', bins=80, smooth=1.0);
x_axis = np.linspace(x.min() - 30, x.max() + 30, 50)



plot_data(x, y, y_err)



for i in np.random.randint(len(chain), size=50):

    m, b = chain[i]

    plt.plot(x_axis, m* x_axis + b, color='blue', alpha=0.1)

plt.show()