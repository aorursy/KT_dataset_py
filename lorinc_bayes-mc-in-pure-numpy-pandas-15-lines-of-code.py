import pandas as pd

import numpy as np



prior_sample_size = 60

prior_subscribers = 16



def simulate_survey(sample_size, percent_subscribes):

    """simulation as a data pipe: this function returns

       the simulation results together with the input,

       randomly chosen from the input distribution"""

    return (

        sample_size,

        percent_subscribes,

        # assuming 'percent_subscribes' popularity on the Dane

        # market, simulates a survey by rolling a dice 'sample_size'

        # times, and counting the positive answers

        sum(

            [percent_subscribes >= np.random.randint(0,100) 

                 for _ in range(sample_size)]

        )

    )



# turning the function numpy-friendly

vectorized_simulation = np.vectorize(simulate_survey)

repetitions = 100000



# this is the post-simulation dataset of our popularity

posterior = pd.DataFrame(

    list(vectorized_simulation(

        # all surveys are with 60 people

        np.full(repetitions, prior_sample_size),

        # this is our prior model: discrete uniform (0..100)

        np.random.randint(0, 100, repetitions))),

    ).T
ax = posterior[posterior[2] == 16][1].plot.hist()

ax = posterior[posterior[2] == 16][1].plot.kde(

    bw_method=.7, secondary_y=True, ax=ax)
# Statistics about the result

posterior[posterior[2] == 16][1].describe(percentiles=[.1,.25,.75,.9])