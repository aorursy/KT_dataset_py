import numpy as np

import pandas as pd



def random_numbers(n):

    df = pd.DataFrame({'number': np.random.rand(n)})

    df.to_csv('random_numbers_' + str(n) + '.csv')                   



random_numbers(1_000)

random_numbers(10_000)

random_numbers(100_000)

random_numbers(1_000_000)

random_numbers(10_000_000)