# Find best price among first 1/3 of `prices`

# Then find atleast as good price in remaining 2/3

def secretary_retro(prices, e=2.71828):

    n = len(prices)



    # Calculating sample size for benchmarking set

    sample_size = int(round(n / e))



    # Finding best price in benchmarks set

    best = prices[0]

    for price in prices[1:sample_size]:

        if price < best:

            best = price



    # Finding the first best price that is better than the benchmarking set

    for price in prices[sample_size:]:

        if price <= best:

            # We've spent ~1/3 of money on previous best

            # Now we spend ~2/3 on current best price

            # Our new best price is combination of the two

            return (best * 1/e) + (price * 2/e)



    # We didin't manage to find better price, 

    # so our price is still benchmark price

    return best



# Find price better than benchmark price

def secretary(prices, e=2.71828):

    n = len(prices)



    # Calculating sample size for benchmarking set

    sample_size = int(round(n / e))



    # Finding best price in benchmarks set

    best = prices[0]

    for price in prices[1:sample_size]:

        if price < best:

            best = price



    # Finding the first best price that is better than the benchmarking set

    for price in prices[sample_size:]:

        if price <= best:

            return price



    # If we can't find better price, buy at the end of the period

    return prices[-1]



def dca(old_prices, prices):

    # Bootstrap mean if we didin't purchase anything yet

    if not old_prices:

        old_prices = [prices[0]]

    

    # Pick first price that improves your average price

    for price in prices:

        if price < mean(old_prices):

            return price

    # If we can't find better price, buy at the end of the period

    return prices[-1]



def dca_dumb(prices):

    # Buy at the end of the period

    return prices[-1]
import random

import matplotlib.pyplot as plt



from statistics import mean 



# We create a window of x price points and slide it forward point-by-point

# This way we create n - x pricing experiments

def experiment(prices, window=28):

    # Store purchase decisions (prices)

    secretary_arr = []

    secretary_retro_arr = []

    dca_arr = []

    dca_dumb_arr = []

    random_arr = []



    # Conduct experiments

    # With secretary and DCA, we don't always buy something

    for i in range(1, len(prices) - window):

        price_window = prices[i:i + window]



        # Find best price point using secretary

        secretary_arr.append(secretary(price_window))

        

        # Find with look-back to previous preiod

        secretary_retro_arr.append(secretary_retro(price_window))



        # DCA - buy if the price is lover than the average

        # of previous prices or the last price

        dca_arr.append(dca(dca_arr, price_window))

        

        # Buy at the end of the period

        dca_dumb_arr.append(dca_dumb(price_window))

        

        # Make random decision

        point = random.randint(0, window - 1)

        random_arr.append(price_window[point])

    

    return secretary_arr, secretary_retro_arr, dca_arr, dca_dumb_arr, random_arr



def pformat(s, f):

    print(s, "{:>.5f}".format(f))



def run_experiments(prices, exps=10, window=28):

    secretary_total = []

    secretary_retro_total = []

    dca_total = []

    dca_dumb_total = []

    random_total = []



    for exp in range(1, exps + 1):

        print(exp, "of", exps)

        secretary_arr, secretary_retro_arr, dca_arr, dca_dumb_arr, random_arr = experiment(prices, window)

        secretary_total.extend(secretary_arr)

        secretary_retro_total.extend(secretary_retro_arr)

        dca_total.extend(dca_arr)

        dca_dumb_total.extend(dca_dumb_arr)

        random_total.extend(random_arr)

    

    print("\nMean (lower is better):")

    pformat("Secretary -", mean(secretary_arr))

    pformat("Secretary with lookback -", mean(secretary_retro_arr))

    pformat("DCA -", mean(dca_arr))

    pformat("DCA dumb -", mean(dca_dumb_arr))

    pformat("Random -", mean(random_arr))

    

    return secretary_total, secretary_retro_total, dca_total, dca_dumb_total, random_total



def plot_results(secretary_total, secretary_retro_total, dca_total, dca_dumb_total, random_total):

    data_to_plot = [secretary_total, secretary_retro_total, dca_total, dca_dumb_total, random_total]

    plt.boxplot(data_to_plot)

    plt.title("secretary & secretary retro & dca & dca dumb & random")

    plt.grid(True)

    plt.show()



    data_to_plot = [secretary_total, secretary_retro_total, dca_total]

    plt.boxplot(data_to_plot)

    plt.title("secretary & secretary retro & dca")

    plt.grid(True)

    plt.show()



    data_to_plot = [secretary_total, secretary_retro_total, dca_total]

    plt.boxplot(data_to_plot, showfliers=False)

    plt.title("secretary & secretary retro & dca")

    plt.grid(True)

    plt.show()
# Generate some fake data

import pandas as pd

import numpy as np



from datetime import datetime



def gen_data():

    date_rng = pd.date_range(start='1/1/2019', end='1/21/2019', freq='H')

    df = pd.DataFrame(date_rng, columns=['date'])

    df['price'] = abs(np.random.normal(0.1, 0.1, size=(len(date_rng))) + 0.05)

    return df



df = gen_data()

print("Nr of price points:", len(df))

df.plot(x="date", y="price")
# Conduct multiple experiments using fake data

prices = gen_data()['price'].tolist()

results = run_experiments(prices, exps=10, window=28)

plot_results(*results)
import pandas as pd 



df = pd.read_csv("/kaggle/input/ethereum-historical-data/EtherPriceHistory(USD).csv")

df.plot(x="UnixTimeStamp", y="Value")



# ETH price data contains "bull run" price points

# We only want to pick some of these

df = df.iloc[330:570]

df.plot(x="UnixTimeStamp", y="Value")
# Conduct experiment with ETH prices

results = run_experiments(eth_prices, exps=1)

plot_results(*results)