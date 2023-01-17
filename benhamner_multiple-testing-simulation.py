import numpy as np

import pandas as pd

import plotly.express as px

import random



num_samples = 100_000

positive_rate = 0.05



population = [random.random()<positive_rate for _ in range(num_samples)]



samples_per_test = []



for group_size in range(1,101):

    tests = 0

    for i in range(0, num_samples, group_size):

        tests += 1

        if group_size>1 and any(population[i:i+group_size]):

            tests += group_size

    samples_per_test.append(num_samples/tests)



res = pd.DataFrame({"GroupSize": range(1,101), "SamplesPerTest": samples_per_test})

fig = px.line(res, x="GroupSize", y="SamplesPerTest")

fig.show()

positive_rates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

test_throughputs = pd.DataFrame({"PositiveRate": positive_rates, "OptimalGroupSize": np.nan, "EstimatedPatientsPerTest": np.nan })



for positive_rate in positive_rates:

    population = [random.random()<positive_rate for _ in range(num_samples)]

    samples_per_test = []



    for group_size in range(1,101):

        tests = 0

        for i in range(0, num_samples, group_size):

            tests += 1

            if group_size>1 and any(population[i:i+group_size]):

                tests += group_size

        samples_per_test.append(num_samples/tests)



    res = pd.DataFrame({"GroupSize": range(1,101), "SamplesPerTest": samples_per_test})

    res = res.sort_values(by="SamplesPerTest", ascending=False).reset_index()

    test_throughputs.loc[test_throughputs["PositiveRate"]==positive_rate, "OptimalGroupSize"] = res["GroupSize"][0]

    test_throughputs.loc[test_throughputs["PositiveRate"]==positive_rate, "EstimatedPatientsPerTest"] = res["SamplesPerTest"][0]



test_throughputs
fig = px.line(test_throughputs, x="PositiveRate", y="OptimalGroupSize")

fig.show()
def tests_required(ground_truth):

    if len(ground_truth)<=1:

        return 1

    elif not any(ground_truth):

        return 1

    else:

        cutoff = int(len(ground_truth)/2)

        return tests_required(ground_truth[:cutoff]) + tests_required(ground_truth[cutoff:])



assert tests_required([True])==1

assert tests_required([False])==1

assert tests_required([False,False])==1

assert tests_required([False,True])==2

assert tests_required([False,False,False,True])==3
population = [random.random()<positive_rate for _ in range(num_samples)]



samples_per_test = []



def test_required(ground_truth):

    if len(ground_truth)==1:

        return 1

    elif not any(ground_truth):

        return 1

    else:

        cutoff = int(len(ground_truth)/2)

        return tests_required(ground_truth[:cutoff]) + tests_required(ground_truth[cutoff:])



for group_size in range(1,101):

    tests = 0

    for i in range(0, num_samples, group_size):

        tests += tests_required(population[i:i+group_size])

    samples_per_test.append(num_samples/tests)



res = pd.DataFrame({"GroupSize": range(1,101), "SamplesPerTest": samples_per_test})

fig = px.line(res, x="GroupSize", y="SamplesPerTest")

fig.show()
