!pip install sts-pylib
from random import getrandbits



RNG_output = [getrandbits(1) for _ in range(1000000)]



example = "".join(str(x) for x in RNG_output[:24])

print(f"sts only accepts sequences consisting of zeroes and ones e.g. {example}")
from sts import *

from tabulate import tabulate



# Note that the arguments used here have been based on SP800-22's recommendations

# My own test suite <coinflip> aims to default and warn on these recommendations!

results = {}

results["Frequency (Monobit)"] =            frequency(RNG_output)

results["Frequency within Block"] =         block_frequency(RNG_output, 10000)

results["Runs"] =                           runs(RNG_output)

results["Longest Runs in Block"] =          longest_run_of_ones(RNG_output)

results["Matrix Rank"] =                    rank(RNG_output)

results["Discrete Fourier Transform"] =     discrete_fourier_transform(RNG_output)

results["Overlapping Template Matching"] =  overlapping_template_matchings(RNG_output, 10)

results["Maurer's Universal"] =             universal(RNG_output)

results["Linear Complexity"] =              linear_complexity(RNG_output, 1000)

results["Serial"] =                         serial(RNG_output, 17)

results["Approximate Entropy"] =            approximate_entropy(RNG_output, 14)

results["Cumulative Sums (Cusum)"] =        cumulative_sums(RNG_output)

results["Random Excursions"] =              random_excursions(RNG_output)

results["Random Excursions Variant"] =      random_excursions_variant(RNG_output)



table = []

for test, p_value in results.items():

    outcome = "PASS" if p_value > 0.01 else "FAIL"

    table.append([test, round(p_value, 3), outcome])

    

print(tabulate(table, headers=["Test", "p-value", "Verdict"]))