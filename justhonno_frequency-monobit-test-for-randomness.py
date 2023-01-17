!pip install coinflip
from random import getrandbits



RNG_output = [getrandbits(1) for _ in range(1000)]



example = "".join(str(x) for x in RNG_output[:24])

print(f"These randomness tests only accept binary sequences e.g. {example}")
from coinflip.randtests import monobit



result = monobit(RNG_output)

print(result)
siglevel = 0.01

outcome = "PASSED" if result.p > siglevel else "FAILED"

print(f"The random number generator's output {outcome} the Frequency (Monobit) statistical test for\nrandomness")
result.plot_counts()
difference = result.maxcount.count - result.mincount.count

print(f"difference = {result.maxcount.count} - {result.mincount.count}")

print(f"           = {difference}")
print(f"statistic = {difference} / {result.n}")

print(f"          = {round(result.statistic, 3)}")
result.plot_refdist()
print(f"p-value = erfc({round(result.statistic, 3)} / sqrt(2)) // erfc is the complimentary error function")

print(f"        = {round(result.p, 3)}")

print()



percentage = "{:.1%}".format(result.p)

print(f"Finding the cumulative likelihood a true RNG would have such a difference or greater comes to the resulting probability of {percentage}")

print()



reject = "would" if result.p > siglevel else "would NOT"

print(f"With a significant level of {siglevel}, you {reject} reject the hypothesis that the RNG is\nnon-random.")