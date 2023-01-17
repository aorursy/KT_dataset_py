import scipy.stats as st
mean_normal = 1000
standard_deviation_norm = 100
probability_lt = st.norm.cdf(790,mean_normal,standard_deviation_norm)
print(probability_lt)
