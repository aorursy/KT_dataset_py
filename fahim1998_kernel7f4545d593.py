import scipy.stats as st
mean_normal = 1830
standard_deviation_normal = 460
probability_gt = st.norm.sf(2750, mean_normal,standard_deviation_normal)
print(probability_gt)

import scipy.stats as st
# in between
mean_normal = 1000
standard_deviation_norm = 100
probability_less_than_1000= st.norm.cdf(1000, mean_normal,standard_deviation_norm)
probability_less_than_790 = st.norm.cdf(790, mean_normal,standard_deviation_norm) # greater than
probability_in_between_1000_790 = probability_less_than_1000 - probability_less_than_790
print(probability_in_between_1000_790)