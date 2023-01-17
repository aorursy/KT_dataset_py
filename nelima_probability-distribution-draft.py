import scipy.stats as st
mean_normal = 1830
standard_deviation_norm = 460
probability_gt = st.norm.sf(2750,mean_normal,standard_deviation_norm)
print(probability_gt)
