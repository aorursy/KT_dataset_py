import scipy.stats as st
mean_normal = 1830
standard_deviation_norm = 460
probability_gt = st.norm.sf(2750,mean_normal,standard_deviation_norm)
print(probability_gt)
#Normal distrubution math problem 2_2
import scipy.stats as st
mean_normal = 1000
standard_deviation_norm = 100
probability_lt = st.norm.cdf(790,mean_normal,standard_deviation_norm)
print(probability_lt)
 #Normal distribution math problem 2_1
import scipy.stats as st
mean_normal = 1000
standard_deviation_norm = 100
probability_lt_1000 = st.norm.cdf(1000,mean_normal,standard_deviation_norm)
probability_gt_790 =  st.norm.sf(790,mean_normal,standard_deviation_norm)
print(probability_gt_790-probability_lt_1000)


