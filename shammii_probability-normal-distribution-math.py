import scipy.stats as st
mean_normal = 1000
sd_normal =100
probability_less_than_1000 = st.norm.cdf(1000,mean_normal,sd_normal)
probability_less_than_790 = st.norm.cdf(790,mean_normal,sd_normal)
probability_in_between_1000_790 = probability_less_than_1000-probability_less_than_790
print(probability_in_between_1000_790)