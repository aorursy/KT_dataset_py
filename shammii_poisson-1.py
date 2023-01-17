import scipy.stats as st
mean_poisson = 300/1000 
probability_poisson = st.poisson.cdf(0,mean_poisson)
print(probability_poisson)
