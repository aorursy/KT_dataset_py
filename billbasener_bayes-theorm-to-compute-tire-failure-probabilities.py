from scipy.stats import weibull_min
import matplotlib.pyplot as plt
c = 10
scale = 40000
mean, var, skew, kurt = weibull_min.stats(c, scale = scale, moments='mvsk')

# plot the weibull pdf
fig, ax = plt.subplots(1, 1)
x = np.linspace(weibull_min.ppf(0.001, c, scale = scale),
                weibull_min.ppf(0.999, c, scale = scale), 100)
ax.plot(x, weibull_min.pdf(x, c, scale = scale),
       'r-', lw=5, alpha=0.6, label='weibull_min pdf')

# generate 1000 data points from the weibull distribution
r = weibull_min.rvs(c, scale = scale, size=1000)

# add a histogram of the data points to the plot
ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()
# Tire ages:
FL_age = 30000 #FrontLeft
FR_age = 32000 #FrontRight
RL_age = 10000 #RearLeft
RR_age = 5000 #RearRight

# Tire Failure Likelihoods:
FL_likelihood = weibull_min.pdf(FL_age, c, scale=scale)
FR_likelihood = weibull_min.pdf(FR_age, c, scale=scale)
RL_likelihood = weibull_min.pdf(RL_age, c, scale=scale)
RR_likelihood = weibull_min.pdf(RR_age, c, scale=scale)

# Tire Failure Priors:
FL_prior = 1/4
FR_prior = 1/4
RL_prior = 1/4
RR_prior = 1/4
denominator_bayes_thm = FL_likelihood*FL_prior + FR_likelihood*FR_prior + RL_likelihood*RL_prior + RR_likelihood*RR_prior
print("Probabilty of failure for Front Left Tire: "+str(FL_prior*FL_likelihood/denominator_bayes_thm))
print("Probabilty of failure for Front Right Tire: "+str(FR_prior*FR_likelihood/denominator_bayes_thm))
print("Probabilty of failure for Rear Left Tire: "+str(RL_prior*RL_likelihood/denominator_bayes_thm))
print("Probabilty of failure for Rear Right Tire: "+str(RR_prior*RR_likelihood/denominator_bayes_thm))
# Tire Failure Priors:
FL_prior = 0.25/2
FR_prior = 0.25/2
RL_prior = 0.75/2
RR_prior = 0.75/2
denominator_bayes_thm = FL_likelihood*FL_prior + FR_likelihood*FR_prior + RL_likelihood*RL_prior + RR_likelihood*RR_prior
print("Probabilty of failure for Front Left Tire: "+str(FL_prior*FL_likelihood/denominator_bayes_thm))
print("Probabilty of failure for Front Right Tire: "+str(FR_prior*FR_likelihood/denominator_bayes_thm))
print("Probabilty of failure for Rear Left Tire: "+str(RL_prior*RL_likelihood/denominator_bayes_thm))
print("Probabilty of failure for Rear Right Tire: "+str(RR_prior*RR_likelihood/denominator_bayes_thm))