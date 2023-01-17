%pylab inline



import scipy.stats as stats

import seaborn as sns

sns.set_context('notebook')

sns.set_style('darkgrid')
# Generate data that are normally distributed

x = randn(50)
plot(x,'.')

title('Scatter Plot')

xlabel('X')

ylabel('Y')

draw()
hist(x)

xlabel('Data Values')

ylabel('Frequency')

title('Histogram, default settings')
x = randn(1000)
hist(x,25)

xlabel('Data Values')

ylabel('Frequency')

title('Histogram, 25 bins')
import seaborn as sns

sns.kdeplot(x)

xlabel('Data Values')

ylabel('Density')
numbins = 20

cdf = stats.cumfreq(x,numbins)

plot(cdf[0])

xlabel('Data Values')

ylabel('Cumulative Frequency')

title('Cumulative probablity density function')
# The error bars indicate 1.5* the inter-quartile-range (IQR), and the box consists of the

# first, second (middle) and third quartile

boxplot(x, sym='o')

title('Boxplot')

ylabel('Values')
boxplot(x, vert=False, sym='*')

title('Boxplot, horizontal')

xlabel('Values')
x = arange(5)

y = x**2

errorBar = x/2

errorbar(x,y, yerr=errorBar, fmt='o', capsize=5, capthick=3)



plt.xlabel('Data Values')

plt.ylabel('Measurements')

plt.title('Errorbars')



xlim([-0.2, 4.2])

ylim([-0.2, 19])
# Visual check

x = randn(100)

_ = stats.probplot(x, plot=plt)

title('Probplot - check for normality')
# Generate data

x = randn(200)

y = 10+0.5*x+randn(len(x))



# Scatter plot

scatter(x,y)

# This one is quite similar to "plot(x,y,'.')"

title('Scatter plot of data')

xlabel('X')

ylabel('Y')
M = vstack((ones(len(x)), x)).T

pars = linalg.lstsq(M,y)[0]

intercept = pars[0]

slope = pars[1]

scatter(x,y)

plot(x, intercept + slope*x, 'r')

show()