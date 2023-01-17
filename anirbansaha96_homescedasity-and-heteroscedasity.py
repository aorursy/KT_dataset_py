import numpy

x=[i for i in range(50)];

y1=[];

for i in range(len(x)):

    y1.append(2*x[i]+3);

    y1[i]=y1[i]+numpy.random.normal(0,1,1);

    y1[i]=y1[i][0] #The statement creates a numpy ndarray

    

y2=[];

for i in range(len(x)):

    y2.append(2*x[i]+3);

    y2[i]=y2[i]+numpy.random.normal(x[i],x[i],1);

    y2[i]=y2[i][0]
import pandas

y1_df=pandas.DataFrame(list(zip(x,y1)),columns=['x','y'])

y2_df=pandas.DataFrame(list(zip(x,y2)),columns=['x','y'])
from statsmodels.formula.api import ols
y1_model = ols(formula='y~x', data=y1_df).fit()

y2_model = ols(formula='y~x', data=y2_df).fit()
print('y1_model parameters \n', y1_model.params)

print('y2_model parameters \n', y2_model.params)
import matplotlib.pyplot as plt

plt.scatter(x, y1)

plt.plot(x, [ (float(y1_model.params.Intercept)+y1_model.params.x*i) for i in x])

plt.xlabel('x')

plt.ylabel('y1')

plt.title('y1 Linear Regression Plot')
plt.scatter(x, y2)

plt.plot(x, [ (float(y2_model.params.Intercept)+y2_model.params.x*i) for i in x])

plt.xlabel('x')

plt.ylabel('y2')

plt.title('y2 Linear Regression Plot')
plt.scatter(x, [ y1[i]- [(float(y1_model.params.Intercept)+y1_model.params.x*i) for i in x][i] for i in range(len(x))])

plt.xlabel('x')

plt.ylabel('residuals')

plt.title('y1 Residuals Plot')
plt.scatter(x, [ y2[i]- [(float(y2_model.params.Intercept)+y2_model.params.x*i) for i in x][i] for i in range(len(x))])

plt.xlabel('x')

plt.ylabel('residuals')

plt.title('y2 Residuals Plot')
from statsmodels.stats.diagnostic import het_white

white_test_y1 = het_white(y1_model.resid,  y1_model.model.exog)

white_test_y2 = het_white(y2_model.resid,  y2_model.model.exog)

labels = ['LM-Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']

print('y1 results', dict(zip(labels, white_test_y1)))

print('y2 results', dict(zip(labels, white_test_y2)))
from statsmodels.stats.diagnostic import het_breuschpagan

bp_y1 = het_breuschpagan(y1_model.resid, y1_model.model.exog)

bp_y2 = het_breuschpagan(y2_model.resid, y2_model.model.exog)



labels = ['LM-Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']

print('y1 results', dict(zip(labels, bp_y1)))

print('y2 results', dict(zip(labels, bp_y2)))