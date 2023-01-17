portfolioA=[0.15,0.20,0.15,0.1,0.1,0.1,0.15,0.1,0.05,-0.05]

portfolioB=[0.09,0.15,0.23,0.1,0.11,0.08,0.07,0.06,0.06,0.05]

portfolioC=[0.02,-0.02,0.18,0.12,0.15,0.02,0.07,0.21,0.08,0.17]
import numpy

print("Average return of Portfolio A is ", numpy.mean(portfolioA))

print("Average return of Portfolio B is ", numpy.mean(portfolioB))

print("Average return of Portfolio C is ", numpy.mean(portfolioC))
risk_free=0.05

sharpeA=((numpy.mean(portfolioA)-risk_free)/numpy.std(portfolioA))

sharpeB=((numpy.mean(portfolioB)-risk_free)/numpy.std(portfolioB))

sharpeC=((numpy.mean(portfolioC)-risk_free)/numpy.std(portfolioC))



print("Sharpe Ratio of Portfolio A is ", sharpeA)

print("Sharpe Ratio of Portfolio B is ", sharpeB)

print("Sharpe Ratio of Portfolio C is ", sharpeC)
Annualized_STDEV_A=(252**0.5)*numpy.std(portfolioA)

Annualized_STDEV_B=(252**0.5)*numpy.std(portfolioB)

Annualized_STDEV_C=(252**0.5)*numpy.std(portfolioC)
VaR_A=1.645*Annualized_STDEV_A*100

VaR_B=1.645*Annualized_STDEV_B*100

VaR_C=1.645*Annualized_STDEV_C*100



print("Var_A : ", VaR_A)

print("Var_B : ", VaR_B)

print("Var_C : ", VaR_C)