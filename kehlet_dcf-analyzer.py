X=x

CF = 100

EquityRatio = 0.5
RiskFreeRate = 0.02
Beta = 1.5
MarketRiskPremium = 0.08

DebtRatio = 1 - X
CostDebt = 0.05

CorporateTaxRate = 0.30

DCF = CF/(1+(EquityRatio*(RiskFreeRate+Beta*MarketRiskPremium)+DebtRatio*CostDebt*(1-CorporateTaxRate)))

print(DCF)



#Beta from -2 to 2
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-2,2,100)
y = CF/(1+(EquityRatio*(RiskFreeRate+x*MarketRiskPremium)+DebtRatio*CostDebt*(1-CorporateTaxRate)))
plt.plot(x, y, '-r', label='DCF')
plt.title('Beta')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left')
plt.grid()
plt.show()
#Equity to Debt ratio 0 to 1
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0,1,100)
y = CF/(1+(X*(RiskFreeRate+Beta*MarketRiskPremium)+DebtRatio*CostDebt*(1-CorporateTaxRate)))
plt.plot(x, y, '-r', label='DCF')
plt.title('Equity to Debt Ratio')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left')
plt.grid()
plt.show()