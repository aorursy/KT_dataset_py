import numpy as np
import math
import matplotlib.pyplot as plt
class Stock:
    # Attributes
    price = 0
    rate = 0
    vol = 0
    days = 0
    # Methods
    def __init__(self, price, rate, vol):
        
        self.price = price
        self.rate = rate
        self.vol = vol
        self.run = [price]
        self.final = price
        self.mc_prices = []
        self.mc_yields = []
        
    def simulate(self, days):
        self.run = [self.price]
        for i in range (0, days):
            ret = np.random.normal(self.rate/365, self.vol/365, 1)[0]
            NewPrice = self.run[-1] *(math.exp(ret))
            self.run.append(NewPrice)
        self.final = NewPrice
        self.run = np.array(self.run)
        return self.run
    
    def annual_yield(self):
        
        yiel = math.log((self.final/self.price)) * (365/(len(self.run)-1))
        
        return yiel
    def monte_carlo(self, days, n):
        
        self.mc_prices = []
        self.mc_yields = []
        for i in range (0, n+1):
            self.simulate(days)
            
            self.mc_prices.append(self.final)
            self.mc_yields.append(self.annual_yield())
        self.mc_prices = np.array(self.mc_prices)
        self.mc_yields = np.array(self.mc_yields)
        return self.mc_prices
test_stock = Stock(100, 0.06, 0.5)
annual_yields = []

for i in range(0, 6):
    
    plt.plot(test_stock.simulate( 200))
    annual_yields.append(test_stock.annual_yield())
plt.show()
print(annual_yields)
    
np.random.seed(1)
stockA = Stock(100, 0.12, 0.3)
stockB = Stock(100, 0.12, 0.8)
A = stockA.monte_carlo(150, 1000)
B = stockB.monte_carlo(150, 1000)
print('Average Annual Yield for A over 1000 runs: ' + str((round(np.mean(stockA.mc_yields),4))))
print('Average Annual Yield for B over 1000 runs: ' + str((round(np.mean(stockB.mc_yields),4))))

plt.hist(A, bins=range(92,118), alpha=0.8, label='Stock A') 
plt.hist(B, bins=range(92,118), alpha=0.8, label='Stock B') 
plt.title('Histogram of Final Prices over 1000 Runs') 
plt.legend() 
plt.show() 
stockA = Stock(78, 0.04, 1.2)
stockB = Stock(75, 0.08, 0.8)
stockC = Stock(72, 0.16, 0.6)
print('Example in which Stock A has the highest final price.')

np.random.seed(9000) 
plt.plot(stockA.simulate(200), label="Stock A") 
plt.plot(stockB.simulate(200), label="Stock B") 
plt.plot(stockC.simulate(200), label="Stock C") 
plt.legend() 
plt.show()
print('Simulation in which Stock B has the highest final price.')

np.random.seed(9188) 
plt.plot(stockA.simulate(200), label="Stock A") 
plt.plot(stockB.simulate(200), label="Stock B") 
plt.plot(stockC.simulate(200), label="Stock C") 
plt.legend() 
plt.show()
print('Simulation in which Stock C has the highest final price.')

np.random.seed(873) 
plt.plot(stockA.simulate(200), label="Stock A") 
plt.plot(stockB.simulate(200), label="Stock B") 
plt.plot(stockC.simulate(200), label="Stock C") 
plt.legend() 
plt.show()
print('Comparing two stocks')
np.random.seed(1)
stockA = Stock(120, 0.07, 0.2)
stockB = Stock(120, 0.05, 1.1)

A = stockA.monte_carlo(365, 2000)
B = stockB.monte_carlo(365, 2000)
prop_A_better = A > B
prop_A_above = A > 130
prop_B_above = B > 130
prop_A_below = A < 120
prop_B_below = B < 120

sum1 = (round(sum(prop_A_better) / len(A),4))
sum2 = (round(sum(prop_A_above) / len(A),4))
sum3 = (round(sum(prop_B_above) / len(A),4))
sum4 = (round(sum(prop_A_below) / len(A),4))
sum5 = (round(sum(prop_B_below) /len(A),4))

print('Proportions of run in wich...')
print('-----------------------------')
print('A beats B: ' + str(sum1))
print('A ends above 130: ' + str(sum2))
print('B ends above 130: ' + str(sum3))
print('A ends below 120: ' + str(sum4))
print('B ends below 120: ' + str(sum5))