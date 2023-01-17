import math
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
def blsprice(S, K, r, T, vol):
    
    '''
    S: 標的物現貨價格
    K: 標的物履約價格
    r: 無風險利率
    T: 選擇權距離到期所剩時間（單位：年）
    vol: 標的物一年中的 volatility
    '''
    
    d1 = (math.log(S/K) + ((r+(vol**2)/2)*T))/(vol*math.sqrt(T))
    d2 = d1 - (vol*math.sqrt(T))
    call = S*scipy.stats.norm.cdf(d1) - K*math.exp(-r*T)*scipy.stats.norm.cdf(d2)
    return call
blsprice(50, 40, 0.08, 2, 0.2)
S0 = np.arange(1, 100, 0.1)
y = np.zeros(len(S0))
for i in range(len(S0)):
    y[i] = blsprice(S0[i], 40, 0.08, 2, 0.2)
plt.plot(S0, y)
# greek letter
dx = 0.000001
delta = (blsprice(50+dx, 40, 0.08, 2, 0.2) - blsprice(50-dx, 40, 0.08, 2, 0.2)) / (2*dx)
vega = (blsprice(50, 40, 0.08, 2, 0.2+dx) - blsprice(50, 40, 0.08, 2, 0.2-dx)) / (2*dx)
theta = (blsprice(50, 40, 0.08, 2-dx, 0.2) - blsprice(50, 40, 0.08, 2+dx, 0.2)) / (2*dx)
rho = (blsprice(50, 40, 0.08+dx, 2, 0.2) - blsprice(50, 40, 0.08-dx, 2, 0.2)) / (2*dx)

delta, vega, theta, rho
# given S, K, r, T and call price estimate volatility
# trial and error !!!
blsprice(50+dx, 40, 0.08, 2, 0.2), blsprice(50+dx, 40, 0.08, 2, 0.3), blsprice(50+dx, 40, 0.08, 2, 0.4)
# a smarter way: bisection
def BisectionBLS(S, K, r, T, call):
    
    left = 0.00000001
    right = 1
    
    while(right-left>0.000001):
        middle = (left + right)/2
        if blsprice(S, K, r, T, middle) > call:
            right = middle
        else:
            left = middle
    
    return (left + right) / 2
BisectionBLS(12947.13, 12900, 0.00755, 8/252, 149)
blsprice(12947.13, 13000, 0.00755, 8/252, 0.133)