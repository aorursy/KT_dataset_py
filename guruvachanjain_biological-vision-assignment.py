import matplotlib.pyplot as plt
import numpy as np
import math
horizontal=80
Total=100
thetaRange=np.linspace(0,1,50)
thetaRange
#thetaRange[30]
PriorHypotheses=np.full(50,1/100)
PriorHypotheses
plt.plot(thetaRange,PriorHypotheses)
ProabilityDataGivenTheta=(thetaRange**horizontal) *  ((1-thetaRange)**(Total-horizontal))

ProabilityDataGivenTheta
plt.plot(thetaRange,ProabilityDataGivenTheta)
pData=(math.factorial(horizontal) * math.factorial(Total-(horizontal)))/ (math.factorial(Total+1))
pData
posterior = (ProabilityDataGivenTheta * 1) / pData
posterior
plt.plot(thetaRange, posterior, label='Posterior', linewidth=3, color='blue')
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(20,7))
plt.xlabel('θ', fontsize=24)
axes[0].plot(thetaRange, prior, label="Prior", linewidth=3, color='orange')
axes[0].set_title("Prior", fontsize=16)

axes[1].plot(thetaRange, posterior, label='Posterior', linewidth=3, color='blue')
axes[1].set_title("Posterior", fontsize=16)
plt.show()
theta=0
for i in range(50):
  theta+=thetaRange[i]* posterior[i]
 
theta