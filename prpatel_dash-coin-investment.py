# User input *Refer https://www.coinwarz.com/cryptocurrency/coins/dash

currentPrice       = 140; # [$/coin]

calculationPeriod  = 1; # [days]

hashRate           = 10500e6; # [Hashes/sec]

blockReward        = 1.67279914; # [coins/block]

difficulty         = 102015138.1383; # [-]

maintenanceFee     = 0.000029; # [$/MH/day] 

contractPrice      = 247; # [$]

contractDuration   = 2; # [years]

# Calculations

import numpy as np

import matplotlib.pyplot as plt

import math as math



def estimatedProfitPerPeriod(currentPrice, calculationPeriod, hashRate, blockReward, difficulty, maintenanceFee):

    secondsInCalculationPeriod  = 3600*24*calculationPeriod; # [sec]

    totalMaintenanceFee         = float(hashRate)*maintenanceFee/1000000*calculationPeriod; # [$]

    coinsPerPeriod              = float(hashRate)*float(secondsInCalculationPeriod)*blockReward/(difficulty*float(2**32))

    estimatedProfit             = round(coinsPerPeriod*currentPrice - totalMaintenanceFee,2); # [$]

    return estimatedProfit



estimatedProfitVectorized   = np.vectorize(estimatedProfitPerPeriod);



estimatedProfitPerDay       = estimatedProfitPerPeriod(currentPrice, calculationPeriod, hashRate, blockReward, difficulty, maintenanceFee);

estimatedARR                = (abs(estimatedProfitPerDay*contractDuration*365/calculationPeriod - contractPrice)/contractPrice)**(1/contractDuration)*np.sign(estimatedProfitPerDay*contractDuration*365/calculationPeriod - contractPrice)

# Results

print("Profit per Day:", estimatedProfitPerDay)

print("Estimated ARR of Contract:", estimatedARR)
# Inputs

priceAtTheTimeOfInvestment = 131; # [$]

initialInvestment          = 55; # [$]

newHashRate                = 5000e6*2; # [Hashes/sec]

newMaintenanceFee          = 0.00001; # [$/MH/day] 

newContractDuration        = 2; # [years]

timeLeftForCurrentContract = 1; # [years]

# Calculations

initialValueOfCurrentContract                = initialInvestment/priceAtTheTimeOfInvestment*currentPrice; # [$]

initialValueOfNewContractAtStart             = 0; # [$]

estimatedProfitPerDayNewContract             = estimatedProfitPerPeriod(currentPrice, calculationPeriod, newHashRate, blockReward, difficulty, newMaintenanceFee);

timeCurrentContract                          = [i for i in range(timeLeftForCurrentContract*365)]; # [days]

timeNewContractAfterCurrentContractExpires   = [i for i in range((newContractDuration - timeLeftForCurrentContract)*365)]; # [days]



profitCurrentContract                        = initialValueOfCurrentContract + estimatedProfitVectorized(currentPrice, timeCurrentContract, hashRate, blockReward, difficulty, maintenanceFee);

profitNewContractCommonTime                  = profitCurrentContract - initialValueOfCurrentContract + initialValueOfNewContractAtStart + estimatedProfitVectorized(currentPrice, timeCurrentContract, newHashRate, blockReward, difficulty, newMaintenanceFee);

profitNewContractAfterCurrentContractExpires = profitNewContractCommonTime[-1] + estimatedProfitVectorized(currentPrice, timeNewContractAfterCurrentContractExpires, newHashRate, blockReward, difficulty, newMaintenanceFee);

profitNewContract                            = [*profitNewContractCommonTime, *profitNewContractAfterCurrentContractExpires]; # [$]

timeNewContract                              = [i for i in range(newContractDuration*365)];

# Comparison of profitability

plt.plot(timeCurrentContract, profitCurrentContract, label='Current Contract')

plt.plot(timeNewContract, profitNewContract, label='New Contract')

plt.title("Comparison of profitability of Contracts", size=14)

plt.xlabel("Number of Days")

plt.ylabel("Profit in $")

plt.legend(loc='lower right')

plt.grid()
