from numpy import random

random.seed(0)



totals = {20:0, 30:0, 40:0, 50:0, 60:0, 70:0}

purchases = {20:0, 30:0, 40:0, 50:0, 60:0, 70:0}

totalPurchases = 0

for _ in range(100000):

    ageDecade = random.choice([20, 30, 40, 50, 60, 70])

    purchaseProbability = float(ageDecade) / 100.0

    totals[ageDecade] += 1

    if (random.random() < purchaseProbability):

        totalPurchases += 1

        purchases[ageDecade] += 1
totals
purchases
totalPurchases
PEF = float(purchases[30]) / float(totals[30])

print('P(purchase | 30s): ' + str(PEF))
PF = float(totals[30]) / 100000.0

print("P(30's): " +  str(PF))
PE = float(totalPurchases) / 100000.0

print("P(Purchase):" + str(PE))
print("P(30's, Purchase)" + str(float(purchases[30]) / 100000.0))
print("P(30's)P(Purchase)" + str(PE * PF))
print((purchases[30] / 100000.0) / PF)