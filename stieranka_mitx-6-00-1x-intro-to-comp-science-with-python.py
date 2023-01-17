balance = 10000
annualInterestRate = 0.2
monthlyPaymentRate = annualInterestRate / 12
month = 0

while month < 12:
    month +=1
    minPayment = (balance * monthlyPaymentRate)
    balance = balance - minPayment
    interest = balance * (annualInterestRate / 12)
    balance = balance + interest
print('Remaining balance: ' + "%.2f" % (balance))
# Paste your code into this box
month = 0
balance = 3329
original = balance
annualInterestRate = 0.2
minPayment = 10

month = 0

while balance != 0 and balance > 0 :
    minPayment += 10
    balance = original
    while month < 12:
        month +=1
        balance = balance - minPayment
        interest = balance * (annualInterestRate / 12)
        balance = balance + interest
    month = 0
print(minPayment)
balance = 1400
annualInterestRate = 0.2

monthInterest = annualInterestRate / 12
lowerBound = balance / 12
higherBound = (balance *(1+monthInterest)**12)/12.0
left = balance
month = 0

while balance != 0:     
    i = ((lowerBound + higherBound) / 2)
    balance = left
    while month < 12:
        month +=1
        balance = balance - i
        interest = balance * (annualInterestRate / 12)
        balance = balance + interest
    month = 0
    if balance <= 0:
        higherBound = i
    elif balance >= 0:
        lowerBound = i
    balance = float("%.2f" % balance)
print('Lowest payment: ' + str("%.2f" % i))