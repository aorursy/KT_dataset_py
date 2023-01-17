# KEYS ARE COIN DENOMINATIONS 1,5,10,25
data = {
    1: 13,
    5: 3,
    10: 1,
    25: 1,
    50: 0,
    100: 0,
    500: 0,
}

print(data[5]) # print quantity of nickels Expect to print 3

# 1. How much is the value of what we have?
# 1. How much is the value of what we have?
total = 0

for key_denomination in data: #LOOPS THROUGH DICT KEYS
    quantity_value = data[key_denomination] # Use key to get value e.g. quantity
    print(f"qv:{quantity_value}")
    subtotal = quantity_value * key_denomination  # subtotal is this quantity from value times this denomination which is the key
    print(f"st:{subtotal}")
    total += subtotal  # update grand total
    print(f"total:{total}")
#     total += data[key_denomination] * key_denomination #OPTION TO DO ALL THREE LINES ON ONE LINE

print(total)

#  EXPECTED OUTPUT
# 63

# Make change
# unlimited number of coins
#

COINS = [25, 10, 5, 1]


def make_change(goal):
    change_to_give = {}

    # while remaining is greater than zero:
    # go through denominations from largest to smallest

    # while remaining less than denomination
    # add quanity until difference less than denomination
    remaining_pennies = goal
    for coin in COINS:  # visit each denom
        while remaining_pennies >= coin:  # determine if current denom issmaller than pennies remain
            remaining_pennies -= coin  # pennies remaining
            if coin not in change_to_give:
                change_to_give[coin] = 0
            change_to_give[coin] += 1
    return change_to_give


print(make_change(93))

# EXPECTED OUTPUT:
# { 25:3, 10:1, 5:1, 1:3 }

print(make_change(76))
# EXPECTED OUTPUT:
# {25: 3, 1: 1}

print(make_change(200))
# EXPECTED OUTPUT:
# {25: 8}