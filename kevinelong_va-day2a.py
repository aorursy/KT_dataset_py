def count_change(denomination_quantities):
    total = 0
    
    # TODO put code here
    
    # loop through the keys
    for key_amount in denomination_quantities:
        
        print(key_amount)
        print(f"k={key_amount}")
        
        # use key to get quantity
        quantity = denomination_quantities[key_amount]
        print(f"q={quantity}")
        
        # calc subtotal by mult qty times key
        subtotal = quantity * key_amount
        print(f"s={subtotal}")
        
        #add subtotal to grand total
        total += subtotal
        
        print(f"t={total}")

    return total


denomination_quantities = {
    1: 3,
    5: 1,
    10: 1,
    25: 3,
}
# print(list(denomination_quantities))

# key_amount = 25
# quantity = denomination_quantities[key_amount]
# subtotal = key_amount * quantity
# print(subtotal)

print(count_change(denomination_quantities))
# Expected result: 93
# Try changing the quantities and adding other denominations

# SOLUTION
# for d in denomination_quantities:
#     q = denomination_quantities[d]
#     subtotal = q * d
#     total += subtotal
