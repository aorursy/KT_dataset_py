def count_change(denomination_quantities):
    total = 0
    # TODO put code here
    for d in denomination_quantities:
        q = denomination_quantities[d]
        subtotal = q * d
        total += subtotal
    return total


denomination_quantities = {
    1: 3,
    5: 1,
    10: 1,
    25: 3,
}

def currency(pennies):
    a = pennies / 100
    return f"${a:.2f}"

 
print(currency(count_change(denomination_quantities)))
# Exepected result: 93
