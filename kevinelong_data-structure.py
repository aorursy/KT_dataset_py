data = {
    "Name" : "Results",
    "Items": [
        {
            "name" : "Apples",
            "price": 12.33,
            "quantity": 12,
        },
        {
            "name" : "Bananas",
            "price": 1.23,
            "quantity": 16,
        }
    ]
}

content_name = data["Name"]
print(content_name)

item_list = data["Items"]
first_item_in_list = item_list[0]
apple_price = first_item_in_list["price"]
print(apple_price)
print(data["Items"][0]["price"])

# 1. get list from data
# 2. print out each item using for loop on the list
# 3. EXTRA provide subtotals
# 4. EXTRA grand total

print()
invoice_name = data['Name']
print(f"Invoice: {invoice_name.upper()}")
grand_total = 0
for item in data["Items"]:
    p = item["price"]
    q = item["quantity"]
    grand_total += p
    extended_price = p * q
    name = item['name']
    print(f"${q:<8.2f} {name:<20} ${p:>8.2f}     ${extended_price:>8.2f}")
print(f"GRAND TOTAL    ${grand_total:>38.2f}")
grid = [
    [1,2,3],
    [4,5,6],
    [7,8,9],
]

value = grid[1][0]
print(value)

