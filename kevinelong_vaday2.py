# We have a change drawer with unlimited change,
# what is the most efficient way to give
# a specific amount of change

DENOMINATIONS = [25, 10, 5, 1]


# def make_change(pennies):
#     output = {}
#     # TODO put code here
#     return output
# ANSWER
def make_change(pennies):
    output = {}
    # TODO put code here
    for d in DENOMINATIONS:
        while pennies >= d:
            pennies -= d
            if d not in output:
                output[d] = 0
            output[d] += 1
    return output


test_cases = [93, 45, 69]

for tc in test_cases:
    print(make_change(tc))

#EXPECTED OUTPUT:

# data = {
#     1: 3,
#     5: 1,
#     10: 1,
#     25: 3,
# }


#ANSWER
# def make_change(pennies):
#     output = {}
#     # TODO put code here
#     for d in DENOMINATIONS:
#         while pennies >= d:
#             pennies -= d
#             if d not in output:
#                 output[d] = 0
#             output[d] += 1
#     return output

# for item in data:
#     last = item["last"]
#     age = int(item["age"])
data = [
    {
        "id": "123",
        "first" : "Kevin",
        "last" : "Long",
        "age" : "53"
    },
    {
        "id": "222",
        "first" : "Nina",
        "last" : "Marie",
        "age" : "42"
    }
]

#SKILLS: Looping, and pulling out data by key

# L1 Loop through print out last names

# L2 bonus print all field formatted with f string
# L3 double bonus print only if age > 50 
#L1
for item in data:
    last = item["last"]
    print(last)
#L2
for item in data:
    first = item['first']
    last = item['last']
    age = item['age']
    print(f"NAME: {first} {last} AGE: {age}")
#L3
for item in data:
    first = item['first']
    last = item['last']
    
    age = int(item['age'])
    
    if age > 50:
        print(f"NAME: {first} {last} AGE: {age}")
name = "KEVIN"
for letter in name:
    print(letter)
name = "ABCabc"
for letter in name:
    number = ord(letter) # ASCII ordinal number
    print(letter, number, bin(number))
address = "127.0.0.1"
parts = address.split(".")
print(parts)
number = int(parts[0])
print(number < 128)
text_list = [
    "Larry",
    "Moe",
    "Curly"
]
print(text_list)
glue = "."
together = glue.join(text_list)
print(together)

# ---

raw_text = "apple orange pear"
print(raw_text)
separator = " "
fruit_list = raw_text.split(separator)
print(fruit_list)

line_glue = "\n"  # new line
document = line_glue.join(fruit_list)

print(document)

data = "abc=123&def=456&hij=789"

# L1 - split on ampersand print the list
# L2 - Loop
# L3 - split on =
# L4 - convert to int
# L5 - output a grand total

# first split on &
# then loop through those
# then split on =
# grab value of the second item [1]
# convert it to int(value)

# how can we use split to access and then add the values together
# list
# split - str.split(sep) return a list (will be called twice)
# sum
# for
def dollars(amount):
    return f"${amount:.2f}"

class CatalogItem:
    def __init__(self, part_number, name, description, price, category = ""):
        self.part_number = part_number
        self.name = name
        self.description = description
        self.price = price
        self.category = category
        
    def __str__(self):
        return f"{self.name} {dollars(self.price)}"
    
cat_item = CatalogItem("123", "Coke", "16oz Bottle", 1.60, "BEVERAGES")

print(cat_item)


class CartItem:
    
    def __init__(self, catalog_item, quantity):
        self.catalog_item = catalog_item
        self.quantity = quantity
        
    def get_subtotal(self):
        return self.quantity * self.catalog_item.price
    
    def __str__(self):
        return f"{self.catalog_item} each - QTY: {self.quantity} SUB: { dollars( self.get_subtotal() ) }"

class Cart:
    def __init__(self):
        self._items:[CartItem] = []
        
    def addItem(self, item):
        self._items.append(item)
        
    def __str__(self):
        output = []
        total = 0
        output.append("\nCART:")
        for item in self._items:
            output.append( "\t" + str(item) )
            total += item.get_subtotal()
        output.append(f"TOTAL: {dollars(total)}")
        return "\n".join(output)
    
cart_item = CartItem(cat_item, 16)
print(cart_item)

cart = Cart()
cart.addItem(CartItem(cat_item, 12))
cart.addItem(CartItem(cat_item, 4))

print(cart)
