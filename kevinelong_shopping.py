def currency(amount):
    output = f"${amount:.2f}"
    return f"{output:>12}"
print(currency(9))
def currency(amount):
    output = f"${amount:.2f}"
    return f"{output:>12}"

class CatalogItem:
    def __init__(self, name, price_each=0, description=""):
        self.name = name
        self.description = description
        self.price_each = price_each

class Catalog:
    def __init__(self):
        self.catalog_item_list = []

class CartItem:
    
    def __init__(self, quantity, catalog_item):
        self.quantity = quantity
        self.catalog_item = catalog_item
        
    def get_subtotal(self):
        return self.quantity * self.catalog_item.price_each

class Cart:
    def __init__(self):
        self.cart_item_list: [CartItem] = []
    
    def get_total(self):
        cart_total = 0
        for cart_item in self.cart_item_list:
            cart_total += cart_item.get_subtotal()
        return cart_total
    
    def print_line(self, q, n, p, s):
        print(f"{q:<12} {n:<40} {currency(p)} {currency(s)}")

    def print_invoice(self):  #  show content
        # itemize invoice items (custome name, po number, store)
        for item in self.cart_item_list:
            
            self.print_line(
                item.quantity,
                item.catalog_item.name,
                item.catalog_item.price_each,
                item.get_subtotal()
            )
            
        print(self.get_total())

catalog = Catalog()

catalog.catalog_item_list.append(CatalogItem("Cool Widget", 12.34, "Widgets are like gadgets but cooler."))
catalog.catalog_item_list.append(CatalogItem("Pencil", 1.25, "Pencils are like pens but more flexible."))

cart = Cart()

cart.cart_item_list.append(CartItem(3,catalog.catalog_item_list[1]))
cart.cart_item_list.append(CartItem(10,catalog.catalog_item_list[0]))

cart.print_invoice()

# how to add size and color?
# how to display catalog items and their indexes
# how to get user input

# checkout = False

# while checkout is False:
    
#     catalog.display()
    
#     print("Enter item index to add or blank line to quit.")
    
#     text = input()
    
#     if text == "":
#         checkout = true
#     else:
#         index = int(text)
#         catalog_item = catalog.catalog_item_list[index]

#         print("Entery Quantity")
#         quantity = input()
#         q = int(quantity)
#         cart.cart_item_list.append(CartItem(q, catalog_item))
        
#     cart.print_invoice()
    
# print("Thanks for your business!")
# q = 12
# p = 34.90
# d = "widget"



# e = (q * p)

# print( f"{q:<10} {d:40} {currency(p)} {currency(e)}")

# q = 1
# p = 4.00
# d = "awsome gadget"

# print( f"{q:<10} {d:40} {currency(p)} {currency(e)}")
