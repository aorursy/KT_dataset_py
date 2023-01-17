# browse - product(name, desc, price_each, brand) list - catalog

# select product - add to cart

# remove item
# adjust quantity
# cart
#     cart_items(quantity, subtotal (qty * price_each))
#     grand_total
    
# checkout payment
# pickup or delivery(store address, or customer delivery address)

# NOUNS (people places or things)
# Catalog, CatalogProduct, Cart, CartLineItem, Brand?
#  Catalog,
# VERBS (action)
# Cart.Remove
# Adjust/Update (qty)
# pickup()
#Browse
#get subtotal
#get grantoal

#AJECTIVE specifics
# isFrozen
# quantity
# price

# IS_A, HAS_A

# Catalog HAS_A Product
# Product HAS_A Price
# CartItem HAS_A Product
# CartItem HAS_A Quantiy

class Brand:
    def __init__(self, name, products):
        self.name = name
        self.logo = ""
        self.motto == ""
        self.rating = 0
        self.products = products
        
class House(Brand):
    def __init__(self, products):
        self.name = "Generic"
        self.logo = "blackandwhite.png"
        self.motto = "Cheap but OK"
        self.rating = 2
        self.products = products
           
class Nike(Brand):
    def __init__(self, products):
        self.name = "Nike"
        self.logo = "swoosh.png"
        self.motto = "Just Do it"
        self.rating = 4
        self.products = products

class Category:
    def __init__(self, name, brands):
        self.name = name
        self.brands = brands
        
class Product:
    def __init__(self, name, price = 0):
        self.name = name
        self.price = price
        
class Catalog:
    def __init__(self):
        self.categories = [
            Category("shoes", [Nike([
                Product("Air Jordan", 99.99)
            ]), House([
                Product("Tennis Shoes", 39.99)
            ])])
        ]


class LineItem:
    def __init__(self, product, quantity = 1):
        self.quantity = quantity
        self.product = product
    
    def get_subtotal(self):
            return self.quantity * self.product.price
        
class Cart:
    def __init__(self):
        self.line_items = []
        
    def get_total(self):
        total = 0
        for item in self.line_items:
            total += item.get_subtotal()
        return total

catalog = Catalog()
cart = Cart()

shoes = catalog.categories[0]
nike_shoes = shoes.brands[0]
my_item = nike_shoes.products[0]
cart.line_items.append(LineItem(my_item, 2))
cart.line_items.append(LineItem(catalog.categories[0].brands[1].products[0], 2))

print(cart.get_total())
