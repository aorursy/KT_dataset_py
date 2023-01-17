# Pizza Planet order

# list of prices
price_size = {'Small': 5.00, 'Medium': 10.00, 'Large': 15.00}
price_topping = {'Pepperoni': 1.25, 'Sausage': 2.00, 'Mystery Meat': .50, 'Salty Vegan': .05}
price_cheese = {'Ricotta': .25, 'Mozzarella': .15, 'Fontina': .12}
price_sauce = {'Red Sauce': .5, 'White Sauce': .10, 'Garlic sauce': .12}
# test print
print(price_size)
print(price_topping)
print(price_cheese)
print(price_sauce)

ordering = True
order = list()
order = []
total = 0

while ordering:
    print('Please select your pizza Size!')
    Size = input('Small, Medium, or Large:')
    Size = Size.title()
    print(Size)
    print(Size[0])
    if Size == 'Small':
        total = total + price_size['Small']
        order.append('Small')
        print('Small has been Selected')
    elif Size == 'Medium':
        total = total + price_size['Small']
        order.append('Small')
        print('Medium has been Selected')
        
    else:
        print('error: ' + Size)
